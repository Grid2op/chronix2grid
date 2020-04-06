"""Class for the economic dispatch framework. Allows to parametrize and run
an economic dispatch based on RES and consumption time series"""

import datetime as dt
import os

import pandas as pd
import plotly.express as px
import pypsa

from chronix2grid.generation.dispatch.utils import RampMode
from chronix2grid.generation.thermal.EDispatch_L2RPN2020.run_economic_dispatch import (
    main_run_disptach)


class Dispatch(pypsa.Network):
    """Wrapper around a pypsa.Network to add higher level methods"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add('Bus', 'node')
        self.add('Load', name='agg_load', bus='node')
        self._env = None  # The grid2op environment when instanciated with from_gri2dop_env
        self._res_load_scenario = None

    @property
    def wind_p(self):
        if self._res_load_scenario is None:
            raise Exception('Cannot access this property before instantiated the Load and'
                            'renewables scenario.')
        return self._res_load_scenario.wind_p

    @property
    def solar_p(self):
        if self._res_load_scenario is None:
            raise Exception('Cannot access this property before instantiated the Load and'
                            'renewables scenario.')
        return self._res_load_scenario.solar_p

    def net_load(self, losses_pct, name):
        if self._res_load_scenario is None:
            raise Exception('Cannot compute net load before instantiated the Load and'
                            'renewables scenario.')
        return self._res_load_scenario.net_load(losses_pct, name)

    @classmethod
    def from_gri2op_env(cls, grid2op_env):
        net = cls()
        net._env = grid2op_env

        carrier_types_to_exclude = ['wind', 'solar']

        for i, generator in enumerate(grid2op_env.name_gen):
            gen_type = grid2op_env.gen_type[i]
            if gen_type not in carrier_types_to_exclude:
                net.add(
                    class_name='Generator', name=generator, bus='node',
                    p_nom=grid2op_env.gen_pmax[i], carrier=grid2op_env.gen_type[i],
                    marginal_cost=grid2op_env.gen_cost_per_MW[i],
                    ramp_limit_up=grid2op_env.gen_max_ramp_up[i] / grid2op_env.gen_pmax[i],
                    ramp_limit_down=grid2op_env.gen_max_ramp_down[i] / grid2op_env.gen_pmax[i],
                )

        return net

    def reset_ramps_from_grid2op_env(self):
        if self._env is None:
            raise Exception('This method can only be applied when Dispatch has been'
                            'instantiated from a grid2op Environment.')
        for i, generator in enumerate(self._env.name_gen):
            if generator in self.generators.index:
                self.generators.loc[generator, 'ramp_limit_up'] = \
                    self._env.gen_max_ramp_up[i] / self._env.gen_pmax[i]
                self.generators.loc[generator, 'ramp_limit_down'] = \
                    self._env.gen_max_ramp_down[i] / self._env.gen_pmax[i]

    def read_hydro_guide_curves(self, hydro_file_path):
        dateparse = lambda x: dt.datetime.strptime(x, '%d/%m/%Y %H:%M')
        hydro_pattern = pd.read_csv(hydro_file_path, usecols=[0, 2, 3],
                                    parse_dates=[0], date_parser=dateparse)
        hydro_pattern.set_index(hydro_pattern.columns[0], inplace=True)
        hydro_names = self.generators[self.generators.carrier == 'hydro'].index

        for extremum in ['min', 'max']:
            hydro_pu = hydro_pattern[[f'p_{extremum}_u'] * len(hydro_names)]
            hydro_pu.columns = hydro_names
            hydro_pu.index = hydro_pu.index.map(
                lambda x: (x.month, x.day, x.hour, x.minute, x.second))
            setattr(self, f'_{extremum}_hydro_pu', hydro_pu)

        self._hydro_file_path = hydro_file_path

    def read_load_and_res_scenario(self, load_path_file, prod_path_file, scenario_name):
        if self._env is None:
            raise Exception('This method can only be applied when Dispatch has been'
                            'instantiated from a grid2op Environment.')
        res_names = dict(
            wind=[name for i, name in enumerate(self._env.name_gen)
                  if self._env.gen_type[i] == 'wind'],
            solar=[name for i, name in enumerate(self._env.name_gen)
                   if self._env.gen_type[i] == 'solar']
        )
        self._res_load_scenario = ChroniXScenario(load_path_file, prod_path_file,
                                                  res_names, scenario_name=scenario_name)

    def make_hydro_constraints_from_res_load_scenario(self):
        if self._res_load_scenario is None or self._hydro_file_path is None:
            raise Exception('This method can only be applied when a Scenario for load'
                            'and renewables has been instantiated and hydro guide'
                            'curves have been read.')
        index_slice = self._res_load_scenario.loads.index.map(
           lambda x: (x.month, x.day, x.hour, x.minute, x.second)
        )
        self._min_hydro_pu = self._min_hydro_pu.loc[index_slice, :]
        self._max_hydro_pu = self._max_hydro_pu.loc[index_slice, :]

    def modify_marginal_costs(self, new_costs):
        """
        Modify marginal costs used for the dispatch given a dictionary
        providing new costs for carriers.

        Parameters
        ----------
        new_costs: dict
            new costs by carrier

        """
        for carrier, new_cost in new_costs.items():
            try:
                targeted_generators = self.generators.carrier.isin([carrier])
                self.generators.loc[targeted_generators, 'marginal_cost'] = new_cost
            except KeyError:
                print(f'Carrier {carrier} is not a valid carrier')
                raise

    def plot_ramps(self):
        caract_gen = self.generators[['p_nom', 'carrier', 'ramp_limit_up']].reset_index()
        caract_gen = caract_gen.rename(columns={'index': 'name'})

        fig = px.scatter(
            caract_gen, x='p_nom', y='ramp_limit_up', color='carrier',
            hover_data=['name']
        )
        return fig

    def simplify_net(self):
        carriers = self.generators.carrier.unique()
        simplified_net = Dispatch()
        for carrier in carriers:
            names = self.generators[self.generators.carrier == carrier].index.tolist()

            gens = self.generators.loc[
                names,
                ['p_nom', 'ramp_limit_up', 'ramp_limit_down', 'marginal_cost']
            ]
            gens['ramp_up_mw'] = gens['p_nom'] * gens['ramp_limit_up']
            gens['ramp_down_mw'] = gens['p_nom'] * gens['ramp_limit_down']
            params = gens.agg(['sum', 'mean'])
            simplified_net.add(
                class_name='Generator', name=carrier, bus='node',
                p_nom=params.loc['sum', 'p_nom'], carrier=carrier,
                marginal_cost=params.loc['mean', 'marginal_cost'],
                ramp_limit_up=params.loc['sum', 'ramp_up_mw'] / params.loc['sum', 'p_nom'],
                ramp_limit_down=params.loc['sum', 'ramp_down_mw'] / params.loc['sum', 'p_nom'],
            )
        simplified_net._hydro_file_path = self._hydro_file_path
        simplified_net._min_hydro_pu = self._min_hydro_pu.iloc[:, 0]
        simplified_net._max_hydro_pu = self._max_hydro_pu.iloc[:, 0]

        print('simplified dispatch by carrier')
        full_ramp = simplified_net.generators['p_nom'] * simplified_net.generators['ramp_limit_up']
        df_full_ramp = pd.DataFrame({'full_ramp': full_ramp})

        print(pd.concat(
            [
                simplified_net.generators[['p_nom', 'ramp_limit_up',
                                           'ramp_limit_down', 'marginal_cost']],
                df_full_ramp
            ], axis=1))
        return simplified_net

    def run(self, load, params, gen_constraints=None,
                     ramp_mode=RampMode.hard, by_carrier=False):
        prods_dispatch, terminal_conditions = main_run_disptach(
            self if not by_carrier else self.simplify_net(),
            load, params, gen_constraints, ramp_mode)
        self._res_load_scenario.prods_dispatch = prods_dispatch
        self.reset_ramps_from_grid2op_env()
        return self._res_load_scenario, terminal_conditions

    def save_results(self, parent_folder_path):
        full_opf_dispatch = pd.concat(
            [self._res_load_scenario.prods_dispatch, self._res_load_scenario.wind_p,
             self._res_load_scenario.solar_p],
            axis=1
        )
        try:
            full_opf_dispatch = full_opf_dispatch[self._env.name_gen].round(2)
        except KeyError:
            # Either we're trying to save results from a simplified dispatch or
            # using the save function before instanciating an env.
            pass
        full_opf_dispatch.to_csv(
            os.path.join(parent_folder_path, self._res_load_scenario.name,
                         "prod_p.csv.bz2"),
            sep=';', index=False
        )


class ChroniXScenario:
    def __init__(self, load_path_file, prod_path_file, res_names, scenario_name):
        self.loads = pd.read_csv(load_path_file, sep=';', index_col=0, parse_dates=True)
        prods = pd.read_csv(prod_path_file, sep=';', index_col=0, parse_dates=True)
        self.wind_p = prods[res_names['wind']]
        self.solar_p = prods[res_names['solar']]
        self.total_res = pd.concat([self.wind_p, self.solar_p], axis=1).sum(axis=1)
        self.prods_dispatch = None  # Will receive the results of the dispatch
        self.name = scenario_name

    def net_load(self, losses_pct, name):
        load_minus_losses = self.loads.sum(axis=1) * (1 + losses_pct/100)
        return (load_minus_losses - self.total_res).to_frame(name=name)


if __name__ == "__main__":
    import os
    import grid2op
    from grid2op.ChronicsHandler import ChangeNothing
    import chronix2grid.generation.thermal.EDispatch_L2RPN2020.run_economic_dispatch as run_economic_dispatch

    INPUT_FOLDER = 'chronix2grid/generation/input'
    CASE = 'case118_l2rpn'
    path_grid = os.path.join(INPUT_FOLDER, CASE)

    losses_pct = 3.0

    env118_blank = grid2op.make(
        "blank",
        grid_path=os.path.join(path_grid, "L2RPN_2020_case118_redesigned.json"),
        chronics_class=ChangeNothing,
    )
    params = {'snapshots': [],
              'step_opf_min': 5,
              'mode_opf': 'day',
              'reactive_comp': 1.025,
              }
    chronics_path_gen = os.path.join(INPUT_FOLDER, "dispatch", str(2012))
    this_path = os.path.join(chronics_path_gen, 'Scenario_0')
    dispatch = Dispatch.from_gri2op_env(env118_blank)
    dispatch.read_hydro_guide_curves(os.path.join(INPUT_FOLDER, 'patterns', 'hydro.csv'))
    dispatch.read_load_and_res_scenario(os.path.join(this_path, 'load_p.csv.bz2'),
                                        os.path.join(this_path, 'prod_p.csv.bz2'))
    dispatch.make_hydro_constraints_from_res_load_scenario()
    net_by_carrier = dispatch.simplify_net()
    agg_load_without_renew = dispatch.net_load(losses_pct, name=dispatch.loads.index[0])

    # Prepare gen constraints for EDispatch module
    hydro_constraints = {'p_max_pu': dispatch._max_hydro_pu.copy(),
                         'p_min_pu': dispatch._min_hydro_pu.copy()}

    opf_dispatch, term_conditions = dispatch.run(
        agg_load_without_renew,
        params=params,
        gen_constraints=hydro_constraints,
        ramp_mode=run_economic_dispatch.RampMode.easy,
        by_carrier=False  # True to run the dispatch only aggregated generators by carrier
    )



