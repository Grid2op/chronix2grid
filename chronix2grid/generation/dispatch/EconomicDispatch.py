# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

"""Class for the economic dispatch framework. Allows to parametrize and run
an economic dispatch based on RES and consumption time series"""

import copy
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import datetime as dt
from collections import namedtuple
import pathlib
from numpy.random import default_rng

import grid2op
from grid2op.Chronics import ChangeNothing
import pandas as pd
import plotly.express as px

from chronix2grid.generation.dispatch.utils import RampMode, add_noise_gen, modify_hydro_ramps, modify_slack_characs
import chronix2grid.constants as cst

DispatchResults = namedtuple('DispatchResults', ['chronix', 'terminal_conditions'])

def init_dispatcher_from_config(env_path, input_folder, dispatcher_class, params_opf):
    # Read grid and gens characs
    env118_withoutchron = grid2op.make(env_path,
                                       test=True,
                                       chronics_class=ChangeNothing)
    # grid_path_parent = pathlib.Path(grid_path).parent.absolute()
    # env118_withoutchron = grid2op.make(str(grid_path_parent),
    #                                 chronics_path=str(grid_path_parent),
    #                                 chronics_class=ChangeNothing)
    # env118_withoutchron = grid2op.make(grid_path)

    # Dispatcher object init
    dispatcher = dispatcher_class.from_gri2op_env(env118_withoutchron)

    dispatcher.read_hydro_guide_curves(
        os.path.join(input_folder, 'patterns', 'hydro_french.csv'))
    return dispatcher

def init_dispatcher_from_config_dataframe(grid_path, input_folder, dispatcher_class, params_opf):
    # Read grid and gens characs
    # env118_withoutchron = grid2op.make("rte_case118_example",
    #                                    test=True,
    #                                    grid_path=grid_path,
    #                                    chronics_class=ChangeNothing)

    # Put infos in dataframe
    # env_df = pd.DataFrame({'name':env118_withoutchron.name_gen,
    #                        'type':env118_withoutchron.gen_type,
    #                        'pmax':env118_withoutchron.gen_pmax,
    #                        'max_ramp_up':env118_withoutchron.gen_max_ramp_up,
    #                        'max_ramp_down':env118_withoutchron.gen_max_ramp_down,
    #                        'cost_per_mw':env118_withoutchron.gen_cost_per_MW})

    # Reading characs from file directly avoids some strange problem with grid2op when launching Chronix2grid several times in the same session
    prods_charac = pd.read_csv(os.path.join(pathlib.Path(grid_path).parent.absolute(),"prods_charac.csv"), decimal = '.')
    env_df = prods_charac.rename(columns = {"Pmax":'pmax',
                                            "marginal_cost":"cost_per_mw"})[["name","type","pmax","max_ramp_up",
                                                                             "max_ramp_down","cost_per_mw"]]

    # Generators temporary adjusts
    env_df = modify_hydro_ramps(env_df, params_opf["hydro_ramp_reduction_factor"])

    if params_opf["slack_p_max_reduction"] != 0. or params_opf['slack_ramp_max_reduction'] != 0.:
        env_df = modify_slack_characs(env_df,
                                                 params_opf["nameSlack"],
                                                 params_opf["slack_p_max_reduction"],
                                                 params_opf["slack_ramp_max_reduction"])

    # Dispatcher object init
    dispatcher = dispatcher_class.from_dataframe(env_df)

    dispatcher.read_hydro_guide_curves(
        os.path.join(input_folder, 'patterns', 'hydro_french.csv'))
    return dispatcher



class Dispatcher(ABC):
    """Abstract Class Dispatcher with some operational method that don't depend on the dispatcher technology (pypsa...)
    and some others that are implemented specifically in inheriting classes"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, load, params, gen_constraints=None,
            ramp_mode=RampMode.hard, by_carrier=False, **kwargs):
        """Run the proper dispatch optimization. Have to be implemented in inheriting classes"""

    @abstractmethod
    def simplify_net(self):
        pass

    @classmethod
    @abstractmethod
    def from_gri2op_env(cls, grid2op_env):
        """Reads grid features from a grid2op environment into a specific object.
        Have to be implemented in inheriting classes according to the type of model"""

    @classmethod
    @abstractmethod
    def from_dataframe(cls, env_df):
        """Reads grid features from a pandas DataFrame into a specific object.
        Have to be implemented in inheriting classes according to the type of model"""

    @property
    def wind_p(self):
        if self._chronix_scenario is None:
            raise Exception('Cannot access this property before instantiated the Load and '
                            'renewables scenario.')
        return self._chronix_scenario.wind_p

    @property
    def solar_p(self):
        if self._chronix_scenario is None:
            raise Exception('Cannot access this property before instantiated the Load and '
                            'renewables scenario.')
        return self._chronix_scenario.solar_p

    @property
    def chronix_scenario(self):
        """

        Returns
        -------
        chronix_scenario: :class:`chronix2grid.generation.dispatch.EconomicDispatch.ChroniXScenario`
            Object that contains all the chronics that are relevant to perform a dispatch
        """
        if self._chronix_scenario is None:
            raise Exception('Cannot access this property before instantiated the Load and '
                            'renewables scenario.')
        return self._chronix_scenario

    @chronix_scenario.setter
    def chronix_scenario(self, chronix_scenario):
        if not isinstance(chronix_scenario, ChroniXScenario):
            raise Exception('The chronix_scenario argument should be an instance of '
                            'ChronixScenario.')
        self._chronix_scenario = chronix_scenario

    def net_load(self, losses_pct, name,include_renewable=True):
        if self._chronix_scenario is None:
            raise Exception('Cannot compute net load before instantiated the Load and'
                            'renewables scenario.')
        return self._chronix_scenario.net_load(losses_pct, name)

    def nlargest_ramps(self, n, losses_pct):
        ramps = self.net_load(losses_pct, "").diff()
        return ramps.nlargest(n, ramps.columns[0])

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

    def reset_ramps_from_dataframe(self):
        if self._df is None:
            raise Exception('This method can only be applied when Dispatch has been'
                            'instantiated from a pandas DataFrame.')
        for i, (generator,pmax, rampup, rampdown) in enumerate(zip(self._df['name'],
                                                                   self._df['pmax'],
                                                                   self._df['max_ramp_up'],
                                                                   self._df['max_ramp_down'])):
            if generator in self.generators.index:
                self.generators.loc[generator, 'ramp_limit_up'] = \
                    rampup / pmax
                self.generators.loc[generator, 'ramp_limit_down'] = \
                    rampdown / pmax

    def read_hydro_guide_curves(self, hydro_file_path):
        """
        Reads realistic hydro pattern that provides seasonal boundaries to the hydro production.
        This constraint in the dispatch problem leads to more realistic hydro production

        Parameters
        ----------
        hydro_file_path: ``str``

        """
        dateparse = lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M')
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

    def read_load_and_res_scenario(self, load_path_file, prod_path_file,
                                   scenario_name, start_date, end_date, dt):
        if self._env is None:
            raise Exception('This method can only be applied when Dispatch has been'
                            'instantiated from a grid2op Environment.')
        res_names = dict(
            wind=[name for i, name in enumerate(self._env.name_gen)
                  if self._env.gen_type[i] == 'wind'],
            solar=[name for i, name in enumerate(self._env.name_gen)
                   if self._env.gen_type[i] == 'solar']
        )
        self._chronix_scenario = ChroniXScenario.from_disk(
            load_path_file, prod_path_file,
            res_names, scenario_name=scenario_name,
            start_date=start_date, end_date=end_date, dt=dt
        )


    def make_hydro_constraints_from_res_load_scenario(self):
        if self._chronix_scenario is None or self._hydro_file_path is None:
            raise Exception('This method can only be applied when a Scenario for load'
                            'and renewables has been instantiated and hydro guide'
                            'curves have been read.')
        index_slice = self._chronix_scenario.loads.index.map(
            lambda x: (x.month, x.day, x.hour, x.minute, x.second)
        )

        self._min_hydro_pu = self._min_hydro_pu.reindex(index_slice).fillna(method='ffill')
        self._max_hydro_pu = self._max_hydro_pu.reindex(index_slice).fillna(method='ffill')

        return {'p_max_pu': self._max_hydro_pu.copy(), 'p_min_pu': self._min_hydro_pu.copy()}

    def modify_marginal_costs(self, new_costs):
        for carrier, new_cost in new_costs.items():
            try:
                targeted_generators = self.generators.carrier.isin([carrier])
                self.generators.loc[targeted_generators, 'marginal_cost'] = new_cost
            except KeyError:
                print(f'Carrier {carrier} is not a valid carrier')
                raise

    def plot_ramps(self):
        caract_gen = self.generators[['p_nom', 'carrier', 'ramp_limit_up']]
        caract_gen = caract_gen.rename(columns={'index': 'name'})
        caract_gen = caract_gen.reset_index()
        try:
            # column "name" might have change of label depending on the versions
            fig = px.scatter(
                caract_gen, x='p_nom', y='ramp_limit_up', color='carrier',
                hover_data=['name']
            )
        except ValueError:
            fig = px.scatter(
                caract_gen, x='p_nom', y='ramp_limit_up', color='carrier',
                hover_data=['Generator']
            )
            
        return fig

    def save_results(self, params, output_folder, prng=None):
        """
        Saves dispatch results in prod_p.csv.bz2, prod_p_forecasted.csv.bz2, load_p.csv.bz2, prices.csv

        Parameters
        ----------
        params: ``dict``
        output_folder: ``str``

        """
        if prng is None:
            prng = default_rng()
        if not self._has_results and not self._has_simplified_results:
            print('The optimization has first to run successfully in order to '
                  'save results.')
        if self._has_results:
            print('Saving results for the grids with individual generators...')
            res_load_scenario = self.chronix_scenario
        else:
            print('Saving results for the grids with aggregated generators by carriers...')
            res_load_scenario = self._simplified_chronix_scenario

        path_metadata_failed = os.path.join(output_folder, "DISPATCH_FAILED")
        if res_load_scenario is None:
            # the backend failed to find a solution
            print('ERROR: the backend failed to find a consistent state. Nothing is saved.')
            with open(path_metadata_failed, "w", encoding="utf-8") as f:
                f.write("The dispatch has failed. We cannot do anything.")
            return
        
        # this did not failed, so I remove it
        if os.path.exists(path_metadata_failed):
            os.remove(path_metadata_failed)
        
        wind_curtail_coeff = 1.0
        solar_curtail_coeff = 1.0
        
        # TODO perf do not recompute the res_load_scenario.wind_p.sum(axis=1)
        if "agg_wind" in res_load_scenario.prods_dispatch:
            wind_curtail_coeff = res_load_scenario.prods_dispatch["agg_wind"] / res_load_scenario.wind_p.sum(axis=1)
            wind_curtail_coeff = wind_curtail_coeff.values.reshape(-1, 1)
        if "agg_solar" in res_load_scenario.prods_dispatch:
            solar_curtail_coeff = res_load_scenario.prods_dispatch["agg_solar"] / res_load_scenario.solar_p.sum(axis=1)
            solar_curtail_coeff = solar_curtail_coeff.values.reshape(-1, 1)
        
        new_wind_after_curtail = res_load_scenario.wind_p * wind_curtail_coeff
        new_solar_after_curtail = res_load_scenario.solar_p * solar_curtail_coeff
        
        full_opf_dispatch = pd.concat(
            [res_load_scenario.prods_dispatch,
             new_wind_after_curtail,
             new_solar_after_curtail
            ],
            axis=1
        )
        
        diff_wind = (res_load_scenario.wind_p - new_wind_after_curtail).sum(axis=1).values
        total_curt = diff_wind.sum()
        total_wind = res_load_scenario.wind_p.sum(axis=1).sum()
        print(f"INFO: wind curtailment max: {diff_wind.max():.2f}MW")
        print(f"INFO: wind curtailment min: {diff_wind.min():.2f}MW")
        print(f"INFO: wind curtailment sum: {total_curt / 12.:.2f}MWh (total {total_wind / 12.:.2f}MWh: {100. * total_curt / total_wind:.2f}%)")
        diff_solar = (res_load_scenario.solar_p - new_solar_after_curtail).sum(axis=1).values
        total_curt = diff_solar.sum()
        total_solar = res_load_scenario.solar_p.sum(axis=1).sum()
        print(f"INFO: solar curtailment max: {diff_solar.max():.2f}MW")
        print(f"INFO: solar curtailment min: {diff_solar.min():.2f}MW")
        print(f"INFO: solar curtailment sum: {total_curt / 12.:.2f}MWh (total { total_curt / total_solar:.2f}MWh: {100. * total_curt / total_solar :.2f}%)")
        
        try:
            if self._env is not None:
                full_opf_dispatch = full_opf_dispatch[self._env.name_gen].round(2)
            else:
                full_opf_dispatch = full_opf_dispatch[self._df['name'].values].round(2)
        except KeyError:
            # Either we're trying to save results from a simplified dispatch or
            # using the save function before instanciating an env.
            pass
        if self._env is not None:
            gen_cap = pd.Series({gen_name: gen_pmax for gen_name, gen_pmax in
                                 zip(self._env.name_gen, self._env.gen_pmax)})
        else:
            gen_cap = pd.Series({gen_name: gen_pmax for gen_name, gen_pmax in
                                 zip(self._df['name'], self._df['pmax'])})

        prod_p_forecasted_with_noise = add_noise_gen(prng,
                                                     copy.deepcopy(full_opf_dispatch),
                                                     gen_cap,
                                                     noise_factor=params['planned_std'])

        # prod_p_forecasted_with_noise.to_csv(
        prod_p_forecasted_with_noise.to_csv(
            os.path.join(output_folder, "prod_p_forecasted.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )
        # prod_p_with_noise.to_csv(
        full_opf_dispatch.to_csv(
            os.path.join(output_folder, "prod_p.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )
        res_load_scenario.marginal_prices.to_csv(
            os.path.join(output_folder, "prices.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )
        res_load_scenario.loads.to_csv(
            os.path.join(output_folder, "load_p.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )
        # save the origin time series        
        pd.concat([res_load_scenario.wind_p, res_load_scenario.solar_p], axis=1).to_csv(
            os.path.join(output_folder, "prod_p_renew_orig.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )

class ChroniXScenario:
    def __init__(self, loads, prods, res_names, scenario_name, loss=None):
        self.loads = loads
        self.wind_p = prods[res_names['wind']]
        self.solar_p = prods[res_names['solar']]
        self.total_res = pd.concat([self.wind_p, self.solar_p], axis=1).sum(axis=1)
        self.prods_dispatch = None  # Will receive the results of the dispatch
        self.marginal_prices = None  # Will receive the marginal prices associated to a dispatch
        self.name = scenario_name
        self.loss = loss

    @classmethod
    def from_disk(cls, load_path_file, prod_path_file, res_names, scenario_name,
                  start_date, end_date, dt, loss_path_file=None):
        loads = pd.read_csv(load_path_file, sep=';')
        prods = pd.read_csv(prod_path_file, sep=';')
        if loss_path_file is not None:
            loss = pd.read_csv(loss_path_file, sep=';')
        else:
            loss = None
        datetime_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq=str(dt) + 'min')
        loads.index = datetime_index[:len(loads)]
        prods.index = datetime_index[:len(prods)]
        return cls(loads, prods, res_names, scenario_name, loss)

    def net_load(self, losses_pct, name,include_renewable=True):
        if self.loss is None:
            load_minus_losses = self.loads.sum(axis=1) * (1 + losses_pct / 100)
        else:
            load_minus_losses = self.loads.sum(axis=1) + self.loss
        if include_renewable:
            return (load_minus_losses - self.total_res).to_frame(name=name)
        else:
            return (load_minus_losses).to_frame(name=name)

    def simplify_chronix(self):
        simplified_chronix = deepcopy(self)
        simplified_chronix.wind_p = simplified_chronix.wind_p.sum(axis=1).to_frame(name='wind')
        simplified_chronix.solar_p = simplified_chronix.solar_p.sum(axis=1).to_frame(name='solar')
        simplified_chronix.name = simplified_chronix.name + 'by_carrier'
        return simplified_chronix


# if __name__ == "__main__":
#
#     INPUT_FOLDER = 'chronix2grid/generation/input'
#     CASE = 'case118_l2rpn'
#     path_grid = os.path.join(INPUT_FOLDER, CASE)
#
#     losses_pct = 3.0
#
#     env118_blank = grid2op.make(
#         test=True,
#         grid_path=os.path.join(path_grid, "L2RPN_2020_case118_redesigned.json"),
#         chronics_class=ChangeNothing,
#     )
#     params = {'snapshots': [],
#               'step_opf_min': 5,
#               'mode_opf': 'week',
#               'reactive_comp': 1.025,
#               }
#     chronics_path_gen = os.path.join(INPUT_FOLDER, "dispatch", str(2012))
#     this_path = os.path.join(chronics_path_gen, 'Scenario_0')
#     dispatch = Dispatcher.from_gri2op_env(env118_blank)
#     dispatch.read_hydro_guide_curves(os.path.join(INPUT_FOLDER, 'patterns', 'hydro.csv'))
#     dispatch.read_load_and_res_scenario(os.path.join(this_path, 'load_p.csv.bz2'),
#                                         os.path.join(this_path, 'prod_p.csv.bz2'),
#                                         'Scenario_0')
#     dispatch.make_hydro_constraints_from_res_load_scenario()
#     net_by_carrier = dispatch.simplify_net()
#     agg_load_without_renew = dispatch.net_load(losses_pct, name=dispatch.loads.index[0])
#
#     # Prepare gen constraints for EDispatch module
#     hydro_constraints = {'p_max_pu': dispatch._max_hydro_pu.copy(),
#                          'p_min_pu': dispatch._min_hydro_pu.copy()}
#
#     opf_dispatch, term_conditions = dispatch.run(
#         agg_load_without_renew,
#         params=params,
#         gen_constraints=hydro_constraints,
#         ramp_mode=run_economic_dispatch.RampMode.easy,
#         by_carrier=False  # True to run the dispatch only aggregated generators by carrier
#     )
#
#     dispatch.save_results(params, '.')
#     test_prods = pd.read_csv('./Scenario_0/prod_p.csv.bz2', sep=";")
#     test_prices = pd.read_csv('./Scenario_0/prices.csv.bz2', sep=";")
