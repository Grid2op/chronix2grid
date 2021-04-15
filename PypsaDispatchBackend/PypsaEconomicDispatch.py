"""Class for the economic dispatch framework. Allows to parametrize and run
an economic dispatch based on RES and consumption time series"""

from collections import namedtuple
import os

import grid2op
from grid2op.Chronics import ChangeNothing
import pandas as pd
import pypsa

from .EDispatch_L2RPN2020.run_economic_dispatch import main_run_disptach

## Dépendances à Chronix2Grid
from chronix2grid.generation.dispatch.EconomicDispatch import Dispatcher
from chronix2grid.generation.dispatch.utils import RampMode

DispatchResults = namedtuple('DispatchResults', ['chronix', 'terminal_conditions'])


class PypsaDispatcher(Dispatcher, pypsa.Network):
    """
    Inheriting from Dispatcher to implement abstract methods thanks to Pypsa API
    Wrapper around a pypsa.Network to add higher level methods
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add('Bus', 'node')
        self.add('Load', name='agg_load', bus='node')
        self._env = None  # The grid2op environment when instanciated with from_gri2dop_env
        self._df = None
        self._chronix_scenario = None
        self._simplified_chronix_scenario = None
        self._has_results = False
        self._has_simplified_results = False

    @classmethod
    def from_gri2op_env(cls, grid2op_env):
        """
        Implements the abstract method of *Dispatcher*

        Parameters
        ----------
        grid2op_env

        Returns
        -------
        net: :class:`pypsa.Network`
        """
        net = cls()
        net._env = grid2op_env

        carrier_types_to_exclude = ['wind', 'solar']

        # PATCH
        # to avoid problems for respecting pmax and ramps when rounding production values in chronics at the end, we apply a correcting factor
        PmaxCorrectingFactor = 1
        RampCorrectingFactor = 0.1

        for i, generator in enumerate(grid2op_env.name_gen):
            gen_type = grid2op_env.gen_type[i]
            if gen_type not in carrier_types_to_exclude:
                p_max = grid2op_env.gen_pmax[i]
                pnom = p_max - PmaxCorrectingFactor
                rampUp = (grid2op_env.gen_max_ramp_up[i] - RampCorrectingFactor) / p_max
                RampDown = (grid2op_env.gen_max_ramp_down[i] - RampCorrectingFactor) / p_max

                net.add(
                    class_name='Generator', name=generator, bus='node',
                    p_nom=pnom, carrier=grid2op_env.gen_type[i],
                    marginal_cost=grid2op_env.gen_cost_per_MW[i],
                    ramp_limit_up=rampUp,
                    ramp_limit_down=RampDown,
                )

        return net

    @classmethod
    def from_dataframe(cls, env_df):
        """
        Implements the abstract method of *Dispatcher*

        Parameters
        ----------
        grid2op_env

        Returns
        -------
        net: :class:`pypsa.Network`
        """
        net = cls()
        net._df = env_df

        carrier_types_to_exclude = ['wind', 'solar']

        # PATCH
        # to avoid problems for respecting pmax and ramps when rounding production values in chronics at the end, we apply a correcting factor
        PmaxCorrectingFactor = 1
        RampCorrectingFactor = 0.1

        for i, (generator, gen_type, p_max, ramp_up, ramp_down, gen_cost_per_MW) in enumerate(zip(env_df['name'],
                                            env_df['type'],
                                            env_df['pmax'],
                                            env_df['max_ramp_up'],
                                            env_df['max_ramp_down'],
                                            env_df['cost_per_mw'])):
            if gen_type not in carrier_types_to_exclude:
                pnom = p_max - PmaxCorrectingFactor
                rampUp = (ramp_up- RampCorrectingFactor) / p_max
                RampDown = (ramp_down - RampCorrectingFactor) / p_max

                net.add(
                    class_name='Generator', name=generator, bus='node',
                    p_nom=pnom, carrier=gen_type,
                    marginal_cost=gen_cost_per_MW,
                    ramp_limit_up=rampUp,
                    ramp_limit_down=RampDown,
                )

        return net


    def run(self, load, params, gen_constraints=None,
            ramp_mode=RampMode.hard, by_carrier=False, **kwargs):
        """
        Implements the abstract method of *Dispatcher*

        Returns
        -------
        simplified_net: :class:`pypsa.Network`
        """
        prods_dispatch, terminal_conditions, marginal_prices = main_run_disptach(
            self if not by_carrier else self.simplify_net(),
            load, params, gen_constraints, ramp_mode, **kwargs)
        if by_carrier:
            self._simplified_chronix_scenario = self._chronix_scenario.simplify_chronix()
            self._simplified_chronix_scenario.prods_dispatch = prods_dispatch
            self._simplified_chronix_scenario.marginal_prices = marginal_prices
            results = self._simplified_chronix_scenario
            self._has_simplified_results = True
            self._has_results = False
        else:
            self._chronix_scenario.prods_dispatch = prods_dispatch
            self._chronix_scenario.marginal_prices = marginal_prices
            results = self._chronix_scenario
            self._has_results = True
            self._has_simplified_results = False
        if self._env is None:
            self.reset_ramps_from_dataframe()
        else:
            self.reset_ramps_from_grid2op_env()

        return DispatchResults(chronix=results, terminal_conditions=terminal_conditions)

    def simplify_net(self):
        """
        Implements the abstract method of *Dispatcher*
        """
        carriers = self.generators.carrier.unique()
        simplified_net = PypsaDispatcher()
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


