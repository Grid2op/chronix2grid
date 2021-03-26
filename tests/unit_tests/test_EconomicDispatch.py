import datetime as dt
import os
import tempfile
import unittest

import pandas as pd
import pathlib

import chronix2grid.constants as cst
from chronix2grid.generation.dispatch.EconomicDispatch import (
            ChroniXScenario, init_dispatcher_from_config)

import grid2op
from grid2op.Chronics import ChangeNothing


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.grid_path = ''
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.CASE = 'case118_l2rpn_wcci'
        self.year = 2012
        self.scenario_name = 'Scenario_0'
        self.grid_path = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME,
                               self.CASE, 'grid.json')
        self.dispatcher = init_dispatcher_from_config(
            self.grid_path,
            os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME),
            cst.DISPATCHER
        )
        self.hydro_file_path = os.path.join(self.input_folder,
                                            cst.GENERATION_FOLDER_NAME,
                                            'patterns',
                                            'hydro_french.csv')

    def test_from_grid2op_env(self):
        grid2op_env = grid2op.make("rte_case118_example",
                                   test=True,
                                   grid_path=self.grid_path,
                                   chronics_class=ChangeNothing)
        dispatcher = cst.DISPATCHER.from_gri2op_env(grid2op_env)
        self.assertTrue(isinstance(dispatcher, cst.DISPATCHER))

    def test_read_hydro_guide_curves(self):
        self.dispatcher.read_hydro_guide_curves(self.hydro_file_path)
        self.assertAlmostEqual(self.dispatcher._max_hydro_pu.iloc[0, 0],
                               0.482099426, places=5)


class TestChronixScenario(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.case = 'case118_l2rpn_wcci'
        self.grid2op_env = grid2op.make(
            "rte_case118_example",
            test=True,
            grid_path=os.path.join(self.input_folder,
                                   cst.GENERATION_FOLDER_NAME,
                                   self.case,
                                   "grid.json"),
            chronics_class=ChangeNothing,
        )
        self.start_date = dt.datetime(2019, 9, 1, 0, 0)
        self.end_date = dt.datetime(2019, 9, 1, 0, 5)
        self.dt = 5

        self.chronics_path_gen = tempfile.mkdtemp()
        self.loads = pd.DataFrame(
            index=[self.start_date, self.end_date],
            columns=['load_1', 'load_2'],
            data=[[9, 10], [11, 12]]
        )
        self.prods = pd.DataFrame(
            index=[self.start_date, self.end_date],
            columns=['gen_1', 'gen_2', 'gen_3', 'gen_4'],
            data=[[1, 2, 3, 4], [5, 6, 7, 8]]
        )
        self.loads.to_csv(
            os.path.join(self.chronics_path_gen, 'load_p.csv.bz2'), sep=';', index=False)
        self.prods.to_csv(
            os.path.join(self.chronics_path_gen, 'prod_p.csv.bz2'), sep=';', index=False)
        self.res_names = dict(wind=['gen_1'], solar=['gen_2', 'gen_4'])
        self.chronix_scenario = ChroniXScenario(
            self.loads, self.prods, self.res_names, 'scen'
        )


    def test_instanciation(self):
        self.assertEqual(self.chronix_scenario.total_res.iloc[0], 7)
        self.assertEqual(self.chronix_scenario.wind_p.columns, ['gen_1'])

    def test_from_disk(self):
        chronix_scenario = ChroniXScenario.from_disk(
            os.path.join(self.chronics_path_gen, 'load_p.csv.bz2'),
            os.path.join(self.chronics_path_gen, 'prod_p.csv.bz2'),
            self.res_names, 'scen', self.start_date,
            self.end_date, self.dt
        )
        self.assertEqual(chronix_scenario.total_res.iloc[0], 7)
        self.assertEqual(chronix_scenario.wind_p.columns, ['gen_1'])

    def test_net_load(self):
        net_load = self.chronix_scenario.net_load(50., 'net_load')
        self.assertEqual(float(net_load.iloc[0]), 21.5)

    def test_simplify_chronix(self):
        simplified_chronix = self.chronix_scenario.simplify_chronix()
        self.assertEqual(float(simplified_chronix.solar_p.iloc[0]), 6)


if __name__ == '__main__':
    unittest.main()
