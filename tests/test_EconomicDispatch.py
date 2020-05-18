import datetime as dt
import json
import os
import tempfile
import unittest

import pandas as pd

from chronix2grid.generation.dispatch.EconomicDispatch import (
    ChroniXScenario, init_dispatcher_from_config, Dispatcher)
import grid2op
from grid2op.Chronics import ChangeNothing


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.grid_path = ''
        self.input_folder = os.path.abspath('input_data/generation')
        self.CASE = 'case118_l2rpn_wcci'
        self.year = 2012
        self.scenario_name = 'Scenario_0'
        with open(os.path.join(self.input_folder, self.CASE, 'params_opf.json'), 'r') as params_opf_jons:
            self.params_opf = json.load(params_opf_jons)
        self.dispatcher = init_dispatcher_from_config(self.params_opf["grid_path"], self.input_folder)
        self.hydro_file_path = os.path.join(self.input_folder, 'patterns',
                                            'hydro_french.csv')

    def test_from_grid2op_env(self):
        grid2op_env = grid2op.make("blank",
                                   grid_path=self.params_opf["grid_path"],
                                   chronics_class=ChangeNothing)
        dispatcher = Dispatcher.from_gri2op_env(grid2op_env)
        self.assertTrue(isinstance(dispatcher, Dispatcher))

    def test_read_hydro_guide_curves(self):
        self.dispatcher.read_hydro_guide_curves(self.hydro_file_path)
        self.assertAlmostEqual(self.dispatcher._max_hydro_pu.iloc[0, 0],
                               0.482099426, places=5)


class TestChronixScenario(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.abspath('input_data/generation')
        self.case = 'case118_l2rpn_wcci'
        self.grid2op_env = grid2op.make(
            "blank",
            grid_path=os.path.join(self.input_folder, self.case,
                                   "L2RPN_2020_case118_redesigned.json"),
            chronics_class=ChangeNothing,
        )

        self.chronics_path_gen = tempfile.mkdtemp()
        self.loads = pd.DataFrame(
            index=[dt.datetime(2019, 9, 1, 0, 0), dt.datetime(2019, 9, 1, 0, 5)],
            columns=['load_1', 'load_2'],
            data=[[9, 10], [11, 12]]
        )
        self.prods = pd.DataFrame(
            index=[dt.datetime(2019, 9, 1, 0, 0), dt.datetime(2019, 9, 1, 0, 5)],
            columns=['gen_1', 'gen_2', 'gen_3', 'gen_4'],
            data=[[1, 2, 3, 4], [5, 6, 7, 8]]
        )
        self.loads.to_csv(
            os.path.join(self.chronics_path_gen, 'load_p.csv.bz2'), sep=';', index=True)
        self.prods.to_csv(
            os.path.join(self.chronics_path_gen, 'prod_p.csv.bz2'), sep=';', index=True)
        self.res_names = dict(wind=['gen_1'], solar=['gen_2', 'gen_4'])
        self.chronix_scenario = ChroniXScenario(self.loads, self.prods, self.res_names, 'scen')

    def test_instanciation(self):
        self.assertEqual(self.chronix_scenario.total_res.iloc[0], 7)
        self.assertEqual(self.chronix_scenario.wind_p.columns, ['gen_1'])

    def test_from_disk(self):
        chronix_scenario = ChroniXScenario.from_disk(
            os.path.join(self.chronics_path_gen, 'load_p.csv.bz2'),
            os.path.join(self.chronics_path_gen, 'prod_p.csv.bz2'),
            self.res_names, 'scen')
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
