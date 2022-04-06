# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import datetime as dt
import os
import pdb
import tempfile
import unittest

import numpy as np
import pandas as pd
import pathlib

import chronix2grid.constants as cst
import chronix2grid.default_backend as def_bk
from chronix2grid.generation.dispatch.EconomicDispatch import (
            ChroniXScenario, init_dispatcher_from_config)
from chronix2grid.generation.dispatch.utils import modify_hydro_ramps, modify_slack_characs
import grid2op
from grid2op.Chronics import ChangeNothing


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.CASE = 'case118_l2rpn_neurips_1x_original'
        # self.grid_path = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME,
        #                               self.CASE, 'grid.json')
        self.grid_path = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME,
                                      self.CASE)
        self.expected_file = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME,
                                          'case118_l2rpn_neurips_1x_modifySlackBeforeChronixGeneration',
                                          'prods_charac.csv')

        self.params_opf = {"hydro_ramp_reduction_factor":2.,
                           "nameSlack":"gen_68_37",
                          "slack_p_max_reduction":150.,
                           "slack_ramp_max_reduction":6.}

    def test_modify_slack_and_hydro(self):
        env = grid2op.make(self.grid_path,
                           chronics_path=self.grid_path,
                           chronics_class=ChangeNothing)
        env_df = pd.DataFrame({'name': env.name_gen,
                               'type': env.gen_type,
                               'pmax': env.gen_pmax,
                               "pmin":env.gen_pmin,
                               'max_ramp_up': env.gen_max_ramp_up,
                               'max_ramp_down': env.gen_max_ramp_down})
        env_df = modify_hydro_ramps(env_df, self.params_opf["hydro_ramp_reduction_factor"])
        env_df = modify_slack_characs(env_df,
                                   self.params_opf["nameSlack"],
                                   self.params_opf["slack_p_max_reduction"],
                                   self.params_opf["slack_ramp_max_reduction"])
        prods = env_df.rename(columns = {"pmax":"Pmax","pmin":"Pmin"})
        prods.sort_values(by="name", inplace=True)
        prods.reset_index(drop=True, inplace=True)
        expected_prods = pd.read_csv(self.expected_file)[prods.columns].fillna(0)
        expected_prods.sort_values(by="name", inplace=True)
        expected_prods.reset_index(drop=True, inplace=True)

        prods = self.round_columns(prods, ["Pmax","Pmin","max_ramp_up","max_ramp_down"], 0)
        expected_prods = self.round_columns(expected_prods, ["Pmax", "Pmin", "max_ramp_up", "max_ramp_down"], 0)

        self.assertTrue(np.all(prods.values==expected_prods.values))

    def round_columns(self, df, cols, decimals):
        for col in cols:
            df[col] = np.round(df[col].values, decimals)
        return df


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.CASE = 'case118_l2rpn_wcci'
        self.year = 2012
        self.scenario_name = 'Scenario_0'
        self.grid_path = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME, self.CASE, 'grid.json')
        print(f"{self.grid_path=}")
        self.dispatcher = init_dispatcher_from_config(
            os.path.split(self.grid_path)[0],
            os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME),
            def_bk.DISPATCHER,
            params_opf = {"hydro_ramp_reduction_factor":1.,
                          "slack_p_max_reduction":0.,
                           "slack_ramp_max_reduction":0.}
        )
        self.grid2op_path = os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME, self.CASE)
        self.hydro_file_path = os.path.join(self.input_folder,
                                            cst.GENERATION_FOLDER_NAME,
                                            'patterns',
                                            'hydro_french.csv')

    def test_from_grid2op_env(self):
        grid2op_env = grid2op.make(self.grid2op_path,
                                   chronics_path=self.grid2op_path,
                                   chronics_class=ChangeNothing)
        dispatcher = def_bk.DISPATCHER.from_gri2op_env(grid2op_env)
        self.assertTrue(isinstance(dispatcher, def_bk.DISPATCHER))

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
        self.grid2op_path = os.path.join(self.input_folder,
                                   cst.GENERATION_FOLDER_NAME,
                                   self.case)
        self.grid2op_env = grid2op.make(self.grid2op_path,
                                        chronics_path = self.grid2op_path,
                                        chronics_class=ChangeNothing)
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
        # self.assertEqual(float(net_load.iloc[0]), 21.5)  # change in the net_load stuff to only count the loads
        self.assertEqual(float(net_load.iloc[0]), 28.5)

    def test_simplify_chronix(self):
        simplified_chronix = self.chronix_scenario.simplify_chronix()
        self.assertEqual(float(simplified_chronix.solar_p.iloc[0]), 6)


if __name__ == '__main__':
    unittest.main()
