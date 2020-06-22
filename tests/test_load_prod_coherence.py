import os
import unittest
import shutil

import numpy as np
import pandas as pd
import pathlib

from chronix2grid import main
from chronix2grid import constants as cst
import chronix2grid.generation.generation_utils as gu
from chronix2grid.config import ResConfigManager


class TestLoadProdCoherence(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'output')
        self.case = 'case118_l2rpn_wcci'
        self.start_date = '2012-07-01'
        self.year = 2012
        self.seed_loads = 1
        self.seed_res = 1
        self.seed_dispatch = 1
        self.n_scenarios = 1
        self.scenario_names = gu.folder_name_pattern(
            cst.SCENARIO_FOLDER_BASE_NAME, self.n_scenarios
        )
        generation_output_folder, kpi_output_folder = main.create_directory_tree(
            self.case, self.start_date, self.output_folder, cst.SCENARIO_FOLDER_BASE_NAME,
            self.n_scenarios, 'LRK', warn_user=False)
        self.generation_output_folder = generation_output_folder
        self.kpi_output_folder = kpi_output_folder
        seeds_for_loads, seeds_for_res, seeds_for_disp = gu.generate_seeds(
            self.n_scenarios, self.seed_loads, self.seed_res, self.seed_dispatch
        )
        self.seeds_for_loads = seeds_for_loads
        self.seeds_for_res = seeds_for_res
        self.seeds_for_disp = seeds_for_disp
        self.ignore_warnings = True
        self.weeks = 1
        self.nb_core = 1
        self.by_n_weeks = 4

    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_load_prod_coherence_winter(self):

        main.generate_per_scenario(
            case=self.case, start_date='2012-01-01', weeks=self.weeks,
            by_n_weeks=self.by_n_weeks, mode='LRT',
            input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

        generation_input_folder = os.path.join(
            self.input_folder, cst.GENERATION_FOLDER_NAME
        )
        res_config_manager = ResConfigManager(
            name="Renewables Generation",
            root_directory=generation_input_folder,
            input_directories=dict(case=self.case, patterns='patterns'),
            required_input_files=dict(case=['prods_charac.csv', 'params.json'],
                                      patterns=['solar_pattern.npy']),
            output_directory=self.generation_output_folder
        )

        params, prods_charac, solar_pattern = res_config_manager.read_configuration()

        scenario_path = os.path.join(
            self.generation_output_folder,
            f'{cst.SCENARIO_FOLDER_BASE_NAME}_0'
        )

        loads = pd.read_csv(
            os.path.join(scenario_path, 'load_p.csv.bz2'),
            sep=';')
        prods = pd.read_csv(
            os.path.join(scenario_path, 'prod_p.csv.bz2'),
            sep=';')

        # Make sure production is higher than consumption
        self.assertEqual((prods.sum(axis=1) - loads.sum(axis=1) < 0).sum(),
                         0)

        prods_dispatch = prods_charac.loc[
            ~prods_charac.type.isin(['wind', 'solar']), :
        ]

        # Make sure physical constraints are respected
        for gen_name in prods_dispatch.name:
            max_ramp_up = prods.loc[:, gen_name].diff().max()
            max_ramp_down = - prods.loc[:, gen_name].diff().min()
            p_max = prods.loc[:, gen_name].max()
            p_min = prods.loc[:, gen_name].min()
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'max_ramp_up'].values[0],
                max_ramp_up
            )
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'max_ramp_down'].values[0],
                max_ramp_down
            )
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'Pmax'].values[0],
                p_max
            )
            self.assertGreaterEqual(
                p_min,
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'Pmin'].values[0]
            )

    def test_load_prod_coherence_summer(self):

        main.generate_per_scenario(
            case=self.case, start_date='2012-07-01', weeks=self.weeks,
            by_n_weeks=self.by_n_weeks, mode='LRT',
            input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

        generation_input_folder = os.path.join(
            self.input_folder, cst.GENERATION_FOLDER_NAME
        )
        res_config_manager = ResConfigManager(
            name="Renewables Generation",
            root_directory=generation_input_folder,
            input_directories=dict(case=self.case, patterns='patterns'),
            required_input_files=dict(case=['prods_charac.csv', 'params.json'],
                                      patterns=['solar_pattern.npy']),
            output_directory=self.generation_output_folder
        )

        params, prods_charac, solar_pattern = res_config_manager.read_configuration()

        scenario_path = os.path.join(
            self.generation_output_folder,
            f'{cst.SCENARIO_FOLDER_BASE_NAME}_0'
        )

        loads = pd.read_csv(
            os.path.join(scenario_path, 'load_p.csv.bz2'),
            sep=';')
        prods = pd.read_csv(
            os.path.join(scenario_path, 'prod_p.csv.bz2'),
            sep=';')

        # Make sure production is higher than consumption
        self.assertEqual((prods.sum(axis=1) - loads.sum(axis=1) < 0).sum(),
                         0)

        prods_dispatch = prods_charac.loc[
            ~prods_charac.type.isin(['wind', 'solar']), :
        ]

        # Make sure physical constraints are respected
        for gen_name in prods_dispatch.name:
            max_ramp_up = prods.loc[:, gen_name].diff().max()
            max_ramp_down = - prods.loc[:, gen_name].diff().min()
            p_max = prods.loc[:, gen_name].max()
            p_min = prods.loc[:, gen_name].min()
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'max_ramp_up'].values[0],
                max_ramp_up
            )
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'max_ramp_down'].values[0],
                max_ramp_down
            )
            self.assertGreaterEqual(
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'Pmax'].values[0],
                p_max
            )
            self.assertGreaterEqual(
                p_min,
                prods_dispatch.loc[prods_dispatch.name == gen_name, 'Pmin'].values[0]
            )
