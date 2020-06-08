import os
import unittest
import shutil

import numpy as np
import pandas as pd
import pathlib

from chronix2grid import main
from chronix2grid import constants as cst
import chronix2grid.generation.generation_utils as gu


class TestMain(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'data')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'output')
        self.case = 'case118_l2rpn_wcci'
        self.start_date = '2012-01-01'
        self.year = 2012
        self.seed_loads = 1
        self.seed_res = 1
        self.seed_dispatch = 1
        self.n_scenarios = 2
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

    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)


    # def test_size_chunks(self):
    #     main.generate_inner(case=self.case, start_date=self.start_date, weeks=8,
    #                         by_n_weeks=4, n_scenarios=2, mode='LRTK',
    #                         root_folder=self.root_folder,
    #                         seed_for_loads=self.seed_loads, seed_for_res=self.seed_res,
    #                         seed_for_dispatch=self.seed_dispatch)
    #     dir_path = os.path.join(self.root_folder,
    #                         'generation', 'output', self.case, str(self.year),
    #                         'Scenario_0')
    #     prod_p = pd.read_csv(os.path.join(dir_path, 'prod_p.csv.bz2'))
    #
    #     prod_p_chunk_0 = pd.read_csv(os.path.join(dir_path, 'chunk_0', 'prod_p.csv.bz2'))
    #     prod_p_chunk_1 = pd.read_csv(os.path.join(dir_path, 'chunk_1', 'prod_p.csv.bz2'))
    #
    #     self.assertTrue(np.array_equal(
    #         prod_p.values,
    #         pd.concat([prod_p_chunk_0, prod_p_chunk_1])
    #     ))

    def test_l(self):
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode='L', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_r(self):
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode='R', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=1)

    def test_lr(self):
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode='RL', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_lrk(self):
        main.generate_per_scenario(
            case=self.case, start_date=self.start_date, weeks=1, by_n_weeks=4,
            mode='LRK', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder,
            generation_output_folder=self.generation_output_folder,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=1)
