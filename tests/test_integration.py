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
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            'output')

        self.start_date = '2012-01-01'
        self.nweeks = 4
        self.year = 2012
        self.n_scenarios = 1
        self.scenario_names = gu.folder_name_pattern(
            cst.SCENARIO_FOLDER_BASE_NAME, self.n_scenarios
        )

        self.seeds_for_loads = [1180859679]
        self.seeds_for_res = [1180859679]
        self.seeds_for_disp = [1180859679]
        self.ignore_warnings = True

        ## 2 cases, rest is equal between both tests
        self.case_noloss = 'case118_l2rpn_neurips_1x_modifySlackBeforeChronixGeneration'
        generation_output_folder, kpi_output_folder = main.create_directory_tree(
            self.case_noloss, self.start_date, self.output_folder, cst.SCENARIO_FOLDER_BASE_NAME,
            self.n_scenarios, 'LRT', warn_user=False)
        self.generation_output_folder_noloss = generation_output_folder
        self.kpi_output_folder_noloss = kpi_output_folder

        self.case_loss = 'case118_l2rpn_neurips_1x'
        generation_output_folder, kpi_output_folder = main.create_directory_tree(
            self.case_loss, self.start_date, self.output_folder, cst.SCENARIO_FOLDER_BASE_NAME,
            self.n_scenarios, 'LRT', warn_user=False)
        self.generation_output_folder_loss = generation_output_folder
        self.kpi_output_folder_loss = kpi_output_folder


    def tearDown(self) -> None:
        a = 0
        # shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_integration_lrt_nolosscorrection(self):
        main.generate_per_scenario(
            case=self.case_noloss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
            mode='LRT', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder_noloss,
            generation_output_folder=self.generation_output_folder_noloss,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

    def test_integration_lrt_withlosscorrection(self):
        main.generate_per_scenario(
            case=self.case_loss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
            mode='LRT', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder_loss,
            generation_output_folder=self.generation_output_folder_loss,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seeds_for_loads,
            seeds_for_res=self.seeds_for_res,
            seeds_for_dispatch=self.seeds_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)

