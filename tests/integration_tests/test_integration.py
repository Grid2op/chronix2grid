import os
import unittest
import shutil
import warnings

import numpy as np
import pandas as pd
import pathlib

from chronix2grid import main
from chronix2grid import constants as cst
import chronix2grid.generation.generation_utils as gu

from chronix2grid.config import ResConfigManager
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend

cst.RENEWABLE_GENERATION_CONFIG = ResConfigManager
cst.RENEWABLE_GENERATION_BACKEND = RenewableBackend


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'output')

        self.start_date = '2012-01-01'
        self.nweeks = 4
        #self.year = 2012
        self.n_scenarios = 1
        self.scenario_names = gu.folder_name_pattern(
            cst.SCENARIO_FOLDER_BASE_NAME, self.n_scenarios
        )

        # Original seed for each step
        seed_for_loads = 1180859679
        seed_for_res = 1180859679
        seed_for_disp = 1180859679
        self.ignore_warnings = True

        # Generated seeds from the first three seeds (but we only generate one scenario)
        seeds_for_loads, seeds_for_res, seeds_for_disp = gu.generate_seeds(
            2, seed_for_loads, seed_for_res, seed_for_disp
        )
        self.seed_for_load = [seeds_for_loads[0]]
        self.seed_for_res = [seeds_for_res[0]]
        self.seed_for_disp = [seeds_for_disp[0]]

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

        # Expected outputs
        self.expected_folder_loss = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'output',"generation",
            "expected_case118_l2rpn_neurips_1x",
            "Scenario_january_0")
        self.expected_folder_noloss = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'output',"generation",
            "expected_case118_l2rpn_neurips_1x_modifySlackBeforeChronixGeneration",
            "Scenario_january_0")
        self.files_tocheck = ['prod_p']


    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_integration_lrt_nolosscorrection(self):

        # Launch module
        main.generate_per_scenario(
            case=self.case_noloss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
            mode='LRT', input_folder=self.input_folder,
            kpi_output_folder=self.kpi_output_folder_noloss,
            generation_output_folder=self.generation_output_folder_noloss,
            scen_names=self.scenario_names,
            seeds_for_loads=self.seed_for_load,
            seeds_for_res=self.seed_for_res,
            seeds_for_dispatch=self.seed_for_disp,
            ignore_warnings=self.ignore_warnings,
            scenario_id=0)
        # Check
        path_out = os.path.join(self.generation_output_folder_noloss, "Scenario_0")
        path_ref = self.expected_folder_noloss
        bool = self.check_frames_equal(path_out, path_ref, self.files_tocheck)
        self.assertTrue(bool)

    def test_integration_lrt_withlosscorrection(self):
        with warnings.catch_warnings(record=True) as w:
            main.generate_per_scenario(
                case=self.case_loss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
                mode='LRT', input_folder=self.input_folder,
                kpi_output_folder=self.kpi_output_folder_loss,
                generation_output_folder=self.generation_output_folder_loss,
                scen_names=self.scenario_names,
                seeds_for_loads=self.seed_for_load,
                seeds_for_res=self.seed_for_res,
                seeds_for_dispatch=self.seed_for_disp,
                ignore_warnings=self.ignore_warnings,
                scenario_id=0)
            path_out = os.path.join(self.generation_output_folder_loss, "Scenario_0")
            path_ref = self.expected_folder_loss
            bool = self.check_frames_equal(path_out, path_ref, self.files_tocheck)
            # Check that we obtain the right result dataframe
            self.assertTrue(bool)
            # Check that we have raised a UserWarning for ramp up (one among all warnings that have been raised)
            boolvec_types = [issubclass(w_.category, UserWarning) for w_ in w]
            self.assertTrue(np.any(boolvec_types))
            boolvec_msg = ["Ramp up" in str(w_.message) for w_ in w]
            self.assertTrue(np.any(boolvec_msg))

    def check_frames_equal(self, path_out,path_ref, files):
        bool = True
        for fil in files:
            df_out = pd.read_csv(os.path.join(path_out, f'{fil}.csv.bz2'), sep=';')
            df_ref = pd.read_csv(os.path.join(path_ref, f'{fil}.csv.bz2'), sep=';')
            bool_ = df_out.equals(df_ref)
            bool = bool_ and bool
        return bool