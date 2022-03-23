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
from numpy.random import default_rng

#from numpy.random import Generator, MT19937
#from numpy.random import MT19937 as default_rng

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

        ###########
        ## Bypassing this to recover same seeds as before
        prng = default_rng()

        # Generated seeds from the first three seeds (but we only generate one scenario)
        seeds_for_loads, seeds_for_res, seeds_for_disp = gu.generate_seeds(
            prng, 2, seed_for_loads, seed_for_res, seed_for_disp
        )
        ########

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

        self.case_all = 'case118_l2rpn_neurips_1x_hydro_loss'
        generation_output_folder, kpi_output_folder = main.create_directory_tree(
            self.case_all, self.start_date, self.output_folder, cst.SCENARIO_FOLDER_BASE_NAME,
            self.n_scenarios, 'LRT', warn_user=False)
        self.generation_output_folder_all = generation_output_folder
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
        self.expected_folder_all = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'output',"generation",
            'expected_case118_l2rpn_neurips_1x_hydro_loss',#"case118_l2rpn_neurips_1x_hyrdo_loss_modifySlackBeforeChronixGeneration",
            "Scenario_january_0")
        self.files_tocheck = ['prod_p']

        # Modification in gen prices to avoid multi solution in OPF
        self.seed_price_noise_noloss = 5
        self.seed_price_noise_loss = 5
        self.seed_price_noise_all = 5
        self.mu = 0
        self.sigma = 0.5
        self.gen_types = ['thermal','hydro', 'nuclear']

        # Truncates dataframe before comparison
        self.id_min = 0
        self.id_max = 5000


    def modify_gen_prices(self, mu, sigma, seed_price_noise, gen_types, case):
        np.random.seed(seed_price_noise)
        noise = np.random.normal(mu, sigma, 32)
        # Read prods_charac
        prods_orig = pd.read_csv(os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME, case, 'prods_charac_original.csv'))

        # Add noise to price columns
        prods_orig.loc[prods_orig['type'].isin(gen_types), 'marginal_cost'] += noise
        prods_orig['marginal_cost'] = prods_orig['marginal_cost'].round(1)

        # Check if all prices are unique
        prices = prods_orig[prods_orig['type'].isin(gen_types)]['marginal_cost']
        if prices.nunique() < len(prices):
            raise ValueError("Prices must be unique - set another seed in test")

        # Write it
        prods_orig.to_csv(os.path.join(self.input_folder, cst.GENERATION_FOLDER_NAME, case, 'prods_charac.csv'), index=False)

    # def tearDown(self) -> None:
    #     shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_integration_l(self):
        self.modify_gen_prices(self.mu, self.sigma, self.seed_price_noise_noloss, self.gen_types, self.case_noloss)

        # Launch module
        main.generate_per_scenario(
            case=self.case_noloss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
            mode='L', input_folder=self.input_folder,
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
        files_to_check=['load_p']
        bool = self.check_frames_equal(path_out, path_ref, files_to_check)
        self.assertTrue(bool)

    def test_integration_r(self):
        self.modify_gen_prices(self.mu, self.sigma, self.seed_price_noise_noloss, self.gen_types, self.case_noloss)

        # Launch module
        main.generate_per_scenario(
            case=self.case_noloss, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
            mode='R', input_folder=self.input_folder,
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
        files_to_check=['solar_p','wind_p']
        bool = self.check_frames_equal(path_out, path_ref, files_to_check)
        self.assertTrue(bool)

    def test_integration_lrt_nolosscorrection(self):
        self.modify_gen_prices(self.mu, self.sigma, self.seed_price_noise_noloss, self.gen_types, self.case_noloss)

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
        self.modify_gen_prices(self.mu, self.sigma, self.seed_price_noise_loss, self.gen_types, self.case_loss)
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

    def test_integration_all(self):
        self.modify_gen_prices(self.mu, self.sigma, self.seed_price_noise_all, self.gen_types, self.case_all)

        with warnings.catch_warnings(record=True) as w:
            main.generate_per_scenario(
                case=self.case_all, start_date=self.start_date, weeks=self.nweeks, by_n_weeks=4,
                mode='LRT', input_folder=self.input_folder,
                kpi_output_folder=self.kpi_output_folder_loss,
                generation_output_folder=self.generation_output_folder_all,
                scen_names=self.scenario_names,
                seeds_for_loads=self.seed_for_load,
                seeds_for_res=self.seed_for_res,
                seeds_for_dispatch=self.seed_for_disp,
                ignore_warnings=self.ignore_warnings,
                scenario_id=0)
            path_out = os.path.join(self.generation_output_folder_all, "Scenario_0")
            path_ref = self.expected_folder_all
            bool = self.check_frames_equal(path_out, path_ref, self.files_tocheck)
            # Check that we obtain the right result dataframe
            self.assertTrue(bool)
            # Check that we have raised a UserWarning for ramp up (one among all warnings that have been raised)
            #boolvec_types = [issubclass(w_.category, UserWarning) for w_ in w]
            #self.assertTrue(np.any(boolvec_types))
            #boolvec_msg = ["Ramp up" in str(w_.message) for w_ in w]
            #self.assertTrue(np.any(boolvec_msg))


    def check_frames_equal(self, path_out,path_ref, files):
        bool = True
        for fil in files:
            df_out = pd.read_csv(os.path.join(path_out, f'{fil}.csv.bz2'), sep=';')
            df_ref = pd.read_csv(os.path.join(path_ref, f'{fil}.csv.bz2'), sep=';')
            bool_ = df_out[self.id_min:self.id_max].equals(df_ref[self.id_min:self.id_max])
            bool = bool_ and bool
        return bool