import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pathlib

from chronix2grid.main import create_directory_tree
import chronix2grid.constants as cst
import chronix2grid.generation.generation_utils as gu


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.output_directory = tempfile.mkdtemp()
        self.n_scenarios = 10

    def test_create_directory_tree(self):
        create_directory_tree('case', 'start_date', self.output_directory,
                              self.n_scenarios, 'L')
        self.assertTrue(os.path.isdir(os.path.join(
            self.output_directory, cst.GENERATION_FOLDER_NAME, 'case', 'start_date'
        )))
        self.assertFalse(os.path.isdir(os.path.join(
            self.output_directory, cst.KPI_FOLDER_NAME, 'case', 'start_date'
        )))
        create_directory_tree('case', 'start_date2', self.output_directory,
                              self.n_scenarios, 'LK')
        self.assertTrue(os.path.isdir(os.path.join(
            self.output_directory, cst.KPI_FOLDER_NAME, 'case', 'start_date2'
        )))
        scen_name_generator = gu.folder_name_pattern(
            cst.SCENARIO_FOLDER_BASE_NAME, self.n_scenarios)
        for i in range(self.n_scenarios):
            scenario_name = scen_name_generator(i)
            path_to_check = os.path.join(
                self.output_directory, cst.GENERATION_FOLDER_NAME, 'case',
                'start_date2', scenario_name
            )
            self.assertTrue(os.path.isdir(path_to_check))
