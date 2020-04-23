import os
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import pathlib

from chronix2grid import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.root_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'chronix2grid')
        print(self.root_folder)
        self.case = 'case118_l2rpn'
        self.start_date = '2012-01-01'
        self.year = 2012
        main.generate_inner(case=self.case, start_date=self.start_date, weeks=8,
                            by_n_weeks=4, n_scenarios=2, mode='LRT',
                            root_folder=self.root_folder, seed_for_loads=None,
                            seed_for_res=None, seed_for_dispatch=None)

    def test_size_chunks(self):
        dir_path = os.path.join(self.root_folder,
                            'generation', 'output', self.case, str(self.year),
                            'Scenario_0')
        prod_p = pd.read_csv(os.path.join(dir_path, 'prod_p.csv.bz2'))

        prod_p_chunk_0 = pd.read_csv(os.path.join(dir_path, 'chunk_0', 'prod_p.csv.bz2'))
        prod_p_chunk_1 = pd.read_csv(os.path.join(dir_path, 'chunk_1', 'prod_p.csv.bz2'))

        self.assertTrue(np.array_equal(
            prod_p.values,
            pd.concat([prod_p_chunk_0, prod_p_chunk_1])
        ))
