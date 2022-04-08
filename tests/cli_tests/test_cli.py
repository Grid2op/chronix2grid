# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import unittest
import shutil
import subprocess

import numpy as np
import pandas as pd
import pathlib

from chronix2grid import main
from chronix2grid import constants as cst
import chronix2grid.generation.generation_utils as gu


class TestCli(unittest.TestCase):
    def setUp(self):
        self.input_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'data', 'input')
        self.output_folder = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(),
            'output')
        os.makedirs(self.output_folder, exist_ok=True)
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
        self.ignore_warnings = True

    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder, ignore_errors=False, onerror=None)

    def test_cli(self):
        cmd = [
            'chronix2grid',
            '--case', 'case118_l2rpn_wcci',
            '--start-date', '2012-01-01',
            '--weeks', str(2),
            '--by-n-weeks', str(4),
            '--n_scenarios', str(self.n_scenarios),
            '--mode', 'LR',
            '--input-folder', self.input_folder,
            '--output-folder', self.output_folder,
            '--seed-for-loads', str(self.seed_loads),
            '--seed-for-res', str(self.seed_res),
            '--seed-for-dispatch', str(self.seed_dispatch),
            '--ignore-warnings',
            '--scenario_name', '',
            '--nb_core', str(2)
        ]

        rv = subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.assertEqual(rv.returncode, 0)
