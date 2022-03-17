# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import json
import os

import numpy as np

import chronix2grid.constants as cst


def parse_seed_arg(seed, arg_name, default_seed):
    if seed is not None:
        try:
            casted_seed = int(seed)
        except TypeError:
            raise RuntimeError(f'The parameter {arg_name} must be an integer')
    else:
        casted_seed = default_seed
    return casted_seed


def generate_default_seed(prng):
    return prng.integers(low=0, high=2 ** 31, dtype=int)


def dump_seeds(output_directory, seeds, scenario_name=''):
    with open(os.path.join(output_directory, scenario_name+'_'+cst.SEEDS_FILE_NAME), 'w') as f:
        json.dump(seeds, f)