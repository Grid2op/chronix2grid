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


def generate_default_seed():
    return np.random.randint(low=0, high=2 ** 31)


def dump_seeds(output_directory, seeds):
    with open(os.path.join(output_directory, cst.SEEDS_FILE_NAME), 'w') as f:
        json.dump(seeds, f)