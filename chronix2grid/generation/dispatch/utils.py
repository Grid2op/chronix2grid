import os
from enum import Enum
import copy

import numpy as np

class RampMode(Enum):
    """
    Encodes the level of complexity of the ramp constraints to apply for
    the economic dispatch
    """
    none = -1
    easy = 0
    medium = 1
    hard = 2


def make_scenario_input_output_directories(input_folder, output_folder, scenario_name):
    os.makedirs(os.path.join(input_folder, scenario_name), exist_ok=True)
    os.makedirs(os.path.join(output_folder, scenario_name), exist_ok=True)
    return os.path.join(input_folder, scenario_name), os.path.join(output_folder, scenario_name)

def modify_hydro_ramps(grid2op_env, hydro_dividing_factor=1., decimals=1):
    for i, generator in enumerate(grid2op_env.name_gen):
        gen_type = grid2op_env.gen_type[i]
        if gen_type == "hydro":
           grid2op_env.gen_max_ramp_up[i] = np.round(grid2op_env.gen_max_ramp_up[i]/hydro_dividing_factor, decimals)
           grid2op_env.gen_max_ramp_down[i] = np.round(grid2op_env.gen_max_ramp_down[i]/hydro_dividing_factor, decimals)
    return grid2op_env

def modify_slack_characs(grid2op_env, slack_name, p_max_reduction=150, ramp_reduction=6, decimals = 1):
    for i, generator in enumerate(grid2op_env.name_gen):
        if generator == slack_name:
            grid2op_env.gen_pmax[i] = np.round(grid2op_env.gen_pmax[i] - p_max_reduction, decimals)
            grid2op_env.gen_max_ramp_up[i] = np.round(grid2op_env.gen_max_ramp_up[i] - ramp_reduction, decimals)
            grid2op_env.gen_max_ramp_down[i] = np.round(grid2op_env.gen_max_ramp_down[i] - ramp_reduction, decimals)
    return grid2op_env

def add_noise_gen(dispatch, gen_cap, noise_factor):
    """ Add noise to opf dispatch to have more
    realistic real-time data

    Parameters
    ----------
    dispatch : dataframe
        Opf PyPSA output
    gen_cap : dataframe
        Maximun capacity for gen
    noise_factor : float
        Noise factor applied to every gen col

    Returns
    -------
    dataframe
        Distpach with noise
    """
    dispatch_new = copy.deepcopy(dispatch)  # dispatch.copy(deep=True)

    # variance_per_col = gen_cap * noise_factor
    print('applying noise to forecast of ' + str(noise_factor) + ' %')
    for col in list(dispatch_new):
        # Check for values greater than zero
        # (means unit has been distpached)
        # only_dispatched_steps = dispatch_new[col][dispatch_new[col] > 0]
        # print(only_dispatched_steps)

        noise = np.random.lognormal(mean=0.0, sigma=noise_factor, size=dispatch_new.shape[0])
        dispatch_new[col] = dispatch[col] * noise
    return dispatch_new.round(2)