# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

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

def modify_hydro_ramps(env_df, hydro_dividing_factor=1., decimals=1):
    rampup_hydro = env_df.loc[env_df['type']=='hydro','max_ramp_up'].values
    rampdown_hydro = env_df.loc[env_df['type'] == 'hydro', 'max_ramp_up'].values

    env_df.loc[env_df['type'] == 'hydro', 'max_ramp_up'] = np.round(rampup_hydro/hydro_dividing_factor, decimals)
    env_df.loc[env_df['type'] == 'hydro', 'max_ramp_down'] = np.round(rampdown_hydro / hydro_dividing_factor, decimals)

    return env_df

def modify_slack_characs(env_df, slack_name, p_max_reduction=0, ramp_reduction=0, decimals = 1):
    slack_pmax = env_df.loc[env_df['name'] == slack_name, 'pmax'].values[0]
    slack_rampup_max = env_df.loc[env_df['name'] == slack_name, 'max_ramp_up'].values[0]
    slack_rampdown_max = env_df.loc[env_df['name'] == slack_name, 'max_ramp_down'].values[0]

    env_df.loc[env_df['name'] == slack_name, 'pmax'] =  np.round(slack_pmax - p_max_reduction, decimals)
    env_df.loc[env_df['name'] == slack_name, 'max_ramp_up'] = np.round(slack_rampup_max - ramp_reduction, decimals)
    env_df.loc[env_df['name'] == slack_name, 'max_ramp_down'] = np.round(slack_rampdown_max - ramp_reduction,
                                                                                 decimals)
    return env_df

def add_noise_gen(prng, dispatch, gen_cap, noise_factor):
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

        noise = prng.lognormal(mean=0.0, sigma=noise_factor, size=dispatch_new.shape[0])
        dispatch_new[col] = dispatch[col] * noise
    return dispatch_new.round(2)