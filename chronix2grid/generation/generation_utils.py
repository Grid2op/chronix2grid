# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import datetime as dt
from functools import partial
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import default_rng

from ..config import DispatchConfigManager, LoadsConfigManager, ResConfigManager


def make_generation_input_output_directories(input_folder, case, year, output_folder):

    dispatch_input_folder_case = os.path.join(input_folder, case, 'dispatch')
    dispatch_input_folder = os.path.join(dispatch_input_folder_case, str(year))
    dispatch_output_folder = os.path.join(output_folder, str(year))

    os.makedirs(dispatch_input_folder_case, exist_ok=True)
    os.makedirs(dispatch_input_folder, exist_ok=True)
    os.makedirs(dispatch_output_folder, exist_ok=True)

    return dispatch_input_folder, dispatch_input_folder_case, dispatch_output_folder


def get_nx_ny_nt(data_type, params, add_dim):
    # Get computation domain size
    Lx = params['Lx']
    Ly = params['Ly']
    T = params['T']

    # Get the decay parameter for each dimension
    dx_corr = params['dx_corr']
    dy_corr = params['dy_corr']
    dt_corr = params[data_type + '_corr']

    # Compute number of element in each dimension
    Nx_comp = int(Lx // dx_corr + 1) + add_dim
    Ny_comp = int(Ly // dy_corr + 1) + add_dim
    Nt_comp = int(T // dt_corr + 1) + add_dim
    return Nx_comp, Ny_comp, Nt_comp


def generate_coarse_noise(prng, params, data_type, add_dim):
    """
    This function generates a spatially and temporally correlated noise.
    Because it may take a lot of time to compute a correlated noise on
    a too fine mesh, we recommend to first compute a correlated signal
    on a coarse mesh, and then use the interpolation function.

    Input:
        params: (dict) Defines the mesh dimensions and
            precision. Also define the correlation scales

    Output:
        (np.array) 3D autocorrelated noise
    """

    Nx_comp, Ny_comp, Nt_comp = get_nx_ny_nt(data_type, params, add_dim)
    
    # Generate gaussian noise input·
    #output = np.random.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp))
    output = prng.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp))

    return output

def interpolate_noise(computation_noise, params, locations, time_scale, add_dim):
    """
    This interpolates an autocarrelated noise mesh, to make it more granular.

    Input:
        computation_noise: (np.array) Autocorrelated signal computed on a coarse mesh
        params: (dict) Defines the mesh dimensions and
            precision. Also define the correlation scales
        locations: (dict) Defines the location of the points of interest in the domain

    Output:
        (dict of np.array) returns one time series per location mentioned in dict locations
    """

    # Get computation domain size
    Lx = params['Lx']
    Ly = params['Ly']
    T = params['T']

    # Get the decay parameter for each dimension
    dx_corr = params['dx_corr']
    dy_corr = params['dy_corr']
    dt_corr = time_scale

    # Compute number of element in each dimension
    Nx_comp = int(Lx // dx_corr + 1) + add_dim
    Ny_comp = int(Ly // dy_corr + 1) + add_dim
    Nt_comp = int(T // dt_corr + 1) + add_dim

    # Get interpolation temporal mesh size
    dt = params['dt']

    # Compute number of element in each dimension
    Nt_inter = T // dt + 1

    # Get coordinates
    x, y = locations

    # Get coordinates of closest points in the coarse mesh
    x_minus = int(x // dx_corr)
    x_plus = int(x // dx_corr + 1)
    y_minus = int(y // dy_corr)
    y_plus = int(y // dy_corr + 1)

    # 1st step : spatial interpolation

    # Initialize output
    output = np.zeros(Nt_comp)

    # Initialize sum of distances
    dist_tot = 0

    # For every close point, add the corresponding time series, weighted by the inverse
    # of the distance between them
    for x_neighbor in [x_minus, x_plus]:
        for y_neighbor in [y_minus, y_plus]:
            dist = 1 / (np.sqrt((x - dx_corr * x_neighbor) ** 2 + (y - dy_corr * y_neighbor) ** 2) + 1)
            output += dist * computation_noise[x_neighbor, y_neighbor, :]
            dist_tot += dist
    output /= dist_tot

    # 2nd step : temporal quadratic interpolation
    t_comp = np.linspace(0, int(T), int(Nt_comp), endpoint=True)
    t_inter = np.linspace(0, int(T), int(Nt_inter), endpoint=True)
    if Nt_comp == 2:
        f2 = interp1d(t_comp, output, kind='linear')
    elif Nt_comp == 3:
        f2 = interp1d(t_comp, output, kind='quadratic')
    elif Nt_comp > 3:
        f2 = interp1d(t_comp, output, kind='cubic')

    if Nt_comp >= 2:
        output = f2(t_inter)

    return output

def natural_keys(text):
    return int([ c for c in re.split('(\d+)', text) ][1])


def time_parameters(weeks, start_date):
    result = dict()
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    result['weeks'] = weeks
    result['start_date'] = start_date
    result['year'] = start_date.year
    return result


def updated_time_parameters_with_timestep(time_parameters, timestep):
    time_parameters['end_date'] = time_parameters['start_date'] + dt.timedelta(
        days=7 * int(time_parameters['weeks'])) - dt.timedelta(minutes=timestep)
    time_parameters['T'] = int(
        pd.Timedelta(
            time_parameters['end_date'] - time_parameters['start_date']
        ).total_seconds() // 60
    )
    return time_parameters


def generate_seeds(prng, n_seeds, seed_for_loads=None, seed_for_res=None, seed_for_disp=None):
    default_seed = prng.integers(low=0, high=2 ** 31, dtype=int)
    if seed_for_loads is not None:
        prng_load = default_rng(seed_for_loads)
    else:
        prng_load = default_rng(default_seed)
    seeds_for_loads = [prng_load.integers(low=0, high=2 ** 31, dtype=int) for _ in range(n_seeds)]

    if seed_for_res is not None:
        prng_res = default_rng(seed_for_res)
    else:
        prng_res = default_rng(default_seed)
    seeds_for_res = [prng_res.integers(low=0, high=2 ** 31, dtype=int) for _ in range(n_seeds)]
    if seed_for_disp is not None:
        prng_disp = default_rng(seed_for_disp)
    else:
        prng_disp = default_rng(default_seed)
    seeds_for_disp = [prng_disp.integers(low=0, high=2 ** 31, dtype=int) for _ in range(n_seeds)]

    return seeds_for_loads, seeds_for_res, seeds_for_disp


def read_all_configurations(weeks, start_date, case, input_folder, output_folder):
    time_params = time_parameters(weeks, start_date)
    year = time_params['year']

    print(f'output_folder: {output_folder}')

    load_config_manager = LoadsConfigManager(
        name="Loads Generation",
        root_directory=input_folder,
        input_directories=dict(case=case, patterns='patterns'),
        required_input_files=dict(case=['loads_charac.csv', 'params.json'],
                                  patterns=['load_weekly_pattern.csv']),
        output_directory=output_folder
    )
    load_config_manager.validate_configuration()

    params, loads_charac, load_weekly_pattern = load_config_manager.read_configuration()

    res_config_manager = ResConfigManager(
        name="Renewables Generation",
        root_directory=input_folder,
        input_directories=dict(case=case, patterns='patterns'),
        required_input_files=dict(case=['prods_charac.csv', 'params.json'],
                                  patterns=['solar_pattern.npy']),
        output_directory=output_folder
    )

    params, prods_charac, solar_pattern = res_config_manager.read_configuration()

    params.update(time_params)
    params = updated_time_parameters_with_timestep(params, params['dt'])

    dispath_config_manager = DispatchConfigManager(
        name="Dispatch",
        root_directory=input_folder,
        output_directory=output_folder,
        input_directories=dict(params=case),
        required_input_files=dict(params=['params_opf.json'])
    )
    dispath_config_manager.validate_configuration()
    params_opf = dispath_config_manager.read_configuration()

    return (year, params, loads_charac, prods_charac, load_weekly_pattern,
            solar_pattern, params_opf)


def folder_name_pattern(base_name, n_scenarios):
    padding_size = len(str(int(n_scenarios)))
    return name_pattern(base_name, padding_size)


def name_pattern(base_name, padding_size):
    prefix = str(base_name) + '_' + '{:0{width}d}'
    return partial(prefix.format, width=padding_size)


def warn_if_output_folder_not_empty(output_folder):
    if len(os.listdir(output_folder)) != 0:
        input(f'The output folder {output_folder} is not empty, '
              f'proceed with caution. '
              'Press Enter to continue or Crtl-C to exit...')
