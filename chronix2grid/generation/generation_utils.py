import re
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def make_generation_input_output_directories(input_folder, case, year, output_folder):

    dispatch_input_folder_case = os.path.join(input_folder, case, 'dispatch')
    dispatch_input_folder = os.path.join(dispatch_input_folder_case, str(year))
    dispatch_output_folder = os.path.join(output_folder, str(year))

    os.makedirs(dispatch_input_folder_case, exist_ok=True)
    os.makedirs(dispatch_input_folder, exist_ok=True)
    os.makedirs(dispatch_output_folder, exist_ok=True)

    return dispatch_input_folder, dispatch_input_folder_case, dispatch_output_folder


def generate_coarse_noise(params, data_type):
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

    # Get computation domain size
    Lx = params['Lx']
    Ly = params['Ly']
    T = params['T']

    # Get the decay parameter for each dimension
    dx_corr = params['dx_corr']
    dy_corr = params['dy_corr']
    dt_corr = params[data_type + '_corr']

    # Compute number of element in each dimension
    Nx_comp = int(Lx // dx_corr + 1)
    Ny_comp = int(Ly // dy_corr + 1)
    Nt_comp = int(T // dt_corr + 1)

    # Generate gaussian noise input·
    output = np.random.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp))

    return output

def interpolate_noise(computation_noise, params, locations, time_scale):
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
    Nx_comp = int(Lx // dx_corr + 1)
    Ny_comp = int(Ly // dy_corr + 1)
    Nt_comp = int(T // dt_corr + 1)

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