from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import copy
import pdb

def compute_loads(loads_charac, temperature_noise, params, load_weekly_pattern):
    # Compute active part of loads
    weekly_pattern = load_weekly_pattern['test'].values
    loads_series = {}
    for i, name in enumerate(loads_charac['name']):
        mask = (loads_charac['name'] == name)
        if loads_charac[mask]['type'].values == 'residential':
            locations = [loads_charac[mask]['x'].values[0], loads_charac[mask]['y'].values[0]]
            Pmax = loads_charac[mask]['Pmax'].values[0]
            loads_series[name] = compute_residential(locations, Pmax, temperature_noise, params, weekly_pattern, index=i)

        if loads_charac[mask]['type'].values == 'industrial':
            raise NotImplementedError("Impossible to generate industrial loads for now.")
            Pmax = loads_charac[mask]['Pmax'].values[0]
            loads_series[name] = compute_industrial(Pmax, params)
    return loads_series

def compute_residential(locations, Pmax, temperature_noise, params, weekly_pattern, index):

    # What if we want more than 1 year???

    # Compute refined signals
    temperature_signal = interpolate_noise(
        temperature_noise,
        params,
        locations,
        time_scale=params['temperature_corr'])

    # Compute seasonal pattern
    Nt_inter = int(params['T'] // params['dt'] + 1)
    t = np.linspace(0, params['T'], Nt_inter, endpoint=True)
    start_year = pd.to_datetime(str(params['start_date'].year) + '/01/01', format='%Y-%m-%d')
    # start_min = int(pd.Timedelta(params['start_date'] - pd.to_datetime('2018/01/01', format='%Y-%m-%d')).total_seconds() // 60)
    start_min = int(pd.Timedelta(params['start_date'] - start_year).total_seconds() // 60)
    seasonal_pattern = 5.5/7 + 1.5/7*np.cos((2*np.pi/(365*24*60))*(t-30*24*60 - start_min))

    # Get weekly pattern
    weekly_pattern = compute_load_pattern(params, weekly_pattern, index)
    std_temperature_noise = params['std_temperature_noise']
    residential_series = Pmax * weekly_pattern * (std_temperature_noise*temperature_signal + seasonal_pattern)

    return residential_series

def compute_load_pattern(params, weekly_pattern, index):
    """
    Loads a typical hourly pattern, and interpolates it to generate
    a smooth solar generation pattern between 0 and 1

    Input:
        computation_params: (dict) Defines the mesh dimensions and
            precision. Also define the correlation scales
            interpolation_params: (dict) params of the interpolation

    Output:
        (np.array) A smooth solar pattern
    """
    # solar_pattern resolution : 1H, 8761

    # Keep only one week of pattern
    index_weekly_perweek = 12 * 24 * 7
    index %= int(weekly_pattern.shape[0] / index_weekly_perweek - 1)
    # print(index)
    weekly_pattern = weekly_pattern[(index * index_weekly_perweek):((index + 1) * index_weekly_perweek)]
    weekly_pattern /= np.mean(weekly_pattern)

    # # # pdb.set_trace()
    # #
    # fig, ax = plt.subplots(figsize=(16, 10))
    # plt.plot(weekly_pattern)
    # # ax.hist(center_reduce(solar_tmp), 1000, density=True, label="Gen", alpha=0.3)
    # # ax.hist(center_reduce(solar_2017), 100, density=True, label="FR", alpha=0.3)
    # plt.title("load {}".format(index))
    # plt.legend()
    # plt.show()
    #
    # plt.plot(weekly_pattern)
    # plt.show()

    start_year = pd.to_datetime(str(params['start_date'].year) + '/01/01', format='%Y-%m-%d')
    T_bis = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // (60))

    Nt_inter_hr = int(T_bis // 5 + 1)
    N_repet = int((Nt_inter_hr - 1) // len(weekly_pattern) + 1)
    stacked_weekly_pattern = weekly_pattern
    for i in range(N_repet - 1):
        stacked_weekly_pattern = np.append(stacked_weekly_pattern, weekly_pattern)

    # The time is in minutes
    t_pattern = np.linspace(0, 60 * 7 * 24 * N_repet, 12 * 7 * 24 * N_repet, endpoint=False)
    f2 = interp1d(t_pattern, stacked_weekly_pattern, kind='cubic')

    Nt_inter = int(params['T'] // params['dt'] + 1)
    start_year = pd.to_datetime(str(params['start_date'].year) + '/01/01', format='%Y-%m-%d')
    start_min = int(pd.Timedelta(params['start_date'] - start_year).total_seconds() // 60)
    end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)
    t_inter = np.linspace(start_min, end_min, Nt_inter, endpoint=True)
    output = f2(t_inter)
    output = output * (output > 0)

    # pdb.set_trace()
    # fig, ax = plt.subplots(figsize=(16, 10))
    # plt.plot(output[:index_weekly_perweek])
    # # ax.hist(center_reduce(solar_tmp), 1000, density=True, label="Gen", alpha=0.3)
    # # ax.hist(center_reduce(solar_2017), 100, density=True, label="FR", alpha=0.3)
    # plt.title("load {}".format(index))
    # plt.legend()
    # plt.show()

    return output

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


def create_csv(dict_, path, forecasted=False, reordering=True, noise=None, shift=False,
               write_results=True, index=True):
    df = pd.DataFrame.from_dict(dict_)
    df.set_index('datetime', inplace = True)
    df = df.sort_index(ascending=True)
    df = df.head(len(df) - 1) # Last value is lonely for another day
    if reordering:
        value = []
        for name in list(df):
            value.append(natural_keys(name))
        new_ordering = [x for _,x in sorted(zip(value,list(df)))]
        df = df[new_ordering]
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)

    df_reactive_power = 0.7 * df
    if noise is not None:
        df *= (1+noise*np.random.normal(0, 1, df.shape))
        df_reactive_power *= (1+noise*np.random.normal(0, 1, df.shape))

    if write_results:
        file_extension = '_forecasted' if forecasted else ''
        df.to_csv(
            os.path.join(path, f'load_p{file_extension}.csv.bz2'),
            index=index, sep=';', float_format='%.1f')
        df_reactive_power.to_csv(
            os.path.join(path, f'load_q{file_extension}.csv.bz2'),
            index=index, sep=';', float_format='%.1f')

    return df

def natural_keys(text):
    return int([ c for c in re.split('(\d+)', text) ][1])

