from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import copy
#import pdb


def compute_wind_series(locations, Pmax, long_noise, medium_noise, short_noise, params, smoothdist):
    # Compute refined signals
    long_scale_signal = interpolate_noise(
        long_noise,
        params,
        locations,
        time_scale=params['long_wind_corr'])
    medium_scale_signal = interpolate_noise(
        medium_noise,
        params,
        locations,
        time_scale=params['medium_wind_corr'])
    short_scale_signal = interpolate_noise(
        short_noise,
        params,
        locations,
        time_scale=params['short_wind_corr'])

    # Compute seasonal pattern
    Nt_inter = int(params['T'] // params['dt'] + 1)
    t = np.linspace(0, params['T'], Nt_inter, endpoint=True)
    start_min = int(
        pd.Timedelta(params['start_date'] - pd.to_datetime('2018/01/01', format='%Y-%m-%d')).total_seconds() // 60)
    seasonal_pattern = np.cos((2 * np.pi / (365 * 24 * 60)) * (t - 30 * 24 * 60 - start_min))

    # Combine signals
    signal = (0.7 + 0.3 * seasonal_pattern) * (0.3 + 0.3 * medium_scale_signal + 0.3 * long_scale_signal)
    signal += 0.04 * short_scale_signal
    signal = 1e-1 * np.exp(4 * signal)
    # signal += np.random.uniform(0, SMOOTHDIST/Pmax, signal.shape)
    signal += np.random.uniform(0, smoothdist, signal.shape)

    # signal *= 0.95
    signal[signal < 0.] = 0.
    signal = smooth(signal)
    wind_series = Pmax * signal

    #mydata = pd.DataFrame(data=0.3 * long_scale_signal)
    #mydata.rename(columns={'0': 'long_scale_signal'}, inplace=True)
    #mydata["medium_scale_signal"] = pd.Series(data=0.3 * medium_scale_signal)
    #mydata["short_scale_signal"] = pd.Series(data=0.04 * short_scale_signal)
    #mydata["base_with_seas_patt"] = pd.Series(data=0.7 + 0.3 * seasonal_pattern)
    #mydata["lm_scale_noise_modulation"] = pd.Series(data=0.3 + 0.3 * medium_scale_signal + 0.3 * long_scale_signal)
    #sign = (0.7 + 0.3 * seasonal_pattern) * (0.3 + 0.3 * medium_scale_signal + 0.3 * long_scale_signal)
    #mydata["res_before_small_scale_noise"] = pd.Series(data=sign)
    #sign += 0.04 * short_scale_signal
    #mydata["res_with_small_scale_noise"] = pd.Series(data=sign)
    # sign = 1e-1 * np.exp(4 * sign)
    #mydata["res_after_exp"] = pd.Series(data=sign)
    #sign[sign < 0.] = 0.
    # sign = smooth(sign)
    #mydata["res_after_trunc_and_smooth"] = pd.Series(data=sign)
    #mydata.to_csv(r"D:\RTE\Challenge\3 - Données\debug\wind\wind_test.csv", decimal=",", sep=";", index=False)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.hist(wind_series, 1000, density=True, label="Sel", alpha=0.3)
    # # ax.hist(center_reduce(solar_tmp), 1000, density=True, label="Gen", alpha=0.3)
    # # ax.hist(center_reduce(solar_2017), 100, density=True, label="FR", alpha=0.3)
    # plt.title("wind")
    # plt.legend()
    # # plt.show()
    return wind_series

def compute_solar_series(locations, Pmax, solar_noise, params, solar_pattern, smoothdist, time_scale):
    """
    """

    # Compute noise at desired locations
    final_noise = interpolate_noise(solar_noise, params, locations, time_scale)

    # Compute solar pattern
    solar_pattern = compute_solar_pattern(params, solar_pattern)

    # Compute solar time series
    signal = solar_pattern*(0.75+0.25*final_noise)
    signal += np.random.uniform(0, smoothdist/Pmax, signal.shape)
    # signal[signal > 1] = 1
    signal[signal < 0.] = 0.
    signal = smooth(signal)
    solar_series = Pmax*signal
    # solar_series[np.isclose(solar_series, 0.)] = 0

    return solar_series

def compute_solar_pattern(params, solar_pattern):
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
    solar_pattern = solar_pattern[:-1]

    start_year = pd.to_datetime(str(params['start_date'].year) + '/01/01', format='%Y-%m-%d')
    end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)

    Nt_inter_hr = int(end_min // 60 + 1)
    N_repet = int((Nt_inter_hr - 1) // len(solar_pattern) + 1)
    stacked_solar_pattern = solar_pattern
    for i in range(N_repet - 1):
        stacked_solar_pattern = np.append(stacked_solar_pattern, solar_pattern)

    # The time is in minutes
    t_pattern = 60 * np.linspace(0, 8760 * N_repet, 8760 * N_repet, endpoint=False)
    f2 = interp1d(t_pattern, stacked_solar_pattern, kind='cubic')

    Nt_inter = int(params['T'] // params['dt'] + 1)
    start_year = pd.to_datetime(str(params['start_date'].year) + '/01/01', format='%Y-%m-%d')
    start_min = int(pd.Timedelta(params['start_date'] - start_year).total_seconds() // 60)
    end_min = int(pd.Timedelta(params['end_date'] - start_year).total_seconds() // 60)

    t_inter = np.linspace(start_min, end_min, Nt_inter, endpoint=True)
    output = f2(t_inter)
    output = output * (output > 0)

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

def smooth(x, alpha=0.5, beta=None):
    """
    smoothing function to avoid weird statistical effect
    we want below alpha: f(x) = x
    f(alpha) = alpha, and f'(alpha) = 1
    f(beta) = 1 and f'(beta) = 0
    f cubic polynomial

    if beta is not provided, beta = 1/alpha

    :param alpha: value (x) where smoothing starts
    :param beta: y when x=1
    """
    x = copy.deepcopy(x)
    if beta is None:
        beta = 1 / alpha

    # def pol(x, alpha, beta):
    #     a = (1 - 2 * (alpha - 1) / (alpha - beta)) / (alpha - beta) ** 2
    #     b = (alpha - 1) / (alpha - beta) ** 2 - a * alpha
    #     return (x - beta) ** 2 * (a * x + b) + 1
    #
    # def pol(x, alpha, beta):
    #     a = ( 1 - 2*(alpha-1)/(alpha-beta)) / (alpha - beta)**2
    #     b = (alpha-1)/(alpha - beta)**2 - a*alpha
    #     return (x-beta)**2*(a*x+b)+1
    # alpha = 0.
    def pol(x, alpha, beta):
        return 1-np.exp(-x)
    # x[x > beta] = beta
    x = pol(x, alpha=alpha, beta=beta)
    return x

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

def create_csv(dict_, path, reordering=True, noise=None, shift=False, with_pdb=False):
    df = pd.DataFrame.from_dict(dict_)
    df.set_index('datetime', inplace=True)
    df = df.sort_index(ascending=True)
    df = df.head(len(df)-1)
    if reordering:
        value = []
        for name in list(df):
            value.append(natural_keys(name))
        new_ordering = [x for _,x in sorted(zip(value,list(df)))]
        df = df[new_ordering]
    if noise is not None:
        df *= (1+noise*np.random.normal(0, 1, df.shape))
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)
    # if with_pdb:
    #     pdb.set_trace()
    df.to_csv(path, index=True, sep=',', float_format='%.1f')

    return df
    
def natural_keys(text):
    return int([ c for c in re.split('(\d+)', text) ][1])