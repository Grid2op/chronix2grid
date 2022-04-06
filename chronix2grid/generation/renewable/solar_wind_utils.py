# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .. import generation_utils as utils
import chronix2grid.constants as cst

def compute_wind_series(prng, locations, Pmax, long_noise, medium_noise, short_noise, params, smoothdist, add_dim):
    # Compute refined signals
    long_scale_signal = utils.interpolate_noise(
        long_noise,
        params,
        locations,
        time_scale=params['long_wind_corr'],
        add_dim=add_dim)
    medium_scale_signal = utils.interpolate_noise(
        medium_noise,
        params,
        locations,
        time_scale=params['medium_wind_corr'],
        add_dim=add_dim)
    short_scale_signal = utils.interpolate_noise(
        short_noise,
        params,
        locations,
        time_scale=params['short_wind_corr'],
        add_dim=add_dim)

    # Compute seasonal pattern
    Nt_inter = int(params['T'] // params['dt'] + 1)
    t = np.linspace(0, params['T'], Nt_inter, endpoint=True)
    start_min = int(
        pd.Timedelta(params['start_date'] - pd.to_datetime('2018/01/01', format='%Y-%m-%d')).total_seconds() // 60)
    seasonal_pattern = np.cos((2 * np.pi / (365 * 24 * 60)) * (t - 30 * 24 * 60 - start_min))

    # Combine signals
    std_short_wind_noise = float(params['std_short_wind_noise'])
    std_medium_wind_noise = float(params['std_medium_wind_noise'])
    std_long_wind_noise = float(params['std_long_wind_noise'])
    signal = (0.7 + 0.3 * seasonal_pattern) * (0.3 + std_medium_wind_noise * medium_scale_signal + std_long_wind_noise * long_scale_signal)
    signal += std_short_wind_noise * short_scale_signal
    signal = 1e-1 * np.exp(4 * signal)
    #signal += prng.uniform(0, SMOOTHDIST/Pmax, signal.shape)
    signal += prng.uniform(0, smoothdist, signal.shape)
    #signal += np.random.uniform(0, smoothdist, signal.shape) #older version - to be removed

    # signal *= 0.95
    signal[signal < 0.] = 0.
    signal = smooth(signal)
    wind_series = Pmax * signal
    wind_series[wind_series > 0.95 * Pmax] = 0.95 * Pmax
    return wind_series

def compute_solar_series(prng, locations, Pmax, solar_noise, params, solar_pattern, smoothdist, time_scale, add_dim, scale_solar_coord_for_correlation=None):

    # Compute noise at desired locations
    if scale_solar_coord_for_correlation is not None:
        locations = [float(scale_solar_coord_for_correlation) * float(locations[0]), float(scale_solar_coord_for_correlation) * float(locations[1])]
    final_noise = utils.interpolate_noise(solar_noise, params, locations, time_scale, add_dim=add_dim)

    # Compute solar pattern
    solar_pattern = compute_solar_pattern(params, solar_pattern)

    # Compute solar time series
    std_solar_noise = float(params['std_solar_noise'])
    if "mean_solar_pattern" in params:
        mean_solar_pattern = float(params["mean_solar_pattern"])
    else:
        # legacy behaviour
        mean_solar_pattern = 0.75
        
    signal = solar_pattern * (mean_solar_pattern + std_solar_noise * final_noise)
    #signal += prng.uniform(0, smoothdist/Pmax, signal.shape) #to be revised: since smmothdist/PMax is very small, the added noise compared to the previous sinal was unsignificant
    #signal += np.random.uniform(0, smoothdist / Pmax, signal.shape) #older version - to be removed
    # signal[signal > 1] = 1
    signal[signal < 0.] = 0.
    signal = smooth(signal)
    solar_series = Pmax * signal
    # solar_series[np.isclose(solar_series, 0.)] = 0
    solar_series[solar_series > 0.95 * Pmax] = 0.95 * Pmax
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


def create_csv(prng, dict_, path, reordering=True, noise=None, shift=False,
               write_results=True, index=False):
    if type(dict_) is dict:
        df = pd.DataFrame.from_dict(dict_)
    else:
        df = dict_.copy()
    df.set_index('datetime', inplace=True)
    df = df.sort_index(ascending=True)
    df = df.head(len(df ) -1)
    if reordering:
        value = []
        for name in list(df):
            value.append(utils.natural_keys(name))
        new_ordering = [x for _ ,x in sorted(zip(value ,list(df)))]
        df = df[new_ordering]
    if noise is not None:
        df *= ( 1 +noise * prng.normal(0, 1, df.shape))
        #df *= (1 + noise * np.random.normal(0, 1, df.shape)) #older version - to be removed
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)
    if write_results:
        df.to_csv(path, index=index, sep=';',
                  float_format=cst.FLOATING_POINT_PRECISION_FORMAT)

    return df

