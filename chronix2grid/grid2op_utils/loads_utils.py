# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import json
import pandas as pd
import numpy as np

from numpy.random import default_rng


from chronix2grid.getting_started.example.input.generation.patterns import ref_pattern_path
from chronix2grid.generation.consumption import ConsumptionGeneratorBackend


def generate_new_loads(load_seed,
                       load_params,
                       forecasts_params,
                       loads_charac,
                       load_weekly_pattern,
                       data_type='temperature',
                       day_lag=6):
    from chronix2grid.generation.consumption.generate_load import get_add_dim
    from chronix2grid.generation.consumption.consumption_utils import (get_seasonal_pattern,
                                                                       compute_load_pattern)
    from chronix2grid.generation.generation_utils import get_nx_ny_nt
    
    nb_load = len(loads_charac['name'])
    
    # generate the independant data on the mesh
    prng = default_rng(load_seed)
    add_dim = get_add_dim(load_params, loads_charac)
    Nx_comp, Ny_comp, Nt_comp = get_nx_ny_nt(data_type, load_params, add_dim)
    Nh_comp = forecasts_params["nb_h"]
    noise_mesh = prng.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp, Nh_comp))
    
    # retrieve the reference curve "bar"
    weekly_pattern = load_weekly_pattern['test'].values
    # Compute seasonal pattern
    seasonal_pattern_unit = get_seasonal_pattern(load_params)
    seasonal_pattern_load = np.tile(seasonal_pattern_unit, (99,1))
    
    # Get weekly pattern for all loads
    pmax_weekly_pattern = []
    for index, name in enumerate(loads_charac['name']):
        mask = (loads_charac['name'] == name)
        Pmax = loads_charac[mask]['Pmax'].values[0]
        tmp_ = Pmax * compute_load_pattern(load_params, weekly_pattern, index, day_lag)
        pmax_weekly_pattern.append(tmp_.reshape(1, -1))
    load_ref = np.concatenate(pmax_weekly_pattern)
    
    import pdb
    pdb.set_trace()
    seasonal_pattern = np.stack([seasonal_pattern_load for _ in range(len(forecasts_params["h"]))], axis=2)
    residential_series = Pmax * weekly_pattern * (std_temperature_noise * temperature_signal + seasonal_pattern)
    
    pass


def generate_loads(path_env,
                   load_seed,
                   forecast_prng,
                   start_date_dt,
                   end_date_dt,
                   dt,
                   number_of_minutes,
                   generic_params,
                   load_q_from_p_coeff=0.7,
                   day_lag=6):
    """
    This function generates the load for each consumption on a grid

    Parameters
    ----------
    path_env : _type_
        _description_
    load_seed : _type_
        _description_
    start_date_dt : _type_
        _description_
    end_date_dt : _type_
        _description_
    dt : _type_
        _description_
    number_of_minutes : _type_
        _description_
    generic_params : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    with open(os.path.join(path_env, "params_load.json"), "r") as f:
        load_params = json.load(f)
    load_params["start_date"] = start_date_dt
    load_params["end_date"] = end_date_dt
    load_params["dt"] = int(dt)
    load_params["T"] = number_of_minutes
    load_params["planned_std"] = float(generic_params["planned_std"])
    
    forecasts_params = {}
    new_forecasts = False
    path_for_ = os.path.join(path_env, "params_forecasts.json")
    if os.path.exists(path_for_):
        with open(path_for_, "r") as f:
            forecasts_params = json.load(f)
        new_forecasts = True
    
    loads_charac = pd.read_csv(os.path.join(path_env, "loads_charac.csv"), sep=",")
    load_weekly_pattern = pd.read_csv(os.path.join(ref_pattern_path, "load_weekly_pattern.csv"), sep=",")
    
    if new_forecasts:
        load_p, load_p_forecasted = generate_new_loads(load_seed,
                                                       load_params,
                                                       forecasts_params,
                                                       loads_charac,
                                                       load_weekly_pattern,
                                                       day_lag=day_lag)
    else:
        load_generator = ConsumptionGeneratorBackend(out_path=None,
                                                     seed=load_seed, 
                                                     params=load_params,
                                                     loads_charac=loads_charac,
                                                     write_results=False,
                                                     load_config_manager=None,
                                                     day_lag=day_lag)
        
        load_p, load_p_forecasted = load_generator.run(load_weekly_pattern=load_weekly_pattern)
    
    load_q = load_p * load_q_from_p_coeff
    load_q_forecasted = load_p_forecasted * load_q_from_p_coeff
    return new_forecasts, load_p, load_q, load_p_forecasted, load_q_forecasted
