# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)
import copy
from datetime import datetime, timedelta
import json
import shutil
import time
import pandas as pd
import os
import cvxpy as cp
import numpy as np


from chronix2grid.generation.renewable.generate_solar_wind import get_add_dim as get_add_dim_renew
from chronix2grid.grid2op_utils.noise_generation_utils import (generate_coords_mesh,
                                                               get_load_mesh_tmp,
                                                               compute_noise,
                                                               get_knn_fitted,
                                                               get_forecast_parameters,
                                                               get_iid_noise,
                                                               resize_mesh_factor,
                                                               get_forecast)
    


def fix_forecast_ramps(new_forecasts,
                       load_p,
                       load_p_forecasted,
                       res_gen_p_df,
                       res_gen_p_forecasted_df,
                       env_for_loss,
                       hydro_constraints):
    
    #### cvxpy
    total_step = 1 # for now
    total_gen = np.sum(env_for_loss.gen_redispatchable)
    load_f = load_p_forecasted.sum(axis=1)
    scaling_factor = env_for_loss.gen_pmax[env_for_loss.gen_redispatchable]
    scale_for_loads =  np.repeat(scaling_factor.reshape(1,-1), total_step, axis=0)
    
    p_min = np.repeat(env_for_loss.gen_pmin[env_for_loss.gen_redispatchable].reshape(1,-1)  / scaling_factor,
                      total_step+1,
                      axis=0)
    p_max = np.repeat(env_for_loss.gen_pmax[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                      total_step+1,
                      axis=0)
    
    ramp_min = np.repeat(-env_for_loss.gen_max_ramp_down[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                         total_step,
                            axis=0)
    ramp_max = np.repeat(env_for_loss.gen_max_ramp_up[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                         total_step,
                         axis=0)
    
    res_gen_p = np.zeros((res_gen_p_forecasted_df.shape[0], total_gen))
    has_error = np.zeros(res_gen_p_forecasted_df.shape[0], dtype=bool)
    t0_errors = []
    errors = []
    for t0 in range(res_gen_p_df.shape[0] - 1):
        # forecast are "consistent from a power system point of view" batch by batch
        # losses are not handled here !
        loss_scale = res_gen_p_df.iloc[t0].sum() / load_p.iloc[t0].sum()
        
        p_t = cp.Variable(shape=(total_step + 1, total_gen), pos=True)
        real_p = cp.multiply(p_t, scale_for_loads)
        load = np.array([res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].sum(),  # value in the env
                         load_f.iloc[t0] * loss_scale - res_gen_p_forecasted_df.iloc[t0, ~env_for_loss.gen_redispatchable].sum() # forecast 5 mins later
                        ]
                        )
        target_vector = 1.0 * np.concatenate((res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable],
                                              res_gen_p_forecasted_df.iloc[t0, env_for_loss.gen_redispatchable]),
                                              axis=0).reshape(2, total_gen)
        
        turned_off_orig = 1.0 * (target_vector == 0.)
        target_vector /= scale_for_loads
        
        constraints = [real_p[0,:] == res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].values,
                       p_t >= p_min,
                       p_t <= p_max,
                       p_t[1:,:] - p_t[:-1,:] >= ramp_min,
                       p_t[1:,:] - p_t[:-1,:] <= ramp_max,
                       cp.sum(real_p, axis=1) == load.reshape(-1),
                       ]
        cost = cp.sum_squares(p_t - target_vector) + cp.norm1(cp.multiply(p_t, turned_off_orig))
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve()
        except cp.error.SolverError as exc_:
            t0_errors.append(t0)
            errors.append(RuntimeError(f"cvxpy failed to find a solution for t0 {t0}, error {exc_}"))
            has_error[t0] = True
        
        # assign the generators
        gen_p_after_optim = real_p.value
        if gen_p_after_optim is None:
            t0_errors.append(t0)
            errors.append(f"cvxpy failed to find a solution for t0 {t0}, and returned None")
            has_error[t0] = True
            
        if not has_error[t0]:
            res_gen_p[t0] = 1.0 * gen_p_after_optim[1,:]
            
    # last value is not used anyway
    # res_gen_p[-1, :] = 1.0 * gen_p_after_optim[1,:]
    has_error[-1] = True
        
    res_gen_p_forecasted_df_res = 1.0 * res_gen_p_forecasted_df
    return res_gen_p_forecasted_df_res, t0_errors, errors


def generate_new_gen_forecasts(prng,
                               forecasts_params,
                               load_params,
                               loads_charac,
                               gens_charac,
                               path_env,
                               res_gen_p_df):
    
    # read the parameters from the inputs
    nb_gen = len(gens_charac['name'])
    nb_h = len(forecasts_params["h"])
    datetime_index = pd.date_range(start=load_params['start_date'],
                                   end=load_params['end_date'],
                                   freq=str(load_params['dt']) + 'min')
    nb_t = datetime_index.shape[0]
    hs_mins, hs, std_hs = get_forecast_parameters(forecasts_params, load_params)
    
    # compute the "real" size of the mesh 
    delta_x, delta_y, range_x, range_y = resize_mesh_factor(loads_charac, gens_charac)
    
    # load the parameters controling the RES
    with open(os.path.join(path_env, "params_res.json"), "r") as f:
        res_params = json.load(f)
        
    # generate the independant data on the mesh
    for data_type in ["solar"]:
        tmp_ = get_iid_noise(None, res_params, forecasts_params,
                             loads_charac, data_type, get_add_dim_renew, prng)
        noise_mesh, (Nx_comp, Ny_comp, Nt_comp, Nh_comp) = tmp_
    
        # get the inteporlation on the mesh
        res_mesh = generate_coords_mesh(Nx_comp,
                                        Ny_comp,
                                        Nt_comp,
                                        Nh_comp,
                                        noise_mesh.size) 
        coords_mesh, rho_mesh_x, rho_mesh_y, rho_mesh_t, rho_mesh_h = res_mesh
        
        # "fit" the kNN    
        model = get_knn_fitted(forecasts_params, coords_mesh, noise_mesh)
    
        mesh_tmp = get_load_mesh_tmp(nb_t, hs, hs_mins,
                                     rho_mesh_t, rho_mesh_h)
        
        # now retrieve the real noise for each load
        gen_this_type = gens_charac.loc[gens_charac["type"] == data_type]
        nb_gen_this_type = gen_this_type.shape[0]
        this_noise = compute_noise(mesh_tmp,
                                   gen_this_type,
                                   model,
                                   range_x, range_y,
                                   delta_x, delta_y,
                                   rho_mesh_x, rho_mesh_y,
                                   nb_t, nb_h, nb_gen_this_type)
        
        # generate all the forecasts
        gen_p_this_type = res_gen_p_df[gens_charac["type"].value == data_type]
        lgen_p_for_this_type = get_forecast(gen_p_this_type, this_noise, hs, std_hs, loads_charac)
        
    return res_gen_p_forecasted_df


def generate_forecasts_gen(new_forecasts,
                           prng,
                           load_p,
                           load_p_forecasted, 
                           res_gen_p_df,
                           sigma,
                           env_for_loss,
                           hydro_constraints,
                           forecasts_params,
                           load_params,
                           loads_charac,
                           gens_charac,
                           path_env):
    if new_forecasts:
        res_gen_p_forecasted_df = generate_new_gen_forecasts(prng,
                                                             forecasts_params,
                                                             load_params,
                                                             loads_charac,
                                                             gens_charac,
                                                             path_env,
                                                             res_gen_p_df)
    else:
        res_gen_p_forecasted_df = res_gen_p_df * prng.lognormal(mean=0.0,
                                                                sigma=sigma,
                                                                size=res_gen_p_df.shape)
        res_gen_p_forecasted_df = res_gen_p_forecasted_df.shift(-1)
        res_gen_p_forecasted_df.iloc[-1] = 1.0 * res_gen_p_forecasted_df.iloc[-2]
        
    res_gen_p_forecasted_df_res, t0_errors, errors = fix_forecast_ramps(new_forecasts,
                                                                        load_p,
                                                                        load_p_forecasted,
                                                                        res_gen_p_df,
                                                                        res_gen_p_forecasted_df,
                                                                        env_for_loss,
                                                                        hydro_constraints)
    return res_gen_p_forecasted_df_res, t0_errors, errors
