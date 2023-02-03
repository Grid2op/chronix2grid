# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import json
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


def fix_forecast_ramps(nb_h,
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
    sum_pmax_renew = env_for_loss.gen_pmax[env_for_loss.gen_renewable].sum()
    
    p_min = np.repeat(env_for_loss.gen_pmin[env_for_loss.gen_redispatchable].reshape(1,-1)  / scaling_factor,
                      total_step + nb_h,
                      axis=0)
    p_max = np.repeat(env_for_loss.gen_pmax[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                      total_step + nb_h,
                      axis=0)
    
    ramp_min = np.repeat(-env_for_loss.gen_max_ramp_down[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                         total_step + (nb_h-1),
                         axis=0)
    ramp_max = np.repeat(env_for_loss.gen_max_ramp_up[env_for_loss.gen_redispatchable].reshape(1,-1) / scaling_factor,
                         total_step + (nb_h-1),
                         axis=0)
    
    res_gen_p = np.zeros((res_gen_p_forecasted_df.shape[0], total_gen))
    amount_curtailed_for = np.zeros((res_gen_p_forecasted_df.shape[0], ))
    has_error = np.zeros(res_gen_p_forecasted_df.shape[0], dtype=bool)
    t0_errors = []
    errors = []
    for t0 in range(res_gen_p_df.shape[0] - 1):
        # forecast are "consistent from a power system point of view" batch by batch
        # losses are not handled here !
        loss_scale = res_gen_p_df.iloc[t0].sum() / load_p.iloc[t0].sum()
        indx_forecasts = np.arange(t0*nb_h, (t0+1)*nb_h)
        
        prod_renew_for = res_gen_p_forecasted_df.iloc[indx_forecasts, env_for_loss.gen_renewable].sum(axis=1)
        loss_for = (load_f.iloc[indx_forecasts] * loss_scale - prod_renew_for)
        
        # curtailment
        curt_t_scaled = cp.Variable(shape=(total_step + nb_h, ), nonneg=True)
        curt_t = cp.multiply(curt_t_scaled, sum_pmax_renew)
        renew = np.array(([res_gen_p_df.iloc[t0, env_for_loss.gen_renewable].sum()] +  # value in the env
                          prod_renew_for.tolist())
                        )
        
        # generation
        p_t = cp.Variable(shape=(total_step + nb_h, total_gen), pos=True)
        real_p = cp.multiply(p_t, scale_for_loads)
        load = np.array(([res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].sum()] +  # value in the env
                         loss_for.tolist())
                        )
        target_vector = 1.0 * np.concatenate((res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].values.reshape(1, total_gen),
                                              res_gen_p_forecasted_df.iloc[indx_forecasts, env_for_loss.gen_redispatchable].values),
                                              axis=0).reshape(1+nb_h, total_gen)
        
        turned_off_orig = 1.0 * (target_vector == 0.)
        target_vector /= scale_for_loads
        
        constraints = [real_p[0,:] == res_gen_p_df.iloc[t0, env_for_loss.gen_redispatchable].values,
                       p_t >= p_min,
                       p_t <= p_max,
                       p_t[1:,:] - p_t[:-1,:] >= ramp_min,
                       p_t[1:,:] - p_t[:-1,:] <= ramp_max,
                       cp.sum(real_p, axis=1) == (load.reshape(-1) + curt_t),
                       curt_t_scaled <= renew / sum_pmax_renew,
                       curt_t_scaled >= 0.,
                       curt_t_scaled[0] == 0.
                       ]
        
        cost = cp.sum_squares(p_t - target_vector) + cp.norm1(cp.multiply(p_t, turned_off_orig)) + cp.sum_squares(curt_t_scaled)  # TODO normalize last stuff
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve()
        except cp.error.SolverError as exc_:
            t0_errors.append(t0)
            errors.append(RuntimeError(f"cvxpy failed to find a solution for t0 {t0}, error {exc_}"))
            has_error[t0] = True
            continue
        
        load_p.iloc[:12].sum(axis=1)
        load_f.iloc[indx_forecasts]
        # assign the generators
        gen_p_after_optim = real_p.value
        if gen_p_after_optim is None:
            t0_errors.append(t0)
            errors.append(f"cvxpy failed to find a solution for t0 {t0}, and returned None")
            has_error[t0] = True
            continue
            
        if not has_error[t0]:
            res_gen_p[t0] = 1.0 * gen_p_after_optim[1,:]
            amount_curtailed_for[indx_forecasts] = curt_t.value[1:]
            
    # last value is not used anyway
    # res_gen_p[-1, :] = 1.0 * gen_p_after_optim[1,:]
    has_error[-1] = True
        
    res_gen_p_forecasted_df_res = 1.0 * res_gen_p_forecasted_df
    # assign controlable generators
    res_gen_p_forecasted_df_res.iloc[~has_error, env_for_loss.gen_redispatchable] = res_gen_p[~has_error,:]
    
    # fix for curtailment
    total_renew = res_gen_p_forecasted_df_res.iloc[:, env_for_loss.gen_renewable].sum(axis=1)
    total_renew[total_renew <= 1.] = 1.  # normalize for value close to 0.
    ratio = (total_renew - amount_curtailed_for) / total_renew
    res_gen_p_forecasted_df_res.iloc[:, env_for_loss.gen_redispatchable] *= np.stack([ratio for _ in 
                                                                                      range(np.sum(env_for_loss.gen_redispatchable))],
                                                                                     axis=1)
    
    return res_gen_p_forecasted_df_res, t0_errors, errors, amount_curtailed_for


def fix_negative(forecast):
    # inplace
    forecast[forecast < 0.] = 0.
    

def fix_pmax(forecast, pmax):
    # forecasts : (nb_gen, nb_t, nb_h)
    # pmax: (nb_gen)
    # not inplace
    pmax_ = np.stack([pmax for _ in range(forecast.shape[1])], axis=1)
    # pmax_: (nb_gen, nb_t)
    pmax_ = np.stack([pmax_ for _ in range(forecast.shape[-1])], axis=2)
    # pmax_: (nb_gen, nb_t, nb_h)
    
    init_shape = forecast.shape
    forecast_ = forecast.ravel()
    pmax_ = pmax_.ravel()
    mask_ = forecast_ > pmax_
    forecast_[mask_] = pmax_[mask_]
    forecast_ = forecast_.reshape(init_shape)
    return forecast_
    
    
def fix_solar(forecast, real_value, pmax):
    # inplace !
    
    # if value < 0. then it's 0.
    fix_negative(forecast)
    
    # if above pmax then pmax
    forecast = fix_pmax(forecast, pmax)
    
    # no solar at night
    real_value_ = np.stack([real_value for _ in range(forecast.shape[-1])], axis=2)
    forecast[real_value_ <= 1e-3] = 0.


def fix_wind(forecast, real_value, pmax):
    # inplace !
    
    # if value < 0. then it's 0.
    fix_negative(forecast)
    
    # if above pmax then pmax
    forecast = fix_pmax(forecast, pmax)


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
    
    res_gen_p_forecasted = np.stack([res_gen_p_df.values.T for _ in hs], axis=2)
    
    forecasts_params["T"] = load_params["T"]
    res_params["T"] = load_params["T"]
    
    # generate the independant data on the mesh
    for data_type, fun_fix in zip(["solar", "wind"],
                                  [fix_solar, fix_wind]):
        hs_mins, hs, std_hs = get_forecast_parameters(forecasts_params, load_params, data_type)
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
        
        # now retrieve the real noise for each gen of this type
        mask_this_type = gens_charac["type"].values == data_type
        gen_carac_this_type = gens_charac.loc[mask_this_type]
        nb_gen_this_type = gen_carac_this_type.shape[0]
        this_noise = compute_noise(mesh_tmp,
                                   gen_carac_this_type,
                                   model,
                                   range_x, range_y,
                                   delta_x, delta_y,
                                   rho_mesh_x, rho_mesh_y,
                                   nb_t, nb_h, nb_gen_this_type)
        
        # generate all the forecasts
        gen_p_this_type = 1.0 * res_gen_p_df.loc[:, mask_this_type].values.T
        gen_p_for_this_type = get_forecast(gen_p_this_type,
                                           this_noise[:, :-1, :],
                                           hs,
                                           std_hs,
                                           gen_carac_this_type,
                                           reshape=False,
                                           keep_first_dim=True)
        
        # fix the value of the forecast (if above pmax or bellow pmin for example)
        fun_fix(gen_p_for_this_type, gen_p_this_type, gen_carac_this_type["Pmax"].values)
        
        res_gen_p_forecasted[mask_this_type, :, :] = gen_p_for_this_type
        
    res_gen_p_forecasted = res_gen_p_forecasted.reshape(nb_gen, (nb_t - 1) * nb_h)
    res_gen_p_forecasted = np.transpose(res_gen_p_forecasted, (1, 0))
    res_gen_p_forecasted_df = pd.DataFrame(res_gen_p_forecasted, columns=gens_charac["name"])
    return res_gen_p_forecasted_df, nb_h


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
        res_gen_p_forecasted_df, nb_h = generate_new_gen_forecasts(prng,
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
        nb_h = 1
        
    tmp_ = fix_forecast_ramps(nb_h,
                              load_p,
                              load_p_forecasted,
                              res_gen_p_df,
                              res_gen_p_forecasted_df,
                              env_for_loss,
                              hydro_constraints)
    res_gen_p_forecasted_df_res, t0_errors, errors, amount_curtailed_for = tmp_
    return res_gen_p_forecasted_df_res, amount_curtailed_for, t0_errors, errors
