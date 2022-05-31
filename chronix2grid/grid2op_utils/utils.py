# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)


# this generates some chronics for a given environment, provided that all necessary files are present in its repo
import copy
from datetime import datetime, timedelta
import json
import shutil
import time
import pandas as pd
import os
import cvxpy as cp
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChangeNothing, FromNPY
from grid2op.Action import DontAct
from grid2op.Opponent import NeverAttackBudget, BaseOpponent
from lightsim2grid import LightSimBackend
import numpy as np
from numpy.random import default_rng

from chronix2grid.generation.consumption import ConsumptionGeneratorBackend
from chronix2grid.generation.renewable import RenewableBackend
from chronix2grid.generation.dispatch.PypsaDispatchBackend import PypsaDispatcher
from chronix2grid.getting_started.example.input.generation.patterns import ref_pattern_path
from chronix2grid.generation.dispatch.EconomicDispatch import ChroniXScenario

import warnings

FLOATING_POINT_PRECISION_FORMAT = '%.1f'

# TODO allow for a "debug" mode where we can save the values for the prices, the renewables generated, the renewables after dispatch 
# and the renewables after the losses
# TODO add a parameter to generate data more correlated for data in the same area but less correlated within different area.


def get_last_scenario_id(env_chronics_dir):
    """This function return the last scenario id identified.
    
    It only works with scenario id formatted like: `WHATEVER_ScenID`, for example: "2050-01-03_0" or "2050_01_03_0"
    but not "2050-01-03-0"

    It also supposes that the directory containing the chronics exists on the hard drive (even if it's empty)
    
    Parameters
    ----------
    env_chronics_dir : _type_
        _description_
    """
    max_ = None
    list_files = os.listdir(env_chronics_dir)
    for el in list_files:
        this_file_path = os.path.join(env_chronics_dir, el)
        if not os.path.isdir(this_file_path):
            continue
        try:
            *date_, scen_id = el.split("_")
        except ValueError:
            continue
        
        try:
            scen_id = int(scen_id)
        except ValueError:
            continue
        
        if max_ is None:
            max_ = scen_id
            
        if scen_id > max_:
            max_ = scen_id
    if max_ is None:
        max_ = -1
    return max_
            
    
def generate_loads(path_env, load_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params,
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
    
    loads_charac = pd.read_csv(os.path.join(path_env, "loads_charac.csv"), sep=",")
    load_weekly_pattern = pd.read_csv(os.path.join(ref_pattern_path, "load_weekly_pattern.csv"), sep=",")
    
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
    return load_p, load_q, load_p_forecasted, load_q_forecasted


def generate_renewable_energy_sources(path_env, renew_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params, gens_charac):
    """This function generates the amount of power produced by renewable energy sources (res). 
    
    It serves as a maximum value for the economic dispatch. 

    Parameters
    ----------
    path_env : _type_
        _description_
    renew_seed : _type_
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
    gens_charac : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    with open(os.path.join(path_env, "params_res.json"), "r") as f:
        renew_params = json.load(f)
    renew_params["start_date"] = start_date_dt
    renew_params["end_date"] = end_date_dt
    renew_params["dt"] = int(dt)
    renew_params["T"] = number_of_minutes
    renew_params["planned_std"] = float(generic_params["planned_std"])
    solar_pattern = np.load(os.path.join(ref_pattern_path, "solar_pattern.npy"))
    renew_backend = RenewableBackend(out_path=None,
                                     seed=renew_seed,
                                     params=renew_params,
                                     loads_charac=gens_charac,
                                     res_config_manager=None,
                                     write_results=False)
    prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = renew_backend.run(solar_pattern=solar_pattern)
    return prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted


def generate_economic_dispatch(path_env, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params, 
                               load_p, prod_solar, prod_wind, name_gen, gen_type, scenario_id, final_gen_p, gens_charac):
    """This function emulates a perfect market where all productions need to meet the demand at the minimal cost.
    
    It does not consider limit on powerline, nor contigencies etc. The power network does not exist here. Only the ramps and
    pmin / pmax are important.

    Parameters
    ----------
    path_env : _type_
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
    load_p : _type_
        _description_
    prod_solar : _type_
        _description_
    prod_wind : _type_
        _description_
    env : _type_
        _description_
    scenario_id : _type_
        _description_
    final_gen_p : _type_
        _description_
    gens_charac : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    with open(os.path.join(path_env, "params_opf.json"), "r") as f:
        opf_params = json.load(f)
    opf_params["start_date"] = start_date_dt
    opf_params["end_date"] = end_date_dt
    opf_params["dt"] = int(dt)
    opf_params["T"] = number_of_minutes
    opf_params["planned_std"] = float(generic_params["planned_std"])
    
    load = pd.DataFrame(load_p.sum(axis=1))
    total_solar = prod_solar.sum(axis=1)
    total_wind = prod_wind.sum(axis=1)
    
    # init the dispatcher
    gens_charac_this = copy.deepcopy(gens_charac)
    gens_charac_this["pmax"] = gens_charac_this["Pmax"]
    gens_charac_this["pmin"] = gens_charac_this["Pmin"]
    gens_charac_this["cost_per_mw"] = gens_charac_this["marginal_cost"]
    economic_dispatch = PypsaDispatcher.from_dataframe(gens_charac_this)
    
    # need to hack it to work...
    n_gen = len(name_gen)
    gen_p_orig = np.zeros((prod_solar.shape[0], n_gen))
    economic_dispatch._chronix_scenario = ChroniXScenario(loads=1.0 * load_p,
                                                          prods=pd.DataFrame(1.0 * gen_p_orig, columns=name_gen),
                                                          scenario_name=scenario_id,
                                                          res_names={"wind": name_gen[gen_type == "wind"],
                                                                     "solar": name_gen[gen_type == "solar"]
                                                                    }
                                                         )
    economic_dispatch.read_hydro_guide_curves(os.path.join(ref_pattern_path, 'hydro_french.csv'))
    hydro_constraints = economic_dispatch.make_hydro_constraints_from_res_load_scenario()
    res_dispatch = economic_dispatch.run(load * (1.0 + 0.01 * float(opf_params["losses_pct"])),
                                         total_solar,
                                         total_wind,
                                         opf_params,
                                         gen_constraints=hydro_constraints,
                                         pyomo=False,
                                         solver_name="cbc")
    
    if res_dispatch is None:     
        error_ = RuntimeError("Pypsa failed to find a solution")
        return None, None, None, error_
    
    # now assign the results
    final_gen_p = 1.0 * final_gen_p  # copy the data frame to avoid modify the original one
    for gen_id, gen_nm in enumerate(name_gen):
        if gen_nm in res_dispatch.chronix.prods_dispatch:
            final_gen_p.iloc[:, gen_id] = 1.0 * res_dispatch.chronix.prods_dispatch[gen_nm].values
    
    #handle curtailment
    mask_wind = total_wind.values > 0.001
    wind_curt = (res_dispatch.chronix.prods_dispatch['agg_wind'].values[mask_wind] / total_wind.values[mask_wind]).reshape(-1,1)
    final_gen_p.iloc[mask_wind, gen_type == "wind"] *= wind_curt
    
    mask_solar = total_solar.values > 0.001  # be carefull not to divide by 0 in case of solar !
    solar_curt = (res_dispatch.chronix.prods_dispatch['agg_solar'].values[mask_solar] / total_solar.values[mask_solar]).reshape(-1,1)
    final_gen_p.iloc[mask_solar, gen_type == "solar"] *= solar_curt
    
    total_wind_curt = total_wind.values.sum() - res_dispatch.chronix.prods_dispatch['agg_wind'].values.sum()
    total_solar_curt = total_solar.values[mask_solar].sum() - res_dispatch.chronix.prods_dispatch['agg_solar'].values[mask_solar].sum()
    return final_gen_p, total_wind_curt, total_solar_curt, None


def _adjust_gens_old(all_loss_orig,
                 env_for_loss,
                 datetimes,
                 total_solar,
                 total_wind,
                 params,
                 env_path,
                 env_param,
                 load_without_loss,
                 load_p, 
                 load_q,
                 gen_p,
                 gen_v,
                 economic_dispatch,
                 diff_,
                 threshold_stop=0.1,  # stop when all generators move less that this
                 max_iter=100,  # declare a failure after this number of iteration
                 iter_quality_decrease=50,  # acept a reduction of the quality after this number of iteration
                 percentile_quality_decrease=99,
                 ):
    """This function is an auxilliary function.
    
    Like its main one (see handle_losses) it is here to make sure that if you run an AC model with the data generated, 
    then the generator setpoints will not change too much 
    (less than `threshold_stop` MW)

    Parameters
    ----------
    all_loss_orig : _type_
        _description_
    env_for_loss : _type_
        _description_
    datetimes : _type_
        _description_
    total_solar : _type_
        _description_
    total_wind : _type_
        _description_
    params : _type_
        _description_
    env_path : _type_
        _description_
    env_param : _type_
        _description_
    load_without_loss : _type_
        _description_
    load_p : _type_
        _description_
    load_q : _type_
        _description_
    gen_p : _type_
        _description_
    gen_v : _type_
        _description_
    economic_dispatch : _type_
        _description_
    diff_ : _type_
        _description_
    threshold_stop : float, optional
        _description_, by default 0.1

    Returns
    -------
    _type_
        _description_
    """
    quality_ = None
    error_ = None
    if np.any(~np.isfinite(gen_p)):
        error_ = RuntimeError("Input data contained Nans !")
        return None, error_, quality_
    all_loss = all_loss_orig
    res_gen_p = 1.0 * gen_p
    iter_num = 0
    hydro_constraints = economic_dispatch.make_hydro_constraints_from_res_load_scenario()
    while True:
        iter_num += 1
        load = load_without_loss + all_loss
        load = pd.DataFrame(load.ravel(), index=datetimes)
        
        # "never" decrease (during iteration) some generators
        min__ = diff_.min()  # this is negative
        gen_max_pu_t = None
        gen_min_pu_t = {gen_nm: np.maximum((res_gen_p[:,gen_id] + min__) / economic_dispatch.generators.loc[gen_nm].p_nom,
                                            env_for_loss.gen_pmin[gen_id] / economic_dispatch.generators.loc[gen_nm].p_nom
                                            )
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        
        ### run the dispatch with the loss
        dispatch_res = economic_dispatch.run(load,
                                             total_solar=total_solar,
                                             total_wind=total_wind,
                                             params=params,
                                             pyomo=False,
                                             solver_name="cbc",
                                             gen_constraints=copy.deepcopy(hydro_constraints),
                                             gen_max_pu_t=gen_max_pu_t,
                                             gen_min_pu_t=gen_min_pu_t,
                                             )
        
        if dispatch_res is None:     
            error_ = RuntimeError(f"Pypsa failed to find a solution at iteration {iter_num}")
            break
        
        # assign the generators
        for gen_id, gen_nm in enumerate(env_for_loss.name_gen):
            if gen_nm in dispatch_res.chronix.prods_dispatch:
                res_gen_p[:, gen_id] = 1.0 * dispatch_res.chronix.prods_dispatch[gen_nm].values
                  
        #handle wind curtailment
        mask_winds = total_wind.values > 0.001
        res_gen_p[mask_winds, :][:,env_for_loss.gen_type == "wind"] *= (dispatch_res.chronix.prods_dispatch['agg_wind'].values[mask_winds] / total_wind.values[mask_winds]).reshape(-1,1)
        total_wind.loc[mask_winds] = 1.0 * dispatch_res.chronix.prods_dispatch["agg_wind"].values[mask_winds]
        
        # re evaluate the losses
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_fixed = grid2op.make(
                env_path,
                test=True,
                # grid_path=grid_path, # assign it the 118 grid
                param=env_param,
                backend=LightSimBackend(),
                chronics_class=FromNPY,
                # chronics_path=path_chronix2grid,
                data_feeding_kwargs={"load_p": load_p,
                                     "load_q": load_q,
                                     "prod_p": 1.0 * res_gen_p,
                                     "prod_v": gen_v}
                )
        diff_ = np.full((env_fixed.max_episode_duration(), env_fixed.n_gen), fill_value=np.NaN)
        all_loss[:] = np.NaN
        
        i = 0
        obs = env_fixed.reset()
        all_loss[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
        diff_[i] = obs.gen_p - res_gen_p[i]

        
        done = False
        while not done:
            obs, reward, done, info = env_fixed.step(env_fixed.action_space())
            i += 1
            if done:
                break
            all_loss[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
            diff_[i] = obs.gen_p - res_gen_p[i]
        
        max_diff_ = np.abs(diff_).max()
        if not np.isfinite(max_diff_):
            error_ = RuntimeError(f"Some nans were found in the generated data at iteration {iter_num}")
            res_gen_p = None
            quality_ = None
            break
            
            
        if max_diff_ <= threshold_stop:
            quality_ = (iter_num,
                        float(np.mean(np.abs(diff_))),
                        float(np.percentile(np.abs(diff_), 95)),
                        float(np.percentile(np.abs(diff_), 99)),
                        float(max_diff_)
            )
            break
        
        if iter_num >= iter_quality_decrease:
            quantile = np.percentile(np.abs(diff_), percentile_quality_decrease)
            if quantile <= threshold_stop:
                quality_ = (iter_num,
                            float(np.mean(np.abs(diff_))),
                            float(np.percentile(np.abs(diff_), 95)),
                            float(np.percentile(np.abs(diff_), 99)),
                            float(np.max(np.abs(diff_)))
                )
                break
                    
        if iter_num >= max_iter:
            error_ = RuntimeError("Too much iterations performed when adjusting for the losses")
            res_gen_p = None
            quality_ = None
            break
        
    return res_gen_p, error_, quality_


def _adjust_gens(all_loss_orig,
                 env_for_loss,
                 datetimes,
                 total_solar,
                 total_wind,
                 params,
                 env_path,
                 env_param,
                 load_without_loss,
                 load_p, 
                 load_q,
                 gen_p,
                 gen_v,
                 economic_dispatch,
                 diff_,
                 threshold_stop=0.1,  # stop when all generators move less that this
                 max_iter=100,  # declare a failure after this number of iteration
                 iter_quality_decrease=50,  # acept a reduction of the quality after this number of iteration
                 percentile_quality_decrease=99,
                 ):
    """This function is an auxilliary function.
    
    Like its main one (see handle_losses) it is here to make sure that if you run an AC model with the data generated, 
    then the generator setpoints will not change too much 
    (less than `threshold_stop` MW)

    Parameters
    ----------
    all_loss_orig : _type_
        _description_
    env_for_loss : _type_
        _description_
    datetimes : _type_
        _description_
    total_solar : _type_
        _description_
    total_wind : _type_
        _description_
    params : _type_
        _description_
    env_path : _type_
        _description_
    env_param : _type_
        _description_
    load_without_loss : _type_
        _description_
    load_p : _type_
        _description_
    load_q : _type_
        _description_
    gen_p : _type_
        _description_
    gen_v : _type_
        _description_
    economic_dispatch : _type_
        _description_
    diff_ : _type_
        _description_
    threshold_stop : float, optional
        _description_, by default 0.1

    Returns
    -------
    _type_
        _description_
    """
    quality_ = None
    error_ = None
    if np.any(~np.isfinite(gen_p)):
        error_ = RuntimeError("Input data contained Nans !")
        return None, error_, quality_
    all_loss = all_loss_orig
    res_gen_p = 1.0 * gen_p
    iter_num = 0
    hydro_constraints = economic_dispatch.make_hydro_constraints_from_res_load_scenario()
    
    # defined some global variable (used for all optimization problems)
    turned_off_orig = 1.0 * (gen_p[:, env_for_loss.gen_redispatchable] == 0.)
    ids_hyrdo = []
    total_gen = np.sum(env_for_loss.gen_redispatchable)
    total_step = total_solar.shape[0]
    gen_id = 0
    for i in range(env_for_loss.n_gen):
        if env_for_loss.gen_redispatchable[i]:
            if env_for_loss.gen_type[i] == "hydro":
                ids_hyrdo.append(gen_id)
            gen_id += 1
    ids_hyrdo = np.array(ids_hyrdo)
    
    # define the constraints    
    scaling_factor = env_for_loss.gen_pmax[env_for_loss.gen_redispatchable]
    p_min = np.repeat(env_for_loss.gen_pmin[env_for_loss.gen_redispatchable].reshape(1,-1)  / scaling_factor,
                        total_step,
                        axis=0)
    p_max = np.repeat(env_for_loss.gen_pmax[env_for_loss.gen_redispatchable].reshape(1,-1) * params["PmaxErrorCorrRatio"] / scaling_factor,
                        total_step,
                        axis=0)
    
    ramp_min = np.repeat(-env_for_loss.gen_max_ramp_down[env_for_loss.gen_redispatchable].reshape(1,-1) * params["RampErrorCorrRatio"]  / scaling_factor,
                            total_step - 1,
                            axis=0)
    ramp_max = np.repeat(env_for_loss.gen_max_ramp_up[env_for_loss.gen_redispatchable].reshape(1,-1) * params["RampErrorCorrRatio"]  / scaling_factor,
                            total_step - 1,
                            axis=0)
    p_max[:, ids_hyrdo] = 1.0 * hydro_constraints["p_max_pu"].values
     
    while True:
        iter_num += 1
        load = load_without_loss + all_loss
        load = pd.DataFrame(load.ravel(), index=datetimes)
        
        # "never" decrease (during iteration) some generators
        min__ = diff_.min()  # this is negative
        load = load_without_loss + all_loss - np.sum(res_gen_p[:,~env_for_loss.gen_redispatchable], axis=1)
        scale_for_loads =  np.repeat(scaling_factor.reshape(1,-1), total_step, axis=0)
        target_vector = res_gen_p[:,env_for_loss.gen_redispatchable] / scaling_factor         
        
        #### cvxpy
        p_t = cp.Variable(shape=(total_step,total_gen), pos=True)
        real_p = cp.multiply(p_t, scale_for_loads)
        
        constraints = [p_t >= p_min,
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
            error_ = RuntimeError(f"cvxpy failed to find a solution at iteration {iter_num}, error {exc_}")
            res_gen_p = None
            quality_ = None
            break
            
        # assign the generators
        gen_p_after_optim = real_p.value
        if gen_p_after_optim is None:
            error_ = RuntimeError(f"cvxpy failed to find a solution at iteration {iter_num}, and returned None.")
            res_gen_p = None
            quality_ = None
            break
            
        id_redisp = 0
        for gen_id, gen_nm in enumerate(env_for_loss.name_gen):
            if env_for_loss.gen_redispatchable[gen_id]:
                res_gen_p[:, gen_id] = 1.0 * gen_p_after_optim[:, id_redisp]
                id_redisp += 1
        
        # re evaluate the losses
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_fixed = grid2op.make(
                env_path,
                test=True,
                # grid_path=grid_path, # assign it the 118 grid
                param=env_param,
                backend=LightSimBackend(),
                chronics_class=FromNPY,
                # chronics_path=path_chronix2grid,
                data_feeding_kwargs={"load_p": load_p,
                                     "load_q": load_q,
                                     "prod_p": 1.0 * res_gen_p,
                                     "prod_v": gen_v},
                opponent_budget_per_ts=0.,
                opponent_init_budget=0.,
                opponent_class=BaseOpponent,
                opponent_budget_class=NeverAttackBudget,
                opponent_action_class=DontAct,
                )
        diff_ = np.full((env_fixed.max_episode_duration(), env_fixed.n_gen), fill_value=np.NaN)
        all_loss[:] = np.NaN
        
        i = 0
        obs = env_fixed.reset()
        all_loss[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
        diff_[i] = obs.gen_p - res_gen_p[i]

        done = False
        while not done:
            obs, reward, done, info = env_fixed.step(env_fixed.action_space())
            i += 1
            if done:
                break
            all_loss[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
            diff_[i] = obs.gen_p - res_gen_p[i]
        
        max_diff_ = np.abs(diff_).max()
        print(f"{iter_num = } : {max_diff_ = :.2f}")
        if not np.isfinite(max_diff_):
            error_ = RuntimeError(f"Some nans were found in the generated data at iteration {iter_num}")
            res_gen_p = None
            quality_ = None
            break
            
        if max_diff_ <= threshold_stop:
            quality_ = (iter_num,
                        float(np.mean(np.abs(diff_))),
                        float(np.percentile(np.abs(diff_), 95)),
                        float(np.percentile(np.abs(diff_), 99)),
                        float(max_diff_)
            )
            break
        
        if iter_num >= iter_quality_decrease:
            quantile = np.percentile(np.abs(diff_), percentile_quality_decrease)
            if quantile <= threshold_stop:
                quality_ = (iter_num,
                            float(np.mean(np.abs(diff_))),
                            float(np.percentile(np.abs(diff_), 95)),
                            float(np.percentile(np.abs(diff_), 99)),
                            float(np.max(np.abs(diff_)))
                )
                break
                    
        if iter_num >= max_iter:
            error_ = RuntimeError("Too much iterations performed when adjusting for the losses")
            res_gen_p = None
            quality_ = None
            break
        
    return res_gen_p, error_, quality_


def _fix_losses_one_scenario(env_for_loss,
                            scenario_id,
                            params,
                            env_path,
                            env_param,
                            load_df,
                            threshold_stop=0.05,  # decide I stop when the data move of less of 0.5 MW at maximum
                            max_iter=100,  # maximum number of iteration
                            iter_quality_decrease=20,  # after 20 iteration accept a degradation in the quality
                            percentile_quality_decrease=99,  # replace the "at maximum" by "percentile 99%"
                            ):
    """This function is an auxilliary function.
    
    Like its main one (see handle_losses) it is here to make sure that if you run an AC model with the data generated, 
    then the generator setpoints will not change too much 
    (less than `threshold_stop` MW)

    Parameters
    ----------
    env_for_loss : _type_
        _description_
    scenario_id : _type_
        _description_
    params : _type_
        _description_
    env_path : _type_
        _description_
    env_param : _type_
        _description_
    load_df : _type_
        _description_
    threshold_stop : float, optional
        _description_, by default 0.5
    max_iter : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    """
    gen_p_orig = np.full((env_for_loss.max_episode_duration(), env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_gen_v = np.full((env_for_loss.max_episode_duration(), env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_load_p = np.full((env_for_loss.max_episode_duration(), env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    final_load_q = np.full((env_for_loss.max_episode_duration(), env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    all_loss_orig = np.zeros(env_for_loss.max_episode_duration())
    max_diff_orig = np.zeros(env_for_loss.max_episode_duration())
    datetimes = np.zeros(env_for_loss.max_episode_duration(), dtype=datetime)
    
    env_for_loss.set_id(scenario_id)
    obs = env_for_loss.reset()
    
    i = 0
    all_loss_orig[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
    final_gen_v[i] = obs.gen_v
    final_load_p[i] = obs.load_p
    final_load_q[i] = obs.load_q
    gen_p_orig[i] = 1.0 * obs.gen_p
    datetimes[i] = obs.get_time_stamp()
    max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data._prod_p[i]))
    
    done = False
    while not done:
        obs, reward, done, info = env_for_loss.step(env_for_loss.action_space())
        if done:
            break
        i += 1
        all_loss_orig[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
        final_load_p[i] = 1.0 * obs.load_p
        final_load_q[i] = 1.0 * obs.load_p
        gen_p_orig[i] = env_for_loss.chronics_handler.real_data._prod_p[i]  # 1.0 * obs.gen_p
        datetimes[i] = obs.get_time_stamp()
        max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data._prod_p[i]))
        
    total_solar = np.sum(gen_p_orig[:, env_for_loss.gen_type == "solar"], axis=1)
    total_wind = np.sum(gen_p_orig[:, env_for_loss.gen_type == "wind"], axis=1)
    load_without_loss = np.sum(final_load_p, axis=1) #  - total_solar - total_wind
    
    # load the right data
    df = pd.read_csv(os.path.join(env_path, "prods_charac.csv"), sep=",")
    df["pmax"] = df["Pmax"]
    df["pmin"] = df["Pmin"]
    df["cost_per_mw"] = df["marginal_cost"]
    economic_dispatch = PypsaDispatcher.from_dataframe(df)
    economic_dispatch.read_hydro_guide_curves(os.path.join(ref_pattern_path, 'hydro_french.csv'))
    economic_dispatch._chronix_scenario = ChroniXScenario(loads=1.0 * load_df,
                                                          prods=pd.DataFrame(1.0 * gen_p_orig, columns=env_for_loss.name_gen),
                                                          scenario_name=scenario_id,
                                                          res_names= {"wind": env_for_loss.name_gen[env_for_loss.gen_type == "wind"],
                                                                      "solar": env_for_loss.name_gen[env_for_loss.gen_type == "solar"]
                                                          }
                                                         )
    
    error_ = None
    total_solar_orig = pd.Series(total_solar.ravel(), index=datetimes)
    total_wind_orig = pd.Series(total_wind.ravel(), index=datetimes)
    
    total_solar = 1.0 * total_solar_orig
    total_wind = 1.0 * total_wind_orig
    res_gen_p = 1.0 * gen_p_orig
    diff_ = 1.0 * max_diff_orig
    diff_ = diff_.reshape(-1,1)
    res_gen_p, error_, quality_ = _adjust_gens(all_loss_orig,
                                               env_for_loss,
                                               datetimes,
                                               total_solar,
                                               total_wind,
                                               params,
                                               env_path,
                                               env_param,
                                               load_without_loss,
                                               final_load_p, 
                                               final_load_q,
                                               gen_p_orig,
                                               final_gen_v,
                                               economic_dispatch,
                                               diff_,
                                               threshold_stop=threshold_stop,
                                               max_iter=max_iter,
                                               iter_quality_decrease=iter_quality_decrease,
                                               percentile_quality_decrease=percentile_quality_decrease)
    
    if error_ is not None:
        # the procedure failed
        return None, error_, None
    
    return res_gen_p, error_, quality_


def handle_losses(path_env,
                  n_gen,
                  name_gen,
                  gens_charac,
                  load_p,
                  load_q,
                  final_gen_p,
                  start_date_dt,
                  dt_dt,
                  scenario_id, 
                  PmaxErrorCorrRatio=0.9,
                  RampErrorCorrRatio=0.95,
                  threshold_stop=0.05,
                  max_iter=100,
                  iter_quality_decrease=20,  # after 20 iteration accept a degradation in the quality
                  percentile_quality_decrease=99,  # replace the "at maximum" by "percentile 99%"
                  ):
    """This function is here to make sure that if you run an AC model with the data generated, then the generator setpoints will not change too much 
    (less than `threshold_stop` MW)

    Parameters
    ----------
    path_env : _type_
        _description_
    n_gen : _type_
        _description_
    gens_charac : _type_
        _description_
    load_p : _type_
        _description_
    load_q : _type_
        _description_
    final_gen_p : _type_
        _description_
    start_date_dt : _type_
        _description_
    dt_dt : _type_
        _description_
    scenario_id : _type_
        _description_
    PmaxErrorCorrRatio : float, optional
        _description_, by default 0.9
    RampErrorCorrRatio : float, optional
        _description_, by default 0.95
    threshold_stop : float, optional
        _description_, by default 0.5
    max_iter : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    """
    
    with open(os.path.join(path_env, "params_opf.json"), "r") as f:
        loss_param = json.load(f)
    loss_param["loss_pct"] = 0.  # losses are handled better in this function
    loss_param["PmaxErrorCorrRatio"] = PmaxErrorCorrRatio
    loss_param["RampErrorCorrRatio"] = RampErrorCorrRatio
    
    # do not treat the slack differently
    loss_param["slack_ramp_limit_ratio"] = loss_param["RampErrorCorrRatio"]
    if "slack_pmin" in loss_param:
        del loss_param["slack_pmin"]
    if "slack_pmax" in loss_param:
        del loss_param["slack_pmax"]
        
    env_param = Parameters()
    env_param.NO_OVERFLOW_DISCONNECTION = True
    gen_v = np.tile(np.array([float(gens_charac.loc[gens_charac["name"] == nm_gen].V) for nm_gen in name_gen ]),
                    load_p.shape[0]).reshape(-1, n_gen)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_for_loss = grid2op.make(
            path_env,
            test=True,
            # grid_path=grid_path, # assign it the 118 grid
            param=env_param,
            backend=LightSimBackend(),
            chronics_class=FromNPY,
            # chronics_path=path_chronix2grid,
            data_feeding_kwargs={"load_p": load_p.values,  # np.concatenate([load_p.values[0].reshape(1,-1), load_p.values]),
                                 "load_q": load_q.values,  # np.concatenate([load_q.values[0].reshape(1,-1), load_q.values]),
                                 "prod_p": 1.0 * final_gen_p.values,  # 1.0 * np.concatenate([final_gen_p.values[0].reshape(1,-1), final_gen_p.values]),
                                 "prod_v": gen_v,  # np.concatenate([gen_v[0].reshape(1,-1), gen_v])}
                                 "start_datetime": start_date_dt,
                                 "time_interval": dt_dt,
            }
            )
    res_gen_p, error_, quality_ = _fix_losses_one_scenario(env_for_loss,
                                                           scenario_id,
                                                           loss_param,
                                                           path_env,
                                                           env_for_loss.parameters,
                                                           load_df=load_p,
                                                           threshold_stop=threshold_stop,
                                                           max_iter=max_iter,
                                                           # after 20 iteration accept a degradation in the quality
                                                           iter_quality_decrease=iter_quality_decrease,  
                                                           # replace the "at maximum" by "percentile 99%"
                                                           percentile_quality_decrease=percentile_quality_decrease,  
                                                           )
    if error_ is not None:
        return None, error_, None
    
    # reformat the generators
    res_gen_p_df = pd.DataFrame(res_gen_p, index=final_gen_p.index, columns=env_for_loss.name_gen)
    
    return res_gen_p_df, error_, quality_

def save_generated_data(this_scen_path,
                        load_p,
                        load_p_forecasted,
                        load_q,
                        load_q_forecasted,
                        prod_p_generated,
                        prod_p_after_dispatch,
                        prod_p,  # after everything, the one the agent sees
                        prod_p_forecasted,
                        debug,
                        sep=';',
                        float_prec=FLOATING_POINT_PRECISION_FORMAT):
    """This function saves the data that have been generated by this script.

    Parameters
    ----------
    this_scen_path : _type_
        _description_
    load_p : _type_
        _description_
    load_p_forecasted : _type_
        _description_
    load_q : _type_
        _description_
    load_q_forecasted : _type_
        _description_
    prod_p_generated: _type_
        generators just after renewable generation (non renewables gens are Nans here !)
    prod_p : _type_
        generators after dispatch and losses !
    prod_p_forecasted : _type_
        _description_
    sep : str, optional
        _description_, by default ';'
    float_prec : _type_, optional
        _description_, by default FLOATING_POINT_PRECISION_FORMAT
    """
    li_dfs = [load_p, load_p_forecasted, load_q, load_q_forecasted, prod_p, prod_p_forecasted]
    li_nms = ["load_p", "load_p_forecasted", "load_q", "load_q_forecasted", "prod_p", "prod_p_forecasted"]
    if debug:
        li_dfs.append(prod_p_generated)
        li_nms.append("prod_p_generated")
        li_dfs.append(prod_p_after_dispatch)
        li_nms.append("prod_p_after_dispatch")
    for df, nm in zip(li_dfs, li_nms):
        df.to_csv(os.path.join(this_scen_path, f'{nm}.csv.bz2'),
                  sep=sep,
                  float_format=float_prec,
                  header=True,
                  index=False)       


def save_meta_data(this_scen_path,
                   path_env,
                   start_date_dt,
                   dt_dt : timedelta,
                   load_seed,
                   renew_seed,
                   gen_p_forecast_seed,
                   quality,
                   total_load,
                   total_gen,
                   losses_mwh,
                   losses_avg,
                   wind_curtailed_opf,
                   wind_curtailed_losses,
                   solar_curtailed_opf,
                   generation_time,
                   saving_time,
                   files_to_copy=("maintenance_meta.json",)):
    """This function saves the "meta data" required for a succesful grid2op run !

    Parameters
    ----------
    this_scen_path : _type_
        _description_
    path_env : _type_
        _description_
    start_date_dt : _type_
        _description_
    dt_dt : timedelta
        _description_
    load_seed : _type_
        _description_
    renew_seed : _type_
        _description_
    gen_p_forecast_seed : _type_
        _description_
    quality: tuple

    """
    with open(os.path.join(this_scen_path, "time_interval.info"), "w", encoding="utf-8") as f:
        # f.write(datetime.strftime(dt_dt, "%H:%M"))  # what I want to do but cannot (TypeError: descriptor 'strftime' for 'datetime.date' objects doesn't apply to a 'datetime.timedelta' object)
        total_s = dt_dt.total_seconds()
        hours = total_s // 3600
        mins = (total_s - 3600 * hours) // 60
        f.write(f"{int(hours):02d}:{int(mins):02d}")
        
    with open(os.path.join(this_scen_path, "start_datetime.info"), "w", encoding="utf-8") as f:
        f.write(datetime.strftime(start_date_dt, "%Y-%m-%d %H:%M"))
    with open(os.path.join(this_scen_path, "_seeds_info.json"), "w", encoding="utf-8") as f:
        json.dump(obj={"load_seed": int(load_seed), "renew_seed": int(renew_seed), "gen_p_forecast_seed": int(gen_p_forecast_seed)},
                  fp=f)
    with open(os.path.join(this_scen_path, "generation_quality.json"), "w", encoding="utf-8") as f:
        iter_num, mean_, percent_95, percent_99, max_ = quality
        json.dump(obj={"iter_num": int(iter_num),
                       "avg": float(mean_),
                       "percent_95": float(percent_95),
                       "percent_99": float(percent_99),
                       "max": float(max_), 
                       "info": ("avg, percent_95, percent_99, max: this 'quality' is the difference between the DC solver and the AC solver. This is the number of "
                                "MW that will differ from the grid2op observation compared to the generated data by chronix2grid.",
                                "total_load, total_gen: total amount of energy consumed / produced for the generated scenario.",
                                "losses_mwh: total amount of losses for the scenario (in energy)",
                                "losses_avg: average (per step) of the loss (avg[loss_this_step / total_generation_this_step])",
                                "wind_curtailed_opf: total (in energy) wind power curtailed by the OPF",
                                "wind_curtailed_losses: total (in energy) wind power curtailed by the loss",
                                "solar_curtailed_opf: total (in energy) solar power curtailed by the OPF",
                                "iter_num: number of iteration of the loss algorithm",
                                "generation_time: total time spent to generate these data (in seconds)",
                                "saving_time: total time spent to save the generated data (in seconds), this excludes the metadata saving time",
                                ),
                       "total_load": float(total_load),
                       "total_gen": float(total_gen),
                       "losses_mwh": float(losses_mwh),
                       "losses_avg": float(losses_avg),
                       "wind_curtailed_opf": float(wind_curtailed_opf),
                       "wind_curtailed_losses": float(wind_curtailed_losses),
                       "solar_curtailed_opf": float(solar_curtailed_opf),
                       "generation_time": float(generation_time),
                       "saving_time": float(saving_time),
                       },
                  fp=f,
                  sort_keys=True,
                  indent=4)
    
    for fn_ in files_to_copy:
        src_path = os.path.join(path_env, fn_)
        if os.path.exists(src_path):
            shutil.copy(src=src_path, 
                        dst=os.path.join(this_scen_path, fn_))
    

def generate_a_scenario(path_env,
                        name_gen,
                        gen_type,
                        output_dir, 
                        start_date,
                        dt,
                        scen_id,
                        load_seed,
                        renew_seed,
                        gen_p_forecast_seed,
                        handle_loss=True,
                        nb_steps=None,
                        PmaxErrorCorrRatio=0.9,
                        RampErrorCorrRatio=0.95,
                        threshold_stop=0.05,
                        max_iter=100,
                        debug=True  # TODO more feature !
                        ):
    """This function generates and save the data for a scenario.
    
    Generation includes:
    - load active value
    - load reactive value
    - renewable generation
    - controlable generation

    Put "outputdir=None" if you don't want to save the data.
    
    
    Parameters
    ----------
    env_name : _type_
        _description_
    name_gen : _type_
        _description_
    gen_type: _type_
        _description_
    output_dir : _type_
        _description_
    start_date : _type_
        _description_
    dt : _type_
        _description_
    scen_id : _type_
        _description_
    load_seed : _type_
        _description_
    renew_seed : _type_
        _description_
    gen_p_forecast_seed : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    beg_ = time.perf_counter()
    scenario_id = f"{start_date}_{scen_id}"
    dt_dt = timedelta(minutes=int(dt))
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d") - dt_dt
    if nb_steps is None:
        end_date_dt = start_date_dt + timedelta(days=7) + 2 * dt_dt
    else:
        end_date_dt = start_date_dt + (int(nb_steps) + 2) * dt_dt
    end_date = datetime.strftime(end_date_dt,  "%Y-%m-%d %H:%M:%S")
    with open(os.path.join(path_env, "params.json"), "r") as f:
        generic_params = json.load(f)
    number_of_minutes = int((end_date_dt - start_date_dt).total_seconds() // 60)
    gens_charac = pd.read_csv(os.path.join(path_env, "prods_charac.csv"), sep=",")
    
    # conso generation
    load_p, load_q, load_p_forecasted, load_q_forecasted = generate_loads(path_env,
                                                                          load_seed,
                                                                          start_date_dt,
                                                                          end_date_dt,
                                                                          dt,
                                                                          number_of_minutes,
                                                                          generic_params,
                                                                          day_lag=6  # TODO 6 because it's 2050
                                                                          )
    
    # renewable energy sources generation
    res_renew = generate_renewable_energy_sources(path_env,renew_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params, gens_charac)
    prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = res_renew

    if prod_solar.isna().any().any():
        error_ = RuntimeError("Nan generated in solar data")
        return error_, None, None, None, None, None, None, None
    if prod_wind.isna().any().any():
        error_ = RuntimeError("Nan generated in wind data")
        return error_, None, None, None, None, None, None, None
    
    # create the result data frame for the generators
    final_gen_p = pd.merge(prod_solar, prod_wind, left_index=True, right_index=True)
    
    for el in name_gen:
        if el in final_gen_p:
            continue
        final_gen_p[str(el)] = np.NaN
    final_gen_p = final_gen_p[name_gen]
    
    # generate economic dispatch
    res_disp = generate_economic_dispatch(path_env, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params,
                                          load_p, prod_solar, prod_wind, name_gen, gen_type, scenario_id, final_gen_p, gens_charac)
    gen_p_after_dispatch, total_wind_curt_opf, total_solar_curt_opf, error_ = res_disp
    
    if error_ is not None:
        # TODO log that !
        return error_, None, None, None, None, None, None, None
    
    # now try to move the generators so that when I run an AC powerflow, the setpoint of generators does not change "too much"
    if handle_loss:
        n_gen = len(name_gen)
        res_gen_p_df, error_, quality_ = handle_losses(path_env,
                                                       n_gen,
                                                       name_gen,
                                                       gens_charac,
                                                       load_p,
                                                       load_q,
                                                       gen_p_after_dispatch,
                                                       start_date_dt,
                                                       dt_dt,
                                                       scenario_id, 
                                                       PmaxErrorCorrRatio=PmaxErrorCorrRatio,
                                                       RampErrorCorrRatio=RampErrorCorrRatio,
                                                       threshold_stop=threshold_stop,
                                                       max_iter=max_iter)
        if error_ is not None:
            # TODO log that !
            return error_, None, None, None, None, None, None, None
    else:
        res_gen_p_df = 1.0 * gen_p_after_dispatch
        quality_ = (-1, float("Nan"), float("Nan"), float("Nan"), float("Nan"))
    
    prng = default_rng(gen_p_forecast_seed)
    res_gen_p_forecasted_df = res_gen_p_df * prng.lognormal(mean=0.0, sigma=float(generic_params["planned_std"]), size=res_gen_p_df.shape)
    res_gen_p_forecasted_df = res_gen_p_forecasted_df.shift(-1)
    res_gen_p_forecasted_df.iloc[-1] = 1.0 * res_gen_p_forecasted_df.iloc[-2]
    end_ = time.perf_counter()
    if output_dir is not None:
        beg_save = time.perf_counter()
        this_scen_path = os.path.join(output_dir, scenario_id)
        if not os.path.exists(this_scen_path):
            os.mkdir(this_scen_path)
        save_generated_data(this_scen_path,
                            load_p,
                            load_p_forecasted,
                            load_q,
                            load_q_forecasted,
                            final_gen_p,  # generated, before economic dispatch
                            gen_p_after_dispatch,  # generated, after economic dispatch (and possibly curtailment)
                            res_gen_p_df,
                            res_gen_p_forecasted_df,
                            debug=debug)
        total_load = float(load_p.sum().sum())
        total_gen = float(res_gen_p_df.sum().sum())
        gen_p_per_step = res_gen_p_df.sum(axis=1)
        proper_MWh_unit = float(dt_dt.total_seconds() / 3600.)
        wind_curtailed_losses = (gen_p_after_dispatch.iloc[:, gen_type=="wind"].sum().sum() - res_gen_p_df.iloc[:, gen_type=="wind"].sum().sum())
        end_save = time.perf_counter()
        save_meta_data(this_scen_path,
                       path_env,
                       start_date_dt,
                       dt_dt,
                       load_seed,
                       renew_seed,
                       gen_p_forecast_seed,
                       quality_,
                       total_load=total_load * proper_MWh_unit,
                       total_gen=total_gen * proper_MWh_unit,
                       losses_mwh=(total_gen - total_load) * proper_MWh_unit,
                       losses_avg=np.mean((gen_p_per_step - load_p.sum(axis=1)) / gen_p_per_step),
                       wind_curtailed_opf=float(total_wind_curt_opf * proper_MWh_unit),
                       wind_curtailed_losses=float(wind_curtailed_losses * proper_MWh_unit),
                       solar_curtailed_opf=float(total_solar_curt_opf * proper_MWh_unit),
                       generation_time=end_ - beg_,
                       saving_time=end_save - beg_save
                       )
        
    return error_, quality_, load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df
