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
import pdb
import shutil
from black import err
import pandas as pd
import os
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChangeNothing, FromNPY
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


def generate_loads(path_env, load_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params,
                   load_q_from_p_coeff=0.7):
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
                                                 load_config_manager=None)
    
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
                               load_p, prod_solar, prod_wind, env, scenario_id, final_gen_p, gens_charac):
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
    gen_p_orig = np.zeros((prod_solar.shape[0], env.n_gen))
    economic_dispatch._chronix_scenario = ChroniXScenario(loads=1.0 * load_p,
                                                          prods=pd.DataFrame(1.0 * gen_p_orig, columns=env.name_gen),
                                                          scenario_name=scenario_id,
                                                          res_names={"wind": env.name_gen[env.gen_type == "wind"],
                                                                     "solar": env.name_gen[env.gen_type == "solar"]
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
        return None, error_
    
    # now assign the results
    final_gen_p = 1.0 * final_gen_p  # copy the data frame to avoid modify the original one
    for gen_id, gen_nm in enumerate(env.name_gen):
        if gen_nm in res_dispatch.chronix.prods_dispatch:
            final_gen_p.iloc[:, gen_id] = 1.0 * res_dispatch.chronix.prods_dispatch[gen_nm].values
            
    #handle curtailment
    final_gen_p.iloc[:, env.gen_type == "wind"] *= (res_dispatch.chronix.prods_dispatch['agg_wind'].values / total_wind.values).reshape(-1,1)   
    final_gen_p.iloc[:, env.gen_type == "solar"] *= (res_dispatch.chronix.prods_dispatch['agg_solar'].values / total_solar.values).reshape(-1,1)
    return final_gen_p, None


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
                max_iter=100,
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
    all_loss = all_loss_orig
    res_gen_p = 1.0 * gen_p
    error_ = None
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
            error_ = RuntimeError("Pypsa failed to find a solution")
            break

        # assign the generators
        for gen_id, gen_nm in enumerate(env_for_loss.name_gen):
            if gen_nm in dispatch_res.chronix.prods_dispatch:
                res_gen_p[:, gen_id] = 1.0 * dispatch_res.chronix.prods_dispatch[gen_nm].values
                
        sum_wind_tmp = total_wind.sum()
        sum_diff = sum_wind_tmp - dispatch_res.chronix.prods_dispatch['agg_wind'].sum() 
        print(f"total curtailed: {sum_diff/12.:.2f}MWh "
              f"({sum_wind_tmp / 12.:.2f}MWh, {sum_diff / sum_wind_tmp:.2f}%)")
        
        #handle wind curtailment
        res_gen_p[:, env_for_loss.gen_type == "wind"] *= (dispatch_res.chronix.prods_dispatch['agg_wind'].values / total_wind.values).reshape(-1,1)
        
        total_wind[:] = 1.0 * dispatch_res.chronix.prods_dispatch["agg_wind"].values
        
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
        print()
        print(f"iter {iter_num}: {diff_.max():.2f}")
        print()
        
        if diff_.max() <= threshold_stop:
            break
        
        if iter_num >= max_iter:
            error_ = RuntimeError("Too much iterations performed")
            break
        
    return res_gen_p, error_

def _fix_losses_one_scenario(env_for_loss,
                            scenario_id,
                            params,
                            env_path,
                            env_param,
                            load_df,
                            threshold_stop=0.5,
                            max_iter=100
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
    
    res_gen_p, error_ = _adjust_gens(all_loss_orig,
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
                                    max_iter=max_iter)
    
    if error_ is not None:
        return None, error_
    
    return res_gen_p, error_


def handle_losses(path_env, env,
                  gens_charac, load_p, load_q, final_gen_p,
                  start_date_dt, dt_dt, scenario_id, 
                  PmaxErrorCorrRatio=0.9,
                  RampErrorCorrRatio=0.95,
                  threshold_stop=0.5,
                  max_iter=100):
    """This function is here to make sure that if you run an AC model with the data generated, then the generator setpoints will not change too much 
    (less than `threshold_stop` MW)

    Parameters
    ----------
    path_env : _type_
        _description_
    env : _type_
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
    gen_v = np.tile(np.array([float(gens_charac.loc[gens_charac["name"] == nm_gen].V) for nm_gen in env.name_gen ]),
                    load_p.shape[0]).reshape(-1, env.n_gen)
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
    res_gen_p, error_ = _fix_losses_one_scenario(env_for_loss,
                                                scenario_id,
                                                loss_param,
                                                path_env,
                                                env_for_loss.parameters,
                                                load_df=load_p,
                                                threshold_stop=threshold_stop,
                                                max_iter=max_iter
                                                )
    # reformat the generators
    res_gen_p_df = pd.DataFrame(res_gen_p, index=final_gen_p.index, columns=env_for_loss.name_gen)
    
    return res_gen_p_df, error_

def save_generated_data(this_scen_path, load_p, load_p_forecasted, load_q, load_q_forecasted, prod_p, prod_p_forecasted,
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
    prod_p : _type_
        _description_
    prod_p_forecasted : _type_
        _description_
    sep : str, optional
        _description_, by default ';'
    float_prec : _type_, optional
        _description_, by default FLOATING_POINT_PRECISION_FORMAT
    """
    for df, nm in zip([load_p, load_p_forecasted, load_q, load_q_forecasted, prod_p, prod_p_forecasted],
                      ["load_p", "load_p_forecasted", "load_q", "load_q_forecasted", "prod_p", "prod_p_forecasted"]):
        df.to_csv(os.path.join(this_scen_path, f'{nm}.csv.bz2'),
                  sep=sep,
                  float_format=float_prec,
                  header=True,
                  index=False)


def save_ref_data(this_scen_path, path_env, start_date_dt, dt_dt : timedelta, load_seed, renew_seed, gen_p_forecast_seed):
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
    """
    with open(os.path.join(this_scen_path, "time_interval.info"), "w", encoding="utf-8") as f:
        # f.write(datetime.strftime(dt_dt, "%H:%M"))  # what I want to do but cannot (TypeError: descriptor 'strftime' for 'datetime.date' objects doesn't apply to a 'datetime.timedelta' object)
        total_s = dt_dt.total_seconds()
        hours = total_s // 3600
        mins = (total_s - 3600 * hours) // 60
        f.write(f"{hours}:{mins}")
        
    with open(os.path.join(this_scen_path, "start_datetime.info"), "w", encoding="utf-8") as f:
        f.write(datetime.strftime(start_date_dt, "%Y-%m-%d %H:%M"))
    with open(os.path.join(this_scen_path, "_seeds_info.json"), "w", encoding="utf-8") as f:
        json.dump(obj={"load_seed": int(load_seed), "renew_seed": int(renew_seed), "gen_p_forecast_seed": int(gen_p_forecast_seed)},
                  fp=f)
    
    for fn_ in ["maintenance_meta.json"]:
        src_path = os.path.join(path_env, fn_)
        if os.path.exists(src_path):
            shutil.copy(src=src_path, 
                        dst=os.path.join(this_scen_path, fn_))
    

def generate_a_scenario(env_name, env, output_dir, 
                        start_date, dt, scen_id,
                        load_seed, renew_seed, gen_p_forecast_seed):
    scenario_id = f"{start_date}_{scen_id}"  # TODO
    path_env = env.get_path_env()
    dt_dt = timedelta(minutes=int(dt))
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d") - dt_dt
    end_date_dt = start_date_dt + timedelta(days=7) + 2 * dt_dt
    end_date = datetime.strftime(end_date_dt,  "%Y-%m-%d %H:%M:%S")
    with open(os.path.join(path_env, "params.json"), "r") as f:
        generic_params = json.load(f)
    number_of_minutes = int((end_date_dt - start_date_dt).total_seconds() // 60)
    gens_charac = pd.read_csv(os.path.join(path_env, "prods_charac.csv"), sep=",")
    
    # conso generation
    load_p, load_q, load_p_forecasted, load_q_forecasted = generate_loads(path_env, load_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params)
    
    # renewable energy sources generation
    res_renew = generate_renewable_energy_sources(path_env,renew_seed, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params, gens_charac)
    prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = res_renew
    
    # create the result data frame for the generators
    final_gen_p = pd.merge(prod_solar, prod_wind, left_index=True, right_index=True)
    for el in env.name_gen:
        if el in final_gen_p:
            continue
        final_gen_p[el] = np.NaN
    final_gen_p = final_gen_p[env.name_gen]
    
    # generate economic dispatch
    gen_p_after_dispatch, error_ = generate_economic_dispatch(path_env, start_date_dt, end_date_dt, dt, number_of_minutes, generic_params, 
                                                              load_p, prod_solar, prod_wind, env, scenario_id, final_gen_p, gens_charac)
    if error_ is not None:
        # TODO log that !
        return error_, None, None, None, None, None, None
    
    # now try to move the generators so that when I run an AC powerflow, the setpoint of generators does not change "too much"
    res_gen_p_df, error_ = handle_losses(path_env, env,
                                         gens_charac, load_p, load_q, gen_p_after_dispatch,
                                         start_date_dt, dt_dt, scenario_id, 
                                         PmaxErrorCorrRatio=0.9,
                                         RampErrorCorrRatio=0.95,
                                         threshold_stop=0.5,
                                         max_iter=100)
    if error_ is not None:
        # TODO log that !
        return error_, None, None, None, None, None, None
    
    prng = default_rng(gen_p_forecast_seed)
    res_gen_p_forecasted_df = res_gen_p_df * prng.lognormal(mean=0.0, sigma=float(generic_params["planned_std"]), size=res_gen_p_df.shape)
    res_gen_p_forecasted_df = res_gen_p_forecasted_df.shift(-1)
    res_gen_p_forecasted_df.iloc[-1] = 1.0 * res_gen_p_forecasted_df.iloc[-2]
    if output_dir is not None:
        this_scen_path = os.path.join(output_dir, scenario_id)
        if not os.path.exists(this_scen_path):
            os.mkdir(this_scen_path)
        save_generated_data(this_scen_path, load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df)
        save_ref_data(this_scen_path, path_env, start_date_dt, dt_dt, load_seed, renew_seed, gen_p_forecast_seed)
        
    return error_, load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df
    
if __name__ == "__main__":
    # required parameters
    env_name = "../example/custom/input/generation/case118_l2rpn_wcci_benjamin" 
    env = grid2op.make(env_name, chronics_class=ChangeNothing)
    output_dir = "/home/donnotben/Documents/chronix2grid_gaetan/getting_started/example/custom/output/fixed_chronics_complete"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    start_date = "2050-01-03"
    dt = "5"
    scen_id = "0"
    seed = 0
    np.random.seed(seed)
    
    li_months = ["2050-01-03", 
             "2050-01-10",
             "2050-01-17",
             "2050-01-24",
             "2050-01-31",
             "2050-02-07",
             "2050-02-14",
             "2050-02-21",
             "2050-02-28",
             "2050-03-07",
             "2050-03-14",
             "2050-03-21",
             "2050-03-28",
             "2050-04-04",
             "2050-04-11",
             "2050-04-18",
             "2050-04-25",
             "2050-05-02", 
             "2050-05-09", 
             "2050-05-16", 
             "2050-05-23", 
             "2050-05-30",
             "2050-06-06",
             "2050-06-13",
             "2050-06-20",
             "2050-06-27",
             "2050-07-04", 
             "2050-07-11", 
             "2050-07-18", 
             "2050-07-25", 
             "2050-08-01", 
             "2050-08-08", 
             "2050-08-15", 
             "2050-08-22", 
             "2050-08-29", 
             "2050-09-05", 
             "2050-09-12", 
             "2050-09-19", 
             "2050-09-26", 
             "2050-10-03", 
             "2050-10-10", 
             "2050-10-17", 
             "2050-10-24", 
             "2050-10-31", 
             "2050-11-07", 
             "2050-11-14", 
             "2050-11-21", 
             "2050-11-28", 
             "2050-12-05",
             "2050-12-12",
             "2050-12-19",
             "2050-12-26",
            ]
    load_seeds = []
    renew_seeds = []
    gen_p_forecast_seeds = []
    for start_date in li_months:
        load_seed, renew_seed, gen_p_forecast_seed = np.random.randint(2**32 - 1, size=3)
        load_seeds.append(load_seed)
        renew_seeds.append(renew_seed)
        gen_p_forecast_seeds.append(gen_p_forecast_seed)
        
    errors = {}
    for i, start_date in enumerate(li_months):
        res_gen = generate_a_scenario(env_name, env, output_dir, start_date, dt, scen_id,
                                      load_seeds[i], renew_seeds[i], gen_p_forecast_seeds[i])
        error_, load_p, load_p_forecasted, load_q, load_q_forecasted, res_gen_p_df, res_gen_p_forecasted_df = res_gen
        if error_ is not None:
            print("=============================")
            print(f"     Error for {scen_id}        ")
            print(f"{error_}")
            print("=============================")
            errors[f'{start_date}_{scen_id}'] = f"{error_}"
            with open(os.path.join(output_dir, "errors.json"), "w", encoding="utf-8") as f:
                json.dump(errors, fp=f)
