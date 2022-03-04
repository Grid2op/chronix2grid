# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from datetime import datetime
import os
from grid2op.Parameters import Parameters
import pandas as pd
import pdb
import warnings
import grid2op
from grid2op.Chronics import FromNPY
import numpy as np
import json
from lightsim2grid import LightSimBackend
from chronix2grid.generation.dispatch.EconomicDispatch import ChroniXScenario
from chronix2grid.generation.dispatch.PypsaDispatchBackend import PypsaDispatcher


from debug_split_loss import save_data


def adjust_gens(all_loss_orig,
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
                econimic_dispatch,
                path_chronix2grid,
                diff_,
                threshold_stop=0.1,  # stop when all generators move less that this
                max_iter=100,
                ):
    all_loss = all_loss_orig
    res_gen_p = 1.0 * gen_p
    error_ = None
    iter_num = 0
    while True:
        iter_num += 1
        load = load_without_loss + all_loss
        load = pd.DataFrame(load.ravel(), index=datetimes)
        
        # never decrease (during iteration) some generators
        min__ = diff_.min()
        gen_max_pu_t = None
        gen_min_pu_t = {gen_nm: np.maximum((res_gen_p[:,gen_id] + min__) / econimic_dispatch.generators.loc[gen_nm].p_nom,
                                            env_for_loss.gen_pmin[gen_id] / econimic_dispatch.generators.loc[gen_nm].p_nom
                                            )
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        
        
        ### run the dispatch with the loss
        dispatch_res = econimic_dispatch.run(load,
                                             total_solar=total_solar,
                                             total_wind=total_wind,
                                             params=params,
                                             pyomo=False,
                                             solver_name="cbc",
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
        
        # diff_plan = res_gen_p - res_gen_p_prev
        # print(f"max diff vs prev: {diff_plan.max():.2f}")
        
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
                chronics_path=path_chronix2grid,
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
                # TODO  res_gen_p has wrong size I think, need to check !
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

def fix_losses_one_scenario(env_for_loss, scenario_id, params, output_path,
                            env_path, env_param, path_chronix2grid,
                            threshold_stop=0.5,
                            max_iter=100
                            ):
    gen_p_orig = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_gen_v = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_load_p = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    final_load_q = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    all_loss_orig = np.zeros(env_for_loss.max_episode_duration() + 1)
    max_diff_orig = np.zeros(env_for_loss.max_episode_duration() + 1)
    datetimes = np.zeros(env_for_loss.max_episode_duration() + 1, dtype=datetime)
    
    env_for_loss.set_id(scenario_id)
    obs = env_for_loss.reset()
    
    i = 0
    all_loss_orig[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
    final_gen_v[i] = obs.gen_v
    final_load_p[i] = obs.load_p
    final_load_q[i] = obs.load_q
    gen_p_orig[i] = 1.0 * obs.gen_p
    datetimes[i] = obs.get_time_stamp()
    max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data.data.prod_p[i]))
    
    done = False
    while not done:
        obs, reward, done, info = env_for_loss.step(env_for_loss.action_space())
        i += 1
        all_loss_orig[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
        final_load_p[i] = 1.0 * obs.load_p
        final_load_q[i] = 1.0 * obs.load_p
        gen_p_orig[i] = env_for_loss.chronics_handler.real_data.data.prod_p[i]  # 1.0 * obs.gen_p
        datetimes[i] = obs.get_time_stamp()
        max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data.data.prod_p[i]))
    
    total_solar = np.sum(gen_p_orig[:, env_for_loss.gen_type == "solar"], axis=1)
    total_wind = np.sum(gen_p_orig[:, env_for_loss.gen_type == "wind"], axis=1)
    load_without_loss = np.sum(final_load_p, axis=1) #  - total_solar - total_wind
    
    # load the right data
    df = pd.read_csv(os.path.join(env_path, "prods_charac.csv"), sep=",")
    df["pmax"] = df["Pmax"]
    df["pmin"] = df["Pmin"]
    df["cost_per_mw"] = df["marginal_cost"]
    econimic_dispatch = PypsaDispatcher.from_dataframe(df)
    econimic_dispatch._chronix_scenario = ChroniXScenario(loads=1.0 * load_without_loss,
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
    
    res_gen_p, error_ = adjust_gens(all_loss_orig,
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
                                    econimic_dispatch,
                                    path_chronix2grid,
                                    diff_,
                                    threshold_stop=threshold_stop,
                                    max_iter=max_iter)
    
    if error_ is not None:
        return error_
    
    # now save it
    save_data(env_for_loss, 
              path_chronix2grid,
              output_path, 
              final_load_p, 
              final_load_q, 
              res_gen_p, 
              final_gen_v)
    return None

def fix_loss_multiple_scenarios(path_env,
                                path_chronix2grid,
                                output_path,
                                scenario_ids,
                                threshold_stop=0.5,
                                max_iter=100):
    ### run a first environment to compute the loss
    env_param = Parameters()
    env_param.NO_OVERFLOW_DISCONNECTION = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_for_loss = grid2op.make(
            path_env,
            test=True,
            # grid_path=empty_env._init_grid_path, # assign it the 118 grid
            chronics_path=path_chronix2grid,
            param=env_param,
            backend=LightSimBackend()
            )

    # now starts the OPF
    with open(os.path.join(path_env, "params_opf.json"), "r") as f:
        params = json.load(f)
    params["loss_pct"] = 0.  # losses are handled better in this function
    params["PmaxErrorCorrRatio"] = 0.9
    params["RampErrorCorrRatio"] = 0.95
    
    # do not treat the slack differently
    params["slack_ramp_limit_ratio"] = params["RampErrorCorrRatio"]
    if "slack_pmin" in params:
        del params["slack_pmin"]
    if "slack_pmax" in params:
        del params["slack_pmax"]
              
    ### now start to split the loss
    for scenario_id in scenario_ids:
        error_ = fix_losses_one_scenario(env_for_loss,
                                         scenario_id,
                                         params,
                                         output_path,
                                         path_env,
                                         env_param,
                                         path_chronix2grid,
                                         threshold_stop=threshold_stop,
                                         max_iter=max_iter
                                         )
        if error_ is not None:
            print(f"ERROR for scenario {scenario_id}: {error_}")
    
if __name__ == "__main__":

    # good but it takes a loooot of time, maybe rewriting it using scipy ? or or tools ?
    
    
    OUTPUT_FOLDER = os.path.join('..', 'example', 'custom', 'output')

    path_chronix2grid = os.path.join(OUTPUT_FOLDER, "all_scenarios")
    path_chronics_fixed = os.path.join(OUTPUT_FOLDER, "fixed_chronics")
    env_name = "case118_l2rpn_wcci_benjamin"
    path_tmp = os.path.join("..", "example", "custom", "input", "generation")
    env_path = os.path.join(path_tmp, env_name)
    grid_path = os.path.join(env_path, "grid.json")
    
    threshold_stop = 0.5
    max_iter = 100
    
    li_months = [
             "2050-01-03", 
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
                                    
    fix_loss_multiple_scenarios(env_path,
                                path_chronix2grid,
                                path_chronics_fixed,
                                scenario_ids=[f"{el}_0" for el in li_months],
                                threshold_stop=threshold_stop,
                                max_iter=max_iter)
    