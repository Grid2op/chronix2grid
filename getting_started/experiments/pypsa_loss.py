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

def adjust_gens(all_loss_orig,
                env_path,
                env_param,
                load_without_loss,
                slack_id,
                load_p, 
                load_q,
                gen_p,
                gen_v,
                threshold_stop=0.1,  # stop when all generators move less that this
                weeks=1,
                max_abs_split=5.0,  # dispatch, on the other generators, at most 5.0 MW
                ):
    all_loss = all_loss_orig
    while True:
        load = load_without_loss + all_loss
        load = pd.DataFrame(load.ravel(), index=datetimes)
        
        
        ## do not modify grid for the losses
        max__ = np.abs(diff_).max(axis=1)  # absorb at least that many things
        sum_gens = np.sum(res_gen_p[:,env_for_loss.gen_redispatchable], axis=1).reshape(-1, 1)
        ratio_gens = res_gen_p / sum_gens
        
        # gen_max_pu_t = {gen_nm: np.minimum((res_gen_p[:,gen_id] + ratio_ * max__ * ratio_gens[:, gen_id]) / econimic_dispatch.generators.loc[gen_nm].p_nom,
        #                                     params["PmaxErrorCorrRatio"])
        #                 for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        
        gen_max_pu_t = {gen_nm: np.minimum((res_gen_p[:,gen_id] + ratio_max * max__.max()) / econimic_dispatch.generators.loc[gen_nm].p_nom,
                                            params["PmaxErrorCorrRatio"])
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
                
        gen_min_pu_t = {gen_nm: np.maximum((res_gen_p[:,gen_id]) / econimic_dispatch.generators.loc[gen_nm].p_nom,
                                            env_for_loss.gen_pmin[gen_id] / econimic_dispatch.generators.loc[gen_nm].p_nom
                                            )
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        
        # never decrease (during iteration) some generators
        min__ = diff_.min()
        print(f"min: {min__:.2f}")
        sum_gens = np.sum(res_gen_p[:,env_for_loss.gen_redispatchable], axis=1).reshape(-1, 1)
        ratio_gens = res_gen_p / sum_gens
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
            ratio_max *= 1.2
            print(f"Pypsa failed to find a solution, increasing ratio_max to {ratio_max} (max value authorized {max__.max():.2f})")
            continue        
            # error_ = RuntimeError("Pypsa failed to find a solution")
            # break
            
        ratio_max = 1.
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
        
        diff_plan = res_gen_p - res_gen_p_prev
        print(f"max diff vs prev: {diff_plan.max():.2f}")
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
                data_feeding_kwargs={"load_p": final_load_p,
                                     "load_q": final_load_q,
                                     "prod_p": 1.0 * res_gen_p,
                                     "prod_v": final_gen_v}
                )
        
        diff_ = np.full((env_fixed.max_episode_duration(), env_fixed.n_gen), fill_value=np.NaN)
        # all_loss = np.full((env_fixed.max_episode_duration() + 1), fill_value=np.NaN)
        all_loss[:] = np.NaN  # = np.full((env_fixed.max_episode_duration() + 1), fill_value=np.NaN)
        
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
        iter_num += 1
        print()
        print(f"max diff after AC = {diff_.max():.2f}")
        print()
        
        # pdb.set_trace()
        res_gen_p_prev = 1.0 * res_gen_p
        if diff_.max() <= 0.5:
            break

if __name__ == "__main__":
    
    
    OUTPUT_FOLDER = os.path.join('..', 'example', 'custom', 'output')

    path_chronix2grid = os.path.join(OUTPUT_FOLDER, "all_scenarios")
    path_chronics_fixed = os.path.join(OUTPUT_FOLDER, "fixed_chronics")
    env_name = "case118_l2rpn_wcci_benjamin"
    path_tmp = os.path.join("..", "example", "custom", "input", "generation")
    env_path = os.path.join(path_tmp, env_name)
    grid_path = os.path.join(env_path, "grid.json")
    scenario_id = "2050-06-06_0"    
    scenario_id = "2050-06-13_0"    
    # scenario_id = "2050-01-03_0"    

    ### run a first environment to compute the loss
    env_param = Parameters()
    env_param.NO_OVERFLOW_DISCONNECTION = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_for_loss = grid2op.make(
            env_path,
            test=True,
            # grid_path=empty_env._init_grid_path, # assign it the 118 grid
            chronics_path=path_chronix2grid,
            param=env_param,
            backend=LightSimBackend()
            )
    
    gen_p_orig = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_gen_v = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_load_p = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    final_load_q = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    all_loss_orig = np.zeros(env_for_loss.max_episode_duration() + 1)
    max_diff_orig = np.zeros(env_for_loss.max_episode_duration() + 1)
    datetimes = np.zeros(env_for_loss.max_episode_duration() + 1, dtype=datetime)
    
    env_for_loss.set_id(scenario_id)
    obs = env_for_loss.reset()
            
    ### now start to split the loss
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
    
    # now starts the OPF
    with open(os.path.join(env_path, "params_opf.json"), "r") as f:
        params = json.load(f)
    params["loss_pct"] = 0.  # losses are handled anove
    params["PmaxErrorCorrRatio"] = 0.9
    params["RampErrorCorrRatio"] = 0.95
    
    # do not treat the slack differently
    params["slack_ramp_limit_ratio"] = params["RampErrorCorrRatio"]
    if "slack_pmin" in params:
        del params["slack_pmin"]
    if "slack_pmax" in params:
        del params["slack_pmax"]
    
    error_ = None
    total_solar_orig = pd.Series(total_solar.ravel(), index=datetimes)
    total_wind_orig = pd.Series(total_wind.ravel(), index=datetimes)
    
    total_solar = 1.0 * total_solar_orig
    total_wind = 1.0 * total_wind_orig
    all_loss = 1.0 * all_loss_orig
    res_gen_p = 1.0 * gen_p_orig
    res_gen_p_prev = 1.0 * gen_p_orig
    iter_num = 1.
    max__ = 1.2 * max_diff_orig.max()
    diff_ = 1.0 * max_diff_orig
    diff_ = diff_.reshape(-1,1)
    
    extra_functionality = None
    ratio_ = 1.
    ratio_max = 1.

        
    if error_ is not None:
        raise error_
    