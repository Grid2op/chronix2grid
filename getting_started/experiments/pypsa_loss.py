# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from datetime import datetime
import enum
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


if __name__ == "__main__":
    
    
    OUTPUT_FOLDER = os.path.join('..', 'example', 'custom', 'output')

    path_chronix2grid = os.path.join(OUTPUT_FOLDER, "all_scenarios")
    path_chronics_fixed = os.path.join(OUTPUT_FOLDER, "fixed_chronics")
    env_name = "case118_l2rpn_wcci_benjamin"
    path_tmp = os.path.join("..", "example", "custom", "input", "generation")
    env_path = os.path.join(path_tmp, env_name)
    grid_path = os.path.join(env_path, "grid.json")
    scenario_id = "2050-06-06_0"    

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
        
    
    final_gen_p = np.full((env_for_loss.max_episode_duration() + 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
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
    final_gen_p[i] = 1.0 * obs.gen_p
    datetimes[i] = obs.get_time_stamp()
    max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data.data.prod_p[i]))
    
    done = False
    while not done:
        obs, reward, done, info = env_for_loss.step(env_for_loss.action_space())
        i += 1
        all_loss_orig[i] = np.sum(obs.gen_p) - np.sum(obs.load_p)
        final_load_p[i] = 1.0 * obs.load_p
        final_load_q[i] = 1.0 * obs.load_p
        final_gen_p[i] = 1.0 * obs.gen_p
        datetimes[i] = obs.get_time_stamp()
        max_diff_orig[i] = np.max(np.abs(obs.gen_p -  env_for_loss.chronics_handler.real_data.data.prod_p[i]))
    
    total_solar = np.sum(final_gen_p[:, env_for_loss.gen_type == "solar"], axis=1)
    total_wind = np.sum(final_gen_p[:, env_for_loss.gen_type == "wind"], axis=1)
    load_without_loss = np.sum(final_load_p, axis=1) - total_solar - total_wind
    load = load_without_loss + all_loss_orig
    
    # load the right data
    df = pd.read_csv(os.path.join(env_path, "prods_charac.csv"), sep=",")
    df["pmax"] = df["Pmax"]
    df["pmin"] = df["Pmin"]
    df["cost_per_mw"] = df["marginal_cost"]
    econimic_dispatch = PypsaDispatcher.from_dataframe(df)
    econimic_dispatch._chronix_scenario = ChroniXScenario(loads=load,
                                                          prods=pd.DataFrame(final_gen_p, columns=env_for_loss.name_gen),
                                                          scenario_name=scenario_id,
                                                          res_names= {"wind": env_for_loss.name_gen[env_for_loss.gen_type == "wind"],
                                                                      "solar": env_for_loss.name_gen[env_for_loss.gen_type == "solar"]
                                                          }
                                                          )
    
    # now starts the OPF
    with open(os.path.join(env_path, "params_opf.json")) as f:
        params = json.load(f)
    params["loss_pct"] = 0.  # losses are handled anove
    params["PmaxErrorCorrRatio"] = 0.9
    params["RampErrorCorrRatio"] = 0.95
    
    # do not treat the slack differently
    params["slack_ramp_limit_ratio"] = 0.95  
    if "slack_pmin" in params:
        del params["slack_pmin"]
    if "slack_pmax" in params:
        del params["slack_pmax"]
    
    
    error_ = None
    total_solar = pd.Series(total_solar.ravel(), index=datetimes)
    total_wind = pd.Series(total_wind.ravel(), index=datetimes)
    all_loss = 1.0 * all_loss_orig
    res_gen_p = 1.0 * final_gen_p
    iter_num = 1.
    max__ = 1.2 * max_diff_orig.max()
    while True:
        load = load_without_loss + all_loss
        load = pd.DataFrame(load.ravel(), index=datetimes)
        # TODO the +/- 10. (or */ 1.2) should change depending on the iteration !
        # max__ /= 2
        gen_max_pu_t = {gen_nm: np.minimum((res_gen_p[:,gen_id] + max__) / env_for_loss.gen_pmax[gen_id],
                                           params["PmaxErrorCorrRatio"])
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        gen_min_pu_t = {gen_nm: np.maximum((res_gen_p[:,gen_id] - max__) / env_for_loss.gen_pmax[gen_id],
                                           env_for_loss.gen_pmin[gen_id] / env_for_loss.gen_pmax[gen_id])
                        for gen_id, gen_nm in enumerate(env_for_loss.name_gen) if env_for_loss.gen_redispatchable[gen_id]}
        
        ### run the dispatch with the loss
        dispatch_res = econimic_dispatch.run(load,
                                             total_solar=None,
                                             total_wind=None,
                                             params=params,
                                             pyomo=False,
                                             solver_name="cbc",
                                             gen_max_pu_t=gen_max_pu_t,
                                             gen_min_pu_t=gen_min_pu_t
                                             )
        if dispatch_res is None:
            error_ = RuntimeError("Pypsa failed to find a solution")
            break
        
        # assign the generators
        for gen_id, gen_nm in enumerate(env_for_loss.name_gen):
            if gen_nm in dispatch_res.chronix.prods_dispatch:
                res_gen_p[:, gen_id] = 1.0 * dispatch_res.chronix.prods_dispatch[gen_nm].values

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
        print(f"max diff = {diff_.max():.2f}")
        max__ = 2.0 * diff_.max()
        pdb.set_trace()
        
    if error_ is not None:
        raise error_
    pdb.set_trace()
    