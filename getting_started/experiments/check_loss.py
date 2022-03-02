# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import grid2op
from grid2op.Chronics import ChangeNothing
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend  # might need "pip install lightsim2grid"
import shutil
from grid2op.Chronics import FromNPY
import os
import numpy as np
import copy
import pandas as pd
import warnings
import pdb
from tqdm import tqdm
OUTPUT_FOLDER = os.path.join('..', 'example', 'custom', 'output')
path_chronics_outputopf = os.path.join(OUTPUT_FOLDER, "all_scenarios")

########
# Detailed configuration to be set in <INPUT_FOLDER>/<CASE>/params.json
weeks = 1
n_scenarios = 1

mode = 'RLTK'
mode = 'RL'
li_months = ["2050-01-01", 
             "2050-01-08",
             "2050-01-15",
             "2050-01-22",
             "2050-02-01",
             "2050-02-08",
             "2050-02-15",
             "2050-02-22",
             "2050-03-01",
             "2050-03-08",
             "2050-03-15",
             "2050-03-22",
             "2050-04-01",
             "2050-04-08",
             "2050-04-15",
             "2050-04-22",
             "2050-05-01", 
             "2050-05-08", 
             "2050-05-15", 
             "2050-05-22", 
             "2050-06-01",
             "2050-06-08",
             "2050-06-15",
             "2050-06-22",
             "2050-07-01", 
             "2050-07-08", 
             "2050-07-15", 
             "2050-07-22", 
             "2050-08-01", 
             "2050-08-08", 
             "2050-08-15", 
             "2050-08-22", 
             "2050-09-01", 
             "2050-09-08", 
             "2050-09-15", 
             "2050-09-22", 
             "2050-10-01", 
             "2050-10-08", 
             "2050-10-15", 
             "2050-10-22", 
             "2050-11-01", 
             "2050-11-08", 
             "2050-11-15", 
             "2050-11-22", 
             "2050-12-01",
             "2050-12-08",
             "2050-12-15",
             "2050-12-22",
            ]

env_name = "case118_l2rpn_wcci_benjamin"
path_tmp = os.path.join("..", "example", "custom", "input", "generation")
output_path = os.path.join(path_tmp, env_name)
grid_path = os.path.join(output_path, "grid.json")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    env118_withoutchron = grid2op.make(
        output_path,
        test=True,
        grid_path=grid_path, # assign it the 118 grid
        chronics_class=ChangeNothing, # tell it to change nothing (not the most usable environment...)
    )

path_chronics_fixed = os.path.join(OUTPUT_FOLDER, "fixed_chronics")
param = env118_withoutchron.parameters
param.NO_OVERFLOW_DISCONNECTION = True
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    env_for_loss = grid2op.make(
        output_path,
        test=True,
        grid_path=grid_path, # assign it the 118 grid
        chronics_path=path_chronics_outputopf,
        param=param,
        backend=LightSimBackend()
        )
      
all_loss = {}
with tqdm(total=len(li_months)) as pbar:
    for month in li_months:
        done = False
        all_loss_ep = np.zeros(weeks * 7 * 288 - 1) * np.NaN
        
        i = 0
        obs = env_for_loss.reset()
        all_loss_ep[i] = 100. * (np.sum(obs.gen_p) - np.sum(obs.load_p)) / np.sum(obs.gen_p)
        while not done:
            obs, reward, done, info = env_for_loss.step(env_for_loss.action_space())
            i += 1
            all_loss_ep[i] = 100. * (np.sum(obs.gen_p) - np.sum(obs.load_p)) / np.sum(obs.gen_p)
        all_loss[env_for_loss.chronics_handler.get_id()] =  1.0 * all_loss_ep
        pbar.update()

total_min = 10000.
total_max = 0.
total_mean = 0.
for k, v in all_loss.items():
    print(f"month {os.path.split(k)[-1]}: {np.min(v):.2f}%, {np.mean(v):.2f}%, {np.max(v):.2f}%")
    total_mean += v.mean()
    total_min = min(total_min, v.min())
    total_max = max(total_max, v.max())
total_mean /= len(all_loss)
print(f"{total_min:.2f}%, {total_mean:.2f}%, {total_max:.2f}%")