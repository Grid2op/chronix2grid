# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import grid2op
import os
import argparse
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Agent import RecoPowerlineAgent

#grid_path='/home/ubuntu/Grid2Op_EnvironmentDesign/grid_design/subgrid'#subgrid.json'#ChroniX2Grid/input_data/generation/case118_l2rpn_wcci/L2RPN_2020_case118_redesigned.json'
#generation_output_folder='/home/ubuntu/Grid2Op_EnvironmentDesign/'#grid_design/subgrid/chronics'#ChroniX2Grid/subnet_chonix'#48years_chronix_v3_SlackCorrection'
#nb_episode = 576#48*12
#NB_CORE = 4



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run a do nothing on generated chronics')
    parser.add_argument('-g', '--grid_path', default='', type=str, help='subnet .json file')
    parser.add_argument('-o', '--ouput_dir', default='', type=str, help='Output dir')
    parser.add_argument('-nc', '--nb_cores', default=1, type=int, help='Numbers of cores for multiprocessing')
    parser.add_argument('-ne', '--nb_episode', default=1, type=int, help='Numbers of cores for multiprocessing')
    parser.add_argument('-a', '--agent_type', default='do_nothing', type=str, help='type of agent, either reco or do_nothing for now')    

    # Parse argumetns CML
    # ++  ++  ++  ++  ++
    args = parser.parse_args()
    grid_path = args.grid_path
    ouput_dir = args.ouput_dir
    NB_CORE = args.nb_cores
    nb_episode=args.nb_episode
    agent_type=args.agent_type
    
    os.makedirs(ouput_dir, exist_ok=True)

    print('the case folder is: '+grid_path)
    print('the output folder will be: '+ouput_dir)
    print('the number of cores used is: '+str(NB_CORE))
    print('the number of scenarios we consider is: '+str(nb_episode))
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
        print("You might need to install the LightSimBackend to gain massive speed up")
    # don't disconnect powerline on overflow, the thermal limit are not set for now, it would not make sens
    param = Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    env=grid2op.make(grid_path,param=param,backend=backend,test=True)
    # If you remove the "GridStateFromFileWithForecasts", from above, chronics will NOT be loaded properly.
    # GridStateFromFileWithForecasts is the format used for the competition, so it is mandatory that this works!
    # WITHOUT ANY MODIFICATIONS

    # Beside the environment should be able to load all data generated, and not one episode.
    # so please look in grid2op for compatible formats. This is not a valid format.

    if(agent_type=='reco'):
        runner = Runner(**env.get_params_for_runner(),agentClass=RecoPowerlineAgent)
    else:
        runner = Runner(**env.get_params_for_runner())
    # do regular computation as you would with grid2op
    res = runner.run(nb_episode=nb_episode,nb_process=NB_CORE, path_save=ouput_dir, pbar=True)

