import os
import shutil

import grid2op
from grid2op.Chronics import Multifolder, GridStateFromFileWithForecasts
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner

#from lightsim2grid import LightSimBackend
#from grid2op.Agent import RecoPowerlineAgent
from grid2op.Agent import DoNothingAgent

def move_chronics_temporarily(scenario_output_folder, grid_path):
    chronics_temporary_path = os.path.join(grid_path,'chronics')
    os.makedirs(chronics_temporary_path, exist_ok=True)
    chronics_temporary_path = os.path.join(chronics_temporary_path, '000')
    print("temporary copy of chronics in "+str(chronics_temporary_path))
    shutil.copytree(scenario_output_folder, chronics_temporary_path)

def remove_temporary_chronics(grid_path):
    chronics_temporary_path = os.path.join(grid_path, 'chronics')
    shutil.rmtree(chronics_temporary_path)



def run_grid2op_simulation_donothing(grid_path, agent_result_path, agent_type = 'do-nothing', nb_core = 1):
    ## Fonction Notebook - Run Grid2op sur les chroniques du scénario et écrire EpisodeData à agent_path_result

    # Récupération des paramètres
    ouput_dir = agent_result_path
    NB_CORE = nb_core # TOUJOURS 1 car la parallélisation se fait sur les scénarii dans le main
    nb_episode = 1 
    # os.makedirs(ouput_dir, exist_ok=True) # Normalement existe déjà

    print('Grid2op simulation for loss')
    print('the case folder is: ' + grid_path)
    print('the output folder will be: ' + ouput_dir)
    print('the number of cores used is: ' + str(NB_CORE))
    print('the number of scenarios we consider is: ' + str(nb_episode))
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
        print("You might need to install the LightSimBackend (provisory name) to gain massive speed up")
    # don't disconnect powerline on overflow, the thermal limit are not set for now, it would not make sens
    param = Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    env = grid2op.make(grid_path,
                       param=param, backend=backend, test=True)
    #env = grid2op.make(grid_path = grid_path, chronics_path = scenario_output_folder,
     #                  param=param, backend=backend, test=True)
    # If you remove the "GridStateFromFileWithForecasts", from above, chronics will NOT be loaded properly.
    # GridStateFromFileWithForecasts is the format used for the competition, so it is mandatory that this works!
    # WITHOUT ANY MODIFICATIONS

    # Beside the environment should be able to load all data generated, and not one episode.
    # so please look in grid2op for compatible formats. This is not a valid format.

    if (agent_type == 'reco'):
        runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)
        #runner = Runner(**env.get_params_for_runner(), agentClass=RecoPowerlineAgent)
    else:
        runner = Runner(**env.get_params_for_runner())
    # do regular computation as you would with grid2op
    res = runner.run(nb_episode=nb_episode, nb_process=NB_CORE, path_save=ouput_dir, pbar=True)

def correct_scenario_loss(scenario_folder_path, agent_result_path, params_loss):
    # Fonction Notebook - lire EpisodeData et Chroniques générées puis amender sur le slack bus generator
    # Subtilité: checker la comparaison entre cet amendement et Pmin/Pmax. Dans params_loss, proposer un mode pour relaxer cette contrainte, sinon renvoyer erreur
    return 0