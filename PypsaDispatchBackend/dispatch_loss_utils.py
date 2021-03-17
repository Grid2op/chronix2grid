import os
import warnings
import shutil
import numpy as np
import pandas as pd

import grid2op
from grid2op.Chronics import Multifolder, GridStateFromFileWithForecasts
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData

#from grid2op.Agent import RecoPowerlineAgent
from grid2op.Agent import DoNothingAgent

import chronix2grid.constants as cst

def move_chronics_temporarily(scenario_output_folder, grid_path):
    chronics_temporary_path = os.path.join(grid_path,'chronics')
    if os.path.exists(chronics_temporary_path):
        warnings.warn("Had to delete a previous chronic temporary path in input data", UserWarning)
        shutil.rmtree(chronics_temporary_path)
    os.makedirs(chronics_temporary_path, exist_ok=True)
    chronics_temporary_path = os.path.join(chronics_temporary_path, '000')
    print("temporary copy of chronics in "+str(chronics_temporary_path))
    shutil.copytree(scenario_output_folder, chronics_temporary_path)

def remove_temporary_chronics(grid_path):
    chronics_temporary_path = os.path.join(grid_path, 'chronics')
    shutil.rmtree(chronics_temporary_path)

def remove_simulation_data(scenario_folder_path):
    simulation_data_folder = os.path.join(scenario_folder_path, 'loss_simulation')
    shutil.rmtree(simulation_data_folder)

def create_or_replace_simulation_data_folder(scenario_folder_path):
    simulation_data_folder = os.path.join(scenario_folder_path, 'loss_simulation')
    if os.path.exists(simulation_data_folder):
        shutil.rmtree(simulation_data_folder)
    os.makedirs(simulation_data_folder, exist_ok=False)
    return simulation_data_folder

def run_grid2op_simulation_donothing(grid_path, agent_result_path, agent_type = 'do-nothing', nb_core = 1):
    """

    :param grid_path (str): path to folder where grid.json and other information on grid are stored
    :param agent_result_path (str): path in which we want to write the result of Grid2op simulation (via serialized EpisodeData object)
    :param agent_type (str): type of grid2op agent to use in simulation. Can be reco for RecoPowerlineAgent or do-nothing for DoNothingAgent
    :param nb_core (int): Simulations can be paralelized on cores. By default, 1 core is used here because paralelization is made previously on chronic scenarios
    :return:
    """
    print('Start grid2op simulation to compute realistic loss on grid')

    # Récupération des paramètres
    NB_CORE = nb_core # TOUJOURS 1 car la parallélisation se fait sur les scénarii dans le main
    nb_episode = 1
    simulation_data_folder = create_or_replace_simulation_data_folder(agent_result_path)

    print('Grid2op simulation for loss')
    print('the case folder is: ' + grid_path)
    print('the output folder will be: ' + simulation_data_folder)
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
    res = runner.run(nb_episode=nb_episode, nb_process=NB_CORE, path_save=simulation_data_folder, pbar=True)
    print('---- end of simulation')

def correct_scenario_loss(scenario_folder_path, agent_result_path, params_opf, grid_path):
    print('Start realistic loss correction from simulation results')

    # Load simulation data
    data_this_episode = EpisodeData.from_disk(os.path.join(agent_result_path, 'loss_simulation'), '000')
    slack_name = params_opf["nameSlack"]
    id_slack = params_opf["idxSlack"]

    # Get gen constraints
    first_obs = data_this_episode.observations[0]
    pmax = first_obs.gen_pmax[id_slack] + params_opf['pmax_margin']
    pmin = max(first_obs.gen_pmin[id_slack] - params_opf['pmin_margin'],0)
    ramp_up = first_obs.gen_max_ramp_up[id_slack] + params_opf['rampup_margin']
    ramp_down = first_obs.gen_max_ramp_down[id_slack] + params_opf['rampdown_margin']

    # Get corrected dispatch prod
    prods_p = pd.DataFrame(np.array([obs.prod_p for obs in data_this_episode.observations]))
    prodSlack = prods_p[id_slack]

    # Get dispatch prods before runner in chronix
    OldProdsDf = pd.read_csv(os.path.join(scenario_folder_path, 'prod_p.csv.bz2'), sep=';')
    OldProdsForecastDf = pd.read_csv(os.path.join(scenario_folder_path, 'prod_p_forecasted.csv.bz2'), sep=';')

    ##correction term
    newProdsDf = OldProdsDf
    newProdsForecastDf = OldProdsForecastDf
    CorrectionLosses = prodSlack - OldProdsDf[slack_name]
    print('maximum compensation for slack before correction in MW: ' + str(CorrectionLosses.abs().max()))
    print('average compensation for slack before correction in MW: ' + str(CorrectionLosses.mean()))
    print('median compensation for slack before correction in MW: ' + str(CorrectionLosses.abs().median()))
    print('min compensation for slack before correction in MW: ' + str(CorrectionLosses.min()))

    # apply correction on slack bus generator
    newProdsDf[slack_name] = OldProdsDf[slack_name] + CorrectionLosses
    newProdsForecastDf[slack_name] = OldProdsForecastDf[slack_name] + CorrectionLosses

    # Check constraints
    violations_message, bool = check_slack_constraints(newProdsDf[slack_name], pmax, pmin, ramp_up, ramp_down)
    if bool:
        if params_opf['early_stopping_mode']:
            remove_temporary_chronics(grid_path)
            raise ValueError(violations_message)
        else:
            warnings.warn(violations_message, UserWarning)
            print("Warning - "+violations_message)


    # Serialization
    newProdsDf.to_csv(
            os.path.join(scenario_folder_path, "prod_p.csv.bz2"),
            sep=';', index=False,
            float_format=cst.FLOATING_POINT_PRECISION_FORMAT
        )

    newProdsForecastDf.to_csv(
        os.path.join(scenario_folder_path, "prod_p_forecasted.csv.bz2"),
        sep=';', index=False,
        float_format=cst.FLOATING_POINT_PRECISION_FORMAT
    )

    print('---- end of loss correction ')
    return newProdsDf, newProdsForecastDf

def check_slack_constraints(prod_p, pmax, pmin, ramp_up, ramp_down):
    msg = "Loss correction violates generator constraints: "
    bool = False

    # Pmax
    dep = max(prod_p-pmax)
    if dep > 0:
        bool = True
        msg += "Pmax + margin is violated with maximum of "+str(dep)+" MW - "

    # Pmin
    dep = min(prod_p-pmin)
    if dep < 0:
        bool = True
        msg += "Pmin - margin is violated with maximum of " + str(dep) + " MW - "

    # Ramp up
    ramps = prod_p.diff()
    ramps_up = ramps[ramps>0]
    dep = max(ramps_up - ramp_up)
    if dep > 0:
        bool = True
        msg += "Ramp up + margin is violated with maximum of " + str(dep) + " MW - "

    # Ramp down
    ramps_down = -1 * ramps[ramps < 0]
    dep = max(ramps_down - ramp_down)
    if dep > 0:
        bool = True
        msg += "Ramp down + margin is violated with maximum of " + str(dep) + " MW - "

    return msg, bool