import os
import shutil

import numpy as np

from .EDispatch_L2RPN2020 import run_economic_dispatch
from .dispatch_loss_utils import run_grid2op_simulation_donothing, correct_scenario_loss, move_chronics_temporarily, \
    remove_temporary_chronics, remove_simulation_data



def main(dispatcher, input_folder, output_folder, grid_folder, seed, params, params_opf):
    """

    Parameters
    ----------
    dispatcher : Dispatcher
        The Dispatcher instance used for running the OPF
    input_folder : str
        The path to the directory containing the inputs for the dispatch
    output_folder
        The path of the directory that will receive the outputs of the dispatch
    grid_folder
        The path of the directory that contains grid information (used only if loss simulation are activated)
    seed : int
        Random seed for parallel execution
    params_opf : dict
        Options for the OPF

    Returns
    -------
    DispatchResults
        The namedtuple return by Dispatcher.run method
    """

    #np.random.seed(seed) # already done before

    hydro_constraints = dispatcher.make_hydro_constraints_from_res_load_scenario()
    agg_load_without_renew = dispatcher.net_load(params_opf['losses_pct'],
                                                 name=dispatcher.loads.index[0])

    dispatch_results = dispatcher.run(
        agg_load_without_renew,
        params=params_opf,
        gen_constraints=hydro_constraints,
        ramp_mode=parse_ramp_mode(params_opf['ramp_mode']),
        by_carrier=params_opf['dispatch_by_carrier'],
        pyomo=params_opf['pyomo'],
        solver_name=params_opf['solver_name']
    )
    dispatcher.save_results(params, output_folder)

    if params_opf["loss_grid2op_simulation"]:
        new_prod_p, new_prod_forecasted_p = simulate_loss(grid_folder, output_folder, params_opf, write_results = True)
        dispatch_results = update_results_loss(dispatch_results, new_prod_p, params_opf)
    return dispatch_results

def update_results_loss(dispatch_results, new_prod_p, params_opf):
    dispatch_results[0].prods_dispatch[params_opf['nameSlack']] = new_prod_p[params_opf['nameSlack']]
    return dispatch_results

def simulate_loss(input_folder, output_folder, params_opf, write_results = True):
    scenario_folder_path = output_folder
    grid_folder_g2op = input_folder
    move_chronics_temporarily(scenario_folder_path, grid_folder_g2op)
    # try:
    run_grid2op_simulation_donothing(grid_folder_g2op, scenario_folder_path,
                                 agent_type=params_opf['agent_type'])
    # except RuntimeError:
    #     remove_temporary_chronics(grid_folder_g2op)
    #     raise RuntimeError("Error in Grid2op simulation, temporary folder deleted")
    dispatch_results_corrected = correct_scenario_loss(scenario_folder_path, scenario_folder_path, params_opf)
    remove_temporary_chronics(grid_folder_g2op)
    #remove_simulation_data(scenario_folder_path)
    return dispatch_results_corrected


def parse_ramp_mode(mode):
    """
    Parse a string representing the difficulty of the ramps in the OPF into
    a RampMode Enum value
    Parameters
    ----------
    mode : str
        The difficulty mode for the ramps in the OPF

    Returns
    -------
    RampMode
        The encoded RampMode value

    """
    if mode == 'hard':
        return run_economic_dispatch.RampMode.hard
    if mode == 'medium':
        return run_economic_dispatch.RampMode.medium
    if mode == 'easy':
        return run_economic_dispatch.RampMode.easy
    if mode == '':
        return run_economic_dispatch.RampMode.none
    raise ValueError(f'mode only takes values from (hard, medium, easy, none), '
                     '{mode} was passed')
