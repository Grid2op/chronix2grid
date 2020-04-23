import os
import shutil

import numpy as np

from .EDispatch_L2RPN2020 import run_economic_dispatch


def main(dispatcher, input_folder, output_folder, seed, params_opf):
    """

    Parameters
    ----------
    dispatcher : Dispatcher
        The Dispatcher instance used for running the OPF
    input_folder : str
        The path to the directory containing the inputs for the dispatch
    output_folder
        The path of the directory that will receive the outputs of the dispatch
    seed : int
        Random seed for parallel execution
    params_opf : dict
        Options for the OPF

    Returns
    -------
    DispatchResults
        The namedtuple return by Dispatcher.run method
    """

    np.random.seed(seed)

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

    dispatcher.save_results(output_folder)

    files_to_move = ['load_p_forecasted.csv.bz2', 'load_q_forecasted.csv.bz2',
                     'load_q.csv.bz2', 'prod_v.csv.bz2']
    for file_to_move in files_to_move:
        shutil.copy(os.path.join(input_folder, file_to_move),
                    os.path.join(output_folder, file_to_move))

    return dispatch_results


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
