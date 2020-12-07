import os
from .loss_utils import run_grid2op_simulation_donothing, correct_scenario_loss, move_chronics_temporarily, remove_temporary_chronics

def main(grid_folder_g2op, scenario_folder_path, params_loss, write_results = True):
    move_chronics_temporarily(scenario_folder_path, grid_folder_g2op)
    run_grid2op_simulation_donothing(grid_folder_g2op, scenario_folder_path,
                                     agent_type=params_loss['agent_type'])
    dispatch_results_corrected = correct_scenario_loss(scenario_folder_path, agent_result_path, params_loss)
    remove_temporary_chronics(grid_folder_g2op)
    return dispatch_results_corrected

def check_chronix(scenario_folder_path):
    bool = True
    if os.path.exists(scenario_folder_path):
        if not (os.path.exists(os.path.join(scenario_folder_path,"load_p.csv.bz2") and
                               os.path.exists(os.path.join(scenario_folder_path,"prod_p.csv.bz2")))):
            bool = False
    else:
         bool = False
    return bool

if __name__ == "__main__":
    # Bouchon
    params_loss = {
        "idxSlack":37,
        "nameSlack": "gen_68_37",
        "adjust_pmin": True,
        "adjust_pmax":True
    }

    losses_pct = 1 #0.4 # Provient du params_opf

    scenario_folder_path = ""
    agent_result_path = ""
    dispatch_results_corrected = main(scenario_folder_path, params_loss, write_results=True)


