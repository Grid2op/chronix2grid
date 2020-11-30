from .loss_utils import run_grid2op_simulation_donothing, correct_scenario_loss

def main(scenario_folder_path, params_loss, write_results = True):
    agent_result_path = "" #TODO: décider où on stocke ça et si c'est de manière permanente ou non. Pour l'instant dans les outputs
    run_grid2op_simulation_donothing(scenario_folder_path, agent_result_path)
    dispatch_results_corrected = correct_scenario_loss(scenario_folder_path, agent_result_path, params_loss)
    return dispatch_results_corrected

def check_chronix(scenario_folder_path):
    # TODO: Check que les chorniques existent et ont fait l'objet de LRT.. Voir que de T en fait ?
     return True

if __name__ == "__main__":
    # Bouchon
    params_loss = {
        "explicit_loss_mode":False,
        "loss_correction_mode":True,
        "idxSlack":37,
        "nameSlack": "gen_68_37",
        "adjust_pmin": True,
        "adjust_pmax":True
    }

    losses_pct = 1 #0.4 # Provient du params_opf

    scenario_folder_path = ""
    agent_result_path = ""
    dispatch_results_corrected = main(scenario_folder_path, params_loss, write_results=True)


