import grid2op

def run_grid2op_simulation_donothing(scenario_folder_path, agent_result_path):
    # Fonction Notebook - Run Grid2op sur les chroniques du scénario et écrire EpisodeData à agent_path_result
    return 0

def correct_scenario_loss(scenario_folder_path, agent_result_path, params_loss):
    # Fonction Notebook - lire EpisodeData et Chroniques générées puis amender sur le slack bus generator
    # Subtilité: checker la comparaison entre cet amendement et Pmin/Pmax. Dans params_loss, proposer un mode pour relaxer cette contrainte, sinon renvoyer erreur
    return 0