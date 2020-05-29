
def check_scenario(n_scenarios, scenario_id):
    if n_scenarios == 1 and scenario_id is None:
        raise RuntimeError("scenario_id must not be None when n_scenarios == 1.")
