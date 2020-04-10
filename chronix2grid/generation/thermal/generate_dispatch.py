import os
import json

# Other Python libraries
import pandas as pd
import numpy as np

# Libraries developed for this module
import generation.thermal.dispatch_utils as disp


def main(i, load_forecasted, prod_solar, prod_wind, output_folder, seed, params,
                          prods_charac, lines, compute_hazards = True):
    """
    This is the dispatch generation function, it allows you to generate producion chronics
    based on loads scenarios and solar and wind fatal generation with an optimal economic dispatch
    It takes into account generators constraints such as ramps and Pmax, but does not take any grid capacity constraints 
    because it has to stay generic for the challenge

    Parameters
    ----------
    i (int): scenario number
    load_forecasted (pandas.DataFrame): forecasted load scenario at each node
    prod_solar_wind_forecasted (pandas.DataFrame): forecasted solar and wind scenario at each node
    destination_folder (string): where results are written
    seed (int): random seed of the scenario
    start_date (datetime.date)
    end_date (datetime.date)
    params (dict): system params such as timestep or mesh characteristics

    Returns
    ------
    pandas.DataFrame

    """

    np.random.seed(seed)
    output_folder = os.path.join(output_folder, 'Scenario_'+str(i))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print('Generating maintenance and hazard signals ...')
    if compute_hazards:
        maintenance = disp.compute_random_event('maintenance', lines, params)
        hazards = disp.compute_random_event('hazards', lines, params)
        disp.create_csv(maintenance, os.path.join(output_folder, 'maintenance.csv.bz2'), reordering=True)
        disp.create_csv(maintenance, os.path.join(output_folder, 'maintenance_forecasted.csv.bz2'), reordering=True)
        disp.create_csv(hazards, os.path.join(output_folder, 'hazards.csv.bz2'), reordering=True)



    