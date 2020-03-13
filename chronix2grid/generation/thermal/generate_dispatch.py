import os
import json

# Other Python libraries
import pandas as pd
import numpy as np

# Libraries developed for this module
import consumption.consumption_utils as conso


def main(i, load_forecasted, prod_solar_wind_forecasted, destination_folder, seed, params, prods_charac):
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

    