import os
import json

# Other Python libraries
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta

# Libraries developed for this module
import consumption.consumption_utils as conso


def main(i, destination_folder, weeks, seed, start_date, end_date, params, loads_charac, load_weekly_pattern):
    """
    This is the load generation function, it allows you to generate consumption chronics based on demand nodes characteristics and on weekly demand patterns.

    Parameters
    ----------
    case
    destination
    seed
    start_date
    end_date

    Returns
    -------

    """

    # Define datetime indices
    datetime_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq=str(params['dt']) + 'min')

    # Generate GLOBAL temperature noise
    print('Computing global auto-correlated spatio-temporal noise ...')
    ## temperature is simply to reflect the fact that loads is correlated spatially, and so is the
    ## real "temperature". It is not the real temperature.
    temperature_noise = conso.generate_coarse_noise(params, 'temperature')

    print('Computing loads ...')
    loads_series = conso.compute_loads(loads_charac, temperature_noise, params, load_weekly_pattern)

    loads_series_df = pd.DataFrame(loads_series)
    loads_series_mat = loads_series_df.values

    # Write results
    scenario_destination_path = os.path.join(destination_folder, 'Scenario_'+str(i))
    if not os.path.exists(scenario_destination_path):
        os.mkdir(scenario_destination_path)
    conso.create_csv(loads_series, os.path.join(scenario_destination_path, 'load_p_forecasted.csv.bz2'),
                  reordering=True,
                  shift=True)
    conso.create_csv(loads_series, os.path.join(scenario_destination_path, 'load_p.csv.bz2'),
                  reordering=True,
                  noise=params['planned_std'])



    
