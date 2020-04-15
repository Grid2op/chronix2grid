# Native python libraries
import os
import json

# Other Python libraries
import pandas as pd
import numpy as np
from datetime import timedelta

# Libraries developed for this module
from .consumption import generate_load as gen_loads
from .renewable import generate_solar_wind as gen_enr
from .thermal import generate_dispatch as gen_dispatch



def read_configuration(input_folder, case, start_date, weeks):
    """
    This functions reads the detailed parameters of the generation in params.json, but also the case settup through files prods_charac.csv, loads_charac.csv and lines.csv
    It processes it and returns usable format for each of those parameters

    Parameters
    ----------
    input_folder (string): peth of folder where inputs are stored
    case (str): name of case to study (must be a folder within input_folder)
    start_date (str): string containing start date of generation (recommended format is YYYY-MM-DD)
    weeks (int): number of weeks on which to generate the chronics

    Returns
    -------
    int: year of generated chronics (if weeks>52, takes the most ancient year)
    dict: dictionnary with parameters, including a formatting of params.json
    pandas.DataFrame: characteristics of generators such as Pmax, carrier and region
    pandas.DataFrame: characteristics of loads node such as Pmax, type of demand and region
    pandas.DataFrame: normalized weekly pattern of load, used as reference for load generation
    pandas.DataFrame: normalized yearly solar production pattern, used as reference for solar chronics generation
    pandas.DataFrame: characteristics of lines

    """

    # Read Json parameters
    print('Importing parameters ...')
    json1_file = open(os.path.join(input_folder, case, 'params.json'))
    json1_str = json1_file.read()
    params = json.loads(json1_str)
    for key, value in params.items():
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = pd.to_datetime(value, format='%Y-%m-%d')

    # Compute date and time parameters
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    params['weeks'] = weeks
    params['start_date'] = start_date
    year = start_date.year
    params['end_date'] = params['start_date'] + timedelta(days=7 * int(weeks)) - timedelta(minutes=params['dt'])
    params['T'] = int(pd.Timedelta(params['end_date'] - params['start_date']).total_seconds() // (60))
    # Nt_inter = int(params['T'] // params['dt'] + 1)

    # Import loads_charac.csv, prods_charac.csv and lines.csv in desired case
    print('Importing loads prods and lines parameters ...')
    try:
        loads_charac = pd.read_csv(os.path.join(input_folder, case, 'loads_charac.csv'), sep = ',')
        names = loads_charac['name']   # to generate error if separator is wrong
        prods_charac = pd.read_csv(os.path.join(input_folder, case, 'prods_charac.csv'), sep = ',')
        lines = pd.read_csv(os.path.join(input_folder, case, 'lines_names.csv'), sep=',')
    except:
        loads_charac = pd.read_csv(os.path.join(input_folder, case, 'loads_charac.csv'), sep=';')
        prods_charac = pd.read_csv(os.path.join(input_folder, case, 'prods_charac.csv'), sep=';')
        lines = pd.read_csv(os.path.join(input_folder, case, 'lines_names.csv'), sep=';')

    # Importing weekly patterns
    load_weekly_pattern = pd.read_csv(os.path.join(input_folder, 'patterns', 'load_weekly_pattern.csv'))
    solar_pattern = np.load(os.path.join(input_folder, 'patterns', 'solar_pattern.npy'))

    return year, params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern, lines


# Call generation scripts n_scenario times with dedicated random seeds
def main(case, year, n_scenarios, params, input_folder, output_folder, prods_charac, loads_charac, lines, solar_pattern, load_weekly_pattern):
    """
    Main function for chronics generation. It works with three steps: load generation, renewable generation (solar and wind) and then dispatch computation to get the whole energy mix

    Parameters
    ----------
    case (str): name of case to study (must be a folder within input_folder)
    year (int): year of generation
    n_scenarios (int): number of desired scenarios to generate for the same timescale
    params (dict): parameters of generation, as returned by function chronix2grid.generation.generate_chronics.read_configuration
    input_folder (str): path of folder containing inputs
    output_folder (str): path where outputs will be written (intermediate folder case/year/scenario will be used)
    prods_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    loads_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    lines (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    solar_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    load_weekly_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration


    Returns
    -------

    """

    print('=====================================================================================================================================')
    print('============================================== CHRONICS GENERATION ==================================================================')
    print('=====================================================================================================================================')

    ## Random seeds:  Make sure the seeds are the same, whether computation is parallel or sequential
    seeds = [np.random.randint(low=0, high=2 ** 31) for _ in range(n_scenarios)]

    ## Folder settings

    # Folder names
    dispatch_input_folder_case = os.path.join(input_folder, case, 'dispatch')
    dispatch_input_folder= os.path.join(dispatch_input_folder_case, str(year))
    dispatch_output_folder = os.path.join(output_folder, str(year))

    # Make sure the output folders exist
    if not os.path.exists(dispatch_input_folder_case):
        os.makedirs(dispatch_input_folder_case)
    if not os.path.exists(dispatch_input_folder):
        os.makedirs(dispatch_input_folder)
    out_folder = os.path.join(dispatch_output_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    ## Launch proper scenarios generation
    for i, seed in enumerate(seeds):
        print("================ Generating scenario number "+str(i)+" ================")
        load, load_forecasted = gen_loads.main(i, dispatch_input_folder, seed, params, loads_charac, load_weekly_pattern, write_results = True)

        prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = gen_enr.main(i, dispatch_input_folder, seed,params, prods_charac, solar_pattern, write_results = True)

        # gen_dispatch.main(i, load, prod_solar, prod_wind, out_folder, seed, params,
        #                      prods_charac, lines, compute_hazards = True)
        print('\n')
    return
