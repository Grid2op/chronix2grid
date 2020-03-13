# Native python libraries
import os
import json

# Other Python libraries
import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta

# Libraries developed for this module
import consumption.generate_load as gen_loads
import renewable.generate_solar_wind as gen_enr
import thermal.generate_dispatch as gen_dispatch


## Temporaire: constantes
# Calculation period of the scenarios (dates in params.json)
YEAR = 2007

# Miscellaneaous configuration
n_scenarios = 1
case = 'case118_l2rpn_2020'

# Chemin des inputs, outputs et outputs intermédiaires
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
DISPATCH_INPUT_FOLDER = os.path.join(INPUT_FOLDER,'dispatch/'+str(YEAR))
DISPATCH_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER,str(YEAR))


# Paramètres de modélisation
SMOOTHDIST = 0.001  # parameter to smooth the distribution, and make it continuous. It is advised not to modify this parameter
COSPHI = 0.7  # parameter used to sample reactive from active


# ======================================================================================================================
## Proper functions
# Read data (independant of the number of scenarios)
def read_configuration(input_folder, case):
    # Json parameters
    print('Importing parameters ...')
    json1_file = open(os.path.join(input_folder, case, 'params.json'))
    json1_str = json1_file.read()
    params = json.loads(json1_str)
    for key, value in params.items():
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = pd.to_datetime(value, format='%Y-%m-%d')

    # date and time parameters
    params['T'] = int(pd.Timedelta(params['end_date'] - params['start_date']).total_seconds() // (60))
    Nt_inter = int(params['T'] // params['dt'] + 1)

    # Import loads_charac.csv and prods_charac.csv
    print('Importing loads and prods parameters ...')
    loads_charac = pd.read_csv(os.path.join(INPUT_FOLDER, case, 'loads_charac.csv'), sep = ';')
    prods_charac = pd.read_csv(os.path.join(INPUT_FOLDER, case, 'prods_charac.csv'), sep = ';')

    # Importing weekly patterns
    load_weekly_pattern = pd.read_csv(os.path.join(INPUT_FOLDER, 'patterns', 'load_weekly_pattern.csv'))
    solar_pattern = np.load(os.path.join(INPUT_FOLDER, 'patterns', 'solar_pattern.npy'))

    return params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern


# Call generation scripts n_scenario times with dedicated random seeds
def main(n_scenarios, params, dispatch_input_folder, dispatch_output_folder, loads_charac, load_weekly_pattern):
    print('=====================================================================================================================================')
    print('============================================== CHRONICS GENERATION ==================================================================')
    print('=====================================================================================================================================')

    # Make sure the seeds are the same, whether computation is parallel or sequential
    seeds = [np.random.randint(low=0, high=2**31) for _ in range(n_scenarios)]

    # Make sure the output folders exist
    main_folder = os.path.join(dispatch_input_folder)
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)

    out_folder = os.path.join(dispatch_output_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Launch proper scenario generation
    for i, seed in enumerate(seeds):
        print("================ Generating scenario number "+str(i)+" ================")
        load, load_forecasted = gen_loads.main(i, dispatch_input_folder, seed, params, loads_charac, load_weekly_pattern)

        prod_solar, prod_solar_forecasted, prod_wind, prod_wind_forecasted = gen_enr.main(i, dispatch_input_folder, seed,
                                                                   params, prods_charac, solar_pattern, SMOOTHDIST)
        # gen_dispatch.main(i, load_forecasted, prod_solar_wind_forecasted, dispatch_output_folder,
        #                    seed, params, prods_charac)
        print('\n')
    return

### Test
params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern = read_configuration(INPUT_FOLDER, case)
main(n_scenarios, params, DISPATCH_INPUT_FOLDER, DISPATCH_OUTPUT_FOLDER, loads_charac, load_weekly_pattern)
