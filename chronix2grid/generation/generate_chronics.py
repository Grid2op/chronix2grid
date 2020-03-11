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

## Temporaire: constantes
# Calculation period of the scenarios
YEAR = 2007
MONTH = 1  # random.randint(1, 12)
DAY = 1  # random.randint(1,29)
WEEKS = 52
start_date = date(YEAR, MONTH, DAY)
end_date = start_date + timedelta(days=7 * WEEKS)

# Miscellaneaous configuration
n_scenarios = 2
case = 'case118_l2rpn_2020'
INPUT_FOLDER = 'input'

# Chemin de
if WEEKS == 52:
    DISPATCH_INPUT_FOLDER = 'input/dispatch/'+str(YEAR)
else:
    DISPATCH_INPUT_FOLDER = 'input/dispatch/' + str(YEAR)+'_'+str(WEEKS)+'Weeks'




# ======================================================================================================================
## Proper functions
# Read data (independant of the number of scenarios)
def read_configuration(start_date, end_date, input_folder, case):
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

    params['start_date'] = pd.to_datetime(start_date)
    params['end_date'] = pd.to_datetime(end_date)

    # date and time parameters
    params['T'] = int(pd.Timedelta(end_date - start_date).total_seconds() // (60))
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
def main(n_scenarios, start_date, end_date, params, dispatch_input_folder, weeks, loads_charac, load_weekly_pattern):
    print('=====================================================================================================================================')
    print('============================================== CHRONICS GENERATION ==================================================================')
    print('=====================================================================================================================================')

    # Make sure the seeds are the same, whether computation is parrallel or sequential
    seeds = [np.random.randint(low=0, high=2**31) for _ in range(n_scenarios)]

    # Make sure the output folder exists
    main_folder = os.path.join(dispatch_input_folder)
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)

    # Launch proper scenario generation
    for i, seed in enumerate(seeds):
        print("================ Generating scenario number "+str(i)+" ================")
        gen_loads.main(i, dispatch_input_folder, weeks, seed, start_date, end_date, params, loads_charac, load_weekly_pattern)
        print('\n')
    return

### Test
params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern = read_configuration(start_date, end_date, INPUT_FOLDER, case)
main(n_scenarios, start_date, end_date, params, DISPATCH_INPUT_FOLDER, WEEKS, loads_charac, load_weekly_pattern)