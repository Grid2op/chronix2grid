# Native python packages
import os

# Other python packages


# Modules
import generation.generate_chronics as gen
import kpi.main as kpis

### CONSTANT
INPUT_FOLDER = 'generation/input'
OUTPUT_FOLDER = 'generation/output'
#CASE = 'case118_l2rpn_2020'
CASE = 'case118_l2rpn'
KPI_INPUT_FOLDER = 'kpi/input'
IMAGES_FOLDER = 'kpi/images'

### LAUNCH CONFIGURATION
# Chronic generation parameters in <INPUT_FOLDER>/<CASE>/params.json
# Warning: only dates and floats in params.json
# KPI calculation parameters in <KPI_INPUT_FOLDER>/paramsKPI.json
start_date = "2012-01-01"
weeks = 4
n_scenarios = 1


### COMPUTATION PHASES
GENERATION_CHRONICS = True    # Il manque la partie dispatch. A integrer dans generation/thermal une fois Camilo prêt
COMPUTE_KPI_ENR_ONLY = False     # Pour visualiser des KPIs sur solar et wind en attendant le dispatch
COMPUTE_ALL_KPI = False         # En attendant que le dispatch soit prêt



# ==============================================================
### PROPER COMPUTATION

## Reading parameters
year, params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern, lines = gen.read_configuration(
    INPUT_FOLDER, CASE, start_date, weeks)

## Chronic generation
if GENERATION_CHRONICS:
    gen.main(CASE, year, n_scenarios, params, INPUT_FOLDER, OUTPUT_FOLDER, prods_charac, loads_charac, lines, solar_pattern, load_weekly_pattern)

## KPI formatting and computing
if COMPUTE_KPI_ENR_ONLY:
    # Get and format solar and wind on all timescale, then compute KPI and save plots
    wind_solar_only = True
    if not os.path.exists(KPI_INPUT_FOLDER):
        os.mkdir(KPI_INPUT_FOLDER)
    kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, IMAGES_FOLDER, year, CASE, n_scenarios, wind_solar_only, params)

elif COMPUTE_ALL_KPI:
    # Get and format monthly optimized chronics
    wind_solar_only = False
    if not os.path.exists(KPI_INPUT_FOLDER):
        os.mkdir(KPI_INPUT_FOLDER)
    kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, IMAGES_FOLDER, year, CASE, n_scenarios, wind_solar_only, params)

