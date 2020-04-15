# Native python packages
import os

# Chronix2grid modules
import generation.generate_chronics as gen
import kpi.main as kpis

# ==============================================================

### CONFIGURATION

## CONSTANT VARIABLES
INPUT_FOLDER = 'generation/input'
OUTPUT_FOLDER = 'generation/output'
#CASE = 'case118_l2rpn_2020'
CASE = 'case118_l2rpn'

KPI_INPUT_FOLDER = 'kpi/input'
IMAGES_FOLDER = "kpi/images'"

## LAUNCH CONFIGURATION
start_date = "2012-01-01"
weeks = 4
n_scenarios = 1

## CHOSE COMPUTATION PHASES
GENERATION_CHRONICS = True    # Il manque la partie dispatch. A integrer dans generation/thermal une fois Camilo prêt
COMPUTE_KPI_LOAD_SOLAR_WIND = True     # Pour visualiser des KPIs sur solar et wind en attendant le dispatch
COMPUTE_KPI_FULL = False         # En attendant que le dispatch soit prêt



# ==============================================================

### PROPER COMPUTATION

## Folders are specific to studied case
OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, CASE)
IMAGES_FOLDER = os.path.join('kpi/images', CASE)

## Reading parameters
year, params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern, lines = gen.read_configuration(
    INPUT_FOLDER, CASE, start_date, weeks)

## Chronic generation
if GENERATION_CHRONICS:
    gen.main(CASE, year, n_scenarios, params, INPUT_FOLDER, OUTPUT_FOLDER, prods_charac, loads_charac, lines, solar_pattern, load_weekly_pattern)

## KPI formatting and computing
if COMPUTE_KPI_LOAD_SOLAR_WIND:
    # Get and format solar and wind on all timescale, then compute KPI and save plots
    wind_solar_only = True
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, IMAGES_FOLDER, year, CASE, n_scenarios, wind_solar_only, params, loads_charac, prods_charac)

elif COMPUTE_KPI_FULL:
    # Get and format dispatched chronics, then compute KPI and save plots
    wind_solar_only = False
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, IMAGES_FOLDER, year, CASE, n_scenarios, wind_solar_only, params, loads_charac, prods_charac)

