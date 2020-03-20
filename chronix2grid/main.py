# Native python packages
import os

# Other python packages


# Modules
import generation.generate_chronics as gen
import generation.kpi_preprocessing as preproc
import kpi.main as kpis

### CONSTANTES
INPUT_FOLDER = 'generation/input'
OUTPUT_FOLDER = 'generation/output'
CASE = 'case118_l2rpn_2020'
KPI_INPUT_FOLDER = 'kpi/input'
IMAGES_FOLDER = 'kpi/images'
COMPARISON = 'eco2mix'

### CONFIG LAUNCH IN generation/<INPUT_FOLDER>/<CASE>/params.json
# Warning: only dates and floats in params.json


### CHOIX DES PHASES A CALCULER
GENERATION_CHRONIQUES = False    # Il manque la partie dispatch. A integrer dans generation/thermal une fois Camilo prêt
COMPUTE_KPI_ENR_ONLY = True     # Pour visualiser des KPIs sur solar et wind en attendant le dispatch
COMPUTE_ALL_KPI = False
KPI_TIMESTEP = '30min'



# ==============================================================
### CALCUL

## Reading parameters
year, n_scenarios, params, loads_charac, prods_charac, load_weekly_pattern, solar_pattern = gen.read_configuration(
    INPUT_FOLDER, CASE)

## Chronic generation
if GENERATION_CHRONIQUES:
    gen.main(year, n_scenarios, params, INPUT_FOLDER, OUTPUT_FOLDER, prods_charac, loads_charac, solar_pattern, load_weekly_pattern)

## KPI formatting and computing
if COMPUTE_KPI_ENR_ONLY:
    # Get and format solar and wind on all timescale, then compute KPI and save plots
    wind_solar_only = True
    if not os.path.exists(KPI_INPUT_FOLDER):
        os.mkdir(KPI_INPUT_FOLDER)
    for i in range(n_scenarios):
        preproc.kpi_format_production(year, i, INPUT_FOLDER, KPI_INPUT_FOLDER, KPI_TIMESTEP, thermal = False)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, IMAGES_FOLDER, year, CASE, i, wind_solar_only, comparison=COMPARISON)

elif COMPUTE_ALL_KPI:
    # Get and format monthly optimized chronics
    wind_solar_only = False
    if not os.path.exists(KPI_INPUT_FOLDER):
        os.mkdir(KPI_INPUT_FOLDER)
    for i in range(n_scenarios):
        preproc.kpi_format_production(year, i, OUTPUT_FOLDER, KPI_INPUT_FOLDER, IMAGES_FOLDER, KPI_TIMESTEP, thermal = True)
        kpis.main(KPI_INPUT_FOLDER, INPUT_FOLDER, year, CASE, i, wind_solar_only, comparison=COMPARISON)

