# ChroniX2Grid

## Configuration of chronics and KPI generation

### Main variables
Within chronix2grid/main.py the main variables have to be set by users.
- **start_date**: day at which chronics generation begins. Recommended format is "YYYY-MM-DD"
- **weeks**: number of weeks to generate. It is recommended to generate a number of weeks inferior or equal to 52
- **n_scenarios**: number of different scenarios to generate for the given time horizon
- **INPUT_FOLDER**: path for folder containing input data for generation (params.json, cases)
- **CASE**: name of case to study. This has to be the name of a folder containing data within *INPUT_FOLDER* 
- **OUTPUT_FOLDER**: path of folder in which we want to write generated chronics
- **KPI_INPUT_FOLDER**: path of folder in which KPI inputs are given. It must include paramsKPI.json and benchmark folders for reference chronics 
- **IMAGES_FOLDER**: path of folder in which we want to write KPI plots (png and interactive html)

### Chronic generation detailed configuration
More detailed configuration has to be set in *INPUT_FOLDER/CASE/params.json* such as timestep, noise and correlation intensity...


### KPI configuration
Some general parameters have to be set in *KPI_INPUT_FOLDER/paramsKPI.json*
- **comparison**: name of benchmark folder to use for KPI reference chronics. For example, benchmark *France* has been implemented with eco2mix and renewable ninja data
- **timestep**: timestep for KPI computation. For example, renewable ninja data requires minimum timestep of 60min
- **night_hours**: dictionary to provide night hours for each season of year
- **seasons**: dictionary to provide months in each season of year


## Separate launch 
Within chronix2grid/main.py one can chose to launch some modules separately by setting boolean values
- **GENERATION_CHRONICS**: if True, will launch generation modules
- **COMPUTE_KPI_SOLAR_WIND_LOAD_ONLY**: if True, will compute KPIs only for wind, solar and loads chronics
- **COMPUTE_KPI_FULL**: if True and if **COMPUTE_KPI_SOLAR_WIND_LOAD_ONLY** is False, will compute all KPIs