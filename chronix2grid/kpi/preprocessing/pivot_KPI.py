import os
import sys
import json
import pandas as pd

from kpi.preprocessing.pivot_utils import chronics_to_kpi, eco2mix_to_kpi, renewableninja_to_kpi, eco2mix_to_kpi_regional

def pivot_format(chronics_folder, kpi_input_folder, year, scenario_num, prods_charac, loads_charac, wind_solar_only, params):

    # Json parameters
    json1_file = open(os.path.join(kpi_input_folder, 'paramsKPI.json'))
    json1_str = json1_file.read()
    paramsKPI = json.loads(json1_str)
    comparison = paramsKPI['comparison']
    timestep = paramsKPI['timestep']

    monthly_pattern = paramsKPI['seasons']
    hours = paramsKPI['night_hours']

    # Format chosen benchmark
    if comparison == 'France':
        corresp_regions = {'R1':"Hauts-de-France", "R2": "Nouvelle-Aquitaine", "R3": "PACA"}
        if wind_solar_only:
            ref_prod, ref_load = renewableninja_to_kpi(kpi_input_folder, timestep, prods_charac, loads_charac, year,
                                                       params)
        else:
            ref_prod, ref_load = eco2mix_to_kpi_regional(kpi_input_folder, timestep, prods_charac, loads_charac, year, params,
                                                         corresp_regions)
    elif comparison == 'Texas':
        print("Arrêt du calcul: la comparaison Texas n'est pas encore implémentée. La comparaison France est disponible")
        sys.exit()
    else:
        print("Please chose one available benchmark in paramsKPI.json/comparison. Given comparison is: "+comparison)
        sys.exit()

    # Format generated chronics

    if wind_solar_only:
        syn_prod, syn_load = chronics_to_kpi(year, scenario_num, chronics_folder, timestep, params,
                                                     thermal=not wind_solar_only, monthly = False)
        return ref_prod, ref_load, syn_prod, syn_load, monthly_pattern, hours
    else:
        syn_prod, syn_load, prices = chronics_to_kpi(year, scenario_num, chronics_folder, timestep, params,
                                             thermal=not wind_solar_only)
        return ref_prod, ref_load, syn_prod, syn_load, monthly_pattern, hours, prices

