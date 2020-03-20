import argparse
import os
import json
import pandas as pd

# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from kpi.eco2mix.preprocessing_eco2mix import format_for_kpi
from kpi.deterministic.kpis import EconomicDispatchValidator



def main(year, case, scenario_num, wind_solar_only, comparison = 'eco2mix'):
    parser = argparse.ArgumentParser(description='Synthetic power system time series generation')

    parser.add_argument('-i', '--syn_dispatch',
                        default='kpi/input/' + str(year) + '/Scenario_' + str(scenario_num) + '/prod_p.csv',
                        type=str,
                        help='Specify csv that is a result of a synthetic dispatch for IEEE 118 grid')

    parser.add_argument('-r', '--ref_dispatch',
                        default='kpi/input/' + str(year) + '/Scenario_' + str(scenario_num) + '/prod_p.csv',
                        type=str, help='CSV path refence dispatch')

    parser.add_argument('-c', '--ref_consumption',
                        default='kpi/input/' + str(year) + '/Scenario_' + str(scenario_num) + '/load_p.csv',
                        type=str, help='CSV path consumption dispatch')

    parser.add_argument('-d', '--destination', default='./',
                        type=str, help='Destination path')

    args = parser.parse_args()

    print('=====================================================================================================================================')
    print('========================================= KPI GENERATION FOR SCENARIO '+str(scenario_num)+' ============================================================')
    print('=====================================================================================================================================')

    # Load grid charac
    prods_charac = pd.read_csv(os.path.abspath('generation/input/'+case+'/prods_charac.csv'), sep=';')
    loads_charac = pd.read_csv(os.path.abspath('generation/input/'+case+'/loads_charac.csv'), sep=';')

    # Read csv
    if comparison == 'eco2mix':
        repo_in = 'kpi/eco2mix/input/eCO2mix_RTE_Annuel-Definitif_'+str(year)+'.csv'
        ref_dispatch = format_for_kpi(repo_in, '30min', prods_charac)
    else:
        ref_dispatch = pd.read_csv(args.ref_dispatch, sep = ';',decimal = '.')
        ref_dispatch['Time'] = pd.to_datetime(ref_dispatch['Time'])
        ref_dispatch.set_index('Time', drop=True, inplace=True)

    syn_dispatch = pd.read_csv(args.syn_dispatch, index_col=0, sep = ';')
    consumption = pd.read_csv(args.ref_consumption, index_col=0, sep = ';')

    # -- + -- + ---
    # DONT FORGET TO ADD THE CONSTRAINT THE REF DISPATCH IT'S ONLY
    # UNITL XXXX-12-31 11:00 TO THE OTHERS DATAFRAMES / ANO BICIESTO
    # -- + -- + ---

    # Set same index as ref dispatch to avoid
    # problem when working in posterior KPI's
    syn_dispatch.index = ref_dispatch.index
    consumption.index = ref_dispatch.index
    
    # Load agg price profile
    if not wind_solar_only:
        prices = pd.read_csv(os.path.abspath('kpi/input/'+str(year)+'+/Scenario_'+str(scenario_num)+'/prices.csv'), sep = ';')
        prices['Time'] = pd.to_datetime(prices['Time'])
        prices.set_index('Time', drop=True, inplace=True)

        # Start Economic dispatch validator
        # -- + -- + -- + -- + -- + -- + --
        print ('(1) Computing KPI\'s...')
        dispatch_validator = EconomicDispatchValidator(consumption,
                                                       ref_dispatch,
                                                       syn_dispatch,
                                                       year,
                                                       scenario_num,
                                                       prods_charac=prods_charac,
                                                       loads_charac=loads_charac,
                                                       prices=prices)
    else:
        # Start Economic dispatch validator
        # -- + -- + -- + -- + -- + -- + --
        print('(1) Computing KPI\'s...')
        dispatch_validator = EconomicDispatchValidator(consumption,
                                                       ref_dispatch,
                                                       syn_dispatch,
                                                       year,
                                                       scenario_num,
                                                       prods_charac=prods_charac,
                                                       loads_charac=loads_charac)

    # Get dispatch temporal view
    dispatch_validator.plot_carriers_pw(curve='reference', stacked=True, max_col_splot=1, save_html=True,
                                        wind_solar_only=wind_solar_only)
    dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=True, max_col_splot=1, save_html=True,
                                        wind_solar_only=wind_solar_only)

    # Get Hydro KPI
    if not wind_solar_only:
        dispatch_validator.hydro_kpi()

    # Get Wind KPI
    dispatch_validator.wind_kpi()

    # Get Solar KPI
    dispatch_validator.solar_kpi()

    # Wind - Solar KPI
    dispatch_validator.wind_load_kpi()

    # Get Nuclear KPI
    if not wind_solar_only:
        dispatch_validator.nuclear_kpi()

    # Write json output file
    # -- + -- + -- + -- + --
    print ('(2) Generating json output file...')
    with open(os.path.join(args.destination, 'ec_validator_output.txt'), 'w') as json_f:
        json.dump(dispatch_validator.output, json_f)

    print ('-Done-\n')
        

if __name__=="__main__":
    main()