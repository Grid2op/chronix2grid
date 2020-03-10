import argparse
import os
import json
import pandas as pd

# import sys
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from kpis import EconomicDispatchValidator

parser = argparse.ArgumentParser(description='Synthetic power system time series generation')

parser.add_argument('-i', '--syn_dispatch', 
                    type=str,
                    help='Specify csv that is a result of a synthetic dispatch for IEEE 118 grid')

parser.add_argument('-r', '--ref_dispatch', default='./grid_charac/prod_p.csv',
                    type=str, help='CSV path refence dispatch')

parser.add_argument('-c', '--ref_consumption', default='./grid_charac/load_p.csv',
                    type=str, help='CSV path consumption dispatch')

parser.add_argument('-d', '--destination', default='./',
                    type=str, help='Destination path')

def main():

    args = parser.parse_args()
    
    

    # Read csv
    ref_dispatch = pd.read_csv(args.ref_dispatch)
    ref_dispatch['Time'] = pd.to_datetime(ref_dispatch['Time'])
    ref_dispatch.set_index('Time', drop=True, inplace=True)

    syn_dispatch = pd.read_csv(args.syn_dispatch, index_col=0)
    consumption = pd.read_csv(args.ref_consumption, index_col=0)

    # -- + -- + ---
    # DONT FORGET TO ADD THE CONSTRAINT THE REF DISPATCH IT'S ONLY
    # UNITL XXXX-12-31 11:00 TO THE OTHERS DATAFRAMES / ANO BICIESTO
    # -- + -- + ---

    # Set same index as ref dispatch to avoid
    # problem when working in posterior KPI's
    syn_dispatch.index = ref_dispatch.index
    consumption.index = ref_dispatch.index
    
    # Load grid charac
    prods_charac = pd.read_csv(os.path.abspath('./grid_charac/prods_charac.csv'))
    loads_charac = pd.read_csv(os.path.abspath('./grid_charac/loads_charac.csv'))
    
    # Load agg price profile
    prices = pd.read_csv(os.path.abspath('./grid_charac/prices.csv'))
    prices['Time'] = pd.to_datetime(prices['Time'])
    prices.set_index('Time', drop=True, inplace=True)

    # Start Economic dispatch validator
    # -- + -- + -- + -- + -- + -- + --
    print ('(1) Computing KPI\'s...')
    dispatch_validator = EconomicDispatchValidator(consumption, 
                                                   ref_dispatch, 
                                                   syn_dispatch,
                                                   prods_charac=prods_charac, 
                                                   loads_charac=loads_charac, 
                                                   prices=prices)

    # Get Hydro KPI
    dispatch_validator.hydro_kpi()

    # Get Wind KPI
    dispatch_validator.wind_kpi()

    # Get Solar KPI
    dispatch_validator.solar_kpi()

    # Wind - Solar KPI
    dispatch_validator.wind_load_kpi()

    # Get Nuclear KPI
    dispatch_validator.nuclear_kpi()

    # Write json output file
    # -- + -- + -- + -- + --
    print ('(2) Generating json outout file...')
    with open(os.path.join(args.destination, 'ec_validator_output.txt'), 'w') as json_f:
        json.dump(dispatch_validator.output, json_f)

    print ('-Done-\n')
        

if __name__=="__main__":
    main()