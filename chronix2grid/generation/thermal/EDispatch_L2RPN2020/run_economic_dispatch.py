import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import pypsa
from datetime import datetime, timedelta

# from .utils import format_dates
# from .utils import import_data
from .utils import get_params
from .utils import adapt_gen_prop
from .utils import run_opf
from .utils import get_indivitual_snapshots_per_mode
# from utils import interpolate
from .utils import add_noise_gen
from .utils import reformat_gen_constraints
from .utils import preprocess_input_data
# from .utils import generate_prod_voltage
# from .utils import generate_reactive_loads
# from .utils import generate_hazard_maintenance
# from .utils import generate_forecasts

params={'snapshots': [],
        'step_opf_min': 5,
        'mode_opf': 'day',
        'reactive_comp': 1.025,
        }

gen_const = gen_constraints={'p_max_pu': None, 'p_min_pu': None}

def main_run_disptach(pypsa_net, 
                      load,
                      params=params,
                      gen_constraints=gen_const,
                     ):
    
    # Define params for opf
    len_chronics = load.shape[0]
    params = get_params(len_chronics, params)
    # Resample input data and set temp index to handle to run
    # OPF by days, week, months
    print ('Preprocessing PyPSA grid..')
    load_, gen_constraints_ = preprocess_input_data(load, gen_constraints, params)
    # Scale gen properties (ramps up/down)
    pypsa_net = adapt_gen_prop(pypsa_net, params['step_opf_min'])
    # Define individual snapshots to formulmate
    # one single opf formulation per period
    grouped_snapshots = get_indivitual_snapshots_per_mode(load_.index, params['mode_opf'])

    start = time.time()
    results = []
    gen_const_2_opf = dict.fromkeys(gen_constraints)
    for snaps in grouped_snapshots:
        # Select partial load and gen contraints
        load_2_opf = load_.loc[snaps]
        gen_const_2_opf.update({k: v.loc[snaps] for k, v in gen_constraints_.items()})
        # Run opf function
        results.append(run_opf(pypsa_net, load_2_opf, gen_const_2_opf, params))
        print ('-- opf succeeded    >Please check always => Objective value (should be greater than zero!')

    # Unpack individual dispatchs
    opf_prod = pd.DataFrame()
    for df in results:
        opf_prod = pd.concat([opf_prod, df], axis=0)

    # Sort by datetime
    opf_prod.sort_index(inplace=True)

    # Create complete prod_p dataframe and interpolate missing rows
    prod_p = opf_prod.copy()

    # Add noise to results
    gen_cap = pypsa_net.generators.p_nom
    prod_p_with_noise = add_noise_gen(prod_p, gen_cap, noise_factor=0.001)

    end = time.time()
    # print('OPF Done......')
    print('Total time {} min'.format(round((end - start)/60, 2)))

    return prod_p_with_noise


# Vars to set up...
PYPSA_CASE = './L2RPN_2020_ieee_118_pypsa_simplified'
REF_DATA_DIR = './reference_data'
DESTINATION_PATH = './OPF_rel/'
POSSIBLE_MODE_OPF = ['day', 'week', 'month']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--grid_path', default=os.path.abspath(PYPSA_CASE), type=str,
                        help='PyPSA grid dir')
    parser.add_argument('--conso_full_path', default=os.path.join(os.path.abspath(REF_DATA_DIR), 'load_2007.csv.bz2'), type=str,
                       help='Specify the consumption or demanda full name')
    parser.add_argument('--gen_const', type=str,
                        help='Specify full path of gen max and min constraints')
    parser.add_argument('--dest_path', default=os.path.abspath(DESTINATION_PATH), type=str,
                         help='Specify the destination dir')
    parser.add_argument('--mode_opf', default='day', type=str,
                        help='Optimization mode, for the opf (possible values are {})'.format(POSSIBLE_MODE_OPF))
    parser.add_argument('--rescaled_min', default=5, type=int,
                        help='Run the optimizer every "rescaled_min" minutes (default 5)'),

    args = parser.parse_args()
    if not args.mode_opf.lower() in POSSIBLE_MODE_OPF:
        raise RuntimeError("Please provide an argument \"mode_opf\" among {}".format(POSSIBLE_MODE_OPF))
    rescaled_min = args.rescaled_min
    try:
        rescaled_min = int(rescaled_min)
    except:
        RuntimeError("\"rescaled_min\" argument should be convertible to an integer.")

    if not rescaled_min % 5 == 0:
        raise RuntimeError("\"rescaled_min\" argument should be multiple of 5 (so 5, or 15 is ok, but not 17 nor 3)")

    # **  **  **  **  ** 
    # Load the PyPSA grid
    net = pypsa.Network(import_name=args.grid_path)

    # Load consumption data without index
    print ('Reading input data -> load...')
    demand = pd.read_csv(args.conso_full_path)
    demand.drop('datetime', axis=1, inplace=True)

    # Check if gen contraints are given
    if args.gen_const:
        gen_max_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_max_pu.csv.bz2')
        gen_min_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_min_pu.csv.bz2')
        gen_const.update({'p_max_pu': gen_max_constraints, 'p_min_pu': gen_min_constraints})

    # Define parameters
    params.update({'step_opf_min': args.rescaled_min, 'mode_opf': args.mode_opf.lower()})
    # Run Economic Dispatch
    prod_p_dispatch = main_run_disptach(net, demand, gen_constraints=gen_const, params=params)

    destination_path = os.path.abspath(args.dest_path)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Save as csv
    prod_p_dispatch.to_csv(os.path.join(destination_path, 'prod_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
    
    print('OPF Done......')