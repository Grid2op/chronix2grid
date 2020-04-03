import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import pypsa
from datetime import datetime, timedelta

from utils import update_gen_constrains, update_params
from utils import preprocess_net
from utils import run_opf
from utils import get_grouped_snapshots
from utils import add_noise_gen
from utils import reformat_gen_constraints
from utils import preprocess_input_data
from utils import interpolate_dispatch

# params={'snapshots': [],
#         'step_opf_min': 5,
#         'mode_opf': 'day',
#         'reactive_comp': 1.025,
#         }

def main_run_disptach(pypsa_net, 
                      load,
                      gen_constraints={'p_max_pu': None, 'p_min_pu': None},
                      params={}):

    # Update gen constrains dict with 
    # values passed by the users and params
    gen_constraints = update_gen_constrains(gen_constraints)  
    params = update_params(load.shape[0], params)

    print ('Preprocessing input data..')
    # Preprocess input data:
    #   - Add date range as index 
    #   - Check whether gen constraints has same lenght as load
    load_, gen_constraints_ = preprocess_input_data(load, gen_constraints, params)
    tot_snap = load_.index

    print ('Adapting PyPSA grid with parameters..')
    # Preprocess net parameters:
    #   - Change ramps according to params step_opf_min (assuming original 
    #     values are normalizing for every 5 minutes)
    #   - It checks for all gen units if commitable variables is False
    #     (commitable as False helps to create a LP problem for PyPSA)
    pypsa_net = preprocess_net(pypsa_net, params['step_opf_min'])

    months = tot_snap.month.unique()
    start = time.time()
    results = []
    for month in months:
        # Get snapshots per month
        snap_per_month = tot_snap[tot_snap.month == month]
        # Filter input data per month
        load_per_month = load_.loc[snap_per_month]
        # Get gen constraints separated and filter by month
        g_max_pu, g_min_pu = gen_constraints_['p_max_pu'], gen_constraints_['p_min_pu']
        g_max_pu_per_month = g_max_pu.loc[snap_per_month]
        g_min_pu_per_month = g_min_pu.loc[snap_per_month]
        # Get grouped snapsshots given monthly snapshots
        snap_per_mode = get_grouped_snapshots(snap_per_month, params['mode_opf'])
        for snaps in snap_per_mode:
            # Truncate input data per mode (day, week, month)
            load_per_mode = load_per_month.loc[snaps]
            gen_max_pu_per_mode = g_max_pu_per_month.loc[snaps]
            gen_min_pu_per_mode = g_min_pu_per_month.loc[snaps]
            # Run opf given in specified mode
            results.append(run_opf(pypsa_net, 
                                   load_per_mode, 
                                   gen_max_pu_per_mode, 
                                   gen_min_pu_per_mode,
                                   params,))

    # Unpack individual dispatchs
    opf_prod = pd.DataFrame()
    for df in results:
        opf_prod = pd.concat([opf_prod, df], axis=0)

    # Sort by datetime
    opf_prod.sort_index(inplace=True)
    # Create complete prod_p dataframe and interpolate missing rows
    prod_p = opf_prod.copy()
    # Apply interpolation in case of step_opf_min greater than 5 min
    if params['step_opf_min'] > 5:
        print ('\n => Interpolating dispatch to have 5 minutes resolution..')
        prod_p = interpolate_dispatch(prod_p)
    # Add noise to results
    gen_cap = pypsa_net.generators.p_nom
    prod_p_with_noise = add_noise_gen(prod_p, gen_cap, noise_factor=0.001)

    end = time.time()
    print('Total time {} min'.format(round((end - start)/60, 2)))
    print('OPF Done......')

    return prod_p_with_noise

# In case to launch by the terminal
# ++  ++  ++  ++  ++  ++  ++  ++  +
# Vars to set up...
# PYPSA_CASE = './L2RPN_2020_ieee_118_pypsa_simplified'
REF_DATA_DIR = './reference_data'
DESTINATION_PATH = './OPF_rel/'
POSSIBLE_MODE_OPF = ['day', 'week', 'month']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--grid_path', default='', type=str,
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

    if not  60 % rescaled_min == 0:
        raise RuntimeError("\"rescaled_min\" argument should be multiple of 5 (so 5, or 15 is ok, but not 17 nor 3)")

    # **  **  **  **  ** 
    # Load the PyPSA grid
    net = pypsa.Network(import_name=args.grid_path)

    # Load consumption data without index
    print ('Reading input data -> load...')
    demand = pd.read_csv(args.conso_full_path)
#     demand.drop('datetime', axis=1, inplace=True)

    # Check if gen contraints are given
    gen_const = {'p_max_pu': None, 'p_min_pu': None}
    if args.gen_const:
        gen_max_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_max_pu.csv.bz2')
        gen_min_constraints = pd.read_csv(os.path.join(args.gen_const), 'gen_min_pu.csv.bz2')
        gen_const.update({'p_max_pu': gen_max_constraints, 'p_min_pu': gen_min_constraints})

    # Define parameters
    params = {'step_opf_min': args.rescaled_min, 
               'mode_opf': args.mode_opf.lower(),
             }
    # Run Economic Dispatch
    prod_p_dispatch = main_run_disptach(net, 
                                        demand, 
                                        gen_constraints=gen_const, 
                                        params=params)

    destination_path = os.path.abspath(args.dest_path)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Save as csv
    prod_p_dispatch.to_csv(os.path.join(destination_path, 'prod_p.csv.bz2'), sep=';', float_format='%.2f', index=False)
    
    print('OPF Done......')