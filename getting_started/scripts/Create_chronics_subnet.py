import os
import re
import time
import argparse
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing
import traceback
import logging
import shutil
from shutil import copyfile

import pandapower as pp
from grid2op.Episode import EpisodeData


SUBNET_PATH = 'grid_design/subgrid/grid.json'

match_names = {2: {'from_or': [('load_32_30', '64_67_183'),
                              ('load_35_34', '68_76_12'),
                              ('load_35_33', '68_74_9'),
                              ('load_35_35', '68_69_171')],
                  'from_ex': [('load_0_1', '14_32_108'),
                              ('load_1_2', '18_33_109'),
                              ('load_5_6', '29_37_117'),
                              ('load_35_36', '67_68_184')]},
              1: {'from_or': [('load_14_12', '14_32_108'),
                              ('load_18_17', '18_33_109'),
                              ('load_22_22', '22_23_94'),
                              ('load_28_26', '29_37_117')],
                  'from_ex': []},
              3: {'from_or': [],
                  'from_ex': [('load_0_1', '22_23_94'),
                             ('load_1_2', '64_67_183'),
                             ('load_2_3', '48_68_169'),
                             ('load_2_4', '48_68_170')]},
}

# Dictionary to map error in lines between chronics and env
fix_line_names = {'7_4_173': '4_7_173',
                  '25_24_174': '24_25_174',
                  '29_16_178': '16_29_178',
                  '63_60_181': '60_63_181',
                  '37_36_179': '36_37_179',
                  '62_58_180': '58_62_180'
                 }

# Define possible basenames for chonixs
gen_basenames = ['prod_p.csv.bz2', 
                 'prod_p_forecasted.csv.bz2',
                 'prod_v.csv.bz2',
                 'prod_v_forecasted.csv.bz2'
                 ]

load_basenames = ['load_p.csv.bz2',
                  'load_p_forecasted.csv.bz2',
                  'load_q.csv.bz2',
                  'load_q_forecasted.csv.bz2'
                  ]

line_basenames = ['maintenance.csv.bz2',
                  'maintenance_forecasted.csv.bz2',
                  'hazards.csv.bz2',
                  ]
                  
def generate_subnet_chronics(AGENT_RESULT_PATH,FULL_DISPATCH_DIR,SUBNET_OUTPUT_DIR,DECIMALS,match_names_region,subnet_original_gen_names,subnet_original_load_names,subnet_original_line_names,gen_map_names,load_map_names,env_name_lines,env_name_loads,env_name_gens,scenario):
    print(f'Generating chronic for subnet => {scenario}')
    # Flag to track errors
    flag_error = False
    # Read agent do nothing results
    if(os.path.exists(os.path.join(AGENT_RESULT_PATH, scenario))):
        
        data_episode = EpisodeData.from_disk(AGENT_RESULT_PATH, scenario)

        # Get interconnection names
        interco_or_names = [l for _, l in match_names_region['from_or']]
        interco_ex_names = [l for _, l in match_names_region['from_ex']]
        interco_or_idx = [int(l.split('_')[-1]) for l in interco_or_names]
        interco_ex_idx = [int(l.split('_')[-1]) for l in interco_ex_names]

        # Read flow in lines from agent
        l_p_or = pd.DataFrame(np.array([obs.p_or[interco_or_idx] for obs in data_episode.observations]), 
                                        columns=interco_or_names)
        l_q_or = pd.DataFrame(np.array([obs.q_or[interco_or_idx] for obs in data_episode.observations]), 
                                        columns=interco_or_names)
        l_p_ex = pd.DataFrame(np.array([obs.p_ex[interco_ex_idx] for obs in data_episode.observations]), 
                                        columns=interco_ex_names)
        l_q_ex = pd.DataFrame(np.array([obs.q_ex[interco_ex_idx] for obs in data_episode.observations]), 
                                        columns=interco_ex_names)

        # Dictionary to map full path names
        path_files = {'gen': [os.path.join(FULL_DISPATCH_DIR, scenario, basename) for basename in gen_basenames],
                      'load': [os.path.join(FULL_DISPATCH_DIR, scenario, basename) for basename in load_basenames],
                      'line': [os.path.join(FULL_DISPATCH_DIR, scenario, basename) for basename in line_basenames],
                      }
        # Create output folder if it doesn't exists
        if not os.path.exists(os.path.join(SUBNET_OUTPUT_DIR, scenario)):
            os.makedirs(os.path.join(SUBNET_OUTPUT_DIR, scenario))
        '''
        Process gen csvs
        '''
        for gen_path_file in path_files['gen']:
            # print (f'{scenario} - {os.path.split(gen_path_file)[-1]}')
            if os.path.exists(gen_path_file):
                try:
                    # Read full csv from chronix2grid generation using full grid
                    gen_full_df = pd.read_csv(gen_path_file, sep=';')
                    # Select subnet gen elements
                    gen_subnet_df = gen_full_df[subnet_original_gen_names].rename(columns=gen_map_names)
                    # Sort cols chronic elemement to match same subnet order
                    gen_subnet_df = gen_subnet_df[env_name_gens]
                    # Write output
                    gen_subnet_df.to_csv(os.path.join(SUBNET_OUTPUT_DIR, scenario, os.path.split(gen_path_file)[-1]), sep=';', index=False)
                except Exception as e:
                    flag_error = True
                    logging.error(f'ERROR IN: {scenario} - file {os.path.split(gen_path_file)[-1] } \n\n {traceback.format_exc()}')
            else:
                continue
        '''
        Process line csvs
        '''
        for line_path_file in path_files['line']:
            if os.path.exists(line_path_file):
                try:
                    # Read full csv from chronix2grid generation using full grid
                    line_full_df = pd.read_csv(line_path_file, sep=';')
                    # Select cols subnet line elements adn rename according to subnet env
                    line_subnet_df = line_full_df[subnet_original_line_names]
                    line_subnet_df.columns = env_name_lines
                    # Write output
                    line_subnet_df.to_csv(os.path.join(SUBNET_OUTPUT_DIR, scenario, os.path.split(line_path_file)[-1]), sep=';', index=False)
                except Exception as e:
                    flag_error = True
                    logging.error(f'ERROR IN: {scenario} - file {os.path.split(line_path_file)[-1] } \n\n {traceback.format_exc()}')
            else:
                continue
        '''
        Process load csvs
        '''
        for load_path_file in path_files['load']:
            # print (f'{scenario} - {os.path.split(load_path_file)[-1]}')
            if os.path.exists(load_path_file):
                try:
                    # Read full csv from chronix2grid generation using full grid
                    load_full_df = pd.read_csv(load_path_file, sep=';')
                    # Select subnet line elements
                    load_subnet_df = load_full_df[subnet_original_load_names].rename(columns=load_map_names)
                    # Get csv file name for add interconnection lines
                    csv_name = os.path.split(load_path_file)[-1].split('.')[0]
                    for ld, ln in match_names_region['from_or']:
                        if csv_name == 'load_p':
                            load_subnet_df[ld] = l_p_or[ln].round(DECIMALS).values
                        if csv_name == 'load_q':
                            load_subnet_df[ld] = l_q_or[ln].round(DECIMALS).values
                        if csv_name == 'load_p_forecasted':
                            load_subnet_df[ld] = np.roll(l_p_or[ln].round(DECIMALS).values, -1, axis=0)
                        if csv_name == 'load_q_forecasted':
                            load_subnet_df[ld] = np.roll(l_q_or[ln].round(DECIMALS).values, -1, axis=0)
                    for ld, ln in match_names_region['from_ex']:
                        if csv_name == 'load_p':
                            load_subnet_df[ld] = l_p_ex[ln].round(DECIMALS).values
                        if csv_name == 'load_q':
                            load_subnet_df[ld] = l_q_ex[ln].round(DECIMALS).values
                        if csv_name == 'load_p_forecasted':
                            load_subnet_df[ld] = np.roll(l_p_ex[ln].round(DECIMALS).values, -1, axis=0)
                        if csv_name == 'load_q_forecasted':
                            load_subnet_df[ld] = np.roll(l_q_ex[ln].round(DECIMALS).values, -1, axis=0)
                    # Sort cols chronic elemement to match same subnet order
                    load_subnet_df = load_subnet_df[env_name_loads]
                    # Write output
                    load_subnet_df.to_csv(os.path.join(SUBNET_OUTPUT_DIR, scenario, os.path.split(load_path_file)[-1]), sep=';', index=False)
                except Exception as e:
                    flag_error = True
                    logging.error(f'ERROR IN: {scenario} - file {os.path.split(load_path_file)[-1] } \n\n {traceback.format_exc()}')

        #copy start_datetime.info
        start_dateTime_file=os.path.join(FULL_DISPATCH_DIR, scenario,'start_datetime.info')
        if (os.path.exists(start_dateTime_file)):
            copyfile(start_dateTime_file,os.path.join(SUBNET_OUTPUT_DIR, scenario,'start_datetime.info'))

        #copy time_interval.info
        time_interval_file=os.path.join(FULL_DISPATCH_DIR, scenario,'time_interval.info')
        if (os.path.exists(time_interval_file)):
            copyfile(time_interval_file,os.path.join(SUBNET_OUTPUT_DIR, scenario,'time_interval.info'))

        if flag_error:
            logging.warning(f'===> {scenario} not saved')
            # Remove scenario in case of errors
            shutil.rmtree(os.path.join(SUBNET_OUTPUT_DIR, scenario))  
        return None
    
    return None

# Load the subnet env
def get_Name_Mappings(SUBNET_PATH):
    grid_path=os.path.join(SUBNET_PATH,'grid.json')

    env_subnet = pp.from_json(grid_path)

    # Get gen and load names and lines
    subnet_original_gen_names = env_subnet.gen.old_name.tolist()
    subnet_original_load_names = list(filter(lambda x: re.match('load_', x), env_subnet.load.old_name))
    subnet_original_line_names = env_subnet.line.old_name.tolist() + env_subnet.trafo.old_name.tolist() 
    # subnet_line_names = env.name_line
    # Fix line names
    for k, v in fix_line_names.items():
        if k in subnet_original_line_names:
            subnet_original_line_names[subnet_original_line_names.index(k)] = v

    # Get map to match old name with new one
    gen_map_names = {k:v for k, v in zip(subnet_original_gen_names, env_subnet.gen.name)}
    load_map_names = {k:v for k, v in env_subnet.load.loc[~env_subnet.load.old_name.str.startswith('interco'), ['old_name', 'name']].values}
    env_name_lines=env_subnet.line.name#env.name_line
    env_name_loads=env_subnet.load.name#env.name_load
    env_name_gens=env_subnet.gen.name#env.name_gen
    
    return(subnet_original_gen_names,subnet_original_load_names,subnet_original_line_names,gen_map_names,load_map_names,env_name_lines,env_name_loads,env_name_gens)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create chronics for subnet given full IEEE 118 disptach')
    parser.add_argument('-n', '--subnet_path', default=SUBNET_PATH, type=str, help='subnet .json file')
    parser.add_argument('-d', '--disptach_dir', default='', type=str, help='Full IEEE 118 dispatch dir')
    parser.add_argument('-a', '--agent_dir', default='', type=str, help='Agent result dir will all scenarios')
    parser.add_argument('-o', '--ouput_dir', default='', type=str, help='Output dir')
    parser.add_argument('-r', '--region', default=2, type=int, help='Subnet region {1, 2 or 3}')
    parser.add_argument('-c', '--cores', default=1, type=int, help='Numbers of cores for multiprocessing')

    # Parse argumetns CML
    # ++  ++  ++  ++  ++
    args = parser.parse_args()
    REGION = args.region
    SUBNET_PATH = args.subnet_path
    FULL_DISPATCH_DIR = args.disptach_dir
    AGENT_RESULT_PATH = args.agent_dir
    SUBNET_OUTPUT_DIR = args.ouput_dir
    ncores = args.cores
    DECIMALS = 2
    # Configure log file
    logging.basicConfig(filename=os.path.join(SUBNET_OUTPUT_DIR, 'logFile.txt'))

    # Get interconnection names
    #interco_or_names = [l for _, l in match_names[REGION]['from_or']]
    #interco_ex_names = [l for _, l in match_names[REGION]['from_ex']]
    #interco_or_idx = [int(l.split('_')[-1]) for l in interco_or_names]
    #interco_ex_idx = [int(l.split('_')[-1]) for l in interco_ex_names]

    # Load the subnet env
    subnet_original_gen_names,subnet_original_load_names,subnet_original_line_names,gen_map_names,load_map_names,env_name_lines,env_name_loads,env_name_gens=get_Name_Mappings(SUBNET_PATH)


    t0 = time.time()
    # Start multiprocessing computation
    multiprocessing_func = partial(generate_subnet_chronics,AGENT_RESULT_PATH,FULL_DISPATCH_DIR,SUBNET_OUTPUT_DIR,DECIMALS,match_names[REGION],subnet_original_gen_names,subnet_original_load_names,subnet_original_line_names,gen_map_names,load_map_names,env_name_lines,env_name_loads,env_name_gens)
    
    scenarios = next(os.walk(FULL_DISPATCH_DIR))[1]
    if(ncores==1):
        for scenario in scenarios:
            generate_subnet_chronics(AGENT_RESULT_PATH,FULL_DISPATCH_DIR,SUBNET_OUTPUT_DIR,DECIMALS,match_names[REGION],subnet_original_gen_names,subnet_original_load_names,subnet_original_line_names,gen_map_names,load_map_names,env_name_lines,env_name_loads,env_name_gens,scenario)
    else:
        pool = multiprocessing.Pool(ncores)

        
        pool.map(multiprocessing_func, scenarios)
        pool.close()
    t1 = time.time()
    print ('Computation done!!!')
    delta_t = round(t1 - t0, 2) / 60
    print (f'Time: {delta_t} min with {ncores} cores')