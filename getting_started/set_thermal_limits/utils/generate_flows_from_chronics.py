import argparse
import os
import numpy as np
import pandas as pd
from lightsim2grid import TimeSerie
from tqdm import tqdm
import grid2op
from grid2op.Parameters import Parameters
from grid2op.Chronics import Multifolder, GridStateFromFileWithForecasts, GridStateFromFile

import warnings
from lightsim2grid.lightSimBackend import LightSimBackend

from collections import defaultdict
import json
import fastparquet as fp
import logging

from merge_df import merge_parquet_files
from OneChangeAgent import load_actions

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_folder",
    type=str,
    help="The directory in which the grid2op env are saved",
)

parser.add_argument(
    "--env_name",
    type=str,
    default="l2rpn_idf_2023_v5",
    help="The env to consider",
)

parser.add_argument(
    "--n_scenarios_to_look_at",
    type=int,
    default=None,
    help="The number of chronics to simulate",
)

parser.add_argument(
    "--get_all_scenario",
    type=bool,
    default=True,
    help="The number of chronics to simulate",
)

parser.add_argument(
    "--action_file_path",
    type=str,
    default=None,
    help="The topological actions to consider individually before flows calculation",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="The path where the generated dataframe is saved",
)

parser.add_argument(
    "--merge_output_files",
    type=bool,
    default=True,
    help="Whether to merge at the end the resulting files into a sole one",
)

def get_n_scenarios(path_chronix, n_scenarios_to_look_at=None, random_seed=None, get_all = False):

    dirs_scenario=sorted(set([el for el in os.listdir(path_chronix) if os.path.isdir(os.path.join(path_chronix, el))]))
    
    if get_all or n_scenarios_to_look_at is None:
        return dirs_scenario
    else:
    
        if random_seed is not None:
            np.random.seed(42)
        selected_scenarios=np.random.choice(dirs_scenario,n_scenarios_to_look_at)
    
        return selected_scenarios
    
def summarize_action(action):
    if len(action) != 0:
        id, positions = action['set_bus']['substations_id']
        return "sub_"+str(id)+"_"+("").join([str(k) for k in positions])
    else:
        return "do_nothing"
    
def get_flows_from_scenario_gridvalue(env, scenario_id, random_seed, path_chronix, return_logs=False):
    LINE_NAMES = env.name_line
    GEN_NAMES = ["_".join([a,b]) for a,b in zip(env.name_gen,env.gen_type)]
    GENTYPE_SET = set(env.gen_type)
    GEN_WIND_W = [k for k,item in enumerate(GEN_NAMES) if any([f"gen_{i}_" in item for i in [3,7,9,26,24]]) and "wind" in item]
    GEN_WIND_NE = [k for k,item in enumerate(GEN_NAMES) if any([f"gen_{i}_" in item for i in [39,48,65,61]])and "wind" in item]
    GEN_WIND_SE = [k for k,item in enumerate(GEN_NAMES) if any([f"gen_{i}_" in item for i in [99,106,84,88]])and "wind" in item]
    
    #compute flows
    time_series = TimeSerie(env)
    Vs = time_series.compute_V(scenario_id=scenario_id, seed=random_seed)
    As = time_series.compute_A()
    df_flows = pd.DataFrame(data=As, columns=LINE_NAMES).round().astype('float32')
    #retrieve prod
    prod_p, load_p, load_q =time_series.get_injections()
    df_gen = pd.DataFrame({gen_type : np.sum(prod_p[:, env.gen_type==gen_type], axis=1) for gen_type in GENTYPE_SET})
    df_gen["wind_w"] = np.sum(prod_p[:, GEN_WIND_W], axis=1)
    df_gen["wind_ne"] = np.sum(prod_p[:, GEN_WIND_NE], axis=1)
    df_gen["wind_se"] = np.sum(prod_p[:, GEN_WIND_SE], axis=1)
    df_gen["load"] = np.sum(load_p, axis=1)
    df_gen = df_gen.round().astype('float32')
    #retrieve calendar info
    for_start_date = GridStateFromFile(os.path.join(path_chronix,scenario_id))
    for_start_date._init_date_time()
    datetimes = pd.to_datetime([for_start_date.start_datetime + i * for_start_date.time_interval for i in range(As.shape[0])])
    df_datetime=pd.DataFrame(dict(
        datetimes=datetimes,
        hour_of_day = datetimes.hour,
        month = datetimes.month,
        year = datetimes.year,
        day_of_year = datetimes.dayofyear,
        day_of_week = datetimes.dayofweek,
        scenario = [scenario_id] * As.shape[0]
    ))
    
    if not return_logs:
        return pd.concat([df_flows, df_gen, df_datetime], axis=1)
    else:
        return pd.concat([df_flows, df_gen, df_datetime], axis=1),(df_flows.shape[0], df_gen.shape[0], df_datetime.shape[0])

def get_flows_from_chronix(args):
    '''Cette fonction prend en entrée les arguments suivants : 
    - arg.data_folder : The directory in which the grid2op env are saved (with chronix)
    - env_name : the environment to consider (e.g. "l2rpn_idf_2023_v5")
    - n_scenarios_to_look_at : int, representing the number of chronics to simulate
    - get_all_scenario : boolean, default True. If True, all generated scenarios are considered. If False, n_scenarios_to_look_at is mandatory
    - action_file_path : default="do_nothing agent", The topological actions to consider individually by best_agent before flows calculation
    - output_dir : path where the generated dataframe is saved
    - merge_output_files : type=bool, default=True, Whether to merge at the end the resulting files into a sole one.

    return Nothing, but generate a dataframe called args.env_name+"_allscenarios.parq"
'''
    #init variables and folders
    output_folder=os.path.join(args.output_dir, args.env_name) # where to save generated dataframes
    random_seed=42
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    path_env = os.path.join(args.data_folder, args.env_name)#f"../generation_donnees/{env_name}"
    path_chronix = os.path.join(path_env,"chronics")#f"../generation_donnees/{env_name}/chronics/"
    
    selected_scenario = get_n_scenarios(path_chronix=path_chronix,
                                       random_seed=random_seed,
                                       n_scenarios_to_look_at=args.n_scenarios_to_look_at)
    
    # Create the grid2op environment
    params = Parameters()
    params.NO_OVERFLOW_DISCONNECTION = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        env_ref = grid2op.make(args.env_name,
                    chronics_path=path_chronix,
                    param=params,
                    backend=LightSimBackend(),
                    chronics_class=Multifolder,
                    data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecasts}
                    )
        
    #load actions
    if args.action_file_path is not None:
        actions=load_actions(args.action_file_path)
        actions.insert(0,{})#add do nothing
    else:
        actions = [{}]
    
    #loop over chronics and changes on topology to generate flows possibilities
    
    logname = os.path.join(output_folder, "logs_"+args.env_name+".log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename = logname,
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    
    start_flag = True
    total_rows = 0
    try:
        for action in tqdm(actions):
            logging.info("start simulation of action %s", summarize_action(action))
            filename = os.path.join(output_folder, args.env_name+"_"+summarize_action(action)+".parq")
            for scenario_id in selected_scenario:
                try:
                    #information extraction
                    env_ref.reset()
                    _ = env_ref.step(env_ref.action_space(action))
                    new_df, logs = get_flows_from_scenario_gridvalue(
                        env=env_ref, scenario_id=scenario_id, 
                        path_chronix=path_chronix, random_seed=random_seed, return_logs=True
                        )
                    #concatenation
                    if start_flag:
                        df = new_df.copy(deep=True)
                        start_flag = False
                    else:
                        df = pd.concat([df, new_df], axis=0, ignore_index=True).reset_index(drop=True)
                        
                    #logs
                    logging.info('With action "%s" and scenario "%s", from %s inputs the final dataframe is created of size %s',
                                summarize_action(action), scenario_id, logs[0], df.shape[0])
                    
                except Exception as e:
                    logging.warning('"%s" not retrieved: "%s"',(scenario_id,summarize_action(action)),str(e))
                    pass
            #appending to parquet file
            append = os.path.exists(filename)
            df["action_before_start"] = [summarize_action(action)]*df.shape[0]
            total_rows += df.shape[0]
            logging.info("Add df to parquet file")
            df.to_parquet(filename, compression='brotli')
            #fp.write(filename, df, append=append, compression='snappy')
            logging.info("all scenarios with grid position after %s done and saved", summarize_action(action))
            start_flag = True
    
                
    except Exception as e:
        logging.error(str(e))
        
    if args.merge_output_files:
        merge_parquet_files(output_folder, filename= args.env_name+"_allscenarios.parq",logger=logging)
    
    return


if __name__ == "__main__":
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    get_flows_from_chronix(args)
            
        
    

