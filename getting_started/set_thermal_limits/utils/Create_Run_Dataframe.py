import pandas as pd
from itertools import chain
import numpy as np
import multiprocessing
from functools import partial
import time
import pyarrow as pa
import pyarrow.parquet as pq
import os


def create_run_df(env,env_name,res_run,agent_name,available_obs=["a_or","load_p","gen_p"]):
    """
        A function to create a full pandas dataframe based on observation available in runner results over several scenarios
        
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The used environment.
            
        env_name: ``str``
            name of environment

        res_run:  ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.BaseAgent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics
              - "obs_to_keep" : dictionnary of observations that were saved for interest. Tied to available_ob

        agent_name: ``str``
            name of run agent
        
        available_obs: list ``str``
            list of avilable observations in res_run / obs_to_keep

        Returns
        -------
        run_df: :class:`pandas.DataFrame`
            a full pandas dataframe with obs from res_run, with aggregated productions (per type) and loads

    """
    
    #run_df=pd.DataFrame()
    n_scenarios_to_look_at=len(res_run)
    max_ts=res_run[0][4]+1
    max_ts_list=np.array([len(re[5]['hour_of_day']) for re in res_run])
    
    n_rows_df=np.sum(max_ts_list)#n_scenarios_to_look_at*max_ts if all scenarios converged
    
    run_df=pd.DataFrame()
    if("a_or" in available_obs):
        run_df = pd.DataFrame(np.array([re[5]["a_or"] for re in res_run]).reshape((n_rows_df,env.n_line))
                              ,columns=env.name_line).astype('float32')

    if("load_p" in available_obs):
        run_df["load"]=np.round(np.array([re[5]["load_p"] for re in res_run]).reshape((n_rows_df,env.n_load))
                                .sum(axis=1)).astype('float32')
    
    if("gen_p" in available_obs):
        res_gen_p=np.array([re[5]["gen_p"] for re in res_run]).reshape((n_rows_df,env.n_gen))
        run_df["nuclear"]=np.round(res_gen_p[:,(env.gen_type=="nuclear")].sum(axis=1)).astype('float32')
        run_df["solar"]=np.round(res_gen_p[:,(env.gen_type=="solar")].sum(axis=1)).astype('float32')
        run_df["wind"]=np.round(res_gen_p[:,(env.gen_type=="wind")].sum(axis=1)).astype('float32')
        run_df["thermal"]=np.round(res_gen_p[:,(env.gen_type=="thermal")].sum(axis=1)).astype('float32')
        run_df["hydro"]=np.round(res_gen_p[:,(env.gen_type=="hydro")].sum(axis=1)).astype('float32')


    if("day_of_week" in available_obs):
        run_df["day_of_week"]=list(chain(*[re[5]["day_of_week"] for re in res_run]))
        
    if("hour_of_day" in available_obs):
        run_df["hour_of_day"]=list(chain(*[re[5]["hour_of_day"] for re in res_run]))
    if("month" in available_obs):
        run_df["month"]=list(chain(*[re[5]["month"] for re in res_run]))
        
    run_df["day_of_year"]=list(chain(*[re[5]["day_of_year"] for re in res_run]))
    run_df["scenario"]=list(chain(*[[re[1]]*max_ts_list[i] for i,re in enumerate(res_run)]))
    run_df["agent"]=agent_name
    run_df["datetimes"]=list(chain(*[re[5]["datetimes"] for re in res_run]))
    run_df["env_name"]=env_name
    
    return run_df


def save_light_df_file(env_output_folder):
    """
        A function to save a list of dataframe parquet files into one file
        
        Parameters
        ----------
        env_output_folder: `str`
            folder where to save the file
     """
    
    file_path=os.path.join(os.path.abspath(env_output_folder),"..",os.path.basename(env_output_folder)+'.file')
    if(os.path.isfile(file_path)):
        print("this file already exist: "+file_path)
    
    else:
        dataset = pq.ParquetDataset(env_output_folder, use_legacy_dataset=False)
        df=dataset.read().to_pandas()
        df_light=pd.DataFrame()
        cols=df.columns[(df.dtypes==np.float32)]
        other_cols=df.columns[(df.dtypes!=np.float32)]
    
        for col in cols:
            #print(col)
            #df[col]=
            #df.loc[col].values=df[col].astype(np.float16, copy=False)#.apply(lambda x: x.astype('float16'))
            #df[col]=np.float16(df[col].values)
            df_light[col]=np.float16(df[col].values)
        df_light[other_cols]=df[other_cols]
    
        print("saving file in: "+file_path)
        df_light.to_feather(file_path)

def get_size(start_path = '.'):
    """
        A function to get the total size of a folder
        
        Parameters
        ----------
        start_path: `str`
            the folder path of interest
        
        Returns
        -------
        total_size: ``float``
            the total size of the folder
     """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
        
def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))

def fast_pd_concat(list_df):
    """
        A function to concatenate quickly pd dataframes with same columns. 
        pd.concat can be very slow for dataframes with lots of columns
        
        Parameters
        ----------
        list_df: ``list`` of pd.DataFaame
            list of pandas dataframew to concatenate row-wise
            

        Returns
        -------
        df: :class:`pandas.DataFrame`
            the concatenated pandas dataframe

    """
    
    COLUMN_NAMES = list_df[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])

    for col in COLUMN_NAMES:
        extracted = (frame[col] for frame in list_df)

        # Flatten and save to df_dict
        df_dict[col] = fast_flatten(extracted)

    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    return df



                                                                  