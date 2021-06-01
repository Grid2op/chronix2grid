import grid2op
from grid2op.Parameters import Parameters
from grid2op.Agent import RecoPowerlineAgent
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
import pandas as pd
import numpy as np
import shutil
import os

try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    backend = LightSimBackend()
except:
    from grid2op.Backend import PandaPowerBackend
    backend = PandaPowerBackend()
    print("You might need to install the LightSimBackend (provisory name) to gain massive speed up")

#we will check that losses were properly adjusted by rerunning grid2op one more time on the chronic 
#and checking that the slack production is not modified much more compare to original one

def check_loss_adjustement(existing_output_case_folder,scenario,scen_id,max_iter,Ouput_folder,grid_case,params_json):


    print('the case folder is: '+existing_output_case_folder)


    # don't disconnect powerline on overflow, the thermal limit are not set for now, it would not make sens
    param = Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
    env=grid2op.make(existing_output_case_folder,param=param)#,backend=backend)

    runner = Runner(**env.get_params_for_runner(),agentClass=RecoPowerlineAgent)
    
    
    #run env
    Ouput_folder_runner_case=os.path.join(Ouput_folder,'runner',grid_case)
    if(os.path.exists(Ouput_folder_runner_case)):
        print('delete folder before creating:'+Ouput_folder_runner_case)
        shutil.rmtree(Ouput_folder_runner_case)
    os.makedirs(Ouput_folder_runner_case, exist_ok=True)
    name_chron, cum_reward, nb_time_step, episode_data =runner.run_one_episode(path_save=Ouput_folder_runner_case,
                         indx=scen_id,
                         max_iter=max_iter,
                         pbar=True)
    
    
    data_this_episode = EpisodeData.from_disk(Ouput_folder_runner_case,scenario)
    
    #check if there are no maintenance because for now we consider chronics without maintenance and attacks
    if not (data_this_episode.observations[0].time_next_maintenance.max()==-1):
        print('WARNING there is a maintenance happeing when running the chronics')
    
    #sclack prod from runner results
    prods_p = pd.DataFrame(np.array([obs.prod_p for obs in data_this_episode.observations]))
    #print(prods_p.iloc[1])
    prodSlack=prods_p[params_json['idxSlack']]
    
    #prod from orginal prod_p.csv file
    prod_p_file_path=os.path.join(existing_output_case_folder,'chronics',scenario,'prod_p.csv.bz2')
    prod_p_old=pd.read_csv(prod_p_file_path,sep=';')
    #print(prod_p_old.iloc[1])
    prod_p_old.columns=prods_p.columns
    prodSlack_old=prod_p_old[params_json['idxSlack']].iloc[0:1001]#.reset_index(drop=True)
    
    delta_prod_slack=(prodSlack-prodSlack_old)[1:]#drop first index which is not really meaningful
    stats=delta_prod_slack.abs().describe()
    print('losses adjustement delta after rerun')
    print(stats)
    
    if(stats['max']>1):
        print('Warning some losses adjustements were off from at least 1MW')
    else:
        print('all losses adjustements are good with precision of less than 1MW')

