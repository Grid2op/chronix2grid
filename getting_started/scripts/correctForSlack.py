import grid2op
import os
import argparse
from lightsim2grid import LightSimBackend
from grid2op.Chronics import Multifolder, GridStateFromFileWithForecasts
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import time


#Scenario='Scenario_april_03'

def CorrectOneScenario(id_slack,slack_name,path_agent_results,chronicsFolder,ScenarioName):
    #multi Processing
    print('start correction scenario: '+ScenarioName)
    data_this_episode = EpisodeData.from_disk(path_agent_results, ScenarioName)

    #prods after runner
    prods_p = pd.DataFrame(np.array([obs.prod_p for obs in data_this_episode.observations]))
    prodSlack=prods_p[id_slack]

    #prods before runner in chronix
    
    prodChronix_file='prod_p.csv.bz2'
    prodForecastChronix_file='prod_p_forecasted.csv.bz2'
    OldProdsDf=pd.read_csv(os.path.join(chronicsFolder,ScenarioName,prodChronix_file),sep=';')
    OldProdsForecastDf=pd.read_csv(os.path.join(chronicsFolder,ScenarioName,prodForecastChronix_file),sep=';')
    #OldProdsDf.head()

    ##correction term
    newProdsDf=OldProdsDf
    newProdsForecastDf=OldProdsForecastDf
    CorrectionLosses=prodSlack-OldProdsDf[slack_name]
    print('maximum compensation for slack before correction in MW: '+str(CorrectionLosses.abs().max()))
    print('average compensation for slack before correction in MW: '+str(CorrectionLosses.mean()))
    print('median compensation for slack before correction in MW: '+str(CorrectionLosses.abs().median()))
    print('min compensation for slack before correction in MW: '+str(CorrectionLosses.min()))

    #apply correction
    newProdsDf[slack_name]=OldProdsDf[slack_name]+CorrectionLosses
    newProdsForecastDf[slack_name]=OldProdsForecastDf[slack_name]+CorrectionLosses

    newProdsDf = newProdsDf.round(decimals=1)
    newProdsDf.to_csv(os.path.join(chronicsFolder,ScenarioName,prodChronix_file),sep=';',index=False)

    newProdsForecastDf = newProdsForecastDf.round(decimals=1)
    newProdsForecastDf.to_csv(os.path.join(chronicsFolder,ScenarioName,prodForecastChronix_file),sep=';',index=False)
    print('end correction scenario: '+ ScenarioName)
    return

###############
## multi processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a do nothing on generated chronics')
    parser.add_argument('-g', '--grid_path', default='', type=str, help='subnet .json file')
    #parser.add_argument('-o', '--ouput_dir', default='', type=str, help='Output dir')
    parser.add_argument('-a', '--agent_results_path', default='', type=str, help='agent result path after runner') 
    parser.add_argument('-nc', '--nb_cores', default=1, type=int, help='Numbers of cores for multiprocessing')
    parser.add_argument('-i', '--id_slack', default=37, type=int, help='indice of slack production in grid case')
    parser.add_argument('-s', '--slack_name', default='gen_68_37', type=str, help='name of slack production in grid case')

    #parser.add_argument('-ne', '--nb_episode', default=1, type=int, help='Numbers of episodes to run on')

    # Parse argumetns CML
    # ++  ++  ++  ++  ++
    args = parser.parse_args()
    grid_path = args.grid_path
    id_slack=args.id_slack
    slack_name=args.slack_name
    agent_results_path = args.agent_results_path
    nb_core = args.nb_cores
    chronicsFolder=os.path.join(grid_path,'chronics')
    #nb_episode=args.nb_episode
    
    ##################
    starttime = time.time()
    pool = multiprocessing.Pool(nb_core)
    
    multiprocessing_func = partial(CorrectOneScenario,id_slack,slack_name, agent_results_path,chronicsFolder)
    
    directoriesList=next(os.walk(chronicsFolder))[1]
    
    pool.map(multiprocessing_func, directoriesList)
    pool.close()
    print('multiprocessing done')  
    print('Time taken = {} seconds'.format(time.time() - starttime))

