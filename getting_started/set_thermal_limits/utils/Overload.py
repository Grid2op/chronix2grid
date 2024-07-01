import pandas as pd
from itertools import chain
import numpy as np
import multiprocessing
from functools import partial
import time
import pyarrow as pa
import pyarrow.parquet as pq
import os



def _overload_info_df(lines_name,thermal_limits,df_analysis,Overloads_Start_end,scenarioStartIndices,indiceOvLine,agent_name):
    """
        A function to create a dataframe with information about overloads (start time, end time, duration,max_rho of overload) for a given line and a given agent of interest
        
        Parameters
        ----------
        df_analysis: :class:`pandas.DataFrame`
            a full pandas dataframe with obs from res_run, with aggregated productions (per type) and loads, for each scenario and each agent
        
        lines_name: :class:`numpy.ndarray`, dtype:str
            name of lines in environment
        
        thermal_limits: :class:`numpy.ndarray`, dtype:float
            array of thermal limits for each powerline
        
        Overloads_Start_end: :class:`list`, dtype:int
            dataframe with power lines in columns with value 1.0 if an overload starts, value -1.0 if overload ends, and 0.0 otherwise        
            
        indiceOvLine: `int`
            indice of overloaded line of interest
        
        agent_name: `str`
            agent name of interest
            
        Returns
        -------
        overloads_info: :class:`pandas.DataFrame`
            the resulting dataframe with information about overloads

    """
    
    df_ov = pd.DataFrame()
    line_name = lines_name[indiceOvLine]
    colName = line_name#colFlows[indiceOvLine]
    thermal_limit = thermal_limits[indiceOvLine]
    
    
    #Overloads_df_agent=Overloads_df[Overloads_df.agent==agent_name]
    
    indexes=Overloads_Start_end.index[np.where(Overloads_Start_end[colName]==1)[0]]
    df_ov['scenarios'] =  df_analysis.scenario[indexes].values #Overloads_df_agent[Overloads_Start_end[colName]==1].scenario.values
    df_ov['agent']=  df_analysis.agent[indexes].values#Overloads_df_agent[Overloads_Start_end[colName]==1].agent.values
    #to have indices in the range 0 to the duration of scenario after in start_indices and end_indices

    start_indices_inDf = Overloads_Start_end[Overloads_Start_end[colName]==1].index
    end_indices_inDf = Overloads_Start_end[Overloads_Start_end[colName]==-1].index
    #print(start_indices_inDf)
    
    scenariosStartIndices_df = [scenarioStartIndices[s_name] for s_name in df_analysis["scenario"][start_indices_inDf]]
    

    if(len(start_indices_inDf)!=len(end_indices_inDf)):
        print("problem with start and end indices: they don t have same length")
        print(len(start_indices_inDf))
        print(len(end_indices_inDf))
        print(line_name)
        #print(start_indices_inDf)
        #print(end_indices_inDf)
        #print(scenariosStartIndices_df)
        raise
       
    df_ov['start_indices'] = (start_indices_inDf-scenariosStartIndices_df).astype('int16')  #scenarioStartIndices[df_ov.scenarios]
    df_ov['end_indices'] = (end_indices_inDf-scenariosStartIndices_df).astype('int16')  #scenarioStartIndices[df_ov.scenarios]
    df_ov['duration'] = df_ov['end_indices']-df_ov['start_indices']
    n_count = df_ov.shape[0]
    
    #.unique()

    #important to have indices from original df dataframe below
    maxDepth_indices_inDf = np.array([df_analysis[colName][start_indices_inDf[i]:end_indices_inDf[i]].idxmax() for i in range(n_count)])
    df_ov['maxDepth_indices'] = (maxDepth_indices_inDf- scenariosStartIndices_df).astype('int16')#
    df_ov['maxDepths'] = (df_analysis[colName][maxDepth_indices_inDf].values/thermal_limit).astype('float16')
    df_ov['maxDepths'] = df_ov['maxDepths'].round(2) 
    # sum_E_overflow
    df_ov['delta_E']=np.array([(df_analysis.loc[start_indices_inDf[i]:(end_indices_inDf[i]-1)][colName].sum()-df_ov.duration[i]*thermal_limit)/thermal_limit for i in range(df_ov.shape[0])]).round(2) 
    df_ov['meanDepths']=df_ov['delta_E']/df_ov['duration']+1
    
    # sum_E_rel_overflow
    
    df_ov['hour_of_day'] = df_analysis['hour_of_day'][maxDepth_indices_inDf].astype('int8').values
    df_ov['day_of_week'] = df_analysis['day_of_week'][maxDepth_indices_inDf].astype('int8').values
    df_ov['month'] = df_analysis['month'][maxDepth_indices_inDf].astype('int8').values
    df_ov['day_of_year'] = df_analysis['day_of_year'][maxDepth_indices_inDf].astype('int8').values
    df_ov['datetimes'] = df_analysis['datetimes'][maxDepth_indices_inDf].values

    df_ov['overload_line_name'] = line_name
    
    return df_ov

# Method to be called in jupyter
# you should only have one kind of 'action' name in df_analysis. you should subest that
def get_overload_info_df(df_analysis,lines_name,thermal_limits,indicesLineToOverload, verbose=True):#nb_core=1
    """
        A function to create a dataframe with information about overloads (start time, end time, duration,max_rho of overload) 
        
        Parameters
        ----------
        df_analysis: :class:`pandas.DataFrame`
            a full pandas dataframe with obs from res_run, with aggregated productions (per type) and loads, for each scenario and each agent
        
        lines_name: :class:`numpy.ndarray`, dtype:str
            name of lines in environment
        
        thermal_limits: :class:`numpy.ndarray`, dtype:float
            array of thermal limits for each powerline
        
        indicesLineToOverload: :class:`list`, dtype:int
            list of ids of lines that are of interest for overloads
            
        nb_core: `int`
            number of cores for multi-processing if faster
        
        verbose: `bool`
            True to display more logs
            
        Returns
        -------
        overloads_info: :class:`pandas.DataFrame`
            the resulting dataframe with information about overloads

    """
    if not ('agent' in df_analysis.columns):
        df_analysis['agent']='do_nothing'
        
    agentNames=df_analysis['agent'].unique()
    #dataframe with flows only
    #Flows_df = df_analysis[lines_name].astype('float16')
    
    #dataframe of overloads
    Overloads=pd.DataFrame()
    for i,l in enumerate(lines_name):
        Overloads[l]=(df_analysis[l]>= thermal_limits[i]).astype('bool')
    #Overloads=(df_analysis[lines_name] >= thermal_limits).astype('bool')#.astype('int8')#cannot be a boolean because need to do a diff after to know when an overload start and end .... .astype(bool)
    print("overload created")
    
    #we will consider that at the beginning and at the end of the scenarios, there are no overloads: they should happen within a scenario. We will hence set the beginning and end of scenarios to not have overloads. This will not hinder the analysis and will help better decipher overloads within a scenario and not in between.
    #Otherwise, this causes problems when computing Overloads.diff(): we don't neceassrily have within a scenario the same number of startIndices and EndIndices for overloads, which later create bugs when creating our overload_info dataframe 
    Overloads['scenario']=df_analysis['scenario']
    def first(df):
        return df.index[0]

    def last(df):
        return df.index[-1]
    overloads_info=pd.DataFrame()
    
    for agent in agentNames:
        print(agent)
        Overloads_agent=Overloads[df_analysis.agent==agent]
        
        #get scenario start index in big dataframe
        pivotDf=Overloads_agent['scenario'].drop_duplicates()
        scenarioStartIndices=dict(zip(list(pivotDf.values),pivotDf.index))
        
        firstLastIndices=pd.concat([Overloads_agent.groupby('scenario').apply(first),Overloads_agent.groupby('scenario').apply(last)],axis=1)
        firstLastIndices.columns=['start','end']
        Overloads_agent=Overloads_agent.drop('scenario',axis=1)
        Overloads_agent.loc[firstLastIndices['start'].values]=False#0
        Overloads_agent.loc[firstLastIndices['end'].values]=False#0

        #dataframe to know when overloads start and ends: now it can be properly computed
        #Overloads_Start_end =Overloads.diff() # inneficient in terms of memory. Don't work well on boolean and dont work on int8 and int16
        Overloads_Start_end = (Overloads_agent.astype('int8')-Overloads_agent.astype('int8').shift(fill_value=0))#.fillna(0).astype('int8')#diff does not work for type int8 and int16...
        start_time = time.time()

        for indice in indicesLineToOverload:  
            #print(scenarioStartIndices)
            overloads_info=pd.concat([overloads_info,_overload_info_df(lines_name,thermal_limits,df_analysis,
                                                 Overloads_Start_end,scenarioStartIndices,indice,agent)])
                          #_overload_info_df(lines_name,thermal_limits,df_analysis,Overloads_df,Overloads_Start_end,scenarioStartIndices,indice)
    overloads_info = overloads_info.reset_index(drop=True)
    print('Dataframe generated!!')  
    if verbose:
        print('Time taken = {} seconds'.format(time.time() - start_time))
    return overloads_info       



#this function will summarize for a given scenario, the longest overloads for each overloaded line
#it will let us find similar scenarios for instance
def MakeSummaryOverload_Scenario(ScenarioName,agent,overloads_info,possiblyOverloadedLines):
    """
        A function to create a dataframe with information about overloads (start time, end time, duration,max_rho of overload) 
        
        Parameters
        ----------
            
        ScenarioName: `str`
            name of scenario of interest
        
        agent: `str`
            name of agent of interest
            
        overloads_info: :class:`pandas.DataFrame`
            the resulting dataframe with information about overloads
            
        possiblyOverloadedLines:class:`list`, dtype:str
            list of line names that are of interest for overloads
            
        Returns
        -------
        summary_overloads_scenario_df: :class:`pandas.DataFrame`
            the resulting dataframe with information about worse overloads for this scenario and agent

    """
    
    overloads_info_scenario=overloads_info[(overloads_info.scenarios==ScenarioName) &(overloads_info.agent==agent)]
    possiblyOverloadedLines
    
    idxMaxLines=[]
    summary_overloads_scenario_df=pd.DataFrame()
    for l in possiblyOverloadedLines:
        
        overloads_info_scenario_line=overloads_info_scenario[overloads_info_scenario.overload_line_name==l]
        if overloads_info_scenario_line.shape[0]!=0:
            idxMaxLine=[overloads_info_scenario_line['duration'].idxmax()]
            #idxMaxLines.append(idxMaxLine)
            summary_overloads_scenario_df=pd.concat([summary_overloads_scenario_df,overloads_info.iloc[idxMaxLine]])
        #else:
        #    #we take a row at random for this line in all the original dataframe and set duration and maxDepth to NaN
        #    row_df=overloads_info[(overloads_info.overload_line_name==l)&(overloads_info.agent==agent)].head(1)
        #    row_df[['maxDepths','start_indices','end_indices',
        #            'maxDepth_indices','hour_of_day','day_of_week']]=np.nan
        #    row_df[['duration']]=0
        #    row_df['scenarios']=None
        #    summary_overloads_scenario_df=pd.concat([summary_overloads_scenario_df,row_df])
    #print(idxMaxLine)
    return summary_overloads_scenario_df.reset_index(drop=True)



#this helps get the best actions at each timesetp considering the ones with least number of overloads and minimal congestion level
def get_simple_overload_df(df_analysis,line_names,indicesLineOverloaded,thermal_limits):
    nb_overloads_df=pd.DataFrame({'nb_total':np.zeros(df_analysis.shape[0])})

    for i,l in enumerate(line_names[indicesLineOverloaded]):
        print("computing for line " + l)
        nb_overloads_df["nb_total"]=nb_overloads_df["nb_total"].add((df_analysis[l]>= thermal_limits[indicesLineOverloaded[i]]).astype('bool'))
    
    print('finish nb total')
    
    max_rhos=(df_analysis[line_names[indicesLineOverloaded]]/thermal_limits[indicesLineOverloaded]).max(axis=1)
    print('finish max rhos')
    
    nb_overloads_df["scenario"]=df_analysis["scenario"]
    nb_overloads_df["agent"]=df_analysis["agent"]
    nb_overloads_df["datetimes"]=df_analysis["datetimes"]
    nb_overloads_df["max_depth"]=max_rhos
    
    #plot per agent
    #print('ploting')
    #matplt.xticks(rotation=90)
    #nb_overloads_df.groupby(["agent"]).sum().plot(kind="bar",title="number of overloads per agent over all scenarios")
    
    
    return nb_overloads_df