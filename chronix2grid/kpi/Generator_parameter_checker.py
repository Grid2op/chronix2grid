# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import pandas as pd
import os
import numpy as np

def EnergyMix_AprioriChecker(env118_withoutchron,Target_EM_percentage, PeakLoad, AverageLoad, CapacityFactor ):
    # # Check the Energy Mix apriori

    # ### Target Energy mix 
    # when summing over scenarios
    # 

    # In[132]:


    #print('\n the target energy mix is')
    #print(Target_EM_percentage)


    # ### Check initial capacity mix

    # In[139]:


    Mix_df=pd.DataFrame({'name':env118_withoutchron.name_gen,'type':env118_withoutchron.gen_type,'pmax':env118_withoutchron.gen_pmax})
    TotalCapacity=Mix_df['pmax'].sum()
    CapacityPerType=Mix_df.groupby(['type'])['pmax'].agg('sum')

    #CapacityNonRenewable=CapacityPerType['thermal']+CapacityPerType['nuclear']+CapacityPerType['hydro']
    CapacityMix=np.round((CapacityPerType/TotalCapacity*100)*10)/10
    CapacityMix.name='capacity_mix'
    #print('\n Total capacity:'+str(np.round(TotalCapacity)))
    #print('Total non renewable capacity:'+str(np.round(CapacityNonRenewable)))
    #print(CapacityPerType)
    #print('\n the capacity mix is:')
    #print(CapacityMix)


    # ### Considering capacity factors

    # We need to define capacity factors then to estimate the share really produced.
    # We can consider that every renewable production than can be produced will be used.
    # According to https://bilan-electrique-2019.rte-france.com/, we can take a capacity factor of:
    #         - 15% for solar
    #         - 25% for wind
    #         
    # We can then consider that nuclear is almost always working near its full capacity, so a capacity factor of 0.95
    # 
    # Then for hydro, the pmax is changing over a year so it is not easy to estimate. 
    # But the storage we have in hydro will be consume anyway during the year anyway. As an indicator in France for dumps in 2019:
    #     - the max hyrdo was 6.5GW (so close to max capacity)
    #     - on average the production was 2.15 GW
    #     - so a capacity factor of 30%
    # 
    # For thermal, we can reach its max capacity when ever needed, if its fast enough. So there is no capacity factor so too speak, but just need to have the right residual capcity to go above peak net demand. 
    # 

    # In[140]:


    #print('\n the capacity factors are:')
    #print(CapacityFactor)


    # In[141]:


    Capacity=pd.concat([Target_EM_percentage, CapacityPerType, CapacityMix, CapacityFactor], axis=1)
    #Capacity['revised_pmax']=np.round((Capacity['pmax'].values/Capacity['capacity_factor'].values*100)*10)/10
    #print("\n revised capacities after taking into account capacity factor")
    #print(Capacity)


    # Finally thermal should be scaled so that nuclear + thermal deals with peak demand. 
    # Peak Demand below is 4200MW

    # In[142]:
    
    EnergyAPriori=Capacity['pmax']/100*Capacity['capacity_factor']
    EnergyAPriori

    MaxMixEnergyNoThermal=EnergyAPriori/EnergyAPriori.sum()*100
    MaxMixEnergyNoThermal

    thermalMix=Target_EM_percentage.loc['thermal'].values
    MaxMixEnergyWithThermal=MaxMixEnergyNoThermal*(1-thermalMix/100)
    MaxMixEnergyWithThermal['thermal']=thermalMix

    Capacity['Apriori_energy_mix']=Capacity['capacity_factor']*Capacity['pmax']/AverageLoad
    
    MixShareNoThermal=Capacity['Apriori_energy_mix'].sum()
    if (MixShareNoThermal>=100):
        Capacity['Apriori_energy_mix']['hydro']=Capacity['Apriori_energy_mix']['hydro']-(MixShareNoThermal-100)
        Capacity['Apriori_energy_mix']['thermal']=0
    else:
        Capacity['Apriori_energy_mix']['thermal']=100-MixShareNoThermal

    Capacity['revised_pmax']=Capacity['target_energy_mix']/Capacity['Apriori_energy_mix']*Capacity['pmax']
    Capacity['revised_pmax']['thermal']=PeakLoad-Capacity['revised_pmax']['nuclear']
    #print("\n revised thermal capacity")
    #print(Capacity)
    
    
    error=np.abs(Capacity['target_energy_mix']-Capacity['Apriori_energy_mix']).sum()
    print('Warning: the differences in your target energy mix and you energy mix a priori are: ' + str(round(error))+'%')

    return Capacity








def Ramps_Pmax_Pmin_APrioriCheckers(env118_withoutchron,Capacity, chronics_path_gen,losses_pct,expected_PeakLoad):



    isThermalInTrouble=False #check that we can satisfy peak demand with thermal and nuclear
    isNuclearInTrouble=False #check that  nuclear pMax is always above min net load, or otherwise that nuclear max ramp down is greater than net load max ramp down
    IsRampUpInTrouble=False #check that thermal and nuclear ramp up is greater than the ramp up of the net demand
    IsRampDownInTrouble=False #check that hydro and nuclear ramp down is greater than the ramp down of the net demand


    # Create dataFrames per type of chronics

    # In[19]:


    #Create DataFrames
    Load_df=pd.DataFrame()
    Load_net=pd.DataFrame()
    Wind_df=pd.DataFrame()
    Solar_df=pd.DataFrame()

    fileList=[f for f in os.listdir(chronics_path_gen) if not f.startswith('.')]
    for subpath in fileList:
        # Load consumption and prod
        if(os.path.isdir(os.path.join(chronics_path_gen,subpath))):
            this_path = os.path.join(chronics_path_gen, subpath)
            load_p = pd.read_csv(os.path.join(this_path, 'load_p.csv.bz2'), sep = ';')
            prod_p = pd.read_csv(os.path.join(this_path, 'prod_p.csv.bz2'), sep = ';')

           # Retrieve wind and solar from prod_p (Balthazar's generator)
            prod_p_wind = prod_p[[el for i, el in enumerate(env118_withoutchron.name_gen) if env118_withoutchron.gen_type[i] in ["wind"]]]
            total_p_wind=prod_p_wind.sum(axis=1)
            prod_p_solar = prod_p[[el for i, el in enumerate(env118_withoutchron.name_gen) if env118_withoutchron.gen_type[i] in ["solar"]]]
            total_p_solar=prod_p_solar.sum(axis=1) 

            total_renew = pd.concat([total_p_wind, total_p_solar], axis=1).sum(axis=1)    

            # Compensate the reactive part in loads
            #load_ = load_p.copy() * (1 + losses_pct/100)
            load_ = load_p.sum(axis=1)
            load_ = load_*(1 + losses_pct/100)

            # Demand for OPF (total - renewable)
            agg_load_without_renew = (load_ - total_renew).to_frame()
            Load_df[subpath]=load_
            Wind_df[subpath]=total_p_wind
            Solar_df[subpath]=total_p_solar

    Load_net=Load_df-Wind_df-Solar_df


    # ## Check renewable energy mix share

    # In[153]:

    MaxSolar=Solar_df.max().max()
    MaxWind=Wind_df.max().max()
    
    #Energy Mix
    TotalLoad=Load_df.sum().sum()
    TotalWind=Wind_df.sum().sum()
    TotalSolar=Solar_df.sum().sum()

    WindShare=np.round((TotalWind/TotalLoad*100)*10)/10
    SolarShare=np.round((TotalSolar/TotalLoad*100)*10)/10

    print('\n the wind share is '+str(WindShare))
    print('the wind share was expected to be '+str(Capacity['target_energy_mix']['wind']))
    print('the solar share is '+str(SolarShare))
    print('the solar share was expected to be '+str(Capacity['target_energy_mix']['solar']))


    # In[154]:


    df_stats=pd.DataFrame(Load_net.values.flatten()).describe()
    df_stats.columns=['Load_net']

    #df_stats['Load_net']=pd.DataFrame(Load_net.values.flatten()).describe()
    df_stats['Solar']=pd.DataFrame(Solar_df.values.flatten()).describe()
    df_stats['Wind']=pd.DataFrame(Wind_df.values.flatten()).describe()
    df_stats['Renewable']=pd.DataFrame(Wind_df.values.flatten()+Solar_df.values.flatten()).describe()
    df_stats['Load']=pd.DataFrame(Load_df.values.flatten()).describe()

    df_stats


    # In[159]:


    print('\n the max load is '+str(df_stats['Load']['max']))
    print('the expected peak load was '+str(expected_PeakLoad))
    print('\n the max net load is '+str(df_stats['Load_net']['max']))

    toleranceThreshold=0.02
    isThermalInTrouble=(abs(expected_PeakLoad-df_stats['Load']['max'])/expected_PeakLoad > toleranceThreshold)

    print('WARNING: if they differ by more than 2%, we should change the capacity of thermal productions')


    # In[170]:


    Ramps_gen_df=pd.DataFrame({'name':env118_withoutchron.name_gen,'type':env118_withoutchron.gen_type,
                               'rampUp':env118_withoutchron.gen_max_ramp_up,'rampDown':env118_withoutchron.gen_max_ramp_down})
    Ramps_genType_df=Ramps_gen_df.groupby(['type'])['rampUp','rampDown'].agg('sum')
    Ramps_genType_df


    # In[157]:


    df_stats_ramps=pd.DataFrame(Load_net.diff().values.flatten()).describe()
    df_stats_ramps.columns=['Load_net']

    #df_stats['Load_net']=pd.DataFrame(Load_net.values.flatten()).describe()
    df_stats_ramps['Solar']=pd.DataFrame(Solar_df.diff().values.flatten()).describe()
    df_stats_ramps['Wind']=pd.DataFrame(Wind_df.diff().values.flatten()).describe()
    df_stats_ramps['Renewable']=pd.DataFrame(Wind_df.diff().values.flatten()+Solar_df.values.flatten()).describe()
    df_stats_ramps['Load']=pd.DataFrame(Load_df.diff().values.flatten()).describe()

    df_stats_ramps


    # In[178]:


    isNuclearPmaxInTrouble=(df_stats['Load_net']['min']<Capacity['pmax']['nuclear'])
    isNuclearRampInTrouble=(df_stats_ramps['Load_net']['max']>Ramps_genType_df['rampDown']['nuclear'])
    isNuclearInTrouble=(isNuclearPmaxInTrouble and isNuclearRampInTrouble)

    print('\n the min net load is '+str(df_stats['Load_net']['min']))
    print('the nuclear capacity is '+str(Capacity['pmax']['nuclear']))
    print('the max net load decrease '+str(df_stats_ramps['Load_net']['max']))
    print('the nuclear max ramp Down is '+str(Ramps_genType_df['rampDown']['nuclear']))
    print('are we in trouble for nuclear:'+ str(isNuclearInTrouble))
    print('WARNING: if the net demand is lowest than nuclear capacity, this will require to decrease nuclear production in the dispatch. If the demand decrease rate is stronger than the nuclear ramp, this might be tricky to do or infeasible')


    # In[185]:


    print('\n the max net load ramp up is '+str(df_stats_ramps['Load_net']['max']))
    print('the max generation ramp up is '+str(Ramps_genType_df['rampUp'].sum()))
    IsRampUpInTrouble=(df_stats_ramps['Load_net']['max'] > Ramps_genType_df['rampUp'].sum())


    # In[190]:
    
    RampDownNucHydro=0
    if('nuclear' in Ramps_genType_df.index):
        RampDownNucHydro+=Ramps_genType_df['rampDown']['nuclear']
    if('hydro' in Ramps_genType_df.index):
        RampDownNucHydro+=Ramps_genType_df['rampDown']['hydro']
    print('\n the max net load ramp down is '+str(df_stats_ramps['Load_net']['min']))
    print('the max nuclear + hydro ramp down is '+str(RampDownNucHydro))
    IsRampDownInTrouble=(df_stats_ramps['Load_net']['min'] > RampDownNucHydro)


    # In[179]:


    #import cufflinks as cf
    #import plotly.offline
    #plotly.offline.init_notebook_mode(connected=True) # for offline mode use
    #cf.go_offline()

    #Wind_df.iplot(kind='scatter', filename='cufflinks/cf-simple-line')
    #Load_net.iplot(kind='scatter', filename='cufflinks/cf-simple-line')

    return [isThermalInTrouble,isNuclearInTrouble,IsRampUpInTrouble,IsRampDownInTrouble]


def Aposteriori_renewableCapacityFactor_Checkers(env118_withoutchron,Capacity, chronics_path_gen):
    
    Wind_df=pd.DataFrame()
    Solar_df=pd.DataFrame()

    fileList=[f for f in os.listdir(chronics_path_gen) if not f.startswith('.')]
    for subpath in fileList:
        
         if(os.path.isdir(os.path.join(chronics_path_gen,subpath))):
            # Load consumption and prod
            this_path = os.path.join(chronics_path_gen, subpath)
            prod_p = pd.read_csv(os.path.join(this_path, 'prod_p.csv.bz2'), sep = ';')

           # Retrieve wind and solar from prod_p (Balthazar's generator
            prod_p_wind = prod_p[[el for i, el in enumerate(env118_withoutchron.name_gen) if env118_withoutchron.gen_type[i] in ["wind"]]]
            total_p_wind=prod_p_wind.sum(axis=1)
            prod_p_solar = prod_p[[el for i, el in enumerate(env118_withoutchron.name_gen) if env118_withoutchron.gen_type[i] in ["solar"]]]
            total_p_solar=prod_p_solar.sum(axis=1) 

            # Demand for OPF (total - renewable)

            Wind_df[subpath]=total_p_wind
            Solar_df[subpath]=total_p_solar
        
    MaxSolar=Solar_df.max().max()
    MaxWind=Wind_df.max().max()
    
    solarCapacityFactor=Solar_df.mean().mean()/Capacity['pmax']['solar']#MaxSolar
    windCapacityFactor=Wind_df.mean().mean()/Capacity['pmax']['wind']#MaxWind
    
    print('\n the max wind production '+str(MaxWind))
    print('\n the expected max wind production was '+str(Capacity['pmax']['wind']))
    print('\n the max solar production '+str(MaxSolar))
    print('\n the expected max solar production was '+str(Capacity['pmax']['solar']))


    print('\n the solar capacity factor is: '+str(solarCapacityFactor))
    
    print('\n the expected solar capacity factor was: '+str(Capacity['capacity_factor']['solar']))
    
    print('\n the wind capacity factor is: '+str(windCapacityFactor))
    
    print('\n the expected wind capacity factor was: '+str(Capacity['capacity_factor']['wind']))
    
    return [solarCapacityFactor,windCapacityFactor]
