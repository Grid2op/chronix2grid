    # Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import warnings
import copy
from multiprocessing import Pool

import numpy as np
    
#obs_var_to_keep=["aor","load_p","gen_p","day_of_week","hour_of_day"]#and datetime with get_time_stamp() + day_of_year

from grid2op.Runner import Runner
from grid2op.Runner.aux_fun import _aux_run_one_episode
from grid2op.Chronics import ChronicsHandler
from grid2op.Action import BaseAction, TopologyAction, DontAct
from grid2op.Exceptions import Grid2OpException, EnvError
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Opponent.OpponentSpace import OpponentSpace
from grid2op.Reward import FlatReward, BaseReward
from grid2op.Rules import AlwaysLegal, BaseRules
from grid2op.Environment import Environment
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, GridValue
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Agent import DoNothingAgent, BaseAgent
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.dtypes import dt_float
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.operator_attention import LinearAttentionBudget

_IS_LINUX=True


def _aux_one_process_parrallel(
    runner,
    episode_this_process,
    process_id,
    path_save=None,
    env_seeds=None,
    agent_seeds=None,
    max_iter=None,
    add_detailed_output=False,
    add_nb_highres_sim=False
):
    """this is out of the runner, otherwise it does not work on windows / macos"""
    chronics_handler = ChronicsHandler(
        chronicsClass=runner.gridStateclass,
        path=runner.path_chron,
        **runner.gridStateclass_kwargs
    )
    parameters = copy.deepcopy(runner.parameters)
    nb_episode_this_process = len(episode_this_process)
    res = [(None, None, None) for _ in range(nb_episode_this_process)]
    for i, ep_id in enumerate(episode_this_process):
        # `ep_id`: grid2op id of the episode i want to play
        # `i`: my id of the episode played (0, 1, ... episode_this_process)
        env, agent = runner._new_env(
            chronics_handler=chronics_handler, parameters=parameters
        )
        try:
            env_seed = None
            if env_seeds is not None:
                env_seed = env_seeds[i]
            agt_seed = None
            if agent_seeds is not None:
                agt_seed = agent_seeds[i]
            print("episode is running: "+str(ep_id))
            name_chron, cum_reward, nb_time_step,max_ts, episode_data, nb_highres_called = _aux_run_one_episode(
                env,
                agent,
                runner.logger,
                ep_id,
                path_save,
                env_seed=env_seed,
                max_iter=max_iter,
                agent_seed=agt_seed,
                detailed_output=add_detailed_output
            )
            id_chron = chronics_handler.get_id()
            max_ts = chronics_handler.max_timestep()
            print("episode has run: "+name_chron)
            if add_detailed_output:
                       
                obs_to_keep={}
                for obs_var in runner.obs_var_to_keep:
                    first_obs_v=getattr(episode_data.observations[0],obs_var)
                    
                    #some run on scenarios have heteregeneous length as they don't go until the end. need to take that into account and fill missing values with Nan
                    obs_dim=1
                    if hasattr(first_obs_v, "__len__"):
                        obs_dim=len(getattr(episode_data.observations[0],obs_var))
                    if obs_dim==1:
                        obs_to_keep[obs_var]=np.array([getattr(obs, obs_var) if hasattr(obs,obs_var) else np.nan for obs in episode_data.observations ])
                    else:
                        obs_to_keep[obs_var]=np.array([getattr(obs, obs_var) if hasattr(obs,obs_var) else np.empty(obs_dim)*np.nan for obs in episode_data.observations ])
                #a_or[:nb_timesteps+1,]=np.array([getattr(obs, "a_or") for obs in episode.observations])
                #load_p[:nb_timesteps+1,]=np.array([getattr(obs, "load_p") for obs in episode.observations])
                #gen_p[:nb_timesteps+1,]=np.array([getattr(obs, "gen_p") for obs in episode.observations])
                #day_of_week[:nb_timesteps+1]=np.array([getattr(obs, "day_of_week") for obs in episode.observations])
                #hour_of_day[:nb_timesteps+1]=np.array([getattr(obs, "hour_of_day") for obs in episode.observations])

                datetimes=[obs.get_time_stamp() if obs is not None else None for obs in episode_data.observations]
                day_of_year=[date.timetuple().tm_yday if date is not None else np.nan for date in datetimes]

                #obs_to_keep=[a_or,load_p,gen_p,day_of_week,day_of_year,datetimes,hour_of_day]
                obs_to_keep["datetimes"]=datetimes
                obs_to_keep["day_of_year"]=day_of_year
                #obs_to_keep={"a_or":a_or,"load_p":load_p,"gen_p":gen_p,"day_of_week":day_of_week,
                #             "day_of_year":day_of_year,"datetimes":datetimes,"hour_of_day":hour_of_day}
                del episode_data
                
                res[i] = (
                    id_chron,
                    name_chron,
                    float(cum_reward),
                    nb_time_step,
                    max_ts,
                    obs_to_keep,#episode_data#obs_to_keep,
                )
                print("episode results are returning: "+name_chron)
            else:
                res[i] = (id_chron, name_chron, float(cum_reward), nb_time_step, max_ts)
        finally:
            env.close()
    
    return res
    
class Runner_Calibration(Runner):
    def __init__(self,
                 obs_var_to_keep,
                 init_env_path: str,
                 init_grid_path: str,
                 path_chron,  # path where chronics of injections are stored
                 name_env="unknown",
                 parameters_path=None,
                 names_chronics_to_backend=None,
                 actionClass=TopologyAction,
                 observationClass=CompleteObservation,
                 rewardClass=FlatReward,
                 legalActClass=AlwaysLegal,
                 envClass=Environment,
                 other_env_kwargs=None,
                 gridStateclass=GridStateFromFile,
                 # type of chronics to use. For example GridStateFromFile if forecasts are not used,
                 # or GridStateFromFileWithForecasts otherwise
                 backendClass=PandaPowerBackend,
                 backend_kwargs=None,
                 agentClass=DoNothingAgent,  # class used to build the agent
                 agentInstance=None,
                 verbose=False,
                 gridStateclass_kwargs={},
                 voltageControlerClass=ControlVoltageFromFile,
                 thermal_limit_a=None,
                 max_iter=-1,
                 other_rewards={},
                 opponent_space_type=OpponentSpace,
                 opponent_action_class=DontAct,
                 opponent_class=BaseOpponent,
                 opponent_init_budget=0.0,
                 opponent_budget_per_ts=0.0,
                 opponent_budget_class=NeverAttackBudget,
                 opponent_attack_duration=0,
                 opponent_attack_cooldown=99999,
                 opponent_kwargs={},
                 grid_layout=None,
                 with_forecast=True,
                 attention_budget_cls=LinearAttentionBudget,
                 kwargs_attention_budget=None,
                 has_attention_budget=False,
                 logger=None,
                 kwargs_observation=None,
                 observation_bk_class=None,
                 observation_bk_kwargs=None,
                 
                 # experimental: whether to read from local dir or generate the classes on the fly:
                 _read_from_local_dir=False,
                 _is_test=False,  # TODO not implemented !!
                 #**kwargs
    ):
        self.obs_var_to_keep=obs_var_to_keep
        
        Runner.__init__(self,
                        init_env_path,
                        init_grid_path,
                        path_chron,  # path where chronics of injections are stored
                        name_env,
                        parameters_path,
                        names_chronics_to_backend,
                        actionClass,
                        observationClass,
                        rewardClass,
                        legalActClass,
                        envClass,
                        other_env_kwargs,
                        gridStateclass,
                        # type of chronics to use. For example GridStateFromFile if forecasts are not used,
                        # or GridStateFromFileWithForecasts otherwise
                        backendClass,
                        backend_kwargs,
                        agentClass,  # class used to build the agent
                        agentInstance,
                        verbose,
                        gridStateclass_kwargs,
                        voltageControlerClass,
                        thermal_limit_a,
                        max_iter,
                        other_rewards,
                        opponent_space_type,
                        opponent_action_class,
                        opponent_class,
                        opponent_init_budget,
                        opponent_budget_per_ts,
                        opponent_budget_class,
                        opponent_attack_duration,
                        opponent_attack_cooldown,
                        opponent_kwargs,
                        grid_layout,
                        with_forecast,
                        attention_budget_cls,
                        kwargs_attention_budget,
                        has_attention_budget,
                        logger,
                        kwargs_observation,
                        observation_bk_class,
                        observation_bk_kwargs,
                        # experimental: whether to read from local dir or generate the classes on the fly:
                        _read_from_local_dir,
                        _is_test,  # TODO not implemented !!
                        
    )
    
    def run_one_episode(
        self,
        indx=0,
        path_save=None,
        pbar=False,
        env_seed=None,
        max_iter=None,
        agent_seed=None,
        episode_id=None,
        detailed_output=False,
        add_nb_highres_sim=False
    ):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Function used to run one episode of the :attr:`Runner.agent` and see how it performs in the :attr:`Runner.env`.

        Parameters
        ----------
        indx: ``int``
            The number of episode previously run

        path_save: ``str``, optional
            Path where to save the data. See the description of :mod:`grid2op.Runner` for the structure of the saved
            file.
        detailed_output: see Runner.run method

        Returns
        -------
        cum_reward: ``np.float32``
            The cumulative reward obtained by the agent during this episode

        time_step: ``int``
            The number of timesteps that have been played before the end of the episode (because of a "game over" or
            because there were no more data)

        """
        self.reset()

        print("episode is running: "+name_chron)
        with self.init_env() as env:
            #if(max_iter is None):
            #    max_iter=env.get_obs().max_step
            
            name_chron, cum_reward, nb_timesteps, episode,nb_highres_called = _aux_run_one_episode(
                env,
                self.agent,
                self.logger,
                indx,
                path_save,
                pbar=pbar,
                env_seed=env_seed,
                max_iter=max_iter,
                agent_seed=agent_seed,
                detailed_output=detailed_output
            )
        print("episode has run: "+name_chron)
        if(detailed_output):
            
            #a_or=np.zeros((nb_timesteps+1,env.n_line))
            #load_p=np.zeros((nb_timesteps+1,env.n_load))
            #gen_p=np.zeros((nb_timesteps+1,env.n_gen))
            #day_of_week=np.zeros(nb_timesteps+1, dtype=int)
            #day_of_year=np.zeros(nb_timesteps+1, dtype=int)
            #datetimes=[]
            #hour_of_day=np.zeros(nb_timesteps+1, dtype=int)
            obs_to_keep=[]
            for obs_var in self.obs_var_to_keep:
                first_obs_v=getattr(episode_data.observations[0],obs_var)
                    
                #some run on scenarios have heteregeneous length as they don't go until the end. need to take that into account and fill missing values with Nan
                obs_dim=1
                if hasattr(first_obs_v, "__len__"):
                    obs_dim=len(getattr(episode_data.observations[0],obs_var))
                if obs_dim==1:
                    obs_to_keep[obs_var]=np.array([getattr(obs, obs_var) if hasattr(obs,obs_var) else np.nan for obs in episode_data.observations ])
                else:
                    obs_to_keep[obs_var]=np.array([getattr(obs, obs_var) if hasattr(obs,obs_var) else np.empty(obs_dim)*np.nan for obs in episode_data.observations ])
            #a_or[:nb_timesteps+1,]=np.array([getattr(obs, "a_or") for obs in episode.observations])
            #load_p[:nb_timesteps+1,]=np.array([getattr(obs, "load_p") for obs in episode.observations])
            #gen_p[:nb_timesteps+1,]=np.array([getattr(obs, "gen_p") for obs in episode.observations])
            #day_of_week[:nb_timesteps+1]=np.array([getattr(obs, "day_of_week") for obs in episode.observations])
            #hour_of_day[:nb_timesteps+1]=np.array([getattr(obs, "hour_of_day") for obs in episode.observations])
            
            datetimes=[obs.get_time_stamp() for obs in episode.observations]
            day_of_year[:nb_timesteps+1]=[date.timetuple().tm_yday for date in datetimes]
            
            #obs_to_keep=[a_or,load_p,gen_p,day_of_week,day_of_year,datetimes,hour_of_day]
            obs_to_keep.append(datetimes)
            obs_to_keep.append(day_of_year)
            
            del episode
        print("episode results are returning: "+name_chron)
            
        return name_chron, cum_reward, nb_timesteps,obs_to_keep,nb_highres_called
    
    
    def _run_parrallel(
        self,
        nb_episode,
        nb_process=1,
        path_save=None,
        env_seeds=None,
        agent_seeds=None,
        max_iter=None,
        episode_id=None,
        add_detailed_output=False,
        add_nb_highres_sim=False
    ):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method will run in parallel, independently the nb_episode over nb_process.

        In case the agent cannot be cloned using `copy.copy`: nb_process is set to 1

        Note that it restarts completely the :attr:`Runner.backend` and :attr:`Runner.env` if the computation
        is actually performed with more than 1 cores (nb_process > 1)

        It uses the python multiprocess, and especially the :class:`multiprocess.Pool` to perform the computations.
        This implies that all runs are completely independent (they happen in different process) and that the
        memory consumption can be big. Tests may be recommended if the amount of RAM is low.

        It has the same return type as the :func:`Runner.run_sequential`.

        Parameters
        ----------
        nb_episode: ``int``
            Number of episode to simulate

        nb_process: ``int``, optional
            Number of process used to play the nb_episode. Default to 1.

        path_save: ``str``, optional
            If not None, it specifies where to store the data. See the description of this module :mod:`Runner` for
            more information

        env_seeds: ``list``
            An iterable of the seed used for the experiments. By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.

        agent_seeds: ``list``
            An iterable that contains the seed used for the environment. By default ``None`` means no seeds are set.
            If provided, its size should match the ``nb_episode``. The agent will be seeded at the beginning of each
            scenario BEFORE calling `agent.reset()`.

        add_detailed_output: see Runner.run method

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.BaseAgent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics
              - "episode_data" : The :class:`EpisodeData` corresponding to this episode run

        """
        if nb_process <= 0:
            raise RuntimeError("Runner: you need at least 2 process to run episodes in parallel")
        force_sequential = False
        tmp = os.getenv(Runner.FORCE_SEQUENTIAL)
        if tmp is not None:
            force_sequential = int(tmp) > 0
        if nb_process == 1 or force_sequential:
            # on windows if i start using sequential, i need to continue using sequential
            # if i start using parallel i need to continue using parallel
            # so i force the usage of the sequential mode
            self.logger.warn(
                "Runner.run_parrallel: number of process set to 1. Failing back into sequential mod."
            )
            return self._run_sequential(
                nb_episode,
                path_save=path_save,
                env_seeds=env_seeds,
                max_iter=max_iter,
                agent_seeds=agent_seeds,
                episode_id=episode_id,
                add_detailed_output=add_detailed_output,
                add_nb_highres_sim=add_nb_highres_sim
            )
        else:
            self._clean_up()

            nb_process = int(nb_process)
            process_ids = [[] for i in range(nb_process)]
            for i in range(nb_episode):
                if episode_id is None:
                    process_ids[i % nb_process].append(i)
                else:
                    process_ids[i % nb_process].append(episode_id[i])

            if env_seeds is None:
                seeds_env_res = [None for _ in range(nb_process)]
            else:
                # split the seeds according to the process
                seeds_env_res = [[] for i in range(nb_process)]
                for i in range(nb_episode):
                    seeds_env_res[i % nb_process].append(env_seeds[i])

            if agent_seeds is None:
                seeds_agt_res = [None for _ in range(nb_process)]
            else:
                # split the seeds according to the process
                seeds_agt_res = [[] for i in range(nb_process)]
                for i in range(nb_episode):
                    seeds_agt_res[i % nb_process].append(agent_seeds[i])

            res = []
            if _IS_LINUX:
                lists = [
                    (
                        self,
                        pn,
                        i,
                        path_save,
                        seeds_env_res[i],
                        seeds_agt_res[i],
                        max_iter,
                        add_detailed_output,
                        add_nb_highres_sim
                    )
                    for i, pn in enumerate(process_ids)
                ]
            else:
                lists = [
                    (
                        Runner(**self._get_params()),
                        pn,
                        i,
                        path_save,
                        seeds_env_res[i],
                        seeds_agt_res[i],
                        max_iter,
                        add_detailed_output,
                        add_nb_highres_sim
                    )
                    for i, pn in enumerate(process_ids)
                ]
            print("parallel run is starting: "+str(process_ids))
            with Pool(nb_process) as p:
                tmp = p.starmap(_aux_one_process_parrallel, lists)
            print("parallel run ended: "+str(process_ids))
            for el in tmp:
                res += el
        return res

