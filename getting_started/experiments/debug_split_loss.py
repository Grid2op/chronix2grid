import grid2op
from grid2op.Chronics import ChangeNothing
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend  # might need "pip install lightsim2grid"
import shutil
from grid2op.Chronics import FromNPY
import os
import numpy as np
import copy
import pandas as pd
import warnings
import pdb

FLOATING_POINT_PRECISION_FORMAT = '%.1f'

def check_controlable_gens(gen_name, env, prods_p_total, weeks=1):
    pmax_ = np.array([env.gen_pmax[np.where(env.name_gen == nm_)[0]] for nm_ in gen_name])
    pmax_ = pmax_.ravel()
    check_pmax = prods_p_total[gen_name] > pmax_

    pmin_ = np.array([env.gen_pmin[np.where(env.name_gen == nm_)[0]] for nm_ in gen_name])
    pmin_ = pmin_.ravel()
    check_pmin = prods_p_total[gen_name] < pmin_

    max_up_ = np.array([env.gen_max_ramp_up[np.where(env.name_gen == nm_)[0]] for nm_ in gen_name])
    max_up_ = max_up_.ravel()
    max_down_ = np.array([env.gen_max_ramp_down[np.where(env.name_gen == nm_)[0]] for nm_ in gen_name])
    max_down_ = max_down_.ravel()
    delta_gen_ = prods_p_total[gen_name].diff()  # prods_p_total.diff(t) = prods_p_total(t) - prods_p_total(t-1)
    check_max_up = delta_gen_ > max_up_
    check_max_down = delta_gen_ < -max_down_
    # remove the "interface" between the months
    check_max_up[::(weeks * 7 * 288 - 1)] = False
    check_max_down[::(weeks * 7 * 288 - 1)] = False
    check_max_up[::(weeks * 7 * 288 )] = False
    check_max_down[::(weeks * 7 * 288)] = False
    return check_pmax, check_pmin, check_max_up, check_max_down

def check_all_controlable_gens(prods_p_total, gen_hydro_name2, gen_nuclear_name2, gen_thermal_name2, env118_withoutchron):
    check_pmax_hydro, check_pmin_hydro, check_max_up_hydro, check_max_down_hydro = check_controlable_gens(
        gen_hydro_name2, env118_withoutchron, prods_p_total)
    check_pmax_nuclear, check_pmin_nuclear, check_max_up_nuclear, check_max_down_nuclear = check_controlable_gens(
        gen_nuclear_name2, env118_withoutchron, prods_p_total)
    check_pmax_thermal, check_pmin_thermal, check_max_up_thermal, check_max_down_thermal = check_controlable_gens(
        gen_thermal_name2, env118_withoutchron, prods_p_total)

    errors = []
    if not np.all(check_pmax_hydro.sum() == 0):
        errors.append((check_pmax_hydro.sum(), f"some hydro are above pmax:\n{check_pmax_hydro.sum()}"))
    if not np.all(check_pmin_hydro.sum() == 0):
        errors.append((check_pmin_hydro.sum(), f"some hydro are below pmin:\n{check_pmin_hydro.sum()}"))
    if not np.all(check_max_up_hydro.sum() == 0):
        errors.append((check_max_up_hydro.sum(), f"some hydro are above max_up:\n{check_max_up_hydro.sum()}"))
    if not np.all(check_max_down_hydro.sum() == 0):
        errors.append((check_max_down_hydro.sum(), f"some hydro are below max_down:\n{check_max_down_hydro.sum()}"))

    if not np.all(check_pmax_nuclear.sum() == 0):
        errors.append((check_pmax_nuclear.sum(), f"some nuclear are above pmax:\n{check_pmax_nuclear.sum()}"))
    if not np.all(check_pmin_nuclear.sum() == 0):
        errors.append((check_pmin_nuclear.sum(), f"some nuclear are below pmin:\n{check_pmin_nuclear.sum()}"))
    if not np.all(check_max_up_nuclear.sum() == 0):
        errors.append((check_max_up_nuclear.sum(), f"some nuclear are above max_up:\n{check_max_up_nuclear.sum()}"))
    if not np.all(check_max_down_nuclear.sum() == 0):
        errors.append((check_max_down_nuclear.sum(), f"some nuclear are below max_down:\n{check_max_down_nuclear.sum()}"))

    if not np.all(check_pmax_thermal.sum() == 0):
        errors.append((check_pmax_thermal.sum(), f"some thermal are above pmax:\n{check_pmax_thermal.sum()}"))
    if not np.all(check_pmin_thermal.sum() == 0):
        errors.append((check_pmin_thermal.sum(), f"some thermal are below pmin:\n{check_pmin_thermal.sum()}"))
    if not np.all(check_max_up_thermal.sum() == 0):
        errors.append((check_max_up_thermal.sum(), f"some thermal are above max_up:\n{check_max_up_thermal.sum()}"))
    if not np.all(check_max_down_thermal.sum() == 0):
        errors.append((check_max_down_thermal.sum(), f"some thermal are below max_down:\n{check_max_down_thermal.sum()}"))

    return errors

def fill_real_gen(target,
                  row_id,
                  obs,
                  env,
                  gen_p_setpoint,
                  share_slack=0.7,
                  prev=None,
                  prev_diff=None,
                  slack_id=None,
                  rounding_margin=0.2,
                  max_abs_split=None):
    target[row_id, env.gen_renewable] = obs.gen_p[env.gen_renewable]
    loss = np.sum(obs.gen_p) - np.sum(obs.load_p)  # actual loss of the grid

    # split what the slack absorbed in the controlable generators        
    to_split = np.sum(obs.gen_p) - np.sum(gen_p_setpoint)
    total_margin = None
    delta_previous = 0.

    can_adjust = copy.deepcopy(~env.gen_renewable)
    can_adjust[slack_id] = False
    slack_abs = gen_p_setpoint[slack_id]  # obs.gen_p[slack_id]  # gen_p_setpoint[slack_id]
    
    max_split_slack = share_slack * to_split
    if prev is not None:
        min_slack, max_slack = [max(obs.gen_pmin[slack_id],
                                    prev[slack_id] - obs.gen_max_ramp_down[slack_id]),
                                min(obs.gen_pmax[slack_id],
                                    prev[slack_id] + obs.gen_max_ramp_up[slack_id])]
        min_slack += 1.0
        max_slack -= 1.0
        if max_split_slack > 0.:
            if slack_abs + max_split_slack > max_slack:
                max_split_slack = max_slack - slack_abs
        if max_split_slack < 0.:
            if slack_abs + max_split_slack < min_slack:
                max_split_slack = min_slack - slack_abs
            
        
    to_split_others = (1.0 - share_slack ) * to_split
    
    if to_split_others > 0.:
        total_margin = np.minimum(obs.gen_max_ramp_up[can_adjust] - delta_previous, 
                                  obs.gen_pmax[can_adjust] - obs.gen_p[can_adjust])
    else:
        total_margin = np.minimum(obs.gen_max_ramp_down[can_adjust] + delta_previous, 
                                  obs.gen_p[can_adjust] - obs.gen_pmin[can_adjust])
            
    if max_abs_split is not None:
        # limit the amount of power i got from this
        if to_split_others > max_abs_split:
            to_split_others = max_abs_split
        elif to_split_others < -max_abs_split:
            to_split_others = -max_abs_split

    if prev is not None:
        delta_previous = gen_p_setpoint[can_adjust] - prev[can_adjust]
        gen_up = delta_previous > 0.
        gen_down = delta_previous < 0.
        
        # there is 4 cases:
        if to_split_others > 0.:
            # generators up: less margin (because the should "get up more")
            total_margin[gen_up] = np.minimum(obs.gen_pmax[can_adjust][gen_up] - gen_p_setpoint[can_adjust][gen_up],
                                              np.maximum(obs.gen_max_ramp_up[can_adjust][gen_up] - delta_previous[gen_up],
                                                         0.)
                                              )
            
            # generators down: more margin (because the should get down, but the split is up)
            total_margin[gen_down] = np.minimum(obs.gen_pmax[can_adjust][gen_down] - gen_p_setpoint[can_adjust][gen_down],
                                                obs.gen_max_ramp_up[can_adjust][gen_down] # - delta_previous[gen_down]
                                                )
        else:
            # generators up: more margin (because the should "get up more")
            total_margin[gen_up] = np.minimum(gen_p_setpoint[can_adjust][gen_up] - obs.gen_pmin[can_adjust][gen_up],
                                              obs.gen_max_ramp_down[can_adjust][gen_up] # + delta_previous[gen_up]
                                              )
            
            # generators down: less margin (because the should get down, but the split is up)
            total_margin[gen_down] = np.minimum(gen_p_setpoint[can_adjust][gen_down] - obs.gen_pmin[can_adjust][gen_down],
                                                np.maximum(obs.gen_max_ramp_down[can_adjust][gen_down] + delta_previous[gen_down],
                                                           0.)
                                                )
            
    total_margin[total_margin < rounding_margin] = 0.
    total_margin[total_margin >= rounding_margin] -= rounding_margin
    
    turned_on = (obs.gen_p[can_adjust] > 0.) 
    if prev_diff is not None:
       turned_on |= (prev_diff[can_adjust] != 0.)
       
    if abs(to_split_others) <= abs(np.sum(total_margin[turned_on])) - rounding_margin:
        # do not turn on turned off generators, because I can
        total_margin[~turned_on] = 0.
    else:
        total_margin[~turned_on] = 0.
        to_split_others = np.sign(to_split_others) * (abs(np.sum(total_margin[turned_on])) - rounding_margin)
        
    sum_margin = np.sum(total_margin)
    redisp_ = 1.0 * gen_p_setpoint[can_adjust]
    
    if sum_margin == 0.:
        print("no margin for your system !")
    else:
        redisp_ += to_split_others * total_margin / sum_margin
    
    target[row_id, can_adjust] = redisp_
    target[row_id, slack_id] = slack_abs + max_split_slack
    return loss

def adjust_gens(all_loss_orig,
                env_path,
                env_param,
                path_chronic,
                slack_id,
                load_p, 
                load_q,
                gen_p,
                gen_v,
                threshold_stop=0.1,  # stop when all generators move less that this
                weeks=1,
                max_abs_split=5.0,  # dispatch, on the other generators, at most 5.0 MW
                ):
    error_ = None
    cond_ = True
    gen_p = 1.0 * gen_p  # not to change input data
    all_loss = 1.0 * all_loss_orig
    prev_max_diff = 100000.
    while cond_:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_fixed = grid2op.make(
                env_path,
                test=True,
                # grid_path=grid_path, # assign it the 118 grid
                param=env_param,
                backend=LightSimBackend(),
                chronics_class=FromNPY,
                chronics_path=path_chronic,
                data_feeding_kwargs={"load_p": load_p,
                                     "load_q": load_q,
                                     "prod_p": 1.0 * gen_p,
                                     "prod_v": gen_v}
                )
            
        # this should be as close to 0. as possible...
        # we might do a second "repartition loop" to make sure it's ok :-)
        i = 0
        final_gen_p_tmp = gen_p * np.NaN
        diff_ = np.full((weeks * 7 * 288 - 1, env_fixed.n_gen), fill_value=np.NaN)
        all_loss = np.zeros(weeks * 7 * 288 - 1)
        obs = env_fixed.reset()
        diff_[i] = obs.gen_p - gen_p[i]
        all_loss[i] = fill_real_gen(final_gen_p_tmp,
                                    i,
                                    obs,
                                    env_fixed,
                                    gen_p_setpoint=gen_p[i],
                                    slack_id=slack_id,
                                    max_abs_split=max_abs_split)
        # prev_ = obs.gen_p
        while True:
            obs, reward, done, info = env_fixed.step(env_fixed.action_space())
            if done:
                break
            i += 1
            all_loss[i] = fill_real_gen(final_gen_p_tmp,
                                        i, obs,
                                        env_fixed,
                                        gen_p_setpoint=gen_p[i],
                                        prev=final_gen_p_tmp[i-1], 
                                        slack_id=slack_id,
                                        max_abs_split=max_abs_split)
            diff_[i] = obs.gen_p - final_gen_p_tmp[i]
        max_diff = np.abs(diff_).max()
        gen_p = 1.0 * final_gen_p_tmp
        if max_diff >= prev_max_diff:
            # print("seems to mess something up... stopping here...")
            cond_ = False
            error_ = RuntimeError("Max difference does not decrease!")
            break
        
        # final_gen_df = pd.DataFrame(final_gen_p, columns=env_for_loss.name_gen)
        # errors = check_all_controlable_gens(final_gen_df, gen_hydro_name2, gen_nuclear_name2, gen_thermal_name2, env118_withoutchron)
        prev_max_diff = copy.deepcopy(max_diff)
        cond_ = max_diff > threshold_stop
    return gen_p, error_


def save_data(env_for_loss,
              path_chronix2grid,
              output_path,
              final_load_p,
              final_load_q,
              final_gen_p,
              final_gen_v,
              sep=';',
              float_prec=FLOATING_POINT_PRECISION_FORMAT):
    scen_id = os.path.split(env_for_loss.chronics_handler.get_id())[-1]
    this_path_chronics_fixed = os.path.join(output_path, scen_id)
    shutil.rmtree(this_path_chronics_fixed, ignore_errors=True)
    os.mkdir(this_path_chronics_fixed)

    pd.DataFrame(final_load_p,
                columns=env_for_loss.name_load).to_csv(os.path.join(this_path_chronics_fixed, f'load_p.csv.bz2'),
                                                        sep=sep,
                                                        float_format=float_prec,
                                                        header=True,
                                                        index=False)
    pd.DataFrame(final_load_q,
                columns=env_for_loss.name_load).to_csv(os.path.join(this_path_chronics_fixed, f'load_q.csv.bz2'),
                                                        sep=sep,
                                                        float_format=float_prec,
                                                        header=True,
                                                        index=False)
    pd.DataFrame(final_gen_p,
                columns=env_for_loss.name_gen).to_csv(os.path.join(this_path_chronics_fixed, f'prod_p.csv.bz2'),
                                                        sep=sep,
                                                        float_format=float_prec,
                                                        header=True,
                                                        index=False)
                
    pd.DataFrame(final_gen_v,
                columns=env_for_loss.name_gen).to_csv(os.path.join(this_path_chronics_fixed, f'prod_v.csv.bz2'),
                                                        sep=sep,
                                                        float_format=float_prec,
                                                        header=True,
                                                        index=False)

    fn_ = "start_datetime.info"
    for fn_ in ["load_p_forecasted.csv.bz2",
                "load_q_forecasted.csv.bz2",
                "start_datetime.info",
                "time_interval.info",
                ]:
        shutil.copy(src=os.path.join(path_chronix2grid, scen_id, fn_), 
                    dst=os.path.join(this_path_chronics_fixed, fn_))
        
        
def fix_losses_one_scenario(empty_env,
                            slack_id,
                            path_env,
                            path_chronix2grid,
                            output_path,
                            scenario_id,
                            gen_hydro_name2,
                            gen_nuclear_name2,
                            gen_thermal_name2,
                            weeks=1,
                            total_errors_threshold=30,
                            ):
    
    param = empty_env.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_for_loss = grid2op.make(
            path_env,
            test=True,
            # grid_path=empty_env._init_grid_path, # assign it the 118 grid
            chronics_path=path_chronix2grid,
            param=param,
            backend=LightSimBackend()
            )
        
    final_gen_p = np.full((weeks * 7 * 288 - 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_gen_v = np.full((weeks * 7 * 288 - 1, env_for_loss.n_gen), fill_value=np.NaN, dtype=np.float32)
    final_load_p = np.full((weeks * 7 * 288 - 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    final_load_q = np.full((weeks * 7 * 288 - 1, env_for_loss.n_load), fill_value=np.NaN, dtype=np.float32)
    all_loss_orig = np.zeros(weeks * 7 * 288 - 1)
    
    env_for_loss.set_id(scenario_id)
    obs = env_for_loss.reset()
    
    ### now start to split the loss
    max_abs_split = 20.0
    i = 0
    all_loss_orig[i] = fill_real_gen(final_gen_p, i, obs, env_for_loss, 
                                    gen_p_setpoint=env_for_loss.chronics_handler.real_data.data.prod_p[i],
                                    slack_id=slack_id,
                                    max_abs_split=max_abs_split)
    final_gen_v[i] = obs.gen_v
    final_load_p[i] = obs.load_p
    final_load_q[i] = obs.load_q
    done = False
    while not done:
        obs, reward, done, info = env_for_loss.step(env_for_loss.action_space())
        i += 1
        all_loss_orig[i] = fill_real_gen(final_gen_p,
                                        i,
                                        obs,
                                        env_for_loss,
                                        gen_p_setpoint=env_for_loss.chronics_handler.real_data.data.prod_p[i],
                                        prev=1.0 * final_gen_p[i-1],
                                        max_abs_split=max_abs_split,
                                        # prev_diff=1.0 * (final_gen_p[i-1] - prev_),
                                        slack_id=slack_id)
        final_gen_v[i] = 1.0 * obs.gen_v
        final_load_p[i] = 1.0 * obs.load_p
        final_load_q[i] = 1.0 * obs.load_p
        
    ###
    # final_gen_df = pd.DataFrame(final_gen_p, columns=env_for_loss.name_gen)
    # errors = check_all_controlable_gens(final_gen_df, gen_hydro_name2, gen_nuclear_name2, gen_thermal_name2, env118_withoutchron)
    # if errors:
    #     for _, el in errors:
    #         print(el)
    #         print()
        # raise RuntimeError("some constraints are not met")
    # print("No errors after iteration 0 !")
    
    ###
    # now adjust the generators schedule so that they do not move
    final_gen_p, error_ = adjust_gens(all_loss_orig,
                                      path_env,
                                      param,
                                      path_chronix2grid,
                                      slack_id,
                                      final_load_p, 
                                      final_load_q,
                                      final_gen_p,
                                      final_gen_v,
                                      max_abs_split=10.,
                                      )
    

    if error_ is not None:
        print(f"ERROR: {scenario_id}: {error_}")
        return
    
    final_gen_df = pd.DataFrame(final_gen_p, columns=env_for_loss.name_gen)
    errors = check_all_controlable_gens(final_gen_df, gen_hydro_name2, gen_nuclear_name2, gen_thermal_name2, empty_env)

    total_errors = sum([sum([sub_el for sub_el in el]) for el, str_ in errors])
    if total_errors > 0 and total_errors < total_errors_threshold:
        print(f"WARNING: {scenario_id}: There will be some violation of some constraints for {total_errors} generators.step in total")
    elif total_errors >= total_errors_threshold:
        for el in errors:
            print(el)
            print()
        print(f"ERROR: {scenario_id} too much generators would be violated")
        return
    # print("No error after the iterations")

    # now save it
    save_data(env_for_loss, 
              path_chronix2grid,
              output_path, 
              final_load_p, 
              final_load_q, 
              final_gen_p, 
              final_gen_v)


def fix_loss_multiple_scenarios(path_env,
                                path_chronix2grid,
                                output_path,
                                scenario_ids,
                                weeks=1):
    ############################
    # this can be refacto for all scenario id
    shutil.rmtree(output_path, ignore_errors=True)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env118_withoutchron = grid2op.make(
            path_env,
            test=True,
            chronics_class=ChangeNothing, # tell it to change nothing (not the most usable environment...)
        )
    
    slack_id = np.where(env118_withoutchron.backend._grid.gen["slack"])[0]
    slack_name = env118_withoutchron.name_gen[slack_id]
    
    gen_solar_name2 = env118_withoutchron.name_gen[env118_withoutchron.gen_type == "solar"]
    gen_wind_name2 = env118_withoutchron.name_gen[env118_withoutchron.gen_type == "wind"]
    gen_hydro_name2 = env118_withoutchron.name_gen[env118_withoutchron.gen_type == "hydro"]
    gen_nuclear_name2 = env118_withoutchron.name_gen[env118_withoutchron.gen_type == "nuclear"]
    gen_thermal_name2 = env118_withoutchron.name_gen[env118_withoutchron.gen_type == "thermal"]
    gen_thermal_name2 = [el for el in gen_thermal_name2 if el != slack_name]
    ###############################
    
    for scenario_id in scenario_ids:
        fix_losses_one_scenario(env118_withoutchron, slack_id,
                                path_env,
                                path_chronix2grid,
                                output_path,
                                scenario_id,
                                gen_hydro_name2,
                                gen_nuclear_name2,
                                gen_thermal_name2,
                                weeks=weeks)
    
if __name__ == "__main__":
    
    OUTPUT_FOLDER = os.path.join('..', 'example', 'custom', 'output')

    path_chronics_outputopf = os.path.join(OUTPUT_FOLDER, "all_scenarios")
    path_chronics_fixed = os.path.join(OUTPUT_FOLDER, "fixed_chronics")
        
    ########
    # Detailed configuration to be set in <INPUT_FOLDER>/<CASE>/params.json
    weeks = 1
    n_scenarios = 1

    env_name = "case118_l2rpn_wcci_benjamin"
    path_tmp = os.path.join("..", "example", "custom", "input", "generation")
    output_path = os.path.join(path_tmp, env_name)
    grid_path = os.path.join(output_path, "grid.json")
    
    li_months = ["2050-01-03", 
             "2050-01-10",
             "2050-01-17",
             "2050-01-24",
             "2050-01-31",
             "2050-02-07",
             "2050-02-14",
             "2050-02-21",
             "2050-02-28",
             "2050-03-07",
             "2050-03-14",
             "2050-03-21",
             "2050-03-28",
             "2050-04-04",
             "2050-04-11",
             "2050-04-18",
             "2050-04-25",
             "2050-05-02", 
             "2050-05-09", 
             "2050-05-16", 
             "2050-05-23", 
             "2050-05-30",
             "2050-06-06",
             "2050-06-13",
             "2050-06-20",
             "2050-06-27",
             "2050-07-04", 
             "2050-07-11", 
             "2050-07-18", 
             "2050-07-25", 
             "2050-08-01", 
             "2050-08-08", 
             "2050-08-15", 
             "2050-08-22", 
             "2050-08-29", 
             "2050-09-05", 
             "2050-09-12", 
             "2050-09-19", 
             "2050-09-26", 
             "2050-10-03", 
             "2050-10-10", 
             "2050-10-17", 
             "2050-10-24", 
             "2050-10-31", 
             "2050-11-07", 
             "2050-11-14", 
             "2050-11-21", 
             "2050-11-28", 
             "2050-12-05",
             "2050-12-12",
             "2050-12-19",
             "2050-12-26",
            ]
    fix_loss_multiple_scenarios(output_path,
                                path_chronics_outputopf,
                                path_chronics_fixed,
                                scenario_ids=[f"{el}_0" for el in li_months]
                                )
