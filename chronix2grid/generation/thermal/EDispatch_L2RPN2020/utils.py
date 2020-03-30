import pandas as pd
import numpy as np
import calendar
import os
import sys
import pypsa
from datetime import datetime, timedelta

# def format_dates(year, month):
#     # Get last of for every month
#     end_day_month = calendar.monthrange(year, month)[1]
#     from_date = f'{year}-{month}-01 00:00'
#     end_date = f'{year}-{month}-{end_day_month} 23:55'
#     # Convert them to datetime
#     FROM_DATE = datetime.strptime(from_date, '%Y-%m-%d %H:%M')
#     END_DATE = datetime.strptime(end_date, '%Y-%m-%d %H:%M')
#     return FROM_DATE, END_DATE

def get_params(num, params):
    snaps = params['snapshots']
    step_opf = int(params['step_opf_min'])
    mode_opf = params['mode_opf']
    # Check some inputs
    if not step_opf % 5 == 0:
        raise RuntimeError("\"step_opf_min\" argument should be multiple of 5")
    if not mode_opf.lower() in ['day', 'week', 'month']:
        raise RuntimeError("Please provide a valid opf mode (day, week, month")
    # Update inputs
    if snaps == []:
        snapshots = pd.date_range(start=f'{2007}-01-01', periods=num, freq='5min')
        params.update({'snapshots': snapshots})
    return params

def preprocess_input_data(load, gen_constraints, params):
    """Prepare input data (load and gen constraints) adding 
    temporary datetime index and resampled according params needs.

    Arguments:
        load {pd.DataFrame} -- Demand 
        gen_constraints {dict} -- Contains two keys {p_max_pu, p_min_pu} which associate
                                  the dataframes with the limits. Dataframes header should have 
                                  same gen index PyPSA name.                
        params {dict} -- Dictionary holding all set of params for opf
    
    Returns:
        {pd.DataFrame} -- Reformated load dataframe
        {dict} -- Reformated dict with dfs for gen constraints
    """  
    # Get load after adapting to step_opf_min  
    new_load = reformat_load(load, params)
    # Get gen contraints after adapting to step_opf_min
    new_gen_const = reformat_gen_constraints(gen_constraints, params, new_load.index)
    return new_load, new_gen_const
    
def reformat_load(load, params):
    """Reformat load dataframe adding temp datetime index
    and resampled it according the params step_opf_min specified.

    Arguments:
        load {pd.DataFrame} -- Demand
        params {dict} -- Dictionary holding all set of params for opf
    
    Returns:
        pd.DataFrame -- Aggregated reformated and resampled demand
    """    
    snapshots = params['snapshots']
    step_opf = params['step_opf_min']
    q_compensation = params['reactive_comp']
    # Set temp index to load and constraints
    load.index = snapshots
    # Resample data and constrains according to params
    load_resampled = load.resample(f'{str(step_opf)}min').apply(lambda x: x[0])
    load_resampled *= q_compensation
    # -- Agregate load in case it is not
    if load_resampled.columns.tolist() == ['agg_load']:
        return load_resampled
    else:
        agg_load = load_resampled.sum(axis=1).to_frame()
        agg_load.columns = ['agg_load']
        return agg_load

def reformat_gen_constraints(gen_constraints, params, new_snaps):
    """Reformat dataframes associated in gen_contraints
    {p_max_pu, p_min_pu} adding tempoary datetime index and resampled them
    acoording to params specified.
    dictionary .
    
    Arguments:
        gen_constraints {dict} -- Contains two keys {p_max_pu, p_min_pu} which associate
                                  the dataframes with the limits. Dataframes header should have 
                                  same gen index PyPSA name.
        params {dict} -- Dictionary holding all set of params for opf
        new_snaps {index} -- load resampled index
    
    Returns:
        {dict} -- Reformated dict containing dfs
    """    
    snapshots = params['snapshots']
    resampled_snapshots = new_snaps
    # Set index in gen contraints 
    for k in gen_constraints:
        if gen_constraints[k] is None:
            # Create an empty gen contraints in case None is pass
            gen_constraints.update({k: pd.DataFrame(index=snapshots)})
        else:
            # Set temp index to gen contraints
            gen_constraints[k].index = snapshots
            gen_constraints.update({k: gen_constraints[k]})
        # Resampled to desired step run opf
        gen_constraints.update({k: gen_constraints[k].loc[resampled_snapshots]})
    return gen_constraints

def adapt_gen_prop(net, every_min, grid_params=5):
    """Scale gen ramps up/down according to step_opf_min param
    (The original grid is prepared that every params are set for
    every 5 minutes)
    
    Arguments:
        net {PyPSA object} -- 
        every_min {int} -- Minutes user wants to run OPF assuming
                           data comes from every 5 min
    
    Keyword Arguments:
        grid_params {int} -- Referece time gen properties is designed
    
    Returns:
        net {PyPSA object} -- Grid with scaled ramps
    """    
    # Adapt ramps according to the time 
    steps = every_min / grid_params
    net.generators.loc[:, ['ramp_limit_up', 'ramp_limit_down']] *= steps
    return net

def get_indivitual_snapshots_per_mode(idx, mode):  
    # Define all posibilities mode
    periods = {'day': idx.groupby(idx.day).values(),
               'week': idx.groupby(idx.week).values(),
               'month': idx.groupby(idx.month).values()
    }
    return periods[mode]

def prepare_net_for_opf(net, load_per_period, gen_const_per_period):
    """Function to set for individual OPF formulation problem:
            - snapshots
            - Aggregated load for all snapshots
            - Gen constraints for all snapshots
    
    Arguments:
        net {PyPSA object}
        load_indiv {pd.DataFrame} -- Load df per day, week or month
        gen_const_indiv {dict} -- Gen constraints per day, week or month
    
    Returns:
        net [PyPSA object] -- Grid with all modifications
    """    
    # Reset any previous value save it in the grid
    net.loads_t.p_set = net.loads_t.p_set.iloc[0:0, 0:0]
    net.generators_t.p_max_pu = net.generators_t.p_max_pu.iloc[0:0, 0:0]
    net.generators_t.p_min_pu = net.generators_t.p_min_pu.iloc[0:0, 0:0]
    # Set snapshot
    # ++  ++  ++ 
    snapshots = load_per_period.index
    net.set_snapshots(snapshots)
    # Set consumption to the grid
    # ++  ++  ++  ++  ++  ++  ++
    net.loads_t.p_set = pd.concat([load_per_period])
    # Set generations pmax, pmin constraints
    # ++  ++  ++  ++  ++  ++  ++  ++  ++  ++
    net.generators_t.p_max_pu = pd.concat([gen_const_per_period['p_max_pu']], axis=1)
    net.generators_t.p_min_pu = pd.concat([gen_const_per_period['p_min_pu']], axis=1)
    # Constrain nuclear power plants
    nuclear_names = net.generators[net.generators.carrier == 'nuclear'].index.tolist()
    for c in nuclear_names:
        net.generators_t.p_max_pu[c] = 1.
        net.generators_t.p_min_pu[c] = 0.4
    return net

def run_opf(net, demand, gen_constraints, params):
    # Show info when running opf
    to_disp = {'day': demand.index.day.unique().values[0],
               'week': demand.index.week.unique().values[0],
               'month': demand.index.month.unique().values[0],
    }
    mode = params['mode_opf']
    print(f'\n--> OPF single formulation by: {mode} - Analyzing {mode} # {to_disp[mode]}')
    # Prepare grid for OPF
    net = prepare_net_for_opf(net, demand, gen_constraints)
    # Run Linear OPF
    rel = net.lopf(net.snapshots, pyomo=False, solver_name='cbc')
    if rel[1] != 'optimal': 
        print ('** OPF failed to find a solution **')
        sys.exit()
    # Get the values
    dispatch = net.generators_t.p.copy()
    return dispatch

def add_noise_gen(dispatch, gen_cap, noise_factor=None):
    """Add noise to dispatch result
    
    Arguments:
        dispatch {pd.DataFrame} -- Df holding dispatch result
        gen_cap {pd.DataFrame} -- Max capacity for generators
    
    Keyword Arguments:
        noise_factor {float} -- Noise factor applied for every gen
    
    Returns:
        {pd.DataFrame} -- Dispatch with noise
    """    
    # Get range of value per columns in df
    # stats = df.agg(['max', 'min'], axis=0)
    # range_ = stats.loc['max'] - stats.loc['min']
    variance_per_col = gen_cap * noise_factor
    for col in dispatch:
        # Check for values greater than zero 
        # (means unit has been distpached)
        only_dispatched_steps = dispatch[col][dispatch[col] > 0]
        noise = np.random.normal(0, variance_per_col.loc[col], only_dispatched_steps.shape[0])
        dispatch.loc[only_dispatched_steps.index, col] = only_dispatched_steps + noise
    return dispatch.round(2)

# def import_data(data_path, from_date, end_date, every_min):
#     dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#     load = pd.read_csv(os.path.join(data_path, 'load_' + str(from_date.year) + '.csv.bz2'), 
#                        parse_dates=['datetime'], date_parser=dateparse)
#     solar = pd.read_csv(os.path.join(data_path, 'selected_solar.csv.bz2'), 
#                         parse_dates=['datetime'], date_parser=dateparse)
#     wind = pd.read_csv(os.path.join(data_path, 'selected_wind.csv.bz2'), 
#                        parse_dates=['datetime'], date_parser=dateparse)
#     # Set corresponding index
#     load.set_index('datetime', inplace=True)
#     solar.set_index('datetime', inplace=True)
#     wind.set_index('datetime', inplace=True)
#     # Truncate according to max dates
#     load_rscl = load.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     solar_rscl = solar.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     wind_rscl = wind.loc[from_date : end_date].resample(f'{str(every_min)}min').apply(lambda x: x[0])
#     load = load.loc[from_date : end_date]
#     snapshots = load.index
#     return wind_rscl, solar_rscl, load_rscl, snapshots

# def interpolate(df, ref_index=None, method='cubic'):
#     # Create dataframe with full index
#     dff = pd.DataFrame(index=ref_index, columns=df.columns)
#     # Replace values in full dataframe
#     dff.loc[df.index, :] = df
#     # Convert them to numeric columns
#     for col in dff:
#         dff[col] = pd.to_numeric(dff[col], errors='coerce')
#     interpolated_df = dff.interpolate(method=method, axis=0)
#     # Force to put zero for very samell values
#     criteria_small_value = 1e-4
#     interpolated_df[interpolated_df < criteria_small_value] = 0
#     return interpolated_df.round(2)

# def generate_prod_voltage(vol, vol_var=1.2, ref_index=None):
#     fill_vol = pd.concat([vol] * ref_index.shape[0], axis=1)
#     prod_v = fill_vol.T
#     prod_v.index = ref_index
#     for col in prod_v:
#         noise = np.random.normal(0, vol_var, prod_v[col].shape[0])
#         prod_v[col] +=noise
#     return prod_v.round(2)

# def generate_reactive_loads(load_p, min_range=12, max_range=25):
#     # Percentage of reactive power 
#     load_q = load_p.copy()
#     q_percen = np.random.randint(min_range, max_range, load_p.shape[1])
#     for i, col in enumerate(load_q):
#         load_q[col] *= q_percen[i] / 100
#     return load_q.round(2)

# def generate_hazard_maintenance(lines_names=None, ref_index=None):
#     hazards = pd.DataFrame(0, index=ref_index, columns=lines_names)
#     maintenance = pd.DataFrame(0, index=ref_index, columns=lines_names)
#     return hazards, maintenance

# def add_noise_forecast(df, noise_factor=0.1):
#     vars_per_col = df.mean(axis=0) * noise_factor
#     for col in df:
#         noise = np.random.normal(0, vars_per_col.loc[col], df.shape[0])
#         df[col] += noise
#     return df.round(2)

# def generate_forecasts(load_p, load_q, prod_p, prod_v, maintenance):
#     load_p_f = load_p.apply(np.roll, shift=1)
#     load_p_ff = add_noise_forecast(load_p_f)
#     load_q_f = load_q.apply(np.roll, shift=1)
#     load_q_ff = add_noise_forecast(load_q_f, noise_factor=0.05)
#     prod_p_f = prod_p.apply(np.roll, shift=1)
#     prod_p_ff = add_noise_forecast(prod_p_f) 
#     prod_v_f = prod_v.apply(np.roll, shift=1)
#     maintenance_f = maintenance.apply(np.roll, shift=1)
#     return load_p_ff, load_q_ff, prod_p_ff, prod_v_f, maintenance_f

