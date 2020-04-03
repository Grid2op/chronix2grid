import pandas as pd
import numpy as np
import calendar
import os
import sys
import warnings
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

def update_gen_constrains(gen_constraints_user):
    """Generator constraint passed by user
    
    Parameters
    ----------
    gen_constraints_user : dict
        Valid keys p_max_pu, p_min_pu
    
    Returns
    -------
    dict
        Updated gen dict constratins
    """    
    gen_constraints = {'p_max_pu': None, 'p_min_pu': None}
    gen_constraints.update(gen_constraints_user)
    return gen_constraints

def update_params(num, params_user):
    """ Function to update parameters dictionary to run
    Linear OPF
    
    Parameters
    ----------
    num : int
        Shape axis 0 of input data
    params_user : dict
        The dictionary contains the following params:
            snapshots     : temporary date range {YEAR 2007 - non leap} to be 
                            set it to load and gen constraints data
            step_opf_min  : Assuming real-time data comes every 5 minutes, the 
                            ste_opf_min is the skipped minutes opf will run
            mode_opf      : Mode OPF formulates as single optimization problem
            reactive_comp : Factor applied to consumption to compensate reactive
                            part not modelled by linear opf
    Returns
    -------
    dict
        Updated parameters dict
    """  
    # Initialize possible param keys if they are
    # not passed by user
    params={'snapshots': [],
            'step_opf_min': 5,
            'mode_opf': 'day',
            'reactive_comp': 1.025,
    }
    params.update(params_user)
    # Get user params
    snaps = params['snapshots']
    step_opf = int(params['step_opf_min'])
    mode_opf = params['mode_opf']
    # Check some inputs
    if not 60 % step_opf == 0:
        raise RuntimeError("\"step_opf_min\" argument might be (5, 10, 15, 20, 30, 60)")
    if not mode_opf.lower() in ['day', 'week', 'month']:
        raise RuntimeError("Please provide a valid opf mode (day, week, month")
    # Create temporary date range to be load to input data
    if snaps == []:
        snapshots = pd.date_range(start=f'{2007}-01-01', periods=num, freq='5min')
        params.update({'snapshots': snapshots})
    return params

def preprocess_input_data(load, gen_constraints, params):
    """ Functions to modify temporarily input data as following:
           - It checks whether load and gen constraints have same lengt.
           - It sets up index in all dataframes with temp data_range format.
           - It resamples data according to param specified -step_opf_min-
           - It modify load df aggregating all loads in unique col named as agg_laod. 
    
    Parameters
    ----------
    load : dataframe
        Consumption to be filled in opf
    gen_constraints : dict
        Dictionary holding gen constraints:
           - p_max_pu: df with all max gen constraints
           - p_min_pu: df with all min gen constraints 
    params : dict
        OPF parameters
    
    Returns
    -------
    df, dict
        Updated input data 
    """    
    # Check lenght between load and gen constraints
    for k in gen_constraints:
        if gen_constraints[k] is not None and gen_constraints[k].shape[0] != load.shape[0]:
            raise RuntimeError(f'Gen constraint in [{k}] does not have same lenght as input data')
    # Get load after adapting to step_opf_min  
    new_load = reformat_load(load, params)
    # Get gen contraints after adapting to step_opf_min
    new_gen_const = reformat_gen_constraints(gen_constraints, params, new_load.index)
    return new_load, new_gen_const
    
def reformat_load(load, params):
    """ Reformat Load input data:
          - Set date_range as index to load df
          - Resamples according to param -step_opf_min-
    
    Parameters
    ----------
    load : dataframe
        Consumption to be filled in opf
    params : dict
        OPF parameters
    
    Returns
    -------
    df
    """  
    if not isinstance(load, pd.DataFrame): 
        raise RuntimeError(f'Load must be a dataframe - > Error!! {type(load)}')
    snapshots, step_opf, q_compensation = params['snapshots'], params['step_opf_min'], params['reactive_comp']
    # Set temp index to load
    load.index = snapshots
    # Resample data and scale to compensate reactive part
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
    """
          - Set date_range as index to all dfs in gen constraints dictionary
          - Resamples according to param -step_opf_min-
    
    Parameters
    ----------
    gen_constraints : dict
        Dictionary holding gen constraints:
           - p_max_pu: df with all max gen constraints
           - p_min_pu: df with all min gen constraints 
    params : dict
        OPF parameters
    new_snaps : Datetime
        Resampled index extracted from input df
    
    Returns
    -------
    dict
        Updated gen dict constraints
    """    
    snapshots = params['snapshots']
    resampled_snapshots = new_snaps
    # Set index in gen contraints 
    for k in gen_constraints:
        if gen_constraints[k] is None:  # If no data is passed. An empty df is created
            gen_constraints.update({k: pd.DataFrame(index=snapshots)})
        else:  # Otherwise it will set up new indexes
            if not isinstance(gen_constraints[k], pd.DataFrame): 
                raise RuntimeError(f'Gen constraints {k} must be a dataframe - > Error!! {type(gen_constraints[k])}')
            gen_constraints[k].index = snapshots
            gen_constraints.update({k: gen_constraints[k]})
        # Finally a resampled df is updated in dict
        gen_constraints.update({k: gen_constraints[k].loc[resampled_snapshots]})
    return gen_constraints

def preprocess_net(net, every_min, input_data_resolution=5):
    """ Function that mainly performs the following tasks:
            - Remove all previous loads modelled in net
              and create a new single load named as agg_load.
            - Adapts the ramps up/down according to the configured
              time to run OPF.
            - Change set up commitable for all gen as False
    
    Parameters
    ----------
    net : PyPSA instance
    every_min : int 
        [description]
    input_data_resolution : int, optional
        [description], by default 5
    
    Returns
    -------
    PyPSA instance
        Modified PyPSA grid
    """      
    # Remove all loads modelled in PyPSA
    # and create one single agg_load.
    net.mremove('Load', names=net.loads.index)  
    net.add('Load', name='agg_load', bus=net.buses.index.tolist()[0])
    # Adapt ramps according to the skipping time
    # configured to run the OPF. 
    steps = every_min / input_data_resolution
    net.generators.loc[:, ['ramp_limit_up', 'ramp_limit_down']] *= steps
    # Change commitable as false
    net.generators.commitable = False
    return net

def get_grouped_snapshots(snapshot, mode): 
    """ Get grouped snapshots per opf mode
    
    Parameters
    ----------
    snapshot : datetime
    mode : str
        [day, week, month]
    """    
    # Define all posibilities mode
    periods = {'day': snapshot.groupby(snapshot.day).values(),
               'week': snapshot.groupby(snapshot.week).values(),
               'month': snapshot.groupby(snapshot.month).values()
    }
    return periods[mode]

def run_opf(net, demand, gen_max, gen_min, params):
    """ Run linear OPF problem in PyPSA considering
    only marginal costs and ramps as LP problem.
    
    Parameters
    ----------
    net : PyPSA instance
    demand : dataframe
        Load to be filled
    gen_max : dataframe
        Generator max constraints in pu
    gen_min : dataframe
        Generator min constraints in pu
    params : dict
        OPF set up parameters
    
    Returns
    -------
    dataframe
        Results of OPF dispatch
    """    
    
    to_disp = {'day': demand.index.day.unique().values[0],
               'week': demand.index.week.unique().values[0],
               'month': demand.index.month.unique().values[0],
    }
    mode = params['mode_opf']
    print(f'\n--> OPF formulation by => {mode} - Analyzing {mode} # {to_disp[mode]}')
    # Reset information previously 
    # saved it in PyPSA instance
    net.loads_t.p_set = net.loads_t.p_set.iloc[0:0, 0:0]
    net.generators_t.p_max_pu = net.generators_t.p_max_pu.iloc[0:0, 0:0]
    net.generators_t.p_min_pu = net.generators_t.p_min_pu.iloc[0:0, 0:0]
    # Set snapshots
    net.set_snapshots(demand.index)
    # ++  ++  ++  ++  ++  ++  ++  ++  ++  ++  ++ 
    # Fill load and gen constraints to PyPSA grid
    net.loads_t.p_set = pd.concat([demand])
    net.generators_t.p_max_pu = pd.concat([gen_max], axis=1)
    net.generators_t.p_min_pu = pd.concat([gen_min], axis=1)
    # ++  ++  ++  ++
    # Run Linear OPF
    rel = net.lopf(net.snapshots, pyomo=False, solver_name='cbc')
    if rel[1] != 'optimal': 
        print ('** OPF failed to find a solution **')
        sys.exit()
    return net.generators_t.p.copy()

def add_noise_gen(dispatch, gen_cap, noise_factor):
    """ Add noise to opf dispatch to have more
    realistic real-time data
    
    Parameters
    ----------
    dispatch : dataframe
        Opf PyPSA output
    gen_cap : dataframe
        Maximun capacity for gen
    noise_factor : float
        Noise factor applied to every gen col
    
    Returns
    -------
    dataframe
        Distpach with noise
    """    
    variance_per_col = gen_cap * noise_factor
    for col in dispatch:
        # Check for values greater than zero 
        # (means unit has been distpached)
        only_dispatched_steps = dispatch[col][dispatch[col] > 0]
        noise = np.random.normal(0, variance_per_col.loc[col], only_dispatched_steps.shape[0])
        dispatch.loc[only_dispatched_steps.index, col] = only_dispatched_steps + noise
    return dispatch.round(2)

def interpolate_dispatch(dispatch, method='quadratic'):
    """Function to interpolate in case opf in running for 
    steps greater than 5 min.
    
    Parameters
    ----------
    dispatch : dataframe
        OPF dispatch result
    method : str, optional
        Pandas interpolation approach, by default 'quadratic'
    
    Returns
    -------
    dataframe
        Interpolated dispatch results
    """    
    # Create dataframe with full index
    dff = dispatch.copy().asfreq('5min')
    # Convert them to numeric columns
    for col in dff:
        dff[col] = pd.to_numeric(dff[col], errors='coerce')
    interpolated_df = dff.interpolate(method=method, axis=0)
    # Force to put zero for very samell values
    criteria_small_value = 1e-4
    interpolated_df[interpolated_df < criteria_small_value] = 0
    return interpolated_df.round(2)