import os
import re

import pandas as pd
import numpy as np


def make_scenario_input_output_directories(input_folder, output_folder, scenario_name):
    os.makedirs(os.path.join(input_folder, scenario_name), exist_ok=True)
    os.makedirs(os.path.join(output_folder, scenario_name, 'chronics'), exist_ok=True)
    return os.path.join(input_folder, scenario_name), os.path.join(output_folder, scenario_name, 'chronics')

def compute_random_event(event_type, lines, params):

    T_min = params['T_' +event_type +'_min']
    dt = params['dt']

    Nt_min = int(T_min // dt)
    Nt_inter = int(params['T'] // params['dt'] + 1)

    dict_ = {}
    df_lines_names = lines

    for name in list(df_lines_names):
        dict_[name] = np.zeros(Nt_inter)

    proba = params['daily_proba_' +event_type] / (24*60*len(list(df_lines_names))/dt)

    A = T_min * np.random.binomial(1, proba, size=(len(list(df_lines_names)), Nt_inter))
    dict_ = dict(zip(list(df_lines_names), A))

    # Select line and time stamp
    return dict_

def create_csv(dict_, path, reordering=True, noise=None, shift=False, with_pdb=False):
    df = pd.DataFrame.from_dict(dict_)
    if reordering:
        value = []
        for name in list(df):
            value.append(natural_keys(name))
        new_ordering = [x for _,x in sorted(zip(value,list(df)))]
        df = df[new_ordering]
    if noise is not None:
        df *= (1+noise*np.random.normal(0, 1, df.shape))
    if shift:
        df = df.shift(-1)
        df = df.fillna(0)
    # if with_pdb:
    #     pdb.set_trace()
    # Drop last row toa void problems with Dispatch
    df.drop(df.index[-1], axis=0, inplace=True)
    df.to_csv(path, index=False, sep=';', float_format='%.1f', )


def natural_keys(text):
    return int([ c for c in re.split('(\d+)', text) ][1])