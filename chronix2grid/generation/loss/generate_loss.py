# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import datetime as dt
import pandas as pd
import copy

def main(input_folder, output_folder, load, prod_solar, prod_wind, params, params_loss, write_results = True):
    """
    :param input_folder (str): input folder in which pattern folder can be found
    :param output_folder (str): output folder to write loss chronic
    :param load(pandas.DataFrame): generated load with Chronix2grid module (for other usage)
    :param prod_solar (pandas.DataFrame): generated solar production with Chronix2grid module (for other usage)
    :param prod_wind (pandas.DataFrame): generated wind production with Chronix2grid module (for other usage)
    :param params (dict): dictionary with parameters concerning chronic generations in general
    :param params_loss (dict): dictionary with parameters concerning loss computation
    :param write_results (bool): whether to serialize results or not in scenario_folder_path (True by default)
    :return: pandas.Series representing provided  loss chronic
    """

    loss_pattern_path = os.path.join(input_folder, 'patterns', params_loss["loss_pattern"])
    loss = generate_valid_loss(loss_pattern_path, params)
    if write_results:
        loss.to_csv(os.path.join(output_folder,'loss.csv.bz2'), sep = ';')
    return loss

def generate_valid_loss(loss_pattern_path, params):
    # It is assumed that provided loss_pattern contains the requested time period and time step
    params = copy.deepcopy(params)
    
    # Reading and parsing dates
    dateparse = lambda x: dt.datetime.strptime(x, '%d/%m/%Y %H:%M')
    loss_pattern = pd.read_csv(loss_pattern_path,
                               usecols=[0, 1],
                               parse_dates=[0],
                               date_parser=dateparse,
                               sep = ';')
    loss_pattern.set_index(loss_pattern.columns[0], inplace=True)

    # Extract subset of loss-pattern corresponding to the period studied
    loss_pattern.index = loss_pattern.index.map(lambda t: t.replace(year=params['year']))  # does not work in some un indentified cases
    datetime_index = pd.date_range(
        start=params['start_date'],
        end=params['end_date'],
        freq=str(params['dt']) + 'min')
    loss_pattern = loss_pattern.loc[datetime_index]
    loss_pattern = loss_pattern.head(len(loss_pattern)-1)  # Last value is lonely for another day (same treatment as load and renewable)
    return pd.Series(loss_pattern[loss_pattern.columns[0]])
