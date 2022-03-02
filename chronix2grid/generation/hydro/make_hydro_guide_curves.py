# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

"""
Utility script to generate hydro guid curves (min, max constraint timeseries
for an economic dispatch problem) from an eco2mix excel file.
Beware that the downloaded .xls file has to be first opened in excel and saved
as a working xlsx file.
"""

import argparse
import datetime as dt

import pandas as pd


def q(percentile):
    """
    Generate a parametrized quantile function to apply over a numpy array
    Parameters.
    ----------
    percentile : float
            The percentile to use in the returned quantile function
    Returns
    -------
    The parametrized quantile function
    """
    def q_leveled(x):
        return x.quantile(percentile)

    # Rename the produced function to be able to use several functions in
    # pandas.DataFrame.groupby.agg method
    q_leveled.__name__ = 'q_{}'.format(str(percentile))

    return q_leveled


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=('generate hydro guid curves (min, max constraint timeseries'
                     'for an economic dispatch problem) from an eco2mix excel file.'))
    parser.add_argument('filepath',
                        type=str,
                        help='Full path to the eco2mix excel file')
    parser.add_argument('--output', nargs=1, type=str,
                        default='output.csv',
                        help='File path for the output csv file (default: "output.csv")')

    args = parser.parse_args()

    if args.filepath.endswith('.xls'):
        print('Open first the .xls file in excel and save it as xlsx before' 
              'running the script')
        exit(0)

    # Read the eco2mix excel file and extract relevant data
    eco2mix = pd.read_excel(io=args.filepath)
    eco2mix = eco2mix[['Date', 'Heures', 'Hydraulique']]
    eco2mix = eco2mix.dropna()
    eco2mix.Heures = pd.to_datetime(eco2mix.Heures, format='%H:%M:%S')
    eco2mix['Datetime'] = eco2mix[['Date', 'Heures']].apply(
        lambda row: dt.datetime(row[0].year,
                                row[0].month,
                                row[0].day,
                                row[1].hour,
                                row[1].minute),
        axis=1
    )
    eco2mix = eco2mix[['Datetime', 'Hydraulique']]
    eco2mix = eco2mix.set_index('Datetime')

    # Normalize the data between 0 and 1
    eco2mix['Hydraulique'] = eco2mix['Hydraulique'] / eco2mix['Hydraulique'].max()

    # Recover the year of the eco2mix data
    year = eco2mix.index.year[0]

    # Recover monthly quantiles
    hydro_monthly = eco2mix.groupby(eco2mix.index.month).agg([q(0.05), q(0.95)])
    hydro_monthly.columns = hydro_monthly.columns.droplevel(0)

    output_range = pd.date_range(
        start=dt.datetime(year, 1, 1),
        end=dt.datetime(year, 12, 31, 23, 55),
        freq='5T')

    # Create the output file
    output = pd.DataFrame(index=output_range, columns=['Month'])
    output['Month'] = output.index.month
    output = output.merge(hydro_monthly, left_on='Month', right_index=True)
    output = output.drop('Month', axis=1)
    output = output.merge(eco2mix, left_index=True, right_index=True, how='left')
    output = output.fillna(method='ffill')

    output.to_csv(args.output[0], sep=',', decimal='.')
