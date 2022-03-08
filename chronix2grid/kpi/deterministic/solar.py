# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
from pathlib import Path
from matplotlib import pyplot as plt


class SolarKPI:

    def __init__(self, kpi_databanks):
        self.kpi_databanks = kpi_databanks

    def __solar_at_night(self,
                         solar_df,
                         params,
                         aggregated=False
                         ):

        '''
        '''

        if aggregated:
            solar_df = solar_df.sum(axis=1)

        # Extract parameters
        monthly_pattern = params['monthly_pattern']
        hours = params['hours']

        # Add month variable solar df
        solar_df_month = pd.concat([solar_df, self.months], axis=1)

        # Iterate over all season per month to check if solar is
        # present during night hours defined by a deterministic
        # criteria per season.
        season_at_night = {}
        for season, month in monthly_pattern.items():

            # Filter by season (including all possible months avaible in season)
            month_filter = solar_df_month.month.isin(month)
            season_solar = solar_df_month.loc[month_filter, solar_df_month.columns != 'month']

            if not season_solar.empty:
                # Filter by season and hours
                sum_over_season = season_solar.drop(season_solar.between_time(hours[season][0], hours[season][1]).index).sum(axis=0)
                percen_over_season = 100 * sum_over_season / season_solar.sum(axis=0)
                percen_over_season = percen_over_season.round(self.precision)

                # Save results in a dictionary
                season_at_night.update({season: percen_over_season})

        return season_at_night

    def __solar_cloudiness(self,
                           solar_df,
                           cloud_quantile,
                           factor_cloud):

        '''
        '''

        # Using all data, we get a unique x quantile
        # in order to grab a long term value
        solar_q = solar_df.quantile(cloud_quantile)

        # Per day, we are interested to get x quantile only when
        # generators are producing energy
        solar_q_perday = solar_df.replace(0, np.nan).resample('D').quantile(cloud_quantile)

        # Measure cloudiness: we compare the quantile values per
        # day with a long them x quantile truncated it a factor
        cloudiness = solar_q_perday <= (solar_q * factor_cloud)

        # # Add month column to get some particular stats
        # # Add months to solar cloudiness measure df
        # month = cloudiness.index.month.to_frame()
        # month.index = cloudiness.index
        # cloudiness['month'] = month

        # Get in percentage the number of days solar
        # generators have been producing below the factor
        # (We considerer as a cloudiness's day)
        # percen_cloud = 100 * cloudiness.groupby('month').drop('month', axis=1).sum() \
        #                    / cloudiness.groupby('month').drop('month', axis=1).count()
        # percen_cloud = percen_cloud.round(self.precision)

        percen_cloud = 100 * cloudiness.groupby(pd.Grouper(freq='M')).sum() \
                       / cloudiness.groupby(pd.Grouper(freq='M')).count()

        percen_cloud.index = percen_cloud.index.month
        percen_cloud.index.rename('month', inplace=True)

        return percen_cloud.round(self.precision)

    def solar_kpi(self,
                  cloud_quantile=0.95,
                  cond_below_cloud=0.57,
                  **kwargs):

        '''

        Parameters:
        ----------

        moderate_cloud_quantile: Quantile applied to the consequitive
                                 difference solar hisrogram to determine
                                 large spiked due to cloudness.
        severe_cloud_quantile:   Quantile applied to the consequitive
                                 difference solar hisrogram to determine
                                 large spiked due to cloudness.

        **kwargs:

            monthly_pattern: Contains a dictionary to define the month
                                a solar time series shoud follow (e.g summer
                                months should have more production rather than
                                sping).

                        monthly_pattern = {'summer': [6,7,8],
                                           'fall': [9,10,11],
                                           'winter': [12,1,2],
                                           'spring': [2,3,4,5]}

            hours: Defines the hours solar is producing energy to the system
                        per season.

                        hours = {'summer': ('07:00', '20:00'),
                                 'fall': ('08:00', '18:00'),
                                 'winter': ('09:30', '16:30')}

        Returns:
        --------

        corr_solar (pd.DataFrame):  Correlation matrix between reference and synthetic data

        season_night_percen (dict): Dictionary containing season percentage solar units has been
                                    producing energy out of hours.

        percen_moderate (dict):     Dictionary per solar unit that contains the number of times in
                                    percentage and per month a specific solar unit has overcome a pre-defined
                                    moderate qunantile given the ref dispatch.

        percen_severe (dict):       Dictionary per solar unit that contains the number of times in
                                    percentage and per month a specific solar unit has overcome a pre-defined
                                    severe qunantile given the ref dispatch.

        '''

        # Get the solar gen names
        solar_filter = self.prod_charac.type.isin(['solar'])
        solar_names = self.prod_charac.name.loc[solar_filter].values

        # From data, extract only wind time series
        solar_ref = self.ref_dispatch[solar_names]
        solar_syn = self.syn_dispatch[solar_names]

        # First KPI
        # -- + -- +
        # Get correlation matrix (10 x 10)
        corr_solar = self._pairwise_corr_different_dfs(solar_ref, solar_syn)

        # Write its value
        # -- + -- + -- +
        self.output['solar_kpi'] = {'solar_corr': corr_solar.to_dict()}

        # Second KPI
        # -- + -- +
        # Get percentage solar at night

        if not kwargs:
            monthly_pattern = {'summer': [6, 7, 8], 'fall': [9, 10, 11],
                               'winter': [12, 1, 2], 'spring': [2, 3, 4, 5]}

            hours = {'summer': ('07:00', '20:00'),
                     'fall': ('08:00', '18:00'),
                     'winter': ('09:30', '16:30'),
                     'spring': ('08:00', '18:00')}

        # Get percentage solar productions for reference data
        params = {'monthly_pattern': monthly_pattern, 'hours': hours}

        solar_night_ref = self.__solar_at_night(solar_ref, params=params)

        # Get percentage solar productions for synthetic data
        solar_night_syn = self.__solar_at_night(solar_syn, params=params)

        # Write output
        # -- + -- + --
        self.output['solar_kpi'] = {'season_solar_at_night_reference': solar_night_ref,
                                    'season_solar_at_night_synthetic': solar_night_syn,
                                    }

        # Third KPI
        # -- + -- +
        # Measure Cloudiness as counting the number of days
        # per month when the x quantile is below a factor
        # specified by the method.

        cloudiness_ref = self.__solar_cloudiness(solar_ref,
                                                 cloud_quantile=cloud_quantile,
                                                 factor_cloud=cond_below_cloud)

        cloudiness_syn = self.__solar_cloudiness(solar_syn,
                                                 cloud_quantile=cloud_quantile,
                                                 factor_cloud=cond_below_cloud)

        # # Write its value
        # # -- + -- + -- +
        self.output['solar_kpi'] = {'cloudiness_reference': cloudiness_ref.to_dict(),
                                    'cloudiness_synthetic': cloudiness_syn.to_dict()
                                    }

        return corr_solar, solar_night_ref, solar_night_syn, cloudiness_ref, cloudiness_syn
