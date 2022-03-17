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


class WindKPI:

    def __init__(self, kpi_databanks):
        self.kpi_databanks = kpi_databanks

    def _pairwise_corr_different_dfs(self, df1, df2):

        n_col_df1 = df1.shape[1]
        n_col_df2 = df2.shape[1]

        tmp_corr = np.zeros((n_col_df1, n_col_df2))
        for i, j in product(range(n_col_df1), range(n_col_df2)):
            tmp_corr[i, j] = df1.iloc[:, i].corr(df2.iloc[:, j])

        corr_wind = pd.DataFrame(tmp_corr, index=df1.columns, columns=df2.columns)

        return corr_wind.round(self.precision)

    def __wind_metric_distrib(self, wind_df):

        '''
        Return:
        -------
        Skewness: measures simmetry
        Kurtosis: measures tailedness
        '''

        # Add month as column to compute stats
        copied_wind_df = wind_df.copy()
        # wind_df_month['month'] = self.months

        # Get Skewness agg per month
        skewness_per_month = copied_wind_df.groupby(pd.Grouper(freq='M')).apply(lambda x: x.skew()).round(2)
        # Set index as month value
        skewness_per_month.index = skewness_per_month.index.month
        skewness_per_month.index.rename('month', inplace=True)

        # Get Kurtosis agg per month
        kurtosis_per_month = copied_wind_df.groupby(pd.Grouper(freq='M')).apply(lambda x: x.kurtosis()).round(2)
        # Set index as month value
        kurtosis_per_month.index = kurtosis_per_month.index.month
        kurtosis_per_month.index.rename('month', inplace=True)

        return skewness_per_month, kurtosis_per_month

    def wind_kpi(self, save_plots=False):

        '''
        Return:
        -------

        fig (plot): Subplot (1x2) containing the aggregated wind production

        corr_wind (pd.dataFrame): Correlation matrix (10x10) between ref data
                                  and synthethic one
        '''

        # First KPI
        # Correlation between wind time series

        # Get the wind gen names
        wind_filter = self.prod_charac.type.isin(['wind'])
        wind_names = self.prod_charac.name[wind_filter]

        # From dispatch, get only wind units
        wind_ref = self.ref_dispatch[wind_names]
        wind_syn = self.syn_dispatch[wind_names]

        # Compute correlation for all elements between both dataframes
        corr_wind = self._pairwise_corr_different_dfs(wind_ref, wind_syn)

        # Write results json output
        # -- + -- + -- + -- + -- +
        self.output['wind_kpi'] = {'corr_wind': corr_wind.to_dict()}

        # Second KPI
        # Measure non linearity of time series
        chaoticness_ref = self.__wind_entropy(wind_ref)
        chaoticness_syn = self.__wind_entropy(wind_syn)

        # Write results
        # -- + -- + --
        self.output['wind_kpi'] = {'non_linearity_reference': chaoticness_ref.to_dict(),
                                   'non_linearity_synthetic': chaoticness_syn.to_dict(),
                                   }

        # Third KPI
        # Meaure the simmetry of wind distribution
        skewness_ref, kurtosis_ref = self.__wind_metric_distrib(wind_ref)
        skewness_syn, kurtosis_syn = self.__wind_metric_distrib(wind_syn)

        # Write results
        # -- + -- + --
        self.output['wind_kpi'] = {'skewness_reference': skewness_ref.to_dict(),
                                   'skewness_synthetic': skewness_syn.to_dict(),
                                   'kurtosis_reference': kurtosis_ref.to_dict(),
                                   'kurtosis_synthetic': kurtosis_syn.to_dict(),
                                   }

        # Four KPI
        # Plot distributions

        # Aggregate time series
        agg_ref_wind = wind_ref.sum(axis=1)
        agg_syn_wind = wind_syn.sum(axis=1)

        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        sns.heatmap(corr_wind, annot=True, linewidths=.5, ax=axes[0])
        sns.distplot(agg_ref_wind, ax=axes[1])
        sns.distplot(agg_syn_wind, ax=axes[2])
        axes[1].set_title('Reference Distribution')
        axes[2].set_title('Synthetic Distribution')

        if save_plots:
            # Save plot as pnd
            extent0 = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent1 = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent2 = axes[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('images/wind_kpi/win_corr_heatmap.png', bbox_inches=extent0.expanded(1.6, 1.6))
            fig.savefig('images/wind_kpi/ref_histogram.png', bbox_inches=extent1.expanded(1.4, 1.4))
            fig.savefig('images/wind_kpi/syn_histogram.png', bbox_inches=extent2.expanded(1.4, 1.4))

        return corr_wind, \
               chaoticness_ref, chaoticness_syn, \
               skewness_ref, skewness_syn, \
               kurtosis_ref, kurtosis_syn

    def wind_load_kpi(self):

        '''
        Return:
        ------

        corr(win, load) Region R1 (pd.DataFrame)
        corr(win, load) Region R2 (pd.DataFrame)
        corr(win, load) Region R3 (pd.DataFrame)
        '''

        # Get the solar gen names
        wind_filter = self.prod_charac.type.isin(['wind'])

        regions = ['R1', 'R2', 'R3']
        corr_rel = []
        for region in regions:
            # Create zone filter for wind and loads
            wind_region_filter = self.prod_charac.zone.isin([region])
            wind_names = self.prod_charac.loc[wind_filter & wind_region_filter]['name']
            loads_names_filter = self.load_charac[self.load_charac.zone == region]['name']

            # Extract only wind units per region
            wind_gens_in_region = self.syn_dispatch[wind_names]
            loads_in_region = self.consumption[loads_names_filter]

            # Compute correlation matrix
            tmp_corr = self._pairwise_corr_different_dfs(wind_gens_in_region, loads_in_region)
            corr_rel.append(tmp_corr)

        # Plot results
        plt.figure(figsize=(18, 4))
        self._plot_heatmap(corr_rel[0],
                           'Correlation Wind Load Region R1',
                           path_png='images/wind_load_kpi/corr_wind_load_R1.png',
                           save_png=False)

        plt.figure(figsize=(18, 4))
        self._plot_heatmap(corr_rel[1],
                           'Correlation Wind Load Region R2',
                           path_png='images/wind_load_kpi/corr_wind_load_R2.png',
                           save_png=False)

        plt.figure(figsize=(12, 3))
        self._plot_heatmap(corr_rel[2],
                           'Correlation Wind Load Region R3',
                           path_png='images/wind_load_kpi/corr_wind_load_R3.png',
                           save_png=False)

        return corr_rel[0], corr_rel[1], corr_rel[2]
