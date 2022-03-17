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
#from entropy import spectral_entropy

class HydroKPI:

    def __init__(self, kpi_databanks):
        self.kpi_databanks = kpi_databanks

    def __hydro_in_prices(self,
                          norm_mw,
                          upper_quantile,
                          lower_quantile,
                          above_norm_cap,
                          below_norm_cap):

        '''
        '''

        # Get number of gen units
        n_units = norm_mw.shape[1]

        # Get the price at upper/lower quantile
        eu_upper_quantile = self.prices.quantile(upper_quantile)
        eu_lower_quantile = self.prices.quantile(lower_quantile)

        # Test if units are above/below normalized capacity
        is_gens_above_cap =  norm_mw.ge(eu_upper_quantile, axis=1)
        is_gens_below_cap =  norm_mw.le(eu_lower_quantile, axis=1)

        # Test if prices are greater/lower than defined quantiles
        is_price_above_q = self.prices > eu_upper_quantile
        is_price_below_q = self.prices < eu_lower_quantile

        # Stacking price bool condition for all hydro units along
        # the columns and rename them as units names
        is_price_above_q_gens = is_price_above_q[['price'] * n_units]
        is_price_above_q_gens.columns = norm_mw.columns

        is_price_below_q_gens = is_price_below_q[['price'] * n_units]
        is_price_below_q_gens.columns = norm_mw.columns

        # Match occurence of price high and full gen disaptch
        high_price_kpi = 100 * (is_price_above_q_gens & is_gens_above_cap).sum(axis=0) \
                         / is_price_above_q_gens.sum(axis=0)

        # Match occurence when price is lower and almost no dispatch
        low_price_kpi = 100 * (is_price_below_q_gens & is_gens_below_cap).sum(axis=0) \
                        / is_price_below_q_gens.sum(axis=0)

        return high_price_kpi.round(self.precision), low_price_kpi.round(self.precision)

    def __hydro_seasonal(self, hydro_mw):

        '''
        '''

        # We aggregate hydro per month if a user want to deliver less MW in some
        # months rather than others.
        #
        # E.g. Typical configuration:
        # 6 month full capacity and
        # other months with 0 MW)
        hydro_mw_month = hydro_mw.copy()
        hydro_mw_month['month'] = self.months
        mw_per_month = hydro_mw_month.groupby('month').sum(axis=0).round(self.precision)

        return mw_per_month

    def hydro_kpi(self,
                  upper_quantile = 0.95,
                  lower_quantile = 0.05,
                  above_norm_cap = 0.9,
                  below_norm_cap = 0.1):

        '''
        Get 4 different Hydro KPI's based on the assumption the total costs
        of the system follow same curve as the consumption.

        Parameters:
        ----------

        upper_quantile (float): Quantile that define high prices.
                                (Prices are considered high whether
                                price(t) is greater than upper_quantile)
        lower_quantile (float): Quantile that define lower prices
                                (Prices are considered low whether
                                price(t) is less than lower_quantile)

        above_cap (float): Per unit (<1) criteria to establish high hydro dispatch
        below_cap (float): Per unit (<1) criteria to establish low hydro dispatch

        Returns:
        --------

        highPrice_kpi (dataframe): Percentage of time a generator is keeping
                                   operating above its predefined capacity

        lowPrice_kpi (dataframe): Percentage of time a generator is keeping
                                   operating below its predefined capacity

        mw_per_month (dataframe): Aggregated sum per month. Used to design
                                  seasonal pattern in hydro plants.
        '''

        # Get Hydro names
        hydro_filter = self.prod_charac.type.isin(['hydro'])
        hydro_names = self.prod_charac.name.loc[hydro_filter].values

        # Normalize MW according to the max value for
        # the reference data and synthetic one
        hydro_ref = self.ref_dispatch[hydro_names]
        hydro_syn = self.syn_dispatch[hydro_names]

        max_mw_ref = hydro_ref.max(axis=0)
        max_mw_syn = hydro_syn.max(axis=0)

        norm_mw_ref = hydro_ref / max_mw_ref
        norm_mw_syn = hydro_syn / max_mw_syn

        # Stats for reference data
        stat_ref_high_price, stat_ref_low_price = self.__hydro_in_prices(norm_mw_ref,
                                                                         upper_quantile,
                                                                         lower_quantile,
                                                                         above_norm_cap,
                                                                         below_norm_cap)

        # Stats for synthetic data
        stat_syn_high_price, stat_syn_low_price = self.__hydro_in_prices(norm_mw_syn,
                                                                         upper_quantile,
                                                                         lower_quantile,
                                                                         above_norm_cap,
                                                                         below_norm_cap)

        # Write results
        # -- + -- + --
        self.output['Hydro'] = {'high_price_for_ref': stat_ref_high_price.to_dict(),
                                'low_price_for_ref': stat_ref_low_price.to_dict(),
                                'high_price_for_syn': stat_syn_high_price.to_dict(),
                                'low_price_for_syn': stat_syn_low_price.to_dict()
                                }

        # Seasonal for reference data
        ref_agg_mw_per_month = self.__hydro_seasonal(hydro_ref)

        # Seasonal for synthetic data
        syn_agg_mw_per_month = self.__hydro_seasonal(hydro_syn)

        # Write results
        # -- + -- + --
        self.output['Hydro'] = {'seasonal_month_for_ref': ref_agg_mw_per_month.to_dict(),
                                'seasonal_month_for_syn': syn_agg_mw_per_month.to_dict()
                                }

        return stat_ref_high_price, stat_ref_low_price, ref_agg_mw_per_month, \
               stat_syn_high_price, stat_syn_low_price, syn_agg_mw_per_month
