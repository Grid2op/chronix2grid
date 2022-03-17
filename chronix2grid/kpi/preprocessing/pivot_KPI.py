# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import os
import sys
import json

from .pivot_utils import chronics_to_kpi, renewableninja_to_kpi, eco2mix_to_kpi_regional, nrel_to_kpi, usa_gan_trainingset_to_kpi


def pivot_format(chronics_folder, kpi_input_folder, year, prods_charac, loads_charac, wind_solar_only, params, case):
    """
    This function contains pivot formatting of reference and synthetic chronics, in a usable format by
    kpi computation object :class: `chronix2grid.kpi.deterministic.kpis.EconomicDispatchValidator`

    Parameters
    ----------
    chronics_folder: ``str``
        path to folder which contains generated chronics
    kpi_input_folder: ``str``
        path to folder of kpi inputs, which contains paramsKPI.json and benchmark folders. paramsKPI.json tells which benchmark to read as reference
    year: ``int``
        year in which results are written
    prods_charac: :class:`pandas.DataFrame`
        characteristics of generators such as Pmax, carrier and region
    loads_charac: :class:`pandas.DataFrame`
        characteristics of loads node such as Pmax, type of demand and region
    wind_solar_only: ``bool``
        True if the generated chronics contain only wind, solar and load chronics, False otherwise
    params: ``dict``
        configuration params computed from params.json, such as timestep or mesh characteristics
    case: ``str``
        identifies the studied case for chronics generation, such as l2rpn_118

    Returns
    -------
    ref_prod: :class:`pandas.DataFrame`
        preprocessed reference productions per generator
    ref_load: :class:`pandas.DataFrame`
        preprocessed reference consumption per load node
    syn_prod: :class:`pandas.DataFrame`
        preprocessed synthetic productions per generator
    syn_load: :class:`pandas.DataFrame`
        preprocessed synthetic consumption per load node
    ref_price: :class:`pandas.DataFrame` or None
        reference price scenario (only if wind_solar_only is False)
    price: :class:`pandas.DataFrame` or None
        synthetic price scenario (only if wind_solar_only is False)
    paramsKPI: ``dict``
        dictionary with useful settings for KPI generation (e.g. season repartition, night hours)
    """

    # Read json parameters for KPI configuration
    kpi_case_input_folder = os.path.join(kpi_input_folder, case)
    json_filepath = os.path.join(kpi_case_input_folder, 'paramsKPI.json')
    with open(json_filepath, 'r') as json_file:
        paramsKPI = json.load(json_file)
    comparison = paramsKPI['comparison']
    timestep = paramsKPI['timestep']

    ## Format chosen benchmark chronics calling designed pivot functions
    if comparison == 'France':
        corresp_regions = {'R1':"Hauts-de-France", "R2": "Nouvelle-Aquitaine", "R3": "PACA"}
        if wind_solar_only:
            ref_prod, ref_load = renewableninja_to_kpi(kpi_case_input_folder, timestep, loads_charac, prods_charac, year,
                                                       params, corresp_regions, case)
        else:
            ref_prod, ref_load, ref_prices = eco2mix_to_kpi_regional(kpi_case_input_folder, timestep, prods_charac, loads_charac, year, params,
                                                         corresp_regions)
    elif comparison == 'Texas':

        if wind_solar_only:
            ref_prod, ref_load  = nrel_to_kpi(kpi_case_input_folder, timestep, prods_charac, loads_charac, params,year)
        else:
            print("Computation stopped: Texas Benchmark not implemented for a whole energy mix. Launch KPI computation in mode wind solar and load only.")
            sys.exit()
    elif comparison == 'USA - Washington State':
        if wind_solar_only:
            ref_prod, ref_load = usa_gan_trainingset_to_kpi(kpi_case_input_folder, timestep, prods_charac, loads_charac, params,year)
        else:
            print("Computation stopped: USA Benchmark not implemented for a whole energy mix. Launch KPI computation with GAN in mode wind solar and load only.")
            sys.exit()
    else:
        print("Please chose one available benchmark in paramsKPI.json/comparison. Given comparison is: "+comparison)
        sys.exit()

    # Format generated chronics
    if wind_solar_only:
        syn_prod, syn_load = chronics_to_kpi(chronics_folder, timestep, params,
                                             thermal=not wind_solar_only)
        return ref_prod, ref_load, syn_prod, syn_load, paramsKPI
    else:
        syn_prod, syn_load, prices = chronics_to_kpi(
            chronics_folder, timestep, params, thermal=not wind_solar_only)
        return ref_prod, ref_load, syn_prod, syn_load, ref_prices, prices, paramsKPI


