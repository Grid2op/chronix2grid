# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

# Python built-in modules
import warnings
import os
import json

# Chronix2grid modules
from .preprocessing.pivot_KPI import pivot_format
from .deterministic.kpis import EconomicDispatchValidator
from ..generation import generation_utils as gu
from .. import constants as cst
from .. import utils as ut


def main(kpi_input_folder, generation_output_folder, scenario_names,
         kpi_output_folder, year, case, n_scenarios, wind_solar_only, params,
         loads_charac, prods_charac, scenario_id=None):
    """
        This is the main function for KPI computation. It formats synthetic and reference chronics and then computes KPIs on it with 2 modes (wind solar and load only or full energy mix). It saves plots in png and html files. It writes a json output

        Parameters
        ----------
        kpi_input_folder (str): path to folder of kpi inputs, which contains paramsKPI.json and benchmark folders. paramsKPI.json tells which benchmark to read as reference
        generation_input_folder (str): path to the input folder of generation module (to read chronics before dispatch)
        generation_output_folder (str): path to the output folder of generation module (to read chronics after dispatch)
        images_repo (str): path to the images folder in which KPI plots will be saved
        year (int): year in which chronics have been generated
        case (str): identify the studied case for chronics generation, such as l2rpn_118
        n_scenario (int): number of scenarios to consider (KPIs are computed for each scenario succesively)
        wind_solar_only (boolean): True if the generated chronics contain only wind, solar and load chronics, False otherwise
        params (dict): configuration params computed from params.json, such as timestep or mesh characteristics
        loads_charac (pandas.DataFrame): characteristics of loads node such as Pmax, type of demand and region
        prods_charac (pandas.DataFrame): characteristics of generators such as Pmax, carrier and region

    """

    ut.check_scenario(n_scenarios, scenario_id)

    print('=====================================================================================================================================')
    print('================================================= KPI GENERATION  ===================================================================')
    print('=====================================================================================================================================')

    warnings.filterwarnings("ignore")


    # Create single zone if no zone is given
    if 'zone' not in prods_charac.columns:
        prods_charac['zone'] = 'R1'
        loads_charac['zone'] = 'R1'

    # Format and compute KPI for each scenario
    for scenario_num in range(n_scenarios):
        if n_scenarios > 1:
            scenario_name = scenario_names(scenario_num)
        else:
            scenario_name = scenario_names(scenario_id)
        print(scenario_name+'...')
        scenario_generation_output_folder = os.path.join(
            generation_output_folder, scenario_name
        )
        scenario_image_folder = os.path.join(
            kpi_output_folder, scenario_name, cst.KPI_IMAGES_FOLDER_NAME
        )
        # Return Warning if KPIs are not computed on full year. Yet, the computation will work
        if params['weeks'] != 52:
            print('Warning: KPI are incomplete. Computation has been made on '+str(params['weeks'])+' weeks, but are meant to be computed on 52 weeks')

        # Read reference and synthetic chronics, but also KPI configuration, in pivot format. 2 modes: with or without full dispatch
        if wind_solar_only:
            # Get reference and synthetic dispatch and loads
            (ref_dispatch, ref_consumption, syn_dispatch, syn_consumption,
             paramsKPI) = pivot_format(
                scenario_generation_output_folder, kpi_input_folder, year,
                prods_charac, loads_charac, wind_solar_only,
                params, case)
            ref_prices = None
            prices = None
        else:
            # Get reference and synthetic dispatch and loads
            (ref_dispatch, ref_consumption, syn_dispatch, syn_consumption,
             ref_prices, prices, paramsKPI) = pivot_format(
                scenario_generation_output_folder, kpi_input_folder, year,
                prods_charac, loads_charac, wind_solar_only,
                params, case)

        ## Start and Run Economic dispatch validator
        # -- + -- + -- + -- + -- + -- + --
        print ('(1) Computing KPI\'s...')
        dispatch_validator = EconomicDispatchValidator(ref_consumption,
                                                       syn_consumption,
                                                       ref_dispatch,
                                                       syn_dispatch,
                                                       year,
                                                       scenario_image_folder,
                                                       prods_charac=prods_charac,
                                                       loads_charac=loads_charac,
                                                       ref_prices=ref_prices,
                                                       syn_prices=prices)


        # Compute dispatch temporal view
        if wind_solar_only:
            max_col = 1
        else:
            max_col = 2
        dispatch_validator.plot_carriers_pw(curve='reference', stacked=True, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)
        dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=True, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)

        dispatch_validator.plot_carriers_pw(curve='reference', stacked=False, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)
        dispatch_validator.plot_carriers_pw(curve='synthetic', stacked=False, max_col_splot=max_col, save_html=True,
                                            wind_solar_only=wind_solar_only)

        # Get Load KPI
        dispatch_validator.load_kpi()

        # Get Wind KPI
        dispatch_validator.wind_kpi()

        # Get Solar KPI
        cloud_quantile = float(paramsKPI['cloudiness_quantile'])
        cond_below_cloud = float(paramsKPI['cloudiness_factor'])
        hours = paramsKPI["night_hours"]
        monthly_pattern = paramsKPI["seasons"]
        dispatch_validator.solar_kpi(cloud_quantile=cloud_quantile, cond_below_cloud=cond_below_cloud,
                                     monthly_pattern=monthly_pattern, hours=hours)

        # Wind - Solar KPI
        dispatch_validator.wind_load_kpi()

        # These KPI only if dispatch has been made
        if not wind_solar_only:
            # Get Energy Mix
            dispatch_validator.energy_mix()

            # Get Hydro KPI
            dispatch_validator.hydro_kpi()

            # Get Nuclear KPI
            dispatch_validator.nuclear_kpi()

            # Get Thermal KPI
            dispatch_validator.thermal_kpi()
            dispatch_validator.thermal_load_kpi()


        # Write json output file
        # -- + -- + -- + -- + --
        print ('(2) Generating json output file...')

        kpi_scenario_output_folder = os.path.join(
            kpi_output_folder, scenario_name)
        ec_validator_path = os.path.join(
            kpi_scenario_output_folder, 'ec_validator_output.json')
        with open(ec_validator_path, 'w') as json_f:
            json.dump(dispatch_validator.output, json_f)

        print ('-Done-\n')
