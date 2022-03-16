# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

# Call generation scripts n_scenario times with dedicated random seeds
from chronix2grid.GeneratorBackend import GeneratorBackend


def main(generator: GeneratorBackend, case, n_scenarios, input_folder, output_folder, scen_names,
         time_params, mode='LRTK', scenario_id=None,
         seed_for_loads=None, seed_for_res=None, seed_for_disp=None):
    """
    Main function for chronics generation. It works with three steps: load generation, renewable generation (solar and wind) and then dispatch computation to get the whole energy mix

    Parameters
    ----------
    generator (GeneratorBackend): implementation class to do the different generation
    case (str): name of case to study (must be a folder within input_folder)
    n_scenarios (int): number of desired scenarios to generate for the same timescale
    params (dict): parameters of generation, as returned by function chronix2grid.generation.generate_chronics.read_configuration
    input_folder (str): path of folder containing inputs
    output_folder (str): path where outputs will be written (intermediate folder case/year/scenario will be used)
    prods_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    loads_charac (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    lines (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    solar_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    load_weekly_pattern (pandas.DataFrame): as returned by function chronix2grid.generation.generate_chronics.read_configuration
    mode (str): options to launch certain parts of the generation process : L load R renewable T thermal

    Returns
    -------

    """
    return generator.run(case, n_scenarios, input_folder, output_folder, scen_names, time_params, mode, scenario_id, seed_for_loads, seed_for_res,
                         seed_for_disp)
