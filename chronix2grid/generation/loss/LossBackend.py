# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from .generate_loss import main


class LossBackend:
    """
    Backend that generates loss simply based on a provided yearly pattern.
    The API provides user the ability to use power consumption, wind and solar productions as input for more complex loss modeling

    Attributes
    ----------
    input_folder: ``str``
        base input folder to get parameters and patterns
    scenario_folder_path: ``str``
        path to output folder for generated chronics
    load: :class:`pandas.DataFrame` or ``dict``
        generated load chronics by L mode
    prod_solar: :class:`pandas.DataFrame` or ``dict``
        generated solar chronics by R mode
    prod_wind: :class:`pandas.DataFrame` or ``dict``
        generated wind chronics by R mode
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys "dt" and  "loss_pattern"
    loss_config_manager: :class:`chronix2grid.config.LossConfigManager`
        config manager used to load specific patterns used for the model (yearly explicit loss pattern)
    write_results: ``bool``
    """
    def __init__(self, input_folder, scenario_folder_path,
                                     load, prod_solar, prod_wind,
                                     params, loss_config_manager, write_results = True):
        self.write_results = write_results
        self.loss_config_manager = loss_config_manager
        self.input_folder = input_folder
        self.params = params
        self.scenario_folder_path = scenario_folder_path
        self.load = load
        self.prod_solar = prod_solar
        self.prod_wind = prod_wind

    def run(self):
        """
        Runs the loss generation model in ``chronix2grid.generation.loss.generate_loss`` and writes loss chronics
        """
        self.loss_config_manager.validate_configuration()
        params_loss = self.loss_config_manager.read_configuration()
        return main(self.input_folder, self.scenario_folder_path,
                             self.load, self.prod_solar, self.prod_wind,
                             self.params, params_loss, write_results=self.write_results)
