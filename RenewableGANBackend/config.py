# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import json
import os

import pandas as pd

from chronix2grid.config import ConfigManager


class ResConfigManagerGan(ConfigManager):
    def __init__(self, name, root_directory, input_directories, output_directory,
                 required_input_files=None):
        super(ResConfigManagerGan, self).__init__(name, root_directory, input_directories,
                                               output_directory, required_input_files)

    def read_configuration(self):
        """
        Reads parameters for :class:`chronix2grid.generation.renewable.RenewableBackendGAN`

            * *model_name_wind* - name of tensorflow serialized model for wind
            * *batch_size_wind*, *n_gens_wind*, *n_timesteps_wind* - dimensions of learning batches of wind networks
            * *n_events_wind* -  number of labels handled as input, representing arbitrary events in learning of wind
            * *dim_input_wind*, *mu_wind*, *sigma_wind* - characteristics of inpus (size of gaussian noise vector, mean and standard deviation)

        ALl the same parameters are expected for solar generation (with suffix *_solar*), that will use the same network architecture but with potentialy différent layout parameters

        Returns
        -------
        params_res: ``dict``
            dictionary of parameters
        prods_charac: :class:`pandas.DataFrame`
        """
        params_file_path = os.path.join(
            self.root_directory,
            self.input_directories['case'], 'params.json')
        network_folder = os.path.join(self.root_directory,
                                      self.input_directories['case'],
                                      'neural_network')
        params_file_path_gan = os.path.join(network_folder, 'paramsGAN.json')

        # Check timestep
        with open(params_file_path, 'r') as params_json:
            params_general = json.load(params_json)
        if params_general["dt"] % 60 != 0:
            raise ValueError('timesteps different from 60min or multiples of 60min are not supported yet with GAN. Please use another model')

        with open(params_file_path_gan, 'r') as params_json:
            params = json.load(params_json)

        for key, value in params.items():
            if key in ["mu_wind","sigma_wind","mu_solar","sigma_solar"]:
                params[key] = float(value)
            elif "model_name" not in key:
                try:
                    params[key] = int(value)
                except ValueError:
                    params[key] = pd.to_datetime(value, format='%Y-%m-%d')

        try:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=',')
            names = prods_charac['name']  # to generate error if separator is wrong

        except:
            prods_charac = pd.read_csv(
                os.path.join(self.root_directory, self.input_directories['case'],
                             'prods_charac.csv'),
                sep=';')
        params["network_folder"] = network_folder

        return params, prods_charac
