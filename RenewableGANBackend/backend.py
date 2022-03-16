# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

class RenewableBackendGAN:
    """
    Backend that generates solar and wind production chronics with a model based on trained neural networks (Generative
    Adversarial Networks or GAN).
    It takes general parameters about generation and specific parameters about how the networks have been trained.
    It uses tensorflow checkpoints in order to load and replay network objects

    .. warning::
        You should install tensorflow 1.15.4 to run this model, which is an optional dependency of chronix2grid

    Attributes
    ----------
    out_path: ``str``
        path to output folder for generated chronics
    seed: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys **"dt", "planned_std", "model_name", "batch_size",
        "n_gens", "n_timesteps", "n_events","dim_inputs", "mu", "sigma", "network_folder"**.
        It has normally been read by a :class:`chronix2grid.config.ResConfigManagerGan` instance
    prods_charac: :class:`pandas.DataFrame`
        data frame with characteristics on wind and solar generators/power plants of the model
    res_config_manager: :class:`chronix2grid.config.ResConfigManagerGan`
    write_results: ``bool``
    """
    def __init__(self, out_path, seed, params, prods_charac, res_config_manager, write_results):
        self.write_results = write_results
        self.prods_charac = prods_charac
        self.res_config_manager = res_config_manager
        self.params = params
        self.seed = seed
        self.out_path = out_path

    def run(self):
        from RenewableGANBackend.generate_solar_wind_gan import main_gan
        """
        Runs the generation model in ``chronix2grid.generation.renewable.generate_solar_wind_gan`` and writes chronics
        """
        return main_gan(self.out_path, self.seed, self.params, self.prods_charac, self.write_results)