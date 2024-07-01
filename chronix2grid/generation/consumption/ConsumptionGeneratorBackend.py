# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from .generate_load import main


class ConsumptionGeneratorBackend:
    """
    Backend that generates load power chronics that represent consumption nodes in the grid, with a spatiotemporal correlated noise model.
    It takes into account specific parameters about the grid, the solar and wind power plants, the regions settings...
    See in documentation *Description of implemented models* for detailed information about the model

    Attributes
    ----------
    out_path: ``str``
        path to output folder for generated chronics
    seed: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys **"dt", "planned_std", "Lx", "Ly", "dx_corr", "dy_corr",
        "temperature_corr", "std_temperature_noise"**
    loads_charac: :class:`pandas.DataFrame`
        data frame with characteristics on load nodes in the simulated grid
    load_config_manager: :class:`chronix2grid.config.LoadsConfigManager`
        config manager used to load specific patterns used for the model (load weekly pattern for residential power consumption)
    write_results: ``bool``
    """
    def __init__(self, out_path, seed, params, loads_charac, load_config_manager, write_results, day_lag=0):
        self.write_results = write_results
        self.load_config_manager = load_config_manager
        self.loads_charac = loads_charac
        self.params = params
        self.seed = seed
        self.out_path = out_path
        self.day_lag = day_lag

    def run(self, load_weekly_pattern=None, return_ref_curve=False, use_legacy=True):
        """
        Runs the generation model in ``chronix2grid.generation.consumption.generate_load`` and writes chronics
        """
        if load_weekly_pattern is None:
            load_weekly_pattern = self.load_config_manager.read_specific()
        return main(self.out_path, self.seed, self.params, self.loads_charac, load_weekly_pattern, self.write_results,
                    day_lag=self.day_lag, return_ref_curve=return_ref_curve, use_legacy=use_legacy)
