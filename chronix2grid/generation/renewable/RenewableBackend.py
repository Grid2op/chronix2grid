# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from .generate_solar_wind import main
import inspect

class RenewableBackend:
    """
    Backend that generates solar and wind production chronics with a spatiotemporal correlated noise model.
    It takes into account specific parameters about the grid, the solar and wind power plants, the regional setting...
    See in documentation *Description of implemented models* for detailed information about the model

    Attributes
    ----------
    out_path: ``str``
        path to output folder for generated chronics
    seed: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys **"dt", "planned_std", "Lx", "Ly", "dx_corr", "dy_corr", "short_wind_corr",
        "medium_wind_corr", "long_wind_corr", "solar_corr", "smoothdist", "std_short_wind_noise", "std_solar_noise",
        "std_medium_wind_noise", "std_long_wind_noise", "year_solar_pattern"**
    prods_charac: :class:`pandas.DataFrame`
        data frame with characteristics on wind and solar generators/power plants of the model
    res_config_manager: :class:`chronix2grid.config.ResConfigManager`
        config manager used to load specific patterns used for the model (solar_pattern)
    write_results: ``bool``
    """
    # def __init__(self, out_path, seed, params, loads_charac, res_config_manager, write_results):
    #     self.write_results = write_results
    #     self.res_config_manager = res_config_manager
    #     self.loads_charac = res_config_manager.prods_charac
    #     self.params = params
    #     self.seed = seed
    #     self.out_path = out_path
    def __init__(self, out_path, seed, params, prods_charac, res_config_manager, write_results):
        self.out_path           = out_path
        self.seed               = seed
        self.params             = params
        self.write_results      = write_results
        self.res_config_manager = res_config_manager
        self.prods_charac       = prods_charac

    def run(self,
            solar_pattern=None,
            return_ref_curve=False,
            return_prng=False,
            tol_zero=0.):
        """
        Runs the generation model in ``chronix2grid.generation.renewable.generate_solar_wind`` and writes chronics
        
        
        # NB `tol_zero` is set to 0.1 for legacy behaviour, otherwise tests do not pass, but this is a TERRIBLE idea.
        """
        # if solar_pattern is None:
        #     solar_pattern = self.res_config_manager.read_specific()

        # print(self.loads_charac.head())  # Exibe as zonas e as coordenadas associadas
        # input("Pressione Enter para continuar...")

        # return main(self.out_path, self.seed, self.params, self.loads_charac,
        #             solar_pattern, self.write_results,
        #             return_ref_curve=return_ref_curve,
        #             return_prng=return_prng,
        #             tol_zero=tol_zero)

        use_zonal = bool(self.params.get('use_zonal_solar_pattern', False))

        # legacy mode: load the single .npy pattern as base_solar_pattern
        if not use_zonal:
            if solar_pattern is not None:
                base_solar_pattern = solar_pattern
            else:
                base_solar_pattern = self.res_config_manager.read_specific()
        else:
            # zonal mode: no .npy needed; main() will use per-zone patterns
            base_solar_pattern = None


        return main(
            self.out_path,
            self.seed,
            self.params,
            self.prods_charac,
            base_solar_pattern,
            self.write_results,
            return_ref_curve=return_ref_curve,
            return_prng=return_prng,
            tol_zero=tol_zero
        )
