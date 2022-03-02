# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

from .generate_dispatch import main


class DispatchBackend: # TODO - PypsaDispatchBackend - devra créer un PypsaDispatcher et l'utiliser. OU ALORS dans les constantes de Chronix2grid choisir la classe de Dispatcher
    """
    Backend that generates nuclear, hydro and thermal from consumption and renewable production constraints by
    computing an optimized economic dispatch.
    Constraints on generators and consumption, solar, wind and loss chronics are provided
    via an instance of :class:`chronix2grid.dispatch.EconomicDispatch.Dispatcher` object

    Attributes
    ----------
    dispatcher: :class:`chronix2grid.dispatch.EconomicDispatch.Dispatcher`
        Object that contains grid and chronics features that represents the dispatch constraints and environment
    scenario_folder_path: ``str``
        path to output folder for generated chronics
    grid_folder: ``str``
        folder in which grid info are stored
    seed_disp: ``int``
    params: ``dict``
        dictionnary with the model parameters. It needs to contain keys **"dt", "planned_std"**
    params_opf: ``dict``
        dictionnary with specific parameters concerning the dispatch optimization (Optimal Power Flow computation)
    """
    def __init__(self,
                 dispatcher,
                 scenario_folder_path,
                 grid_folder,
                 seed_disp,
                 params,
                 params_opf):
        self.dispatcher = dispatcher
        self.params = params
        self.params_opf = params_opf
        self.seed_disp = seed_disp
        self.scenario_folder_path = scenario_folder_path
        self.grid_folder = grid_folder

    def run(self):
        """
        Runs Economic dispatch as in ``chronix2grid.dispatch.generate_dispatch``

        .. warning::
            The dispatch optimization can rely on pypsa simulation. If it is the case you should ensure pypsa dependencies are installed

        .. note::
            As a final step, loss can be simulated thanks to grid2op. It is achieved if "loss_grid2op_simulation" is True in params_opf.
            You should then provide keys **"idxSlack","nameSlack","early_stopping_mode","pmin_margin","pmax_margin","rampup_margin",
            "rampdown_margin","agent_type"**

        """
        return main(self.dispatcher, self.scenario_folder_path, self.scenario_folder_path,
                    self.grid_folder, self.seed_disp, self.params, self.params_opf)
