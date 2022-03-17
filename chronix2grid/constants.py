# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

"""
This file gathers constants related to directories and files that chronix2grid
reads/writes. It also defines the ConfigManager and the Backend used for each step of generation process (LRDT)

.. note::
    For the moment you can only change the backend for renewable generation and the associated config manager.

    Switch from RenewableBackend to RenewableBackendGAN and from ResConfigManager to ResConfigManagerGan

    Note that this would require tensorflow 1.15.4 which is an optional dependency of chronix2grid
"""

##############################################
################## NAMES #####################
##############################################

GENERATION_FOLDER_NAME = 'generation'
KPI_FOLDER_NAME = 'kpi'
KPI_IMAGES_FOLDER_NAME = 'images'

DEFAULT_OUTPUT_FOLDER_NAME = 'output'
DEFAULT_INPUT_FOLDER_NAME = 'getting_started/example/input'

SCENARIO_FOLDER_BASE_NAME = 'Scenario'

SEEDS_FILE_NAME = 'seeds_info.json'

FLOATING_POINT_PRECISION_FORMAT = '%.1f'

TIME_STEP_FILE_NAME = 'time_interval.info'

REFERENCE_ZONE = 'France'

GRID_FILENAME = 'grid.json'
