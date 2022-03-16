# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

##############################################
################ BACKENDS ####################
##############################################

from chronix2grid.generation.consumption.ConsumptionGeneratorBackend import ConsumptionGeneratorBackend
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend
from chronix2grid.generation.loss.LossBackend import LossBackend
from chronix2grid.generation.dispatch.DispatchBackend import DispatchBackend
from chronix2grid.config import GeneralConfigManager, LoadsConfigManager, ResConfigManager, LossConfigManager, DispatchConfigManager

GENERAL_CONFIG = GeneralConfigManager

#### LOAD (L) ####
LOAD_GENERATION_CONFIG = LoadsConfigManager
LOAD_GENERATION_BACKEND = ConsumptionGeneratorBackend

#### RENEWABLE - SOLAR AND WIND (R) ####
# from RenewableGANBackend.config import ResConfigManagerGan
RENEWABLE_GENERATION_CONFIG = ResConfigManager #ResConfigManagerGan #ResConfigManager
# from RenewableGANBackend.backend import RenewableBackendGAN
RENEWABLE_GENERATION_BACKEND = RenewableBackend #RenewableBackendGAN # RenewableBackend

#### LOSS (D) ####
LOSS_GENERATION_CONFIG = LossConfigManager
LOSS_GENERATION_BACKEND = LossBackend

#### DISPATCH - HYDRO, THERMAL, NUCLEAR (T) ####
DISPATCH_GENERATION_CONFIG = DispatchConfigManager
HYDRO_GENERATION_BACKEND = None

from chronix2grid.generation.dispatch.PypsaDispatchBackend import PypsaDispatcher
DISPATCHER = PypsaDispatcher
DISPATCH_GENERATION_BACKEND = DispatchBackend

#### KPI (K) ####
RENEWABLE_NINJA_REFERENCE_FOLDER = 'renewable_ninja'
GAN_TRAINING_SET_REFERENCE_FOLDER = 'GAN_training_data'
