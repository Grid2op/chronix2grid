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
DEFAULT_INPUT_FOLDER_NAME = 'input'

SCENARIO_FOLDER_BASE_NAME = 'Scenario'

SEEDS_FILE_NAME = 'seeds_info.json'

FLOATING_POINT_PRECISION_FORMAT = '%.1f'

TIME_STEP_FILE_NAME = 'time_interval.info'

REFERENCE_ZONE = 'France'

GRID_FILENAME = 'grid.json'

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
# from chronix2grid.config import ResConfigManagerGan
RENEWABLE_GENERATION_CONFIG = ResConfigManager #ResConfigManagerGan #ResConfigManager
# from chronix2grid.generation.renewable.RenewableBackend import RenewableBackendGAN
RENEWABLE_GENERATION_BACKEND = RenewableBackend #RenewableBackendGAN # RenewableBackend

#### LOSS (D) ####
LOSS_GENERATION_CONFIG = LossConfigManager
LOSS_GENERATION_BACKEND = LossBackend

#### DISPATCH - HYDRO, THERMAL, NUCLEAR (T) ####
DISPATCH_GENERATION_CONFIG = DispatchConfigManager
HYDRO_GENERATION_BACKEND = None
from PypsaDispatchBackend.PypsaEconomicDispatch import PypsaDispatcher
DISPATCHER = PypsaDispatcher
DISPATCH_GENERATION_BACKEND = DispatchBackend

#### KPI (K) ####
RENEWABLE_NINJA_REFERENCE_FOLDER = 'renewable_ninja'
GAN_TRAINING_SET_REFERENCE_FOLDER = 'GAN_training_data'