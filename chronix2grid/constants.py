"""
This file gathers constants related to directories and files that chronix2grid
reads/writes. It also defines the ConfigManager and the Backend used for each step of generation process (LRDT)
"""
from chronix2grid.generation.consumption.ConsumptionGeneratorBackend import ConsumptionGeneratorBackend
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend, RenewableBackendGAN
from chronix2grid.generation.loss.LossBackend import LossBackend
from chronix2grid.generation.dispatch.DispatchBackend import DispatchBackend
from chronix2grid.config import GeneralConfigManager, LoadsConfigManager, ResConfigManager, ResConfigManagerGan, LossConfigManager, DispatchConfigManager

GENERATION_FOLDER_NAME = 'generation'
KPI_FOLDER_NAME = 'kpi'
KPI_IMAGES_FOLDER_NAME = 'images'

DEFAULT_OUTPUT_FOLDER_NAME = 'output'
DEFAULT_INPUT_FOLDER_NAME = 'input'

SCENARIO_FOLDER_BASE_NAME = 'Scenario'

SEEDS_FILE_NAME = 'seeds_info.json'

FLOATING_POINT_PRECISION_FORMAT = '%.1f'

TIME_STEP_FILE_NAME = 'time_interval.info'

RENEWABLE_NINJA_REFERENCE_FOLDER = 'renewable_ninja'

GAN_TRAINING_SET_REFERENCE_FOLDER = 'GAN_training_data'

REFERENCE_ZONE = 'France'

GRID_FILENAME = 'grid.json'

GENERAL_CONFIG = GeneralConfigManager
RENEWABLE_GENERATION_CONFIG = ResConfigManagerGan #ResConfigManagerGan #ResConfigManager
LOAD_GENERATION_CONFIG = LoadsConfigManager
LOSS_GENERATION_CONFIG = LossConfigManager
DISPATCH_GENERATION_CONFIG = DispatchConfigManager

LOAD_GENERATION_BACKEND = ConsumptionGeneratorBackend
RENEWABLE_GENERATION_BACKEND = RenewableBackendGAN #RenewableBackendGAN # RenewableBackend
LOSS_GENERATION_BACKEND = LossBackend
DISPATCH_GENERATION_BACKEND = DispatchBackend
HYDRO_GENERATION_BACKEND = None
