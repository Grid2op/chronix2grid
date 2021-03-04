"""
This file gathers constants related to directories and files that chronix2grid
reads/writes.
"""
from chronix2grid.generation.renewable.RenewableBackend import RenewableBackend, RenewableBackendGAN
from chronix2grid.config import ResConfigManager, ResConfigManagerGan

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

REFERENCE_ZONE = 'France'

GRID_FILENAME = 'grid.json'

RENEWABLE_GENERATION_CONFIG = ResConfigManagerGan #ResConfigManager
RENEWABLE_GENERATION_BACKEND = RenewableBackendGAN # RenewableBackend
