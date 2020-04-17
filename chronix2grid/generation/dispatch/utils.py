import os
from enum import Enum


class RampMode(Enum):
    """
    Encodes the level of complexity of the ramp constraints to apply for
    the economic dispatch
    """
    none = -1
    easy = 0
    medium = 1
    hard = 2


def make_scenario_input_output_directories(input_folder, output_folder, scenario_name):
    os.makedirs(os.path.join(input_folder, scenario_name), exist_ok=True)
    os.makedirs(os.path.join(output_folder, scenario_name), exist_ok=True)
    return os.path.join(input_folder, scenario_name), os.path.join(output_folder, scenario_name)
