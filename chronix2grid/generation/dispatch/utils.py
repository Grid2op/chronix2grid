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