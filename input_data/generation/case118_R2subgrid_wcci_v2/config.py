from grid2op.Action import TopologyAndDispatchAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend

config = {
    "backend": PandaPowerBackend,
    "action_class": TopologyAndDispatchAction,
    "observation_class": None,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits":[ 43.3, 205.2, 341.2, 204. , 601.4, 347.1, 319.6, 301.4, 330.3,       274.1, 307.4, 178.2, 354.3, 133.6, 174.9, 152.6,  81.8, 204.3,       561.5, 561.5, 103.7, 179.8, 197.2, 244.3, 164.8, 100.4, 125.7,       278.2, 274. ,  89.9, 352.1, 168.7, 127.5, 158.1,  88.4, 109.1,       148.5, 129.6, 136.1,  86. , 313.2, 207.4, 599.1, 206.8, 233.7,       395.8, 516.7, 656.4, 583. , 583. , 263.1, 222.6, 322.8, 340.6,       305.2, 360.1, 395.8, 274.2, 605.5]
}
