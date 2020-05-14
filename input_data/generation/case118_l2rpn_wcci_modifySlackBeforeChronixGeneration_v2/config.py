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
    "thermal_limits": [  44.5,  206.6,  336.7,  204.8,  569. ,  332.8,  304.6,  274.8,
        322.4,  265.7,  245.1,  177.3,  333.6,  134.7,  182.4,  177.5,
        102.2,  236. ,  205.5,  205.5,  137.7,  198.7,  137.1,  170.1,
        122.1,   73.8,  149.4,  203.4,  200.6,  138.3,  691. ,  187.6,
         87.2,  108.1,   94.2,   72.2,  142. ,  121.7,  126.7,   88.8,
        279.3,  310.8,  572.5,  183.7,  240. ,  255.6,  486.1,  577.8,
        603.4,  603.4,  245.3,  205. ,  294.4,  332.9,  299.6,  941.8,
        613.5,  606.8, 1396. ]
}
