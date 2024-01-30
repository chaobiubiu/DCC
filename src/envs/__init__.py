from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from .smac import MultiAgentEnv, StarCraft2Env
# from src.smac_plus import ForagingEnv
from .mpe.multiagent.mpe_env import MPEEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "mpe": partial(env_fn, env=MPEEnv)
}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
