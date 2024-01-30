REGISTRY = {}

from .rnn_agent import RNNAgent
from .dcc_low_agent import DCCAgent
from .roma_agent import RomaAgent
from .ldsa_agent import LDSAAgent
from .maven_agent import MavenAgent
from .hsd_agent import HSDAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dcc_agent"] = DCCAgent
REGISTRY["roma_agent"] = RomaAgent
REGISTRY["ldsa_agent"] = LDSAAgent
REGISTRY["maven_agent"] = MavenAgent
REGISTRY["hsd_agent"] = HSDAgent
