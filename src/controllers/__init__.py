REGISTRY = {}

from .basic_controller import BasicMAC
from .rnn_controller import RNNMAC
from .dcc_controller import DCCMAC
from .roma_controller import ROMAMAC
from .ldsa_controller import LDSAMAC
from .maven_controller import MavenMAC
from .hsd_controller import HSDMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["rnn_mac"] = RNNMAC
REGISTRY["dcc_mac"] = DCCMAC
REGISTRY["roma_mac"] = ROMAMAC
REGISTRY["ldsa_mac"] = LDSAMAC
REGISTRY["maven_mac"] = MavenMAC
REGISTRY["hsd_mac"] = HSDMAC