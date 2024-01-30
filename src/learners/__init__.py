from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .rnn_learner import RNNQLearner
from .dcc_learner import DCCLearner
from .roma_learner import ROMALearner
from .macc_learner import MACCLearner
from .ldsa_learner import LDSALearner
from .maven_learner import MavenLearner
from .hsd_learner import HSDLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["rnn_q_learner"] = RNNQLearner
REGISTRY["dcc_learner"] = DCCLearner
REGISTRY["roma_learner"] = ROMALearner
REGISTRY["macc_learner"] = MACCLearner
REGISTRY["ldsa_learner"] = LDSALearner
REGISTRY["maven_learner"] = MavenLearner
REGISTRY["hsd_learner"] = HSDLearner
