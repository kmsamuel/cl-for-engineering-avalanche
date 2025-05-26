from .agem import AGEMPlugin, AGEMPluginRAADL
from .cwr_star import CWRStarPlugin
from .evaluation import EvaluationPlugin
from .ewc import EWCPlugin, EWCPluginRAADL
from .gdumb import GDumbPlugin
from .gem import GEMPlugin, GEMPluginRAADL, GEMPluginDRIVAERNET
from .lwf import LwFPlugin, LwFPluginRAADL
from .replay import ReplayPlugin
from .replay_tracker import ReplayTracker
from .strategy_plugin import SupervisedPlugin
from .synaptic_intelligence import SynapticIntelligencePlugin
from .gss_greedy import GSS_greedyPlugin
from .cope import CoPEPlugin, PPPloss
from .lfl import LFLPlugin
from .early_stopping import EarlyStoppingPlugin
from .lr_scheduling import LRSchedulerPlugin
from .generative_replay import (
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
)
from .rwalk import RWalkPlugin
from .mas import MASPlugin
from .bic import BiCPlugin
from .mir import MIRPlugin, RegressionMIRPlugin
from .from_scratch_training import FromScratchTrainingPlugin
from .rar import RARPlugin
# from .tabpfn_replay import *
# from .tabpfn_adaptive_replay import *
# from .adaptive_replay import *
# from .adaptive_replay_expclusters import *
from .adaptive_replay_plugins import *
from .gdumb_clustering import *
from .update_ncm import *
from .update_fecam import *
from .feature_distillation import *