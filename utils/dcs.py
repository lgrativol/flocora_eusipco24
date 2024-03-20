from dataclasses import dataclass
from typing import List,Any,Dict
from torch.utils.data import Dataset
# from utils.models import NaiveScheduler

@dataclass
class Info():
    model        : str
    dataset_name : str
    feature_maps : int
    input_shape  : List[int]
    num_classes  : int 
    batchn       : bool = False
    fedbn        : bool = False

    def __hash__(self):
        return hash((self.model))
    
# @dataclass
# class ClInfo(Info):
#     lr_sched : NaiveScheduler = None

@dataclass
class ServerInfo(Info):
    test_set : Dataset = None
    num_clients : int = 0
    num_rounds   : int = 0
    knn_set     : Dataset = None

@dataclass
class KDInfo():
    temp        : float
    epochs      : int
    fed_dir     : str
    batch_size  : int
    alpha       : float
    lr          : float
    wd          : float
    train_set   : Dataset = None

@dataclass
class LoraInfo():
    alpha : int
    r : int
    target_modules : List[str]
    modules_to_save : List[str]
    lora_type : str
    rank_pattern    : Dict[str,int] = None

@dataclass
class FlInfo():
    exp_name     : str
    saddr        : str
    device       : str
    num_rounds   : int
    cid          : str
    fed_dir      : str
    no_thread    : bool
    prune        : bool = False
    prune_srv    : bool = False
    server_model : str = None
    strategy     : str = None
    lora_config  : LoraInfo = None
    nworkers     : int = 2,
    apply_quant  : bool = False,
    quant_bits   : int = 8