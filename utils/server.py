import flwr as fl
from utils.dcs import Info
from main_ray import args
from utils.lora import *
from utils.utils import inst_model_info

if(args.wandb):
    import wandb
def start_server(srv_addr, strategy, num_rounds, server_queue):

    if(args.wandb):
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_prj_name,
            # track hyperparameters and run metadata
            config=args
        )

    """Start the server."""
    if server_queue != None:
        server_queue.put(
            fl.server.start_server(
                server_address=srv_addr,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )
        )
    else:
        return fl.server.start_server(
            server_address=srv_addr,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

def build_clients(clients_models,input_shape,num_classes):
    cls_models = []
    for cl in clients_models:
        cls_models.append(
            Info(
                model=cl,
                dataset_name=args.dataset,
                feature_maps=args.feature_maps,
                input_shape=input_shape,
                num_classes=num_classes,
                batchn=args.batchn,
            )
        )
    return cls_models

def build_df_lora_config(protos : Info,lora_alpha,lora_r):
    lora_config = []
    for p in protos:
        target_modules, modules_to_save = get_target_save(inst_model_info(p),p.model)

        lc = LoraInfo(alpha=lora_alpha,
                                r=lora_r,
                                target_modules=target_modules,
                                modules_to_save=modules_to_save)
        lora_config.append(lc)


def build_prototypes(protos,input_shape,num_classes):
    prototypes = []
        
    for p in protos:
        prototypes.append(
            Info(
                model=p,
                dataset_name=args.dataset,
                feature_maps=args.feature_maps,
                input_shape=input_shape,
                num_classes=num_classes,
                batchn=args.batchn,
            )
        )

    return prototypes

def start_client(info, fl_info):
    from client import FlowerClient

    client = FlowerClient(info, fl_info)
    client.start_client()