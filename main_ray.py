import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
from json import dumps
from pathlib import Path

from torch import device as torch_device
from torch.utils.data import random_split
from utils.models import do_model_pool
from utils.dataset import import_dataset, do_fl_partitioning, get_df_dist, get_ct_dist
from utils.utils import *
from utils.file_name import gen_filename
from utils.server import *
from args import args
from log import logger, HFILE
from utils.strats import Evaluate,get_model_size,EvaluateLora
from utils.simple_quant import original_msg_size,quant_msg_size
client_lr = args.cl_lr


def fit_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""
    global client_lr
    if args.milestones != 0:
        if server_round in args.milestones:
            client_lr *= args.lr_step
    logger.debug(f"Client lr {client_lr}, round {server_round}")
    config = {
        "epochs": args.cl_epochs,  # number of local epochs
        "batch_size": args.cl_bs,
        "cl_lr": client_lr,
        "cl_momentum": args.cl_mmt,
        "cl_wd":args.cl_wd,
        "server_round": server_round,
        "prate" : args.prate,
    }
    return config


def eval_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""

    config = {
        "server_round": server_round,
    }
    return config

def build_server_info(test_set,knn_set=None):
    return ServerInfo(
        model=args.model,
        dataset_name=args.dataset,
        feature_maps=args.feature_maps,
        input_shape=input_shape,
        num_classes=num_classes,
        batchn=args.batchn,
        test_set=test_set,
        knn_set=knn_set,
        num_clients=args.num_clients,
    )

if __name__ == "__main__":
    saddr = "0.0.0.0:8080"
    processes = []

    pool_size = args.num_clients

    create_all_dirs()

    # Dataset
    train_path, num_classes, input_shape = import_dataset(args.dataset, is_train=True,skip_gen_training=args.skip_gen_training,path_to_data=args.dataset_path)
    test_set = import_dataset(args.dataset, is_train=False,skip_gen_training=args.skip_gen_training,path_to_data=args.dataset_path)

    if args.file_name == "":
        file_name = gen_filename()
        args.file_name = file_name
    else:
        if args.id_exp != "":
            file_name = "exp_" + args.id_exp + "_" + args.file_name
        else:
            file_name = args.file_name

    if args.alpha_inf:
        alpha = float("inf")
    else:
        alpha = args.alpha

    args_dict = vars(args)

    logger.info(f"Starting experiment - {file_name}")
    logger.debug(dumps(args_dict, indent=2), extra=HFILE)

    server_model, clients_models = do_model_pool(model=args.model,pool_size=pool_size)

    fed_dir, class_dst, _ = do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        alpha=alpha,
        num_classes=num_classes,
        val_ratio=args.val_ratio,
        seed=args.seed,        
        is_cinic=args.dataset == "cinic10"
    )

    pclass = [f"{i} : {x}" for i, x in enumerate(class_dst)]
    pclass = "\n".join(pclass)

    logger.info(f"Class distribution : \n{pclass}")

    protos = list(set(clients_models))
    # configure the strategy

    # Common parameters
    kwargs_dict = {
        "fraction_fit": args.samp_rate,
        "fraction_evaluate": 0.0,
        "min_fit_clients": int(pool_size * args.samp_rate),
        "min_evaluate_clients": pool_size,
        "min_available_clients": pool_size,  # All clients should be available
        "initial_parameters": [],
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": eval_config,
        "evaluate_fn": None,
        "drop_random": args.drop_random,
        "fedbn": args.fedbn,
    }

    device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = None
    lora_config = None

    model_size = -1
    trainable_params = 100
    total_nb_params = -1

    if args.strategy == "fedavg":
        from utils.strats import Evaluate
        from strategies.fedavg import FedAvg

        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )
        model_size = original_msg_size(server_model)
        total_nb_params = model_size//4

        if args.checkpoint:
            try:
                from flwr.common import ndarrays_to_parameters

                params = np.load("checkpoint/server.npy", allow_pickle=True)
                kwargs_dict["initial_parameters"] = ndarrays_to_parameters(params)
            except:
                kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)
        else:
            kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)

        kwargs_dict["evaluate_fn"] = Evaluate(server_model, test_set, device)

        del server_model
        strategy = FedAvg(**kwargs_dict)
    elif args.strategy == "fedlora" or  args.strategy == "fedloha":
        from strategies import FedLora
        from utils.lora import inject_low_rank
        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )

        target_modules, modules_to_save,rank_pattern = gen_rank_pattern(server_model,r=args.lora_r,mode=args.lora_ablation_mode,ratio= args.loha_ratio)
        
        lora_config = LoraInfo(alpha=args.lora_alpha,
                               r=args.lora_r,
                               target_modules=target_modules,
                               modules_to_save=modules_to_save,
                               lora_type= args.strategy[3:],
                               rank_pattern=rank_pattern)

        if args.from_pretrained:
            try:
                server_model= load_pretrained(server_model,args.model)
            except:
                pass

        server_model = inject_low_rank(server_model,lora_config)

        _trainable,_total = server_model.get_nb_trainable_parameters()
        total_nb_params = _total
        trainable_params = 100 * _trainable / _total

        if args.apply_quant:
            model_size = quant_msg_size(server_model,bits=args.quant_bits)
        else:
            model_size = original_msg_size(server_model)


        kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)

        # evaluate = get_evaluate_fn(server_model, test_set, device)
        # kwargs_dict["evaluate_fn"] = get_evaluate_fn(server_model, test_set, device)
        kwargs_dict["evaluate_fn"] = EvaluateLora(server_model,lora_config, test_set, device)

        del server_model
        strategy = FedLora(**kwargs_dict)
    elif args.strategy == "fedprox":
        from utils.strats import get_evaluate_fn
        from strategies import FedProx

        server_model = server_model(
            args.feature_maps, input_shape, num_classes, batchn=args.batchn
        )

        kwargs_dict["initial_parameters"] = get_tensor_parameters(server_model,args.fedbn)
        kwargs_dict["evaluate_fn"] = get_evaluate_fn(server_model, test_set, device)
        kwargs_dict.update({"proximal_mu" : args.mu})
        del server_model,kwargs_dict["drop_random"],kwargs_dict["fedbn"]
        strategy = FedProx(**kwargs_dict)
    else:
        logger.error(f"Unknown strategy {args.strategy}")
        exit(-1)


    def client_fn(cid):
        from client import FlowerClient
        info = Info(
            model=clients_models[int(cid)],
            dataset_name=args.dataset,
            feature_maps=args.feature_maps,
            input_shape=input_shape,
            num_classes=num_classes,
            batchn=args.batchn,
            fedbn=args.fedbn,
        )

        if args.only_cpu:
            client_device = "cpu"
        else:
            client_device = "cuda"

        fl_info = FlInfo(
            exp_name=file_name,
            saddr=saddr,
            device=client_device,
            num_rounds=args.num_rounds,
            cid=str(cid),
            fed_dir=Path(fed_dir),
            no_thread=args.no_thread,
            server_model=args.model,
            prune=args.prune,
            prune_srv=args.prune_srv,
            strategy=args.strategy,
            lora_config = lora_config,
            nworkers=args.nworkers,
            apply_quant=args.apply_quant,
            quant_bits=args.quant_bits,
        )

        return FlowerClient(info,fl_info)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}
    client_resources = {
        "num_cpus": 1,
        "num_gpus": args.ray_gpu
    }

    if(args.wandb):
        wandb.init(
            entity=args.entity,
            # set the wandb project where this run will be logged
            project=args.wandb_prj_name,
            # track hyperparameters and run metadata
            config=args
        )

        wandb.config["model_size_bytes"] = model_size
        wandb.config["total_nb_params"] = total_nb_params
        wandb.config["trainable"] = trainable_params

    # start simulation
    hist =  fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    tell_history(hist, file_name, infos=args_dict, path=args.path_results)

    logger.info("The End")
