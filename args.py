import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Federated Learning Simulation")

## FL
parser.add_argument("--num_rounds", type=int, default=2,help="number of rounds for a federated learning training")
parser.add_argument("--num_clients", type=int, default=10,help="number of clients")
parser.add_argument("--alpha",type=float, default=100,help="alpha used for LDA")
parser.add_argument("--alpha_inf",action="store_true",help="special flag to use alpha as infinity") # uniform
parser.add_argument("--val_ratio", type=float, default=0.0,help="validationd dataset split")
parser.add_argument("--dataset", type = str, default='cifar10',help="which dataset to use")
parser.add_argument("--samp_rate", type=float, default=1.0,help="client's sample rate")
parser.add_argument("--strategy", type = str, default='feddf',help="which strategy to use")
parser.add_argument("--fedbn", action="store_true",help="fed bn strategy for local batchnorm")
parser.add_argument("--drop_random", action="store_true",help="drop client's results with random guess")
parser.add_argument("--bb_only", action="store_true",help="to aggregate backbone only")
parser.add_argument("--pr_dim",type=int, default=256,help="projection dim for fedct")
parser.add_argument("--ema", type=float,default=0.0,help="ema's decay")
parser.add_argument("--dataset_path", type = str, default='./data',help="dataset path")
parser.add_argument("--from_pretrained", action="store_true",help="secretly badx2")

## Model
parser.add_argument("--model", type = str, default='resnet8',help="model to use (resnet18, resnet20,qresnet12)")
parser.add_argument("--feature_maps", type=int, default=16,help="number of feature maps for the model")
parser.add_argument("--batchn", action="store_true",help="to use batch norm or group norm")

## Client
parser.add_argument("--cl_lr", type=float, default=0.01,help="client's learning rate")
parser.add_argument("--cl_mmt", type=float, default=0.9,help="client's momentum")
parser.add_argument("--cl_wd", type=float, default=0.0,help="client's weight decay")
parser.add_argument("--cl_epochs", type=int, default=1,help="number of local epochs in each client")
parser.add_argument("--cl_bs", type=int, default=16,help="client's batch size")
parser.add_argument("--only_cpu", action="store_true",help="to force the use of only cpu in the client")

## Pruning
parser.add_argument("--prune", action="store_true",help="flag to activate pruning")
parser.add_argument("--prune_srv", action="store_true",help="flag to activate pruning on the server side")
parser.add_argument("--prate", type=float, default=0.1,help="pruning rate")

## Others
parser.add_argument("--milestones", type=int, default=0, nargs='+',
    help="milestones for cl_lr, can be int (then milestones every X epochs) or list. 0 means no milestones")
parser.add_argument("--lr_step", type=float,default=0.1,help="reduction for the learning rate")
parser.add_argument("--path_results", type=str,default="results/",help="folder to save results")
parser.add_argument("--id_exp", type=str,default="",help="id for multiple runs")
parser.add_argument("--no_thread", action="store_true",help="to launch the fit/eval in the current thread")
parser.add_argument("--file_name", type=str,default="",help="experience's name (optional)")
parser.add_argument("--skip_gen_training", action="store_true",help="to skip training pt data")
parser.add_argument("--checkpoint", action="store_true",help="secretly bad")
parser.add_argument("--seed", type=int,default=1234,help="reduction for the learning rate")
parser.add_argument("--freq_checkpoint", type=int,default=9999999,help="reduction for the learning rate")
parser.add_argument("--nworkers", type=int,default=2,help="num workers dataloader")
parser.add_argument("--ray_gpu", type=float, default=0.005,help="ray argument for num_gpus")

##Lora
parser.add_argument("--lora_alpha", type=int,default=16,help="alpha parameter in a lora adapter")
parser.add_argument("--lora_r", type=int,default=16,help="common dimmension in a lora adapter")
parser.add_argument("--lora_keepa", action="store_true",help="during merge, if matrix A should be reseted or not")
parser.add_argument("--lora_merging_period", type=int,default=1,help="period to merge models")
parser.add_argument("--apply_lora", action="store_true",help="experimental, to apply lora to FedDF")
parser.add_argument("--log_a_sim", action="store_true",help="log a similarity between rounds")
parser.add_argument("--loha_ratio", type=float,default=0.0,help="loha compression rate")
parser.add_argument("--lora_ablation_mode", type=int,default=2,help="used for ablations, (0) => freeze all layers and apply lora to conv+linear, "\
                                                                                        "(1) => apply lora to all and unfreeze norms layers "\
                                                                                        "(2) => (Default) apply lora to conv and unfreeze norm+linear layers, "\
                                                                                        "(3) => lora to conv, except first, and unfreeze first conv layer+norm+linear")

## Quant
parser.add_argument("--apply_quant", action="store_true",help="to enable fakequant at client level")
parser.add_argument("--quant_bits", type=int,default=8,help="number of bits for fakequant")

## Wandb
parser.add_argument("--wandb", action="store_true",help="to used wandb")
parser.add_argument("--entity", type=str,help="wandb entity name")
parser.add_argument("--wandb_prj_name", type=str,help="wandb project name")

##FedProx
parser.add_argument("--mu", type=float,default=0.01,help="fedprox proximal factor")

args = parser.parse_args()

Path(args.path_results).mkdir(parents=True,exist_ok=True)

if(args.fedbn):
    args.batchn = args.fedbn # if fedbn, force batchnorm use

if(args.milestones != 0 and len(args.milestones)==1):
    args.milestones = np.arange(2,
        args.num_rounds + 1, args.milestones[0]).tolist()

if(args.path_results[-1]!="/"):
    args.path_results+="/"

# parse input arguments
