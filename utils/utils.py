import torch
import numpy as np
from collections import OrderedDict
from utils.models import model_selection
from utils.dcs import *
from models.projector import Project
import math
from functools import reduce

SCALING_FACTOR = 1.2

def get_random_guess_perf(dataset):
    if dataset == "cifar10":
        return 1 / 10 * SCALING_FACTOR
    elif dataset == "cifar100":
        return 1 / 100 * SCALING_FACTOR
    elif "imagenet" in dataset:
        return 1 / 1000 * SCALING_FACTOR
    else:
        raise NotImplementedError

def adjust_learning_rate(args, optimizer, len_loader, step):
    max_steps = args.kd_epochs * len_loader
    base_lr = args.kd_lr #* args.batch_size / 256

    warmup_steps = 10 * len_loader
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_tensor_parameters(model,fedbn=False):
    from flwr.common.parameter import ndarrays_to_parameters

    return ndarrays_to_parameters(
        get_params(model,fedbn)
    )

def get_params(model,fedbn=False):
    """Get model weights as a list of NumPy ndarrays."""

    if(fedbn):
        return [val.cpu().numpy() for name, val in model.state_dict().items() if 'bn' not in name]
    else:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

def count_params(model,trainable = False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def set_params(model, params, fedbn = False, bb_only = False):
    """Set model weights from a list of NumPy   ndarrays."""
    
    # keys = model.state_dict().keys()
    
    if(bb_only):
        keys = model.state_dict().keys()
        params_dict = dict(zip(keys, params))
        linear_keys = [k for k in params_dict.keys() if "linear" in k]
        [params_dict.pop(k) for k in linear_keys] # pop layers linear layers
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict.items()})

        model.load_state_dict(state_dict, strict=False)
    elif(fedbn):
        keys = [k for k in model.state_dict().keys() if 'bn' not in k]
        params_dict = zip(keys, params)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=False)
    else:
        keys = model.state_dict().keys()
        params_dict = zip(keys, params)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)


def pile_str(line, item):
    return "_".join([line, item])


def tell_history(hist, file_name, infos=None, path=""):
    _, acc_cent = zip(*hist.metrics_centralized["accuracy"])
    losses_cent = hist.losses_centralized
    losses_dis = hist.losses_distributed
    try:
        acc_dis = hist.metrics_distributed["dist_acc"]
    except:
        acc_dis = 0.0

    acc_cent = np.asarray(acc_cent, dtype=object)

    infos["accuracy_cent"] = acc_cent
    infos["accuracy_dist"] = acc_dis
    infos["losses_cent"] = losses_cent
    infos["losses_dis"] = losses_dis

    with open(path + file_name + ".npy", "wb") as f:
        np.save(f, infos)


def inst_model_info(model_info: Info, use_proj: bool = False, out_dim: int = -1):
    model = model_selection(model_info.model)
    model = model(
        model_info.feature_maps,
        model_info.input_shape,
        model_info.num_classes,
        model_info.batchn,
    )

    if use_proj:
        model = Project(model, 
                        input_dim=model.features_dim, 
                        out_dim=out_dim
                )

    return model


def inst_model_lora_info(model_info: Info, lora_config : LoraInfo):
    from utils.lora import inject_low_rank
    model = model_selection(model_info.model)
    model = model(
        model_info.feature_maps,
        model_info.input_shape,
        model_info.num_classes,
        model_info.batchn,
    )


    return inject_low_rank(model,lora_config)

def create_all_dirs():
    from pathlib import Path
    from args import args

    Path.mkdir(Path("./data"), parents=True, exist_ok=True)
    Path.mkdir(Path(args.path_results), parents=True, exist_ok=True)
    Path.mkdir(Path("./checkpoint"), parents=True, exist_ok=True)


def train(net, trainloader, epochs, optimizer, criterion, device):
    """Train the network on the training set."""

    net.train()
    for _ in range(epochs):
        for images, labels, _ in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out, _ = net(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()


def test(model, test_loader, device):
    if not isinstance(model,list):
        model =  [model]

    for m in model:
        m.eval()
        m.to(device)
    outputs=[]
    losses = torch.zeros(len(model))
    accuracies = torch.zeros(len(model))
    en_loss, en_accuracy, total,accuracy_top_5 = 0, 0, 0,0
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs=[]
            for m in model:
                out, _ = m(data)
                outputs.append(out)
            if(len(model)>1):
                en_output = sum(outputs)/len(model)
            else:
                en_output = outputs[0]

            for i,out in enumerate(outputs):
                losses[i] += criterion(out, target).item() * data.shape[0]
                pred = out.argmax(dim=1, keepdim=True)
                accuracies[i] += pred.eq(target.view_as(pred)).sum().item()

            en_loss += criterion(en_output, target).item() * data.shape[0]
            pred = en_output.argmax(dim=1, keepdim=True)
            en_accuracy += pred.eq(target.view_as(pred)).sum().item()

            total += target.shape[0]
            # preds = output.sort(dim = 1, descending = True)[1][:,:5]
            # for i in range(preds.shape[0]):
            #     if target[i] in preds[i]:
            #         accuracy_top_5 += 1

    # return results

    return {
        "test_loss": en_loss / total,
        "test_acc": en_accuracy / total,
        "test_acc_top_5": accuracy_top_5 / total,
        "losses": losses/total,
        "accuracies": accuracies/total,
    }

def quick_plot(file_name, threshold=0.7):
    import matplotlib.pyplot as plt

    for i, name in enumerate(file_name):
        vec = np.load(name, allow_pickle=True).item()
        acc = vec["accuracy_cent"]
        rounds = range(len(acc))
        max_idx = acc.argmax()
        plt.plot(rounds, acc, label=f"run {i}")
        round_threshold = np.argmax(acc > threshold)
        print(
            f"Run {i} : Max accuracy {acc[max_idx]} @ round {max_idx+1}, "
            + f"it reaches {threshold} @ round {round_threshold} - {name}"
        )
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def save_model(file_name, data):
    # file_name must contains path ex: "checkpoint/server.npy"
    obj_data = np.array(data, dtype=object)
    np.save(file_name, obj_data)

def ema(prev_weights,results,decay = 0.9):
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    ema_weights = [x*decay + y*(1-decay)
                    for x,y in zip(weights_prime,prev_weights)]
    
    return ema_weights

def load_pretrained(model,model_name,path="pretrained"):

    path_to_pretrained = f"./{path}/{model_name}.pt"
    state_dict = torch.load(path_to_pretrained, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()
    ##Correcting for the number of classes, it should not be a problem
    ##in future implementations, where it would be trained with ssl
    state_dict["fc.weight"] = model_state_dict["fc.weight"]
    state_dict["fc.bias"] = model_state_dict["fc.bias"]

    model.load_state_dict(state_dict)
    print("### Successfully Loaded from pretrained ###")
    return model