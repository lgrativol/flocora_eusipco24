# from args import args
import torch
import torch.nn
import math
import numpy as np

def eff_sparsity(params):
    flat_params,_= get_flat_sorted_params(params)
    _,counts = np.unique(flat_params,return_counts=True)
    return counts[0]/len(flat_params)

def threshold_zero(vec,thres):
    vec[np.abs(vec)<thres] = 0
    return vec

def get_flat_sorted_params(params):
    return  torch.cat([torch.from_numpy(i).flatten().abs() for i in params]).sort()
    #return np.concatenate(np.asanyarray(params),axis=None)

def prune_threshold(params,prate):

    sorted,_ = get_flat_sorted_params(params)
    threshold = sorted[int(len(sorted)*prate)]

    for i,p in enumerate(params):
        params[i] = threshold_zero(p,threshold.item())

    return params

def unflatten(flatv,dims):

    arr = []
    cum = 0
    for dim in dims:
        size = np.asarray(dim).prod()
        reshaped = (flatv[cum:cum+size]).reshape(dim)
        arr.append(reshaped)
        cum+=size
    return arr

def gen_prates(start,stop,pmin,pmax,reg=False,rel_sizes=None):
    size = stop-start +1
    step = (pmax-pmin)/(size) #linear
    
    prates = np.arange(pmin,pmax,step)
    
    if(reg):
        rel_sizes+=1
        prates = prates * rel_sizes
    
    return prates

# def prune_threshold_layer_wise(start,stop,params,prates):
#     j = 0
#     for i,p in enumerate(params):
#         if (i >= start and i <= stop) :
#             if (p.size > 1000 or not args.w_only):
#                 sorted = np.abs(p).flatten()
#                 sorted.sort()
#                 threshold = sorted[int(len(sorted)*prates[j])]
#                 params[i] = threshold_zero(p,threshold.item())
#             j+=1
#     return params

def prune_norm_threshold(params,prate):
    original_dims = [p.shape for p in params]
    normalized_params = [(p / np.linalg.norm(p)) for p in params]

    _,idx = get_flat_sorted_params(normalized_params)
    prune_idxs = idx[:int(len(idx)*prate)]
    
    flat_params = np.concatenate(params,axis=None)
    flat_params[prune_idxs] = 0
    return unflatten(flat_params,original_dims)

def prune(params,prate):
    return prune_threshold(params,prate)

# def get_a(batch_idx, current_epoch, max_epoch, dataset_length):

#     max_batch = max_epoch * dataset_length
#     current_batch = (current_epoch * dataset_length) + batch_idx
#     exponent = math.log(args.a_max / args.a_min) / max_batch
#     return args.a_min * math.exp(current_batch * exponent)

# def train_swd(net, trainloader, epochs, device, lr = 0.01, momentum = 0.9):

#     """Train the network on the training set."""
#     prate = args.prate
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(),lr=lr, momentum=momentum,weight_decay=args.wd)
#     net.train()
#     regularizer = SWD(None,get_unstructured_mask,prate)
#     for epoch in range(epochs):
#         for batch_idx, (images, labels,_) in enumerate(trainloader):

#             regularizer.set_a(get_a(batch_idx, epoch, epochs,len(trainloader)))

#             images, labels = images.to(device,non_blocking=True), labels.to(device,non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
#             out,_ = net(images)
#             loss = criterion(out, labels)

#             if args.wd == 0 and args.mu > 0:
#                 loss += args.mu * regularizer(net)
#             else:
#                 loss += args.wd * regularizer(net)

#             loss.backward()
#             optimizer.step()


def get_unstructured_mask(model, target):
    parameters = torch.cat([i.abs().flatten() for n, i in model.named_parameters() if 'bn' not in n]).sort()[0]
    ths = parameters[int(target * len(parameters))]
    return [(param.data.abs() >= ths).float() for param in model.parameters()]

get_mask = get_unstructured_mask

class SWD():

    def __init__(self,a,get_mask,target):
        self.a = a
        self.get_mask = get_mask
        self.target = target

    def swd(self, model):
        mask = self.get_mask(model, self.target)
        total = 0
        for p, m in zip(model.parameters(), mask):
            total += 0.5 * self.a * torch.pow(p * (1 - m).float(), 2).sum()
        return total

    def __call__(self,model):
        return self.swd(model)

    def set_a(self, a):
        self.a = a

    def get_target(self):
        return self.target