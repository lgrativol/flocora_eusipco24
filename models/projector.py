import torch.nn as nn
import torch
class Project(nn.Module):
    def __init__(self,net, input_dim, out_dim=128, apply_bn=False, device="cpu"):
        super(Project, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.out_features = out_dim
        self.net=net
        if apply_bn:
            self.linear = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.linear = nn.Sequential(self.linear1,self.relu, self.linear2)
        self.linear = self.linear.to(device)

    def forward(self, x):
        out,features = self.net(x)
        return out,self.linear(features)

class HeadProject(nn.Module):
    def __init__(self, input_dim, out_dim=128, apply_bn=False, device="cpu"):
        super(HeadProject, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.out_features = out_dim
        if apply_bn:
            self.linear = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.linear = nn.Sequential(self.linear1, self.relu, self.linear2)
        self.linear = self.linear.to(device)

    def forward(self, x):
        return self.linear(x)
    
class HydraProject(nn.Module):
    def __init__(self,nb_heads, input_dim, out_dim=128, apply_bn=False, device="cpu"):
        super(HydraProject, self).__init__()
        self.nb_heads = nb_heads
        self.out_features = out_dim
        self.heads = []
        for _ in range(nb_heads):
            linear1 = nn.Linear(input_dim, input_dim)
            linear2 = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(input_dim)
            relu = nn.ReLU()
            if apply_bn:
                linear = nn.Sequential(linear1, bn, relu, linear2)
            else:
                linear = nn.Sequential(linear1, relu, linear2)
            linear = linear.to(device)
            self.heads.append(linear)

    def to(self,device):
        for h in self.heads:
            h.to(device)

    def eval(self):
        for h in self.heads:
            h.eval()

    def train(self):
        for h in self.heads:
            h.train()

    def forward(self, feats):
        pjs = []
        for h in self.heads:
            pjs.append(h(feats))
        return torch.stack(pjs)
    

class NetLessProject(nn.Module):
    def __init__(self,input_dim, out_dim=128, apply_bn=False, device="cpu"):
        super(NetLessProject, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.out_features = out_dim
        if apply_bn:
            self.linear = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.linear = nn.Sequential(self.linear1,self.relu, self.linear2)
        self.linear = self.linear.to(device)

    def forward(self, x):
        return self.linear(x)