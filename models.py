import torch
import torch.nn as nn
import numpy as np



class PolNet(nn.Module):
    def __init__(self, sd, ad):
        super(PolNet, self).__init__()
        self.f1 = nn.Linear(sd, 256)
        self.f2 = nn.Linear(256, 128)
        self.m = nn.Linear(128, ad)
        self.ls = nn.Linear(128, ad)

        # Initialization
        nn.init.orthogonal_(self.f1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.f2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.m.weight, gain=0.01)
        nn.init.orthogonal_(self.ls.weight, gain=0.01)
        
    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        m = torch.tanh(self.m(x))
        ls = self.ls(x)
        ls = torch.clamp(ls, -20, 2)
        return m, ls

class ValNet(nn.Module):
    def __init__(self, sd):
        super(ValNet, self).__init__()
        self.f1 = nn.Linear(sd, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, 1)

        nn.init.orthogonal_(self.f1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.f2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.f3.weight, gain=1.0)
        
    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        return self.f3(x)
