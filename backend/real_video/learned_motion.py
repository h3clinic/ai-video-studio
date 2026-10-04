"""Causal learned Gaussian velocity dynamics; no future-track archive input."""
import torch
from torch import nn


class LearnedGaussianMotion(nn.Module):
    def __init__(self,hidden=32):
        super().__init__(); self.hidden=hidden
        self.initialize_memory=nn.Sequential(nn.Conv2d(9,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,hidden,3,padding=1),nn.Tanh())
        self.gates=nn.Conv2d(9+hidden,2*hidden,3,padding=1)
        self.candidate=nn.Conv2d(9+hidden,hidden,3,padding=1)
        self.head=nn.Sequential(nn.Conv2d(hidden,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,3,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad(): self.head[-1].bias[2]=4

    @staticmethod
    def features(fields,velocity,acceleration):
        return torch.cat((fields[:,:2],fields[:,6:9],velocity*64,acceleration*64),1)

    def initialize(self,first,second,third,dt=1.):
        if first.shape!=second.shape or first.shape!=third.shape or first.ndim!=4 or first.shape[1]!=9 or dt<=0:
            raise ValueError('Three matching Gaussian observations and positive dt required')
        velocity=(third[:,:2]-second[:,:2])/dt
        previous=(second[:,:2]-first[:,:2])/dt
        acceleration=(velocity-previous)/dt
        memory=self.initialize_memory(self.features(third,velocity,acceleration))
        return dict(fields=third.clone(),velocity=velocity,acceleration=acceleration,memory=memory)

    def step(self,state,dt=1.,reset_memory=False):
        if not 0<dt<=1: raise ValueError('Use 0 < dt <= 1')
        fields,velocity,acceleration,memory=(state[k] for k in ['fields','velocity','acceleration','memory'])
        if reset_memory: memory=torch.zeros_like(memory)
        x=self.features(fields,velocity,acceleration)
        reset,update=self.gates(torch.cat((x,memory),1)).sigmoid().chunk(2,1)
        candidate=self.candidate(torch.cat((x,reset*memory),1)).tanh()
        memory=(1-update)*memory+update*candidate
        action=self.head(memory)
        damp=action[:,2:3].sigmoid().pow(dt)
        next_velocity=(damp*velocity+dt*.01*action[:,:2].tanh()).clamp(-.12,.12)
        next_fields=torch.cat((fields[:,:2]+dt*next_velocity,fields[:,2:]),1)
        return dict(fields=next_fields,velocity=next_velocity,acceleration=(next_velocity-velocity)/dt,memory=memory)
