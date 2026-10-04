import torch
from torch import nn
import torch.nn.functional as F
from .geometry import exp_so3, rotation_6d, local, world
from .data import NOISE_DIM, DT


class GaussianInitializer(nn.Module):
    """Learned noise + finite control -> initial dot attributes. No stored asset."""
    def __init__(self,width=96):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(NOISE_DIM+4,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,19))
        with torch.no_grad():
            self.net[-1].bias[3:9].copy_(torch.tensor([1.,0.,0.,0.,1.,0.]))

    def forward(self,z,c):
        cond=torch.cat((c['kind'],c['rate']),-1)[:,None,:].expand(*z.shape[:2],4)
        x=self.net(torch.cat((z,cond),-1))
        return dict(p=x[...,:3]+c['center'][:,None,:],R=rotation_6d(x[...,3:9]),
                    velocity=x[...,9:12],scale=0.01+0.08*torch.sigmoid(x[...,12:15]),
                    color=torch.sigmoid(x[...,15:18]),alpha=torch.sigmoid(x[...,18:19]))


class VectorDynamics(nn.Module):
    """Local-frame neighbor messages -> per-dot GRU -> vector dynamics.

    Hidden state is scalar under global rigid transformations; geometric outputs
    are local vectors transported through the generated frame. Dense cdist is
    a prototype limitation, not a claim of subquadratic neighbor search.
    """
    def __init__(self,hidden=32,width=96,neighbors=6):
        super().__init__()
        self.hidden=hidden
        self.neighbors=neighbors
        self.edge=nn.Sequential(nn.Linear(16,width//2),nn.SiLU(),nn.Linear(width//2,16),nn.SiLU())
        # local position, velocity, axis (9); scale(3); kind/rate(4); messages(16)
        self.encode=nn.Sequential(nn.Linear(32,width),nn.SiLU())
        self.gru=nn.GRUCell(width,hidden)
        self.head=nn.Sequential(nn.Linear(hidden,width),nn.SiLU(),nn.Linear(width,6))
        nn.init.normal_(self.head[-1].weight,std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def initial_memory(self,state):
        return state['p'].new_zeros(*state['p'].shape[:2],self.hidden)

    def messages(self,state):
        p,R=state['p'],state['R']
        b,n,_=p.shape
        if n==1:
            return p.new_zeros(b,n,16)
        dist=torch.cdist(p,p)
        dist=dist+torch.eye(n,device=p.device)[None]*1e6
        ids=dist.topk(min(self.neighbors,n-1),largest=False).indices
        batches=torch.arange(b,device=p.device)[:,None,None]
        pj=p[batches,ids]
        Rj=R[batches,ids]
        vj=state['velocity'][batches,ids]
        Ri=R[:,:,None,:,:]
        rel=local(Ri,pj-p[:,:,None,:])
        relv=local(Ri,vj-state['velocity'][:,:,None,:])
        orientation=(Ri.transpose(-1,-2)@Rj).flatten(-2)
        e=torch.cat((rel,relv,orientation,rel.norm(dim=-1,keepdim=True)),-1)
        return self.edge(e).mean(2)

    def forward(self,state,h,c,dt=DT,zero_memory=False):
        p,R=state['p'],state['R']
        b,n,_=p.shape
        axis=c['axis'][:,None,:].expand(b,n,3)
        cond=torch.cat((c['kind'],c['rate']),-1)[:,None,:].expand(b,n,4)
        inputs=torch.cat((local(R,p-c['center'][:,None,:]),local(R,state['velocity']),
                          local(R,axis),state['scale'],cond,self.messages(state)),-1)
        if zero_memory:
            h=torch.zeros_like(h)
        h=self.gru(self.encode(inputs).reshape(b*n,-1),h.reshape(b*n,-1)).reshape(b,n,-1)
        action=self.head(h)
        # Bounded learned local acceleration and angular velocity, no teacher dynamics.
        acceleration=2.5*torch.tanh(action[...,:3])
        omega=2.0*torch.tanh(action[...,3:])
        a_world=world(R,acceleration)
        velocity=state['velocity']+dt*a_world
        out=dict(state)
        out['p']=p+dt*state['velocity']+0.5*dt*dt*a_world
        out['velocity']=velocity
        out['R']=R@exp_so3(dt*omega)
        return out,h


class Prototype(nn.Module):
    def __init__(self,hidden=32,width=96,neighbors=6):
        super().__init__()
        self.config=dict(hidden=hidden,width=width,neighbors=neighbors)
        self.initializer=GaussianInitializer(width)
        self.dynamics=VectorDynamics(hidden,width,neighbors)

    @torch.no_grad()
    def generate(self,z,control,steps=120,dt=DT):
        state=self.initializer(z,control)
        h=self.dynamics.initial_memory(state)
        yield state,h
        for _ in range(steps):
            state,h=self.dynamics(state,h,control,dt)
            yield state,h


def state_bytes(state,h):
    return sum(x.numel()*x.element_size() for x in state.values())+h.numel()*h.element_size()


def init_loss(pred,target):
    return (F.mse_loss(pred['p'],target['p'])+0.4*F.mse_loss(pred['R'],target['R'])+
            0.5*F.mse_loss(pred['velocity'],target['velocity'])+
            10*F.mse_loss(pred['scale'],target['scale'])+0.2*F.mse_loss(pred['color'],target['color'])+
            0.05*F.mse_loss(pred['alpha'],target['alpha']))


def motion_loss(pred,target):
    return (4*F.mse_loss(pred['p'],target['p'])+0.5*F.mse_loss(pred['R'],target['R'])+
            0.5*F.mse_loss(pred['velocity'],target['velocity']))
