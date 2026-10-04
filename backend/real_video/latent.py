"""Learned compact code decodes to Gaussian parameters, never directly to RGB."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv3d(channels,channels,3,padding=1),
                                  nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv3d(channels,channels,3,padding=1))

    def forward(self,x): return x+self.layers(x)


class GaussianCodec(nn.Module):
    """No encoder-decoder skip connections: all information passes through z."""
    def __init__(self,latent_channels=16,width=32):
        super().__init__()
        self.config=dict(latent_channels=latent_channels,width=width)
        self.encoder=nn.Sequential(nn.Conv3d(9,width,3,padding=1),Block(width),
                                   nn.Conv3d(width,width*2,3,stride=(1,2,2),padding=1),Block(width*2),
                                   nn.Conv3d(width*2,width*3,3,stride=2,padding=1),Block(width*3),
                                   nn.GroupNorm(8,width*3),nn.SiLU(),nn.Conv3d(width*3,latent_channels*2,1))
        self.decoder_input=nn.Sequential(nn.Conv3d(latent_channels,width*3,3,padding=1),Block(width*3))
        self.decoder_mid=nn.Sequential(nn.Conv3d(width*3,width*2,3,padding=1),Block(width*2))
        self.decoder_output=nn.Sequential(nn.Conv3d(width*2,width,3,padding=1),Block(width),
                                          nn.GroupNorm(8,width),nn.SiLU(),nn.Conv3d(width,9,3,padding=1))

    def encode(self,fields):
        mean,logvar=self.encoder(fields).chunk(2,1)
        return mean,logvar.clamp(-8,4)

    def decode(self,z):
        h=self.decoder_input(z)
        h=self.decoder_mid(F.interpolate(h,size=(8,16,16),mode='nearest'))
        return self.decoder_output(F.interpolate(h,size=(8,32,32),mode='nearest'))

    def forward(self,fields,stochastic=True):
        mean,logvar=self.encode(fields)
        z=mean+torch.randn_like(mean)*(0.5*logvar).exp() if stochastic else mean
        return self.decode(z),mean,logvar


def kl_loss(mean,logvar):
    return 0.5*(mean.square()+logvar.exp()-1-logvar).mean()


def inference_decoder(config,state):
    """Load on CPU, then remove the analysis encoder before any GPU transfer.

    Retains the exact trained decoder computation. Not a compressed-weight claim.
    """
    codec=GaussianCodec(**config).eval().requires_grad_(False)
    codec.load_state_dict(state)
    del codec.encoder
    return codec


def rendered_loss(rendered,target):
    loss=rendered.new_zeros(())
    for size in (64,32,16,8):
        p=F.interpolate(rendered.flatten(0,1),size=size,mode='area')
        q=F.interpolate(target.flatten(0,1),size=size,mode='area')
        loss=loss+(p-q).abs().mean()/4
    return loss+0.25*((rendered[:,1:]-rendered[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean()
