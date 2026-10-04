"""Training-only feedback through Gaussian splatting; no RGB decoder at inference."""
import torch
import torch.nn.functional as F
from .representation import render_fields


def field_loss(prediction,target,rgb_weight=1.0):
    weights=prediction.new_ones(9)
    weights[6:]=rgb_weight
    per_channel=(prediction-target).square().mean((0,2,3,4))
    return (per_channel*weights).sum()/weights.sum()


def appearance_loss(prediction,noisy,alpha_bar,mean,std,reference_rgb,max_examples=2):
    """Only low-noise examples; predicted x0 remains a Gaussian field.

    reference_rgb is B,T,3,H,W, training data only. Alpha-bar weights reduce
    image regression where the conditioning input contains little signal.
    """
    eligible=torch.where(alpha_bar.flatten()>0.5)[0][:max_examples]
    if len(eligible)==0: return prediction.sum()*0
    a=alpha_bar[eligible].sqrt()
    s=(1-alpha_bar[eligible]).sqrt()
    clean=(a*noisy[eligible]-s*prediction[eligible]).clamp(-6,6)
    rendered=render_fields(clean*std+mean)
    target=reference_rgb[eligible]
    multiscale=rendered.new_zeros(len(eligible))
    for size in (64,32,16,8):
        p=F.interpolate(rendered.flatten(0,1),size=size,mode='area').reshape(len(eligible),8,3,size,size)
        q=F.interpolate(target.flatten(0,1),size=size,mode='area').reshape_as(p)
        multiscale=multiscale+(p-q).abs().mean((1,2,3,4))/4
    temporal=((rendered[:,1:]-rendered[:,:-1])-(target[:,1:]-target[:,:-1])).abs().mean((1,2,3,4))
    return ((multiscale+0.25*temporal)*alpha_bar[eligible].flatten()).mean()
