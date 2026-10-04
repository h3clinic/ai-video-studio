import math
import torch
import numpy as np
from real_video.compose_dog_gaussians import GaussianReplay, dog_transform, tensor_bytes
from real_video.capture_scene_gaussians import fill_enclosed_holes


def test_replay_vectors_refresh_and_immutable_bank():
    base=torch.tensor([[.5,.5,-5.,-5.,1.,0.,.2,.3,.4,1.]])
    packet=dict(frames=2,segments=[dict(first_frame=0,base=base,states=[dict(delta=torch.tensor([[.1,.2,0.,0.,math.pi/2]]),visibility=torch.tensor([255],dtype=torch.uint8))]),
                                dict(first_frame=1,base=base+0,states=[dict(delta=torch.zeros(1,5),visibility=torch.tensor([128],dtype=torch.uint8))])])
    replay=GaussianReplay(packet)
    first=replay.at(0,'cpu'); second=replay.at(1,'cpu')
    assert torch.allclose(first[0,:2],torch.tensor([.6,.7]))
    assert torch.allclose(first[0,4:6],torch.tensor([0.,1.]),atol=1e-6)
    assert torch.equal(base[0,:2],torch.tensor([.5,.5]))
    assert torch.allclose(second[0,:2],base[0,:2])
    assert abs(second[0,9].item()-128/255)<1e-6
    assert tensor_bytes(packet)>base.numel()*base.element_size()
    try: replay.at(2,'cpu')
    except IndexError: pass
    else: raise AssertionError('Out of range replay allowed')


def test_transform_preserves_colour_and_scales_covariance():
    points=torch.tensor([[.5,.5,-5.,-5.,1.,0.,.2,.3,.4,1.],[1.,1.,-5.,-5.,0.,1.,.5,.6,.7,1.]])
    result=dog_transform(points,points,scale=.5)
    assert torch.equal(result[:,4:],points[:,4:])
    assert torch.allclose(result[:,2:4],points[:,2:4]+math.log(.5))
    assert torch.allclose(result[1,:2]-result[0,:2],(points[1,:2]-points[0,:2])*.5)


def test_mask_closes_internal_holes_not_exterior_leg_gaps():
    mask=np.zeros((12,12),np.float32); mask[2:10,2:10]=1
    mask[4,4]=0; mask[8:,6]=0
    fixed=fill_enclosed_holes(mask)
    assert fixed[4,4]==1
    assert fixed[8,6]==0
    assert mask[4,4]==0
