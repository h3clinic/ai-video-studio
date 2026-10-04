import torch
from real_video.train_temporal_wan import temporal_flow_loss


def test_exact_flow_matches_clean_temporal_changes():
    torch.manual_seed(3)
    clean=torch.randn(1,2,5,3,4); noise=torch.randn_like(clean); sigma=.6
    loss,flow,motion=temporal_flow_loss(noise-clean,(1-sigma)*clean+sigma*noise,clean,noise,sigma)
    assert loss < 1e-12 and flow < 1e-12 and motion < 1e-12


def test_flicker_is_penalized_not_rewarded():
    clean=torch.zeros(1,2,5,3,4); noise=torch.ones_like(clean); sigma=.5
    prediction=noise.clone(); prediction[:,:,1::2]+=2
    loss,flow,motion=temporal_flow_loss(prediction,(1-sigma)*clean+sigma*noise,clean,noise,sigma)
    assert motion>0 and loss>flow


def test_static_video_keeps_pooled_centers_and_border_occupancy():
    import numpy as np
    from real_video.train_temporal_wan import video_conditions
    image=np.random.default_rng(4).integers(0,256,(32,48,3),dtype=np.uint8)
    frames=np.repeat(image[None],5,axis=0)
    condition,tracks,visibility=video_conditions(frames,np.ones((5,32,48),np.float32))
    assert torch.allclose(tracks[:,0],tracks[:,0].round(),atol=1e-7)
    assert torch.allclose(condition[:,:,:1],condition[:,:,1:],atol=1e-4)
