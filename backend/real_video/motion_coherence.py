"""Observed-boundary-weighted planar neighbour supervision, not anatomy.

Edges use saved grid correspondence. The three observed states set weights;
future positions are targets during training/evaluation only. A changed edge
is allowed when the target changes: this is not rest-shape rigidity.
"""
import torch


def edges(x):
    return x[..., :, 1:]-x[..., :, :-1], x[..., 1:, :]-x[..., :-1, :]


def signed_area(position):
    x=position[:, :, :-1, 1:]-position[:, :, :-1, :-1]
    y=position[:, :, 1:, :-1]-position[:, :, :-1, :-1]
    return x[:, :1]*y[:, 1:2]-x[:, 1:2]*y[:, :1]


def weighted_mean(value,weight):
    return (value*weight).sum()/weight.sum().clamp_min(1e-8)


def make_context(first,second,third):
    """Only three observations; colour/velocity gates are heuristics, not parts."""
    if first.shape!=second.shape or first.shape!=third.shape or third.ndim!=4 or third.shape[1]!=9:
        raise ValueError('Three matching B,9,H,W fields required')
    velocity=third[:,:2]-second[:,:2]
    colour_edges=edges(third[:,6:9]); velocity_edges=edges(velocity*64)
    edge_weights=tuple(torch.exp(-c.square().sum(1,keepdim=True)/.15-v.square().sum(1,keepdim=True)/4).detach()
                       for c,v in zip(colour_edges,velocity_edges))
    area=signed_area(third[:,:2]).detach()
    typical=1/(third.shape[-1]*third.shape[-2])
    cell_weight=torch.minimum(edge_weights[0][...,:-1,:],edge_weights[1][...,:,:-1])
    cell_weight=cell_weight*(area.abs()>typical*.1)
    safe_area=torch.where(area.abs()>typical*.1,area,torch.ones_like(area))
    return dict(edge_weights=edge_weights,cell_weight=cell_weight,reference_area=safe_area)


def coherence_loss(predicted,target,context):
    pe=edges(predicted[:,:2]); te=edges(target[:,:2])
    edge_error=sum(weighted_mean(((p-t)*64).square().sum(1,keepdim=True),w)
                   for p,t,w in zip(pe,te,context['edge_weights']))/2
    pr=signed_area(predicted[:,:2])/context['reference_area']
    tr=signed_area(target[:,:2])/context['reference_area']
    # Penalize inversion only where the tracked target retains its orientation.
    w=context['cell_weight']*(tr>.05)
    fold_penalty=weighted_mean(torch.relu(.05-pr).square(),w)
    return edge_error+.1*fold_penalty


def geometry_metrics(predicted,target,context):
    pe=edges(predicted[:,:2]); te=edges(target[:,:2])
    edge_error=sum(weighted_mean(((p-t)*64).square().sum(1,keepdim=True),w)
                   for p,t,w in zip(pe,te,context['edge_weights']))/2
    stretch=sum(weighted_mean(((p.norm(dim=1,keepdim=True)-t.norm(dim=1,keepdim=True))*64).abs(),w)
                for p,t,w in zip(pe,te,context['edge_weights']))/2
    pr=signed_area(predicted[:,:2])/context['reference_area']
    tr=signed_area(target[:,:2])/context['reference_area']
    return dict(edge_mse_pixels64=edge_error,edge_length_mae_pixels64=stretch,
                fold_fraction=weighted_mean((pr<=0).float(),context['cell_weight']),
                target_fold_fraction=weighted_mean((tr<=0).float(),context['cell_weight']),
                fold_mismatch=weighted_mean(((pr<=0)!=(tr<=0)).float(),context['cell_weight']))


def passes_gate(candidate,baseline):
    """Predeclared validation gate; not a broad video-quality certification."""
    checks=dict(edge_improved=candidate['edge_mse_pixels64']<=.95*baseline['edge_mse_pixels64'],
                trajectory_retained=candidate['weighted_position_epe_pixels64']<=1.02*baseline['weighted_position_epe_pixels64'],
                image_retained=candidate['rgb_mse']<=1.01*baseline['rgb_mse'],
                fold_not_worse=candidate['fold_mismatch']<=baseline['fold_mismatch']+.002,
                motion_retained=candidate['mean_displacement_pixels64']>=.8*baseline['mean_displacement_pixels64'])
    return dict(passed=all(checks.values()),checks=checks)
