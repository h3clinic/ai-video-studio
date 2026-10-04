import math
import torch

from real_video.gaussian_scene_objects import (
    GaussianSceneObject, merge_scene_objects, object_bounds, overlap_report,
    placement_report, render_scene,
)


def asset():
    return dict(position=torch.tensor([[-1., 0., -.3], [1., 2., .3]]),
        covariance=torch.diag(torch.tensor([.04, .09, .16])).repeat(2, 1, 1),
        normal=torch.tensor([[0., 1., 0.], [0., 1., 0.]]),
        colour=torch.tensor([[1., 0., 0.], [0., 1., 0.]]), opacity=torch.ones(2)*.9,
        ids=torch.tensor([8, 15]), face_id=torch.tensor([0, 3]),
        barycentric=torch.tensor([[.2, .3, .5], [.5, .2, .3]]))


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid value accepted')


def test_sim3_correct_covariance_normal_and_stable_material_binding():
    a = asset()
    obj = GaussianSceneObject(12, a, 'a'*64)
    r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    t = torch.tensor([2., 4., -1.])
    moved = obj.transform(scale=2., rotation=r, translation=t)
    assert torch.allclose(moved['position'], 2*(a['position'] @ r.T)+t)
    assert torch.allclose(moved['covariance'], 4*(r @ a['covariance'] @ r.T))
    assert torch.allclose(moved['normal'], a['normal'] @ r.T)
    for key in ['ids', 'face_id', 'barycentric', 'colour', 'opacity']:
        assert torch.equal(moved[key], a[key])
    assert moved['object_id'] == 12 and moved['asset_sha256'] == 'a'*64


def test_canonical_appearance_and_ids_do_not_alias_input_or_output():
    a = asset()
    obj = GaussianSceneObject(0, a, 'b'*64)
    a['position'].zero_()
    a['colour'].zero_()
    moved = obj.transform()
    assert float(moved['colour'].sum()) == 2.
    moved['position'].zero_()
    moved['ids'].zero_()
    fresh = obj.transform()
    assert torch.equal(fresh['ids'], torch.tensor([8, 15]))
    assert float(fresh['position'].abs().sum()) > 0


def test_frame_scale_asset_supported_without_covariance():
    a = asset()
    del a['covariance']
    a['frame'] = torch.eye(3).repeat(2, 1, 1)
    a['scale'] = torch.tensor([[.2, .3, .4]]).repeat(2, 1)
    obj = GaussianSceneObject(1, a, 'c'*64)
    moved = obj.transform(scale=3.)
    assert torch.allclose(moved['covariance'], asset()['covariance']*9.)
    assert torch.allclose(moved['scale'], a['scale']*3.)


def test_grounded_explicit_height_bounds_are_not_claimed_metres():
    obj = GaussianSceneObject(1, asset(), 'c'*64)
    moved = obj.grounded(1.6, center_xz=(3., -2.), ground_y=.1)
    lo, hi = object_bounds(moved)
    assert torch.allclose((lo+hi)[[0, 2]]*.5, torch.tensor([3., -2.]))
    assert abs(float(lo[1])-.1) < 1e-6
    assert abs(float(hi[1]-lo[1])-1.6) < 1e-6
    result = placement_report(moved, (-10., 0., -10.), (10., 3., 10.), height_range=(1.5, 1.7), ground_y=.1)
    assert result['accepted'] and not result['physical_scale_recovered']
    assert not placement_report(moved, (-1., 0., -1.), (1., 3., 1.), ground_y=.1)['accepted']


def test_overlap_and_splat_support_bounds_are_explicit():
    obj = GaussianSceneObject(1, asset(), 'd'*64)
    a, b = obj.transform(), obj.transform(translation=torch.tensor([3., 0., 0.]))
    assert not overlap_report(a, b)['interior_aabb_overlap']
    assert abs(overlap_report(a, b)['aabb_separation']-1.) < 1e-6
    assert overlap_report(a, a)['interior_aabb_overlap']
    lo0, hi0 = object_bounds(a)
    lo3, hi3 = object_bounds(a, sigma=3.)
    assert bool((lo3 < lo0).all()) and bool((hi3 > hi0).all())


def test_identity_pairs_survive_other_object_addition_and_order():
    a = GaussianSceneObject(9, asset(), 'e'*64).transform()
    b = GaussianSceneObject(2, asset(), 'f'*64).transform()
    alone, together, reverse = merge_scene_objects([a]), merge_scene_objects([a, b]), merge_scene_objects([b, a])
    assert torch.equal(together['identity_pairs'], reverse['identity_pairs'])
    assert torch.equal(together['identity_pairs'][together['object_ids'] == 9], alone['identity_pairs'])
    assert together['bindings'] == {'2': 'f'*64, '9': 'e'*64}
    reject(lambda: merge_scene_objects([a, a]))


def test_shared_depth_occlusion_is_not_object_layer_order():
    a = asset()
    a = {k: v[:1].clone() for k, v in a.items()}
    a['position'][:] = torch.tensor([0., 0., .4])
    a['covariance'][:] = torch.eye(3)*.03
    a['colour'][:] = torch.tensor([1., 0., 0.])
    near = GaussianSceneObject(8, a, 'a'*64).transform()
    a['position'][:] = torch.tensor([0., 0., -.4])
    a['colour'][:] = torch.tensor([0., 0., 1.])
    far = GaussianSceneObject(1, a, 'b'*64).transform()
    eye, target = torch.tensor([0., 0., 3.]), torch.zeros(3)
    image, alpha, cache = render_scene([near, far], eye, target, height=32, width=32, ground=False, return_cache=True)
    reverse, _ = render_scene([far, near], eye, target, height=32, width=32, ground=False)
    assert torch.equal(image, reverse)
    assert image[16, 16, 0] > image[16, 16, 2]
    assert set(cache['identity_pairs'][:, 0].tolist()) == {1, 8}
    assert bool((alpha > 0).any())


def test_invalid_geometry_similarity_and_binding_rejected():
    obj = GaussianSceneObject(1, asset(), 'a'*64)
    for scale in [0., -1., float('nan'), True]:
        reject(lambda scale=scale: obj.transform(scale=scale))
    reject(lambda: obj.transform(rotation=torch.diag(torch.tensor([-1., 1., 1.]))))
    reject(lambda: obj.transform(rotation=torch.eye(3)*2.))
    reject(lambda: GaussianSceneObject(1, asset(), 'not-a-hash'))
    a = asset(); a['ids'][1] = a['ids'][0]
    reject(lambda: GaussianSceneObject(1, a, 'a'*64))
    a = asset(); a['covariance'][0, 0, 0] = -1.
    reject(lambda: GaussianSceneObject(1, a, 'a'*64))
    reject(lambda: obj.grounded(0.))
