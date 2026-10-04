import torch

from real_video.adaptive_gaussian_renderer import footprint_bounds, project_guarded, render
from real_video.gaussian3d import render as legacy_render


def fixture():
    torch.manual_seed(319)
    p = torch.rand(8, 3)*.5
    p[:, 1] += .4
    covariance = torch.eye(3)[None].repeat(8, 1, 1)*.0009
    return dict(position=p, covariance=covariance, colour=torch.rand(8, 3), opacity=torch.rand(8)*.8+.1,
                eye=torch.tensor([0., 1., 3.]), target=torch.tensor([0., .6, 0.]), height=19, width=23, ground=False)


def test_fixed_radius_matches_existing_renderer():
    a = fixture()
    for ground in (False, True):
        a['ground'] = ground
        expected = legacy_render(**a, radius=3)
        actual = render(**a, fixed_radius=3, fragment_budget=31)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)


def test_chunk_size_invariance_and_budget():
    a = fixture()
    x = render(**a, fragment_budget=7, return_stats=True)
    y = render(**a, fragment_budget=100_000, return_stats=True)
    for xx, yy in zip(x[:2], y[:2]):
        torch.testing.assert_close(xx, yy, atol=2e-6, rtol=2e-6)
    assert x[2]['largest_candidate_chunk'] <= 7


def test_adaptive_matches_exhaustive_large_stencil():
    a = fixture()
    a['covariance'] *= 30
    expected = legacy_render(**a, radius=30)
    actual = render(**a, fragment_budget=53)
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)


def test_large_splat_restores_truncated_support():
    a = fixture()
    a = {k: v[:1] if k in ('position', 'covariance', 'colour', 'opacity') else v for k, v in a.items()}
    a['covariance'] *= 200
    short = render(**a, fixed_radius=3)[1]
    full = render(**a, fragment_budget=17)[1]
    assert int((full > 1e-3).sum()) > int((short > 1e-3).sum())*2


def test_rotated_ellipse_bounds_include_all_thresholded_pixels():
    mean = torch.tensor([[7.2, 8.1]])
    cov = torch.tensor([[[9., 5.], [5., 4.]]])
    lo, hi, size = footprint_bounds(mean, cov, torch.tensor([.8]), 21, 24)
    yy, xx = torch.meshgrid(torch.arange(21), torch.arange(24), indexing='ij')
    delta = torch.stack((xx, yy), -1).float()+.5-mean
    alpha = .8*torch.exp(-.5*torch.einsum('hwi,ij,hwj->hw', delta, torch.linalg.inv(cov[0]), delta))
    assert not torch.any((alpha > 1e-4) & ((xx < lo[0, 0]) | (xx > hi[0, 0]) | (yy < lo[0, 1]) | (yy > hi[0, 1])))


def test_input_state_is_unchanged():
    a = fixture()
    before = {k: v.clone() for k, v in a.items() if isinstance(v, torch.Tensor)}
    render(**a)
    for key, original in before.items():
        assert torch.equal(a[key], original)


def test_far_off_axis_near_plane_cannot_flood_image():
    args = dict(position=torch.tensor([[5., 9., .021]]), covariance=torch.eye(3)[None]*.02,
        colour=torch.tensor([[0., 1., 0.]]), opacity=torch.tensor([.995]),
        eye=torch.zeros(3), target=torch.tensor([0., 0., 1.]), height=19, width=23, ground=False)
    unguarded = render(**args, slope_guard=None)[1]
    guarded = render(**args)[1]
    assert float(unguarded.max()) > .9
    assert float(guarded.max()) == 0
    projection_args = [args[k] for k in ('position', 'covariance', 'eye', 'target', 'height', 'width')]
    x, y = project_guarded(*projection_args, slope_guard=None), project_guarded(*projection_args)
    assert torch.equal(x[0], y[0]) and torch.equal(x[2], y[2])


def test_empty_and_behind_camera():
    a = fixture()
    a['position'][:, 2] = 10
    image, alpha = render(**a)
    assert torch.isfinite(image).all() and alpha.max() == 0
    for key in ('position', 'covariance', 'colour', 'opacity'):
        a[key] = a[key][:0]
    empty_image, empty_alpha = render(**a)
    torch.testing.assert_close(empty_image, image)
    assert empty_alpha.max() == 0


def test_explicit_work_limit_rejects_not_silent_truncation():
    try:
        render(**fixture(), max_candidate_pixels=1)
    except ValueError as exc:
        assert 'no hidden radius clamp' in str(exc)
    else:
        raise AssertionError('Expected explicit bounded-work rejection')


def test_invalid_inputs_rejected():
    a = fixture()
    for key, value in [('fragment_budget', 0), ('alpha_threshold', 0.), ('fov', 180.), ('fixed_radius', -1), ('slope_guard', .5)]:
        try:
            render(**a, **{key: value})
        except ValueError:
            pass
        else:
            raise AssertionError(key)


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith('test_')]
    for name, fn in tests:
        fn()
        print('PASS', name)
    print(f'{len(tests)} CPU tests passed')
