import unittest
import torch
from real_video.gaussian_object_agent import GaussianObjectAgentMemory, IncrementalGaussianFrame, diagonal_curvature_step
from real_video.gaussian_scene_objects import render_scene


def asset(x=0., z=0., colour=(1., .3, .05)):
    return dict(position=torch.tensor([[x, 0., z]], dtype=torch.float32),
        covariance=torch.eye(3)[None]*.01, colour=torch.tensor([colour]),
        opacity=torch.tensor([.95]), ids=torch.tensor([7]))


class ObjectAgentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.memory=GaussianObjectAgentMemory({1:asset(), 2:asset(.8, -.2, (.2,.8,.2))})

    def lease(self):
        return self.memory.lease(1, max_points=8, lower=(-1,-1,-1), upper=(1,1,1))

    def test_object_view_is_small_owned_and_nonaliasing(self):
        view=self.lease();view['asset']['colour'].zero_()
        state,_=self.memory.snapshot()
        self.assertGreater(float(state[1]['colour'].sum()),0)
        self.assertEqual(view['bytes'],72)
        self.assertEqual(len(view['asset']['position']),1)

    def test_replacement_preserves_other_object_and_invalidates_lifetime(self):
        before,_=self.memory.snapshot();a=self.lease();b=self.lease()
        result=self.memory.replace(a['token'],asset(.1,colour=(.8,.05,.03)))
        after,versions=self.memory.snapshot()
        for key in ('position','covariance','colour','opacity','ids'):
            self.assertTrue(torch.equal(before[2][key],after[2][key]))
        self.assertEqual(versions,{1:1,2:0})
        self.assertEqual(result['invalidated_material_revision'],0)
        with self.assertRaises(ValueError):self.memory.replace(b['token'],asset())
        with self.assertRaises(ValueError):self.memory.replace(a['token'],asset())

    def test_invalid_geometry_rejected_atomically(self):
        a=self.lease();bad=asset();bad['covariance'][0,0,0]=-1
        with self.assertRaises(ValueError):self.memory.replace(a['token'],bad)
        self.assertEqual(self.memory.snapshot()[1][1],0)
        self.memory.replace(a['token'],asset())

    def test_capability_extent_and_budget(self):
        a=self.lease()
        with self.assertRaises(ValueError):self.memory.replace('not-a-lease',asset())
        with self.assertRaises(ValueError):self.memory.replace(a['token'],asset(.9))
        bad={k:v.repeat((9,)+(1,)*(v.ndim-1)) for k,v in asset().items()}
        with self.assertRaises(ValueError):self.memory.replace(a['token'],bad)

    def test_incremental_matches_full_render_and_preserves_other_pixels(self):
        eye,target=torch.tensor([0.,0.,3.]),torch.zeros(3)
        cache=IncrementalGaussianFrame(self.memory,eye,target)
        before=cache.image();a=self.lease()
        self.memory.replace(a['token'],asset(.2,colour=(.8,.05,.03)))
        metrics=cache.update();current=cache.image()
        full,_=render_scene(self.memory.snapshot()[0].values(),eye,target,height=64,width=96,ground=False)
        self.assertTrue(torch.allclose(current,full,atol=2e-6,rtol=2e-6))
        self.assertTrue(torch.equal(before[:10],current[:10]))
        self.assertLess(metrics['rasterized_pixels'],64*96)
        self.assertGreater(float((before-current).abs().max()),.01)
        self.assertEqual(cache.update()['rasterized_pixels'],0)

    def test_occluder_is_included_in_dirty_region(self):
        memory=GaussianObjectAgentMemory({1:asset(0,-.2),2:asset(0,.2,(.1,.2,.9))})
        eye,target=torch.tensor([0.,0.,3.]),torch.zeros(3)
        cache=IncrementalGaussianFrame(memory,eye,target)
        lease=memory.lease(1,max_points=2,lower=(-1,-1,-1),upper=(1,1,1))
        memory.replace(lease['token'],asset(0,-.2,(.9,.01,.02)))
        result=cache.update()
        full,_=render_scene(memory.snapshot()[0].values(),eye,target,height=64,width=96,ground=False)
        self.assertEqual(result['rasterized_gaussians'],2)
        self.assertTrue(torch.allclose(cache.image(),full,atol=2e-6,rtol=2e-6))

    def test_removing_visible_footprint_repairs_background(self):
        eye,target=torch.tensor([0.,0.,3.]),torch.zeros(3)
        cache=IncrementalGaussianFrame(self.memory,eye,target)
        lease=self.memory.lease(1,max_points=2,lower=(-1,-1,-1),upper=(1,1,5))
        self.memory.replace(lease['token'],asset(0,4))
        cache.update()
        full,_=render_scene(self.memory.snapshot()[0].values(),eye,target,height=64,width=96,ground=False)
        self.assertTrue(torch.allclose(cache.image(),full,atol=2e-6,rtol=2e-6))

    def test_curvature_step_bounded_and_invalid_rejected(self):
        g=torch.tensor([1.,-1.]);d=torch.tensor([0.,100.])
        step=diagonal_curvature_step(g,d,max_step=.1)
        self.assertAlmostEqual(float(step[0]),-.1,places=6)
        self.assertTrue(0<float(step[1])<.011)
        with self.assertRaises(ValueError):diagonal_curvature_step(g,-torch.ones(2))


if __name__=='__main__':unittest.main()
