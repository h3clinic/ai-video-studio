import unittest
import torch
from real_video.gaussian3d import render
from real_video.fit_appearance3d import FixedColourOperator,camera_from,outward_sign


class Appearance3DTests(unittest.TestCase):
    def test_winding_reversal_changes_visibility_normal_sign(self):
        v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        f=torch.tensor([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        self.assertEqual(float(outward_sign(v,f)),1.)
        self.assertEqual(float(outward_sign(v,f.flip(1))),-1.)
        with self.assertRaises(ValueError): outward_sign(v,f[:-1])

    def test_cache_uses_original_ids_and_exact_alpha_weights(self):
        p=torch.tensor([[0.,0.,0.],[0.,0.,1.]])
        c=torch.tensor([[.1,.2,.9],[.8,.2,.1]]); cov=torch.eye(3).expand(2,3,3)*.02
        rgb,alpha,cache=render(p,cov,c,torch.ones(2)*.9,torch.tensor([0.,0.,4.]),torch.zeros(3),20,20,ground=False,return_cache=True)
        expect=rgb+(1-alpha[...,None])*(.5-torch.tensor([.15,.19,.24]))
        actual=FixedColourOperator(cache,alpha)(c)
        torch.testing.assert_close(actual,expect)

    def test_colour_gradient(self):
        colour=torch.tensor([[.2,.4,.6]],requires_grad=True)
        cache=dict(pixel=torch.tensor([0,1]),ids=torch.tensor([0,0]),weight=torch.tensor([.8,.3]))
        op=FixedColourOperator(cache,torch.tensor([[.8,.3]]))
        op(colour).sum().backward()
        torch.testing.assert_close(colour.grad,torch.full((1,3),1.1))

    def test_camera_gradient_finite(self):
        x=torch.tensor([0.,0.,.6,0.,.3],requires_grad=True)
        eye,target=camera_from(x,torch.tensor(0.))
        p=torch.tensor([[.1,.2,.1]])
        _,alpha=render(p,torch.eye(3)[None]*.02,torch.ones(1,3),torch.ones(1)*.8,eye,target,20,20,ground=False)
        alpha.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(float(x.grad.abs().sum()),0)
