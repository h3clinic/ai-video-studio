import unittest
import torch
from real_video.gaussian3d import rotation,forward_kinematics,skin,project,render


class Gaussian3DTests(unittest.TestCase):
    def test_rotation_is_proper(self):
        r=rotation(torch.tensor([0.,1.,0.]),torch.tensor(.7))
        torch.testing.assert_close(r.T@r,torch.eye(3)); torch.testing.assert_close(torch.det(r),torch.tensor(1.))

    def test_skeleton_bone_length(self):
        j=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        r=rotation(torch.tensor([[0.,0.,1.]]).expand(3,-1),torch.tensor([.2,.5,-.4]))
        t=forward_kinematics(j,[-1,0,1],r)
        torch.testing.assert_close((t[1:,:3,3]-t[:-1,:3,3]).norm(dim=1),torch.ones(2))

    def test_skin_identity(self):
        p=torch.randn(5,3); cov=torch.eye(3).expand(5,3,3)
        j=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
        q,c,_=skin(p,cov,j,[-1,0],torch.eye(3).expand(2,3,3),torch.tensor([[0,1]]).expand(5,-1),torch.full((5,2),.5))
        torch.testing.assert_close(q,p); torch.testing.assert_close(c,cov)

    def test_perspective_depth(self):
        p=torch.tensor([[1.,0.,0.],[1.,0.,-2.]])
        m,_,_,_,_=project(p,torch.eye(3).expand(2,3,3)*.001,torch.tensor([0.,0.,4.]),torch.zeros(3),32,32)
        self.assertGreater(m[0,0],m[1,0])

    def test_depth_occlusion_and_order(self):
        p=torch.tensor([[0.,0.,1.],[0.,0.,0.]])
        c=torch.tensor([[1.,0.,0.],[0.,0.,1.]]); cov=torch.eye(3).expand(2,3,3)*.04
        eye=torch.tensor([0.,0.,4.]); target=torch.zeros(3); op=torch.ones(2)*.99
        a,_=render(p,cov,c,op,eye,target,32,32,ground=False)
        b,_=render(p.flip(0),cov,c.flip(0),op,eye,target,32,32,ground=False)
        torch.testing.assert_close(a,b); self.assertGreater(a[16,16,0],a[16,16,2])

    def test_camera_parallax(self):
        p=torch.tensor([[.1,0.,.4],[.1,0.,-.4]])
        cov=torch.eye(3).expand(2,3,3)*.001
        m,*_=project(p,cov,torch.tensor([3.,0.,0.]),torch.zeros(3),32,32)
        self.assertGreater(abs(float(m[0,0]-m[1,0])),5)

    def test_offcenter_projection_matches_viewport_translation(self):
        p=torch.tensor([[0.,0.,0.]])
        args=(p,torch.eye(3)[None]*.01,torch.tensor([0.,0.,4.]),torch.zeros(3),32,32)
        a,*_=project(*args); b,*_=project(*args,principal=[20.,10.])
        torch.testing.assert_close(b-a,torch.tensor([[4.,-6.]]))

    def test_hidden_gaussians_are_not_deleted_or_modified(self):
        p=torch.tensor([[0.,0.,1.],[0.,0.,0.],[0.,0.,8.]])
        original=p.clone(); cov=torch.eye(3).expand(3,3,3)*.01
        render(p,cov,torch.ones(3,3),torch.ones(3)*.9,torch.tensor([0.,0.,4.]),torch.zeros(3),16,16,ground=False)
        torch.testing.assert_close(p,original)

    def test_empty_visible_set(self):
        rgb,alpha=render(torch.tensor([[0.,0.,8.]]),torch.eye(3)[None]*.01,torch.ones(1,3),torch.ones(1),torch.tensor([0.,0.,4.]),torch.zeros(3),16,16,ground=False)
        self.assertTrue(torch.isfinite(rgb).all()); self.assertEqual(float(alpha.sum()),0.)

    def test_rigid_skin_covariance_transport(self):
        p=torch.tensor([[1.,2.,3.]]); cov=torch.diag(torch.tensor([.01,.02,.03]))[None]
        q=rotation(torch.tensor([[0.,1.,0.]]),torch.tensor([.7]))
        moved,c,_=skin(p,cov,torch.zeros(1,3),[-1],q,torch.zeros(1,1,dtype=torch.long),torch.ones(1,1))
        torch.testing.assert_close(moved,(q[0]@p.T).T)
        torch.testing.assert_close(c,q@cov@q.transpose(-1,-2))
