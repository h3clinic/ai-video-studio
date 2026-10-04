import unittest
import torch
from real_video.controller_rig import dual_quaternion_skin, two_bone, from_to, DualSurfaceSkinner, CatRetargeter, guarded_rig
from real_video.gaussian3d import rotation
from tests.test_surface_skin3d import example


class ControllerRigTests(unittest.TestCase):
    def test_rest_ik_reference_preserves_posed_cat(self):
        vertices=torch.tensor([[0.,0.,0.]])
        rig=guarded_rig(vertices)
        # Deliberately very different proportions and pose from the cat.
        source=torch.arange(81,dtype=torch.float32).reshape(27,3)/100
        targeter=CatRetargeter(rig,source,mode='rest_ik')
        local,shift,feet,error,contacts=targeter.step(source,torch.zeros(3))
        torch.testing.assert_close(local,torch.eye(3).repeat(17,1,1),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(feet,rig['joints'][[7,10,13,16]],atol=2e-6,rtol=2e-6)
        self.assertFalse(any(contacts))

    def test_dual_skin_rigid(self):
        vertices=torch.tensor([[1.,0,0],[0.,1,0]])
        r=rotation(torch.tensor([0.,0,1]),torch.tensor(.6))
        t=torch.tensor([[.4,.2,-.1]])
        moved=dual_quaternion_skin(vertices,torch.ones(2,1),r[None],t)
        torch.testing.assert_close(moved,vertices@r.T+t)

    def test_surface_rest(self):
        a,r=example(); skin=DualSurfaceSkinner(a,r)
        p,c,_,v,areas=skin(torch.eye(3).repeat(2,1,1))
        torch.testing.assert_close(v,a['mesh_vertices'])
        torch.testing.assert_close(c,skin.covariance)
        torch.testing.assert_close(areas,torch.ones(2))

    def test_shared_edge(self):
        a,r=example(); skin=DualSurfaceSkinner(a,r)
        rot=rotation(torch.tensor([[0.,0,1]]).repeat(2,1),torch.tensor([0.,.7]))
        p,c,*_=skin(rot)
        torch.testing.assert_close(p[0],p[1])
        self.assertTrue(bool((torch.linalg.eigvalsh(c)>0).all()))

    def test_ik_lengths_and_reach(self):
        hip=torch.zeros(3); target=torch.tensor([.8,-1.,0]); pole=torch.tensor([1.,0,0])
        k,f,e=two_bone(hip,target,pole,torch.tensor(.8),torch.tensor(.7))
        torch.testing.assert_close(k.norm(),torch.tensor(.8))
        torch.testing.assert_close((f-k).norm(),torch.tensor(.7))
        self.assertLess(float(e),1e-5)

    def test_ik_unreachable_reports(self):
        k,f,e=two_bone(torch.zeros(3),torch.tensor([0.,-5.,0]),torch.tensor([1.,0,0]),torch.tensor(1.),torch.tensor(1.))
        self.assertGreater(float(e),2.9)
        torch.testing.assert_close(k.norm(),torch.tensor(1.))
        torch.testing.assert_close((f-k).norm(),torch.tensor(1.))

    def test_from_to_opposite(self):
        a=torch.tensor([1.,0,0]); r=from_to(a,-a)
        torch.testing.assert_close(r@a,-a)
        torch.testing.assert_close(torch.linalg.det(r),torch.tensor(1.))
