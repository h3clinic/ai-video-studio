import unittest
import torch
from real_video.surface_skin3d import SurfaceSkinner
from real_video.gaussian3d import rotation


def example():
    vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=torch.tensor([[0,1,2],[1,3,2]])
    # Same physical shared-edge point represented from both triangles.
    bary=torch.tensor([[0.,.5,.5],[.5,0.,.5]])
    asset=dict(mesh_vertices=vertices,mesh_faces=faces,face_id=torch.tensor([0,1]),barycentric=bary,
               frame=torch.eye(3).expand(2,3,3),scale=torch.tensor([[.1,.2,.03]]).expand(2,3))
    rig=dict(joints=torch.tensor([[0.,0.,0.],[1.,0.,0.]]),parents=[-1,0],
             vertex_weights=torch.tensor([[1.,0.],[.5,.5],[.8,.2],[0.,1.]]))
    return asset,rig


class SurfaceSkin3DTests(unittest.TestCase):
    def test_rest_identity(self):
        a,r=example(); op=SurfaceSkinner(a,r)
        p,c,_,v,area=op(torch.eye(3).expand(2,3,3))
        torch.testing.assert_close(v,a['mesh_vertices']); torch.testing.assert_close(p,torch.tensor([[.5,.5,0.]]).expand(2,3))
        torch.testing.assert_close(c,op.covariance); torch.testing.assert_close(area,torch.ones(2))

    def test_global_rigid_transport(self):
        a,r=example(); op=SurfaceSkinner(a,r); q=rotation(torch.tensor([0.,1.,0.]),torch.tensor(.7))
        local=torch.stack((q,torch.eye(3))); p,c,_,v,area=op(local,torch.tensor([.1,.2,.3]))
        torch.testing.assert_close(v,a['mesh_vertices']@q.T+torch.tensor([.1,.2,.3]))
        torch.testing.assert_close(c,q@op.covariance@q.T); torch.testing.assert_close(area,torch.ones(2))

    def test_shared_edge_remains_connected_under_articulation(self):
        a,r=example(); op=SurfaceSkinner(a,r)
        q=rotation(torch.tensor([[0.,0.,1.]]).expand(2,3),torch.tensor([0.,.6]))
        p,c,_,_,_=op(q)
        torch.testing.assert_close(p[0],p[1]); self.assertTrue((torch.linalg.eigvalsh(c)>0).all())

    def test_reject_bad_weights(self):
        a,r=example(); r['vertex_weights'][0]*=2
        with self.assertRaises(ValueError): SurfaceSkinner(a,r)

    def test_unused_degenerate_face_is_preserved_not_inverted(self):
        a,r=example(); a['mesh_faces']=torch.cat((a['mesh_faces'],torch.tensor([[0,0,0]])))
        op=SurfaceSkinner(a,r); self.assertEqual(op.unused_degenerate_faces,1)
        p,c,_,_,area=op(torch.eye(3).expand(2,3,3))
        self.assertTrue(torch.isfinite(c).all()); self.assertEqual(len(area),2)
