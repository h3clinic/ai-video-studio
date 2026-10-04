import unittest
import torch
from real_video.fit_motion3d import batched_fk, rotation_vectors, bind_asset, to_condition, lift_landmarks, project_points, mesh_weights, LANDMARKS
from real_video.gaussian3d import forward_kinematics, skin


class FittedMotionTests(unittest.TestCase):
    def test_batched_fk_matches_public_geometry(self):
        joints=torch.tensor([[0.,1.,0.],[.4,1.,0.],[.4,.5,.1]])
        parents=[-1,0,1]; rv=torch.randn(4,3,3)*.25
        local=rotation_vectors(rv); translation=torch.randn(4,3)*.1
        xyz,rot=batched_fk(joints,parents,local,translation)
        for i in range(4):
            reference=forward_kinematics(joints,parents,local[i])
            torch.testing.assert_close(xyz[i],reference[:,:3,3]+translation[i])
            torch.testing.assert_close(rot[i],reference[:,:3,:3])

    def test_batched_fk_gradients(self):
        rest=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
        rv=torch.zeros(2,3,3,requires_grad=True); shift=torch.zeros(2,3,requires_grad=True)
        xyz,_=batched_fk(rest,[-1,0,1],rotation_vectors(rv),shift)
        xyz[:,:,0].sum().backward()
        self.assertTrue(torch.isfinite(rv.grad).all()); self.assertGreater(float(rv.grad.abs().sum()),0)

    def test_rigid_bone_uses_proximal_transform(self):
        # Point halfway down forearm must rotate around elbow, not wrist.
        joints=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        local=rotation_vectors(torch.tensor([[0.,0.,0.],[0.,0.,torch.pi/2],[0.,0.,0.]]))
        point=torch.tensor([[1.5,0.,0.]]); cov=torch.eye(3)[None]*.01
        out,_,_=skin(point,cov,joints,[-1,0,1],local,torch.tensor([[1]]),torch.ones(1,1))
        torch.testing.assert_close(out,torch.tensor([[1.,.5,0.]]),atol=1e-6,rtol=1e-6)

    def test_source_crop_mapping(self):
        camera=dict(x0=20,y0=30,pad_x=10,pad_y=40,size=200,width=512)
        value=to_condition(torch.tensor([[20.,30.],[220.,230.]]),camera)
        torch.testing.assert_close(value,torch.tensor([[25.6,102.4],[537.6,614.4]]))

    def test_bilateral_lift_preserves_projection(self):
        torch.manual_seed(28)
        position=torch.rand(400,3)-torch.tensor([.5,0.,.5])
        camera=dict(eye=torch.tensor([0.,.5,2.]),target=torch.tensor([0.,.5,0.]),
                    height=480,width=832,fov=42.,size=832,x0=0,y0=0,pad_x=0,pad_y=0)
        points=torch.tensor(LANDMARKS)
        joints=lift_landmarks(position,camera,points)
        torch.testing.assert_close(project_points(joints,camera),points,atol=1e-4,rtol=1e-5)
        self.assertTrue((joints[[5,6,7,11,12,13],2]>joints[[8,9,10,14,15,16],2]).all())

    def test_continuous_mesh_weights_are_convex_and_barycentric(self):
        grid=torch.linspace(-.5,.5,8)
        x,y=torch.meshgrid(grid,grid,indexing='ij')
        vertices=torch.stack((x.flatten(),y.flatten()+.5,x.flatten()*.1),1)
        faces=[]
        for i in range(7):
            for j in range(7):
                a=i*8+j; faces.extend([[a,a+1,a+8],[a+1,a+9,a+8]])
        asset=dict(mesh_vertices=vertices,mesh_faces=torch.tensor(faces),face_id=torch.arange(6),
                   barycentric=torch.tensor([[.2,.3,.5]]).repeat(6,1))
        joints=vertices[torch.linspace(0,63,17).long()]
        vw,index,weights=mesh_weights(asset,joints)
        self.assertEqual(tuple(vw.shape),(64,17)); self.assertEqual(tuple(index.shape),(6,17))
        self.assertGreaterEqual(float(vw.min()),0.)
        torch.testing.assert_close(vw.sum(1),torch.ones(64))
        torch.testing.assert_close(weights.sum(1),torch.ones(6))
        exact=(vw[asset['mesh_faces'][asset['face_id']]]*asset['barycentric'][...,None]).sum(1)
        torch.testing.assert_close(weights,exact)


if __name__=='__main__': unittest.main()
