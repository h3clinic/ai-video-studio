import unittest
import numpy as np
import torch
from real_video.weld_band import fair_band,transport

def grid():
    x,y=np.meshgrid(np.arange(7),np.arange(7));v=np.column_stack((x.ravel(),y.ravel(),np.zeros(49))).astype(np.float32)
    f=[]
    for j in range(6):
        for i in range(6):
            k=j*7+i;f.extend([[k,k+1,k+7],[k+1,k+8,k+7]])
    m=(v[:,0]>1)&(v[:,0]<5)&(v[:,1]>1)&(v[:,1]<5)
    return v,np.array(f),m

class WeldTests(unittest.TestCase):
    def test_transport_preserves_ids_colors_and_untouched_face(self):
        old=torch.tensor([[0.,0,0],[1.,0,0],[0.,1,0],[3.,0,0],[4.,0,0],[3.,1,0]])
        new=old.clone();new[2,2]=.1;faces=torch.tensor([[0,1,2],[3,4,5]])
        a=dict(face_id=torch.tensor([0,1]),ids=torch.tensor([17,18]),barycentric=torch.ones(2,3)/3,
            position=old[faces].mean(1),covariance=torch.eye(3).repeat(2,1,1)*.01,
            normal=torch.tensor([[0.,0,1],[0.,0,1]]),colour=torch.rand(2,3),opacity=torch.ones(2))
        out,n=transport(a,old,new,faces);self.assertEqual(n,1)
        for k in ['ids','colour','opacity','barycentric','face_id']:self.assertTrue(torch.equal(a[k],out[k]))
        self.assertTrue(torch.equal(a['position'][1],out['position'][1]))
        self.assertTrue(torch.equal(a['covariance'][1],out['covariance'][1]))
        self.assertTrue(bool((torch.linalg.eigvalsh(out['covariance'])>0).all()))
    def test_fixed_context_and_energy(self):
        v,f,m=grid();v[24,2]=.7
        out,r=fair_band(v,f,m,max_displacement=.4)
        self.assertTrue(np.array_equal(out[~m],v[~m]));self.assertEqual(r['flipped_faces'],0)
        self.assertLess(r['laplacian_energy_after'],r['laplacian_energy_before'])
        self.assertLessEqual(r['max_displacement'],.40001)
    def test_rigid_equivariance(self):
        v,f,m=grid();v[24,2]=.7;q=np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32);t=np.array([4,2,3])
        a,_=fair_band(v,f,m);b,_=fair_band(v@q.T+t,f,m)
        np.testing.assert_allclose(b,a@q.T+t,atol=1e-6)
    def test_requires_fixed_context(self):
        v,f,m=grid()
        for bad in [np.zeros(len(v),bool),np.ones(len(v),bool)]:
            with self.assertRaises(ValueError):fair_band(v,f,bad)
