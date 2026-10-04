import sys
import unittest
import numpy as np
import torch
from real_video.instantmesh_adapter import REPO,regular_grid,chunk_geometry,export_gaussians
sys.path.insert(0,str(REPO))
from src.models.geometry.rep_3d.flexicubes import FlexiCubes
from src.models.renderer.synthesizer_mesh import TriplaneSynthesizer
from src.models.encoder.dino import ViTModel
from transformers import ViTConfig

class InstantMeshTests(unittest.TestCase):
    def test_grid_parity(self):
        fc=FlexiCubes('cpu')
        for r in [2,3,8]:
            a,b=fc.construct_voxel_grid(r);c,d=regular_grid(fc,r)
            torch.testing.assert_close(a,c,rtol=0,atol=0);self.assertTrue(torch.equal(b,d))
    def test_chunk_geometry_parity(self):
        torch.manual_seed(31);m=TriplaneSynthesizer(2,16).eval()
        p=torch.randn(1,3,2,8,8);xyz=torch.rand(1,25,3)*2-1;indices=torch.randint(25,(11,8))
        with torch.no_grad():
            a=m.get_geometry_prediction(p,xyz,indices);b=chunk_geometry(m,p,xyz,indices,chunk=7)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
    def test_dino_current_transformers_forward(self):
        m=ViTModel(ViTConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,image_size=32,patch_size=16),add_pooling_layer=False).eval()
        with torch.no_grad():out=m(pixel_values=torch.randn(1,3,32,32),adaln_input=torch.zeros(1,32))
        self.assertEqual(out.last_hidden_state.shape,(1,5,32))
    def test_gaussian_binding_and_colour(self):
        v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,1.]])
        f=np.array([[0,1,2]]);c=np.array([[1.,0.,0.]]*3)
        a,rotated=export_gaussians(v,f,c,128)
        positions=(torch.tensor(rotated,dtype=torch.float32)[torch.tensor(f)[a['face_id']]]*a['barycentric'][...,None]).sum(1)
        torch.testing.assert_close(positions,a['position']);torch.testing.assert_close(a['colour'],torch.tensor(c[0],dtype=torch.float32).expand(128,3))
        self.assertTrue((torch.linalg.eigvalsh(a['covariance'])>0).all())
