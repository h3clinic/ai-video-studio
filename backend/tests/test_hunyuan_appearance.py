import numpy as np
import torch
import trimesh
import unittest
from real_video.hunyuan_appearance import sample_surface


def test_surface_binding_and_covariance():
    mesh=trimesh.creation.icosphere(subdivisions=1)
    a=sample_surface(mesh.vertices,mesh.faces,1000)
    reconstructed=(torch.tensor(mesh.vertices[mesh.faces[a['face_id']]])*a['barycentric'][...,None]).sum(1)
    assert torch.allclose(reconstructed.float(),a['position'],atol=1e-6)
    assert torch.allclose(a['barycentric'].sum(1),torch.ones(1000))
    assert (a['barycentric']>=0).all()
    assert torch.linalg.eigvalsh(a['covariance']).min()>0
    assert torch.allclose(a['covariance'],a['covariance'].transpose(1,2),atol=1e-8)
    assert len(torch.unique(a['ids']))==1000
    b=sample_surface(mesh.vertices,mesh.faces,1000)
    assert torch.equal(a['position'],b['position'])


class HunyuanAppearanceTests(unittest.TestCase):
    def test_surface_binding_and_covariance(self):
        test_surface_binding_and_covariance()
