import unittest
import torch
import torch.nn.functional as F
from real_video.texture_restore import tiled

class NearestModel:
    scale=4
    def __call__(self,x):return F.interpolate(x,scale_factor=4,mode='nearest')

class LocalModel:
    scale=2
    def __call__(self,x):
        return F.interpolate(F.avg_pool2d(x,3,1,1),scale_factor=2,mode='nearest')

class TextureRestoreTests(unittest.TestCase):
    def test_invalid_tiling_rejected(self):
        for tile,pad in [(0,1),(1,-1)]:
            with self.assertRaises(ValueError):tiled(NearestModel(),torch.zeros(1,3,2,2),tile,pad)
    def test_uneven_tiles_preserve_alignment(self):
        x=torch.arange(3*19*23,dtype=torch.float32).reshape(1,3,19,23)
        torch.testing.assert_close(tiled(NearestModel(),x,7,3),NearestModel()(x))

    def test_padding_preserves_local_context(self):
        x=torch.rand(1,3,19,23)
        torch.testing.assert_close(tiled(LocalModel(),x,7,3),LocalModel()(x))

    def test_input_is_not_modified(self):
        x=torch.rand(1,3,8,8);before=x.clone()
        tiled(NearestModel(),x,4,2)
        torch.testing.assert_close(x,before,rtol=0,atol=0)
