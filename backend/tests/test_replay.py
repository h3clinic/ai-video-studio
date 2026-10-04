import unittest
from unittest.mock import patch
import numpy as np
import torch
from real_video.latent import GaussianCodec,inference_decoder
from real_video.replay_fit import temporal_basis,expand_coefficients,physical_fields,replay
from real_video.representation import encode_clip,render_fields,render_fields_chunked


class ReplayTests(unittest.TestCase):
    def test_anchored_positions_are_bounded_and_learnable(self):
        coefficients=torch.randn(9,4,4,4,requires_grad=True)
        fields=physical_fields(coefficients,temporal_basis(8,4),position_limit=0.03125)
        self.assertLessEqual(fields[:2].abs().max().item(),0.03125)
        fields.square().mean().backward()
        self.assertTrue(torch.isfinite(coefficients.grad).all())
        self.assertGreater(coefficients.grad[:2].abs().sum().item(),0)

    def test_chunked_splatting_preserves_pixels_and_gradients(self):
        fields=(torch.randn(1,9,3,8,8)*0.1).requires_grad_()
        whole=render_fields(fields)
        streamed=render_fields_chunked(fields,frame_chunk=1)
        torch.testing.assert_close(whole,streamed,rtol=1e-6,atol=1e-6)
        whole.sum().backward(); gradient=fields.grad.clone(); fields.grad=None
        streamed.sum().backward()
        torch.testing.assert_close(fields.grad,gradient,rtol=1e-5,atol=1e-5)

    def test_inference_decoder_exact_and_encoder_removed(self):
        original=GaussianCodec(width=8).eval()
        decoder=inference_decoder(original.config,original.state_dict())
        self.assertFalse(hasattr(decoder,'encoder'))
        z=torch.randn(1,16,4,8,8)
        torch.testing.assert_close(original.decode(z),decoder.decode(z),rtol=0,atol=0)
        self.assertLess(sum(p.numel() for p in decoder.parameters()),sum(p.numel() for p in original.parameters()))

    def test_temporal_basis_is_orthonormal_and_full_rank_roundtrip(self):
        basis=temporal_basis(8,8)
        torch.testing.assert_close(basis.T@basis,torch.eye(8),atol=1e-6,rtol=1e-6)
        fields=torch.randn(9,8,4,4)
        coefficients=torch.einsum('tk,cthw->ckhw',basis,fields)
        torch.testing.assert_close(expand_coefficients(coefficients,basis),fields,atol=3e-6,rtol=1e-5)

    def test_replay_never_reads_source_and_chunking_preserves_pixels(self):
        packet=dict(coefficients=torch.randn(9,2,4,4)*0.1,frames=3,size=32,radius=3)
        with patch('real_video.replay_fit.read_video',side_effect=AssertionError('Source read')):
            a=replay(packet,device='cpu',chunk=1)
            b=replay(packet,device='cpu',chunk=3)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        self.assertEqual(a.shape,(3,3,32,32))

    def test_native_resolution_encoder_and_pixel_centres(self):
        frame=np.full((1,128,128,3),127,dtype=np.uint8)
        field=torch.from_numpy(encode_clip(frame,grid=64))[None]
        output=render_fields(field,size=128,radius=5,min_log_scale=-2.)
        torch.testing.assert_close(output,torch.full_like(output,127/255),atol=1e-5,rtol=1e-5)
        self.assertEqual(field.shape,(1,9,1,64,64))


if __name__=='__main__': unittest.main()
