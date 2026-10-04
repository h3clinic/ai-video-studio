import unittest
from unittest.mock import patch
import torch
from real_video.latent import GaussianCodec,kl_loss
from real_video.model import GaussianVideoDenoiser,sample
from real_video.representation import render_fields,decode_state,mirror_fields
from real_video.prepare_windows import validate_windows


class GaussianLatentTests(unittest.TestCase):
    def test_extra_windows_reject_heldout_sources_and_duplicates(self):
        source=dict(file='training.avi',split='train',label='Action',group='g1',sha256='abc',frame_indices=list(range(8,16)))
        record=dict(source,frame_indices=list(range(8)),source_frames=16)
        packet=dict(fields=torch.zeros(1,9,8,32,32),metadata=[record],labels=['Action'],dataset_protocol_sha256='protocol')
        self.assertTrue(validate_windows(packet,[source],['Action'],'protocol'))
        record['file']='heldout.avi'
        with self.assertRaisesRegex(ValueError,'Non-training'): validate_windows(packet,[source],['Action'],'protocol')
        record['file']='training.avi'; record['frame_indices']=source['frame_indices']
        with self.assertRaisesRegex(ValueError,'Duplicate'): validate_windows(packet,[source],['Action'],'protocol')

    def test_mirror_fields_matches_mirrored_render(self):
        f=torch.randn(1,9,8,32,32)*0.1
        f[:,4]=0.6; f[:,5]=0.8
        torch.testing.assert_close(mirror_fields(mirror_fields(f)),f)
        expected=render_fields(f).flip(-1)
        actual=render_fields(mirror_fields(f))
        self.assertLess((expected-actual).abs().mean().item(),1e-4)

    def test_dropout_disabled_for_inference(self):
        model=GaussianVideoDenoiser(classes=2,width=8,channels=16,dropout=0.15).eval()
        x=torch.randn(1,16,4,8,8); t=torch.tensor([300]); y=torch.tensor([1])
        torch.testing.assert_close(model(x,t,y),model(x,t,y),rtol=0,atol=0)

    def test_codec_shapes_compression_and_gradients(self):
        codec=GaussianCodec(width=8)
        x=torch.randn(1,9,8,32,32)
        fields,mu,logvar=codec(x)
        self.assertEqual(mu.shape,(1,16,4,8,8))
        self.assertEqual(fields.shape,x.shape)
        self.assertEqual(x.numel()//mu.numel(),18)
        ((fields-x).square().mean()+0.0001*kl_loss(mu,logvar)).backward()
        for name,p in codec.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)

    def test_latent_noise_generation_has_no_encoder_path(self):
        codec=GaussianCodec(width=8).eval()
        prior=GaussianVideoDenoiser(classes=2,width=8,channels=16).eval()
        with patch.object(codec,'encode',side_effect=AssertionError('Inference encoder called')):
            z=sample(prior,torch.tensor([1]),seed=813,steps=2,shape=(4,8,8))
            fields=codec.decode(z)
            video=render_fields(fields)
        self.assertEqual(video.shape,(1,8,3,64,64))
        self.assertTrue(torch.isfinite(video).all())
        r=decode_state(fields[:,:,0])['R']
        torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-5,rtol=1e-5)

    def test_standard_normal_posterior_has_zero_kl(self):
        self.assertEqual(kl_loss(torch.zeros(3),torch.zeros(3)).item(),0)


if __name__=='__main__': unittest.main()
