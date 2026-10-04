import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from real_video.latent import GaussianCodec
from real_video.model import GaussianVideoDenoiser
from real_video.session import GaussianVideoSession


class SessionTests(unittest.TestCase):
    def test_source_free_reuse_output_ownership_and_close(self):
        codec=GaussianCodec(width=8); prior=GaussianVideoDenoiser(classes=2,width=8,channels=16)
        packet=dict(labels=['A','B'],latent_shape=(4,8,8),config=prior.config,model=prior.state_dict(),
                    codec_config=codec.config,codec=codec.state_dict(),mean=torch.zeros(1,9,1,1,1),
                    std=torch.ones(1,9,1,1,1),latent_mean=torch.zeros(1,16,1,1,1),latent_std=torch.ones(1,16,1,1,1))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tiny.pt'; torch.save(packet,path)
            with patch.object(GaussianCodec,'encode',side_effect=AssertionError('No encoder at generation')):
                session=GaussianVideoSession(path,device='cpu',cuda_graph=False,precast_weights=False)
                with patch('torch.load',side_effect=AssertionError('No reloading after setup')):
                    first=session.generate('A',17,steps=2); snapshot=first['fields'].clone()
                    second=session.generate('B',18,steps=2); repeated=session.generate('A',17,steps=2)
                torch.testing.assert_close(first['fields'],snapshot,rtol=0,atol=0)
                torch.testing.assert_close(first['fields'],repeated['fields'],rtol=0,atol=0)
                self.assertFalse(torch.equal(first['latent'],second['latent']))
                self.assertEqual(first['video'].shape,(8,3,64,64))
                self.assertEqual(first['fields'].device.type,'cpu')
                self.assertFalse(hasattr(session.decoder,'encoder'))
                with self.assertRaises(ValueError): session.generate('missing',1,steps=2)
                session.close(); session.close()
                with self.assertRaisesRegex(RuntimeError,'closed'): session.generate('A',17,steps=2)


if __name__=='__main__': unittest.main()
