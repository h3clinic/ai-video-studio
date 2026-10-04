import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from real_video.checkpoint_io import save_training_checkpoint,load_verified


class CheckpointTests(unittest.TestCase):
    def test_optimizer_and_rng_recover_next_update(self):
        torch.manual_seed(901)
        model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(0.15),torch.nn.Linear(8,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        def update():
            optimizer.zero_grad()
            loss=model(torch.randn(3,4)).square().mean()
            loss.backward(); optimizer.step()
        update()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),out,1,True)
            update(); expected={k:v.clone() for k,v in model.state_dict().items()}
            restored=load_verified(out/'last.pt')
            model.load_state_dict(restored['model']); optimizer.load_state_dict(restored['optimizer']); torch.set_rng_state(restored['rng'])
            update()
            for name,value in model.state_dict().items(): torch.testing.assert_close(value,expected[name],rtol=0,atol=0)

    def test_versions_are_verified_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            save_training_checkpoint({'model':{'x':torch.tensor([1.])}},out,1,True)
            save_training_checkpoint({'model':{'x':torch.tensor([2.])}},out,2,False)
            self.assertEqual(load_verified(out/'best.pt')['model']['x'].item(),1)
            self.assertEqual(load_verified(out/'last.pt')['model']['x'].item(),2)
            self.assertEqual(len(list((out/'checkpoints').glob('*.pt'))),2)
            with self.assertRaises(FileExistsError): save_training_checkpoint({},out,2,False)

    def test_failed_write_preserves_last_good_weights(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            save_training_checkpoint({'model':{'x':torch.tensor([1.])}},out,1,True)
            with patch('real_video.checkpoint_io.torch.save',side_effect=OSError('Simulated write failure')):
                with self.assertRaises(OSError): save_training_checkpoint({'x':torch.tensor([2.])},out,2,True)
            self.assertEqual(load_verified(out/'last.pt')['model']['x'].item(),1)
            self.assertEqual(list(out.rglob('*.pending')),[])

    def test_checksum_detects_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            save_training_checkpoint({'x':torch.tensor([1.])},out,1,True)
            (out/'last.pt').write_bytes(b'\x00'*128)
            with self.assertRaisesRegex(ValueError,'digest mismatch'): load_verified(out/'last.pt')
            self.assertEqual(load_verified(out/'checkpoints/step_00000001.pt')['x'].item(),1)

    def test_nonfinite_weights_are_not_published(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                save_training_checkpoint({'x':torch.tensor([float('nan')])},out,1,True)
            self.assertFalse((out/'last.pt').exists())


if __name__=='__main__': unittest.main()
