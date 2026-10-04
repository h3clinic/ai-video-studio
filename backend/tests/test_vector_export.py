"""CLI-level immutable vector export and post-export recovery checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from real_video.checkpoint_io import save_inference_checkpoint, load_verified, digest
from real_video.vector_recording import main
from tests.test_vector_recording import example


class VectorExportTests(unittest.TestCase):
    def test_export_audit_and_recovery_preserve_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); asset=example()
            asset_path=root/'asset.pt'; motion_path=root/'motion.pt'; output=root/'vectors.pt'
            save_inference_checkpoint(asset,asset_path)
            vertices=torch.stack([asset['mesh_vertices'],asset['mesh_vertices']+.03])
            save_inference_checkpoint(dict(vertices=vertices,fps=16.,asset_sha256=digest(asset_path)),motion_path)
            argv=['vector_recording',str(motion_path),'--asset',str(asset_path),'--out',str(output)]
            with patch('sys.argv',argv): main()
            checksum=digest(output)
            audit=json.loads(output.with_suffix('.audit.json').read_text())
            self.assertEqual(audit['packet_bytes'],output.stat().st_size)
            self.assertEqual(audit['sha256'],checksum)
            self.assertLess(audit['position_replay_max_error'],1e-6)
            self.assertTrue(torch.equal(load_verified(output)['ids'],asset['ids']))
            with patch('sys.argv',argv),self.assertRaises(FileExistsError): main()
            # Simulate interrupted audit writing, not a failed tensor export.
            output.with_suffix('.audit.json').unlink()
            with patch('sys.argv',argv+['--audit-only']): main()
            self.assertEqual(digest(output),checksum)

    def test_mismatched_asset_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); asset=example()
            save_inference_checkpoint(asset,root/'asset.pt')
            save_inference_checkpoint(dict(vertices=torch.stack([asset['mesh_vertices']]*2),fps=16.,asset_sha256='incorrect'),root/'motion.pt')
            argv=['vector_recording',str(root/'motion.pt'),'--asset',str(root/'asset.pt'),'--out',str(root/'out.pt')]
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'hashes differ'): main()
            self.assertFalse((root/'out.pt').exists())


if __name__=='__main__': unittest.main()
