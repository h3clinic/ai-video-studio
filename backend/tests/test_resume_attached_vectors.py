import tempfile
import unittest
from pathlib import Path
import torch
from unittest.mock import patch
from real_video.checkpoint_io import save_inference_checkpoint, digest
from real_video.resume_attached_vectors import advance, resume, resume_input_guard, forecast_model_provenance, continuation_tensor_bytes
from real_video.vector_motion_network import VectorMotionNetwork
from real_video.vector_motion_residual import ResidualVelocityNetwork


class ResumeAttachedTests(unittest.TestCase):
    def test_saved_resume_uses_only_checkpoint_asset_weights(self):
        self.check_model_resume(VectorMotionNetwork(hidden=8, residual_scale=.002, damping_init=.75).eval())

    def test_residual_architecture_resume_uses_retained_state(self):
        model = ResidualVelocityNetwork(hidden=8, residual_scale=.02).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.head[-1].weight, std=.03)
        self.check_model_resume(model)

    def check_model_resume(self, model):
        torch.manual_seed(917)
        rest = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,0.]])
        asset = dict(mesh_vertices=rest, mesh_faces=torch.tensor([[0,1,2],[1,3,2]]))
        observed = torch.stack((rest[:,:2]*.5-.02, rest[:,:2]*.5-.01, rest[:,:2]*.5))[None]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ap, mp, cp = root/'asset.pt', root/'weights.pt', root/'continuation.pt'
            save_inference_checkpoint(asset, ap)
            save_inference_checkpoint(dict(model=model.state_dict(), config=model.config,
                                           architecture=getattr(model,'architecture','damped_velocity_v1')), mp)
            packet = dict(format_version=1, state=model.initialize(observed, torch.eye(4)[None]),
                          current_vertices=rest.clone(), canonical_control_reference=observed[0,2].clone(),
                          bound_vertex_index=torch.arange(4), bound_depth=torch.ones(4), focal=480., camera_rotation=torch.eye(3),
                          constraint_weight=torch.ones(4)*10, steps_completed=0,
                          asset_path=str(ap), asset_sha256=digest(ap), model_path=str(mp), model_sha256=digest(mp))
            save_inference_checkpoint(packet, cp)
            with torch.no_grad():
                expected, vertices, controls = advance(model, asset, packet, 2)
                # initialize must NOT be used in resume; only retained state.
                with patch.object(VectorMotionNetwork, 'initialize', side_effect=AssertionError('Seed initialization forbidden')):
                    actual, saved_vertices, saved_controls, reads = resume(cp, 2)
            self.assertEqual(set(reads), {str(p.resolve()) for p in [cp, ap, mp]})
            self.assertEqual(len(reads), 3)
            self.assertEqual(continuation_tensor_bytes(packet), continuation_tensor_bytes(actual))
            torch.testing.assert_close(vertices, saved_vertices, atol=0, rtol=0)
            torch.testing.assert_close(controls, saved_controls, atol=0, rtol=0)
            for key in expected['state']:
                torch.testing.assert_close(expected['state'][key], actual['state'][key], atol=0, rtol=0)

    def test_forecast_mode_provenance_is_explicit(self):
        forecast=dict(controls={'residual_learned':None,'previous_learned':None,'constant_velocity':None},
                      vertices={'residual_learned':None,'previous_learned':None,'constant_velocity':None},
                      model_paths={'residual_learned':'new.pt','previous_learned':'old.pt'},new_model_sha256='newhash',old_model_sha256='oldhash')
        self.assertEqual(forecast_model_provenance(forecast,'residual_learned'),(Path('new.pt').resolve(),'newhash'))
        self.assertEqual(forecast_model_provenance(forecast,'previous_learned'),(Path('old.pt').resolve(),'oldhash'))
        with self.assertRaises(ValueError):
            forecast_model_provenance(forecast,'constant_velocity')

    def test_guard_rejects_seed_and_image_reads(self):
        import cv2
        import numpy as np
        from PIL import Image
        with resume_input_guard(Path('only_checkpoint.pt')):
            for fn, arg in [(torch.load, 'observed_seed.pt'), (cv2.VideoCapture, 'source.mp4'),
                            (cv2.imread, 'source.png'), (Image.open, 'source.png'), (np.load, 'future.npy')]:
                with self.assertRaises(AssertionError):
                    fn(arg)


if __name__ == '__main__':
    unittest.main()
