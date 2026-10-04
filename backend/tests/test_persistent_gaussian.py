import unittest
import torch
from real_video.persistent_gaussian import PersistentGaussianDynamics, GaussianMemorySession, state_bytes, tracked_fields_to_state, render_state
from real_video.representation import render_fields


class PersistentTests(unittest.TestCase):
    def inputs(self):
        raw = torch.zeros(1,9,8,8); raw[:,4]=1
        first = tracked_fields_to_state(raw)
        second = first.clone(); second[:,0]+=.002
        return first, second

    def test_fixed_memory_and_unit_frames_over_long_rollout(self):
        model = PersistentGaussianDynamics()
        first,second = self.inputs()
        session = GaussianMemorySession(model, model.initialize(first,second))
        count = state_bytes(session.state)
        for _ in range(128):
            frame = session.advance()
            self.assertEqual(state_bytes(session.state), count)
        self.assertLess((frame[:,4:6].norm(dim=1)-1).abs().max().item(),1e-6)
        self.assertTrue(torch.isfinite(frame).all())
        self.assertFalse(any(v.requires_grad for v in session.state.values()))

    def test_snapshot_exact_resume_and_output_ownership(self):
        model = PersistentGaussianDynamics()
        a,b = self.inputs(); session = GaussianMemorySession(model,model.initialize(a,b))
        first = session.advance(); saved = first.clone(); packet = session.snapshot()
        expected = session.advance(); session.advance(); session.restore(packet)
        self.assertTrue(torch.equal(session.advance(),expected))
        self.assertTrue(torch.equal(first,saved))

    def test_legacy_render_conversion(self):
        torch.manual_seed(8)
        raw=torch.randn(1,9,8,8)*.1; raw[:,4]+=1
        explicit=render_state(tracked_fields_to_state(raw))
        original=render_fields(raw.unsqueeze(2))[:,0]
        self.assertLess((explicit-original).abs().max().item(),2e-5)

    def test_recurrent_gradients(self):
        model=PersistentGaussianDynamics(); a,b=self.inputs()
        state=model.initialize(a,b)
        for _ in range(3): state=model.step(state)
        state['fields'].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__': unittest.main()
