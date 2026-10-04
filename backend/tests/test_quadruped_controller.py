import sys
import unittest
from pathlib import Path
import torch
from real_video.quadruped_controller import QuadrupedController

BUNDLE=Path('artifacts/real_video/neural_motion_controller/v1')


@unittest.skipUnless((BUNDLE/'controller.ts').exists(),'Run export_neural_controller first')
class ReleasedControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.controller=QuadrupedController(bundle=BUNDLE)
        cls.initial=cls.controller.snapshot()

    def setUp(self): self.controller.restore(self.initial)

    def test_exact_resume(self):
        c=self.controller; c.step(); state=c.snapshot(); expected=c.step()[1]
        c.restore(state); actual=c.step()[1]
        self.assertTrue(torch.equal(expected,actual))

    def test_commands_change_predictions(self):
        c=self.controller; walk=c.step('Walk',.7)[1]
        c.restore(self.initial); idle=c.step('Idle',0)[1]
        self.assertGreater(float((walk-idle).abs().mean()),1e-4)

    def test_bounded_state_and_no_vendor_import(self):
        c=self.controller
        size=lambda: sum(t.numel()*t.element_size() for t in c.snapshot().values())
        before=size()
        for _ in range(300): c.step()
        self.assertEqual(size(),before)
        self.assertTrue(torch.isfinite(c.position).all())
        self.assertLessEqual(len(c.latencies),256)
        self.assertNotIn('ai4animation',sys.modules)

    def test_reject_unknown_command(self):
        with self.assertRaises(ValueError): self.controller.step('Fly',1.)
