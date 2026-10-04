import copy
import unittest
import torch
from real_video.motion_only_session import MotionOnlySession,command,tensor_bytes,BUNDLE


class CommandTests(unittest.TestCase):
    def test_commands(self):
        self.assertEqual(command('walk')['speed'],.7)
        for action,speed in [('jump',1),('idle',1),('walk',float('nan')),('trot',True)]:
            with self.assertRaises(ValueError):command(action,speed)


@unittest.skipUnless((BUNDLE/'controller.ts').exists(),'Released local bundle required')
class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.session=MotionOnlySession();cls.initial=cls.session.snapshot()
    def setUp(self):self.session.restore(self.initial)
    def test_exact_resume_and_orthogonal_controls(self):
        s=self.session;s.set_action('walk');s.advance();saved=s.snapshot()
        expected=s.advance();s.restore(saved);actual=s.advance()
        for key in ('local_rotation','root_translation','ik_reach_error'):
            self.assertTrue(torch.equal(expected[key],actual[key]))
        r=actual['local_rotation']
        torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=2e-5,rtol=2e-5)
    def test_no_appearance_history_or_growing_state(self):
        s=self.session;before=tensor_bytes(s.snapshot());s.set_action('trot')
        for _ in range(30):s.advance()
        self.assertEqual(tensor_bytes(s.snapshot()),before)
        self.assertEqual(s.frame_index,90)
        self.assertFalse(s.snapshot()['appearance_updated'])
    def test_restore_rejects_wrong_asset_and_nonfinite_without_mutation(self):
        s=self.session
        for fault in ('asset','nonfinite'):
            bad=copy.deepcopy(self.initial)
            if fault=='asset':bad['bindings']['asset']['sha256']='wrong'
            else:bad['controller']['position'][0,0]=float('nan')
            with self.assertRaises(ValueError):s.restore(bad)
        self.assertEqual(s.frame_index,0)
    def test_quality_gate(self):
        with self.assertRaises(ValueError):self.session.render_permission()
        self.assertFalse(self.session.render_permission(diagnostic=True)['quality_accepted'])


if __name__=='__main__':unittest.main()
