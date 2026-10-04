import unittest
from real_video.studio_agent_events import project_trace


class TraceTests(unittest.TestCase):
    def test_pass_does_not_claim_video_or_175_agents(self):
        result=project_trace(dict(agents={'check':dict(role='verifier',parent=None)},
            events=[dict(agent='check',kind='pass',t=1,msg='valid plan')]))
        self.assertEqual(result['agent_count'],1)
        self.assertEqual(result['video_status'],'unverified')
        self.assertEqual(result['agents'][0]['activity'],'decision_verified')
    def test_unknown_agent_rejected(self):
        with self.assertRaises(ValueError):project_trace(dict(agents={},events=[dict(agent='ghost',kind='pass')]))
