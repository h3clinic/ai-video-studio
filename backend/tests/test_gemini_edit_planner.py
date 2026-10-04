import json
import unittest
from unittest.mock import patch
from real_video.gemini_edit_planner import propose_once


class PlannerTests(unittest.TestCase):
    def args(self):
        return dict(model='gemini-test',key='AIza'+'a'*35,instruction='Replace fruit',target='fruit',
            inventory={'fruit':dict(protected=False,binding_verified=True,revision=0)},
            frames=[dict(index=0,mime='image/png',bytes=b'\x89PNG\r\n\x1a\nfixture')])
    def test_mock_request_header_no_key_url_no_autoexecution(self):
        proposal=dict(candidates=[],unknown_parts=['unmapped peel'],reviewed_frames=[0])
        raw=json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(proposal)}]}}]}).encode()
        with patch('urllib.request.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value=raw
            result=propose_once(**self.args())
            request=factory.return_value.open.call_args.args[0]
            self.assertNotIn(self.args()['key'],request.full_url)
            self.assertEqual(request.get_header('X-goog-api-key'),self.args()['key'])
            self.assertFalse(result['plan']['scope_verified']);self.assertFalse(result['execution_performed'])
            self.assertEqual(factory.return_value.open.call_count,1)
            payload=json.loads(request.data)
            self.assertIn('responseSchema',payload['generationConfig'])
            self.assertNotIn('additionalProperties',json.dumps(payload['generationConfig']['responseSchema']))
    def test_protected_replacement_misclassified_as_target_stays_preserved(self):
        from real_video.elimination_scope import plan_elimination
        inventory={'orange':dict(protected=False,binding_verified=False,revision=0),
                   'apple':dict(protected=True,binding_verified=False,revision=0)}
        proposal=dict(candidates=[dict(entity_id='apple',relation='target',confidence=1.0,
            frames=[0],reason='Incorrect model label')],unknown_parts=[],reviewed_frames=[0])
        plan=plan_elimination('orange',inventory,proposal,[0])
        self.assertEqual(plan['remove'],['orange'])
        self.assertIn('apple',plan['preserve'])
        self.assertFalse(plan['scope_verified'])
        self.assertFalse(plan['executable'])
    def test_validation_before_network(self):
        with patch('urllib.request.build_opener') as factory:
            args=self.args();args['model']='../other-server'
            with self.assertRaises(ValueError):propose_once(**args)
            factory.assert_not_called()
