import json
import unittest
from real_video.gaussian_program import ROOT
from real_video.agentvideo_bridge import load_upstream
from real_video.agentvideo_decision_stage import run_sizes
from real_video.agentvideo_gemini import GeminiLLM,CallBudget


class DecisionStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT.parents[1]/'work/agentvideo-reference'
        if not path.is_dir():raise unittest.SkipTest('Explicit pinned upstream checkout not installed')
        cls.upstream=load_upstream(path)

    def test_original_verifier_retries_and_preserves_ancestry(self):
        script=dict(prompt='a donkey eating an apple',setting='farm',duration_s=5,
            subjects=[dict(noun='donkey',count=1,role='actor'),dict(noun='apple',count=1,role='prop')],
            beats=[dict(who='donkey',t=[0,1],does='eats an apple')],brief='Donkey eats apple')
        answers=iter([dict(length_m=-2,height_m=1.5,legs=4),
                      dict(length_m=2,height_m=1.5,legs=4),dict(length_m=.08,height_m=.08,legs=0)])
        def transport(*args):
            return dict(candidates=[dict(finishReason='STOP',content=dict(parts=[dict(text=json.dumps(next(answers)))]))])
        client=GeminiLLM('gemini-test',budget=CallBudget(calls=3),transport=transport)
        bus=self.upstream['Bus'](echo=False)
        result=run_sizes(self.upstream,script,client,bus)
        self.assertEqual(client.budget.used,3)
        self.assertEqual(bus.agents['decision']['parent'],'idea')
        self.assertTrue(any(e['kind']=='fail' for e in bus.events))
        self.assertEqual(result['estimates']['donkey']['length_m'],2)
        self.assertFalse(result['scene_modified'])
        self.assertFalse(result['bindings_verified'])

    def test_invalid_saved_script_makes_no_call(self):
        client=GeminiLLM('gemini-test',budget=CallBudget())
        bus=self.upstream['Bus'](echo=False)
        with self.assertRaises((ValueError,KeyError)):
            run_sizes(self.upstream,dict(prompt='donkey',subjects=[],beats=[],setting=''),client,bus)
        self.assertEqual(client.budget.used,0)
