import unittest
from real_video.agentvideo_gemini import GeminiLLM, CallBudget


def reply(text='{"ok":true}', reason='STOP'):
    return dict(candidates=[dict(finishReason=reason,content=dict(parts=[dict(text=text)]))])


class ProviderTests(unittest.TestCase):
    def test_preserves_conversation_and_complete_prefix(self):
        seen=[]
        def transport(model,payload):seen.append(payload);return reply()
        client=GeminiLLM('gemini-test',budget=CallBudget(),transport=transport)
        client.chat([dict(role='system',content='laws',persist=True),dict(role='user',content='task'),
            dict(role='assistant',content='bad'),dict(role='user',content='fix')],prefill='{')
        self.assertEqual(seen[0]['contents'][1]['role'],'model')
        self.assertEqual(seen[0]['systemInstruction']['parts'][0]['text'],'laws')
        self.assertIn('COMPLETE',seen[0]['contents'][-1]['parts'][-1]['text'])
    def test_shared_budget_and_no_retry(self):
        budget=CallBudget(calls=1)
        a=GeminiLLM('gemini-test',budget=budget,transport=lambda *a:reply())
        b=GeminiLLM('gemini-test',budget=budget,transport=lambda *a:reply())
        a.chat([dict(role='user',content='task')])
        with self.assertRaises(RuntimeError):b.chat([dict(role='user',content='task')])
        self.assertEqual(budget.used,1)
    def test_rejects_truncated_answer(self):
        a=GeminiLLM('gemini-test',budget=CallBudget(),transport=lambda *a:reply('{"ok":', 'MAX_TOKENS'))
        with self.assertRaises(RuntimeError):a.chat([dict(role='user',content='task')])
        self.assertEqual(a.usage[0]['finish_reason'],'MAX_TOKENS')
    def test_invalid_input_does_not_spend(self):
        budget=CallBudget()
        a=GeminiLLM('gemini-test',budget=budget)
        with self.assertRaises(ValueError):a.chat([dict(role='assistant',content='task')])
        self.assertEqual(budget.used,0)
    def test_reserves_actual_reasoning_headroom(self):
        seen=[]
        def transport(model,payload):seen.append(payload);return reply()
        budget=CallBudget(calls=3,output_tokens=2048)
        a=GeminiLLM('gemini-test',budget=budget,transport=transport)
        a.chat([dict(role='user',content='script')],max_tokens=150)
        self.assertEqual(seen[0]['generationConfig']['maxOutputTokens'],2048)
        self.assertEqual(budget.tokens,0)
        with self.assertRaises(RuntimeError):a.chat([dict(role='user',content='script')])
    def test_empty_candidates_recorded_and_rejected(self):
        a=GeminiLLM('gemini-test',budget=CallBudget(),transport=lambda *a:dict(candidates=[]))
        with self.assertRaisesRegex(RuntimeError,'NO_CANDIDATE'):a.chat([dict(role='user',content='script')])
        self.assertEqual(len(a.usage),1)
    def test_transport_failure_remains_budgeted(self):
        def fail(*args):raise RuntimeError('service unavailable')
        budget=CallBudget(calls=1)
        a=GeminiLLM('gemini-test',budget=budget,transport=fail)
        with self.assertRaises(RuntimeError):a.chat([dict(role='user',content='script')])
        self.assertEqual(budget.used,1)
        self.assertEqual(a.usage[0]['finish_reason'],'TRANSPORT_ERROR')
        self.assertIsNone(a.usage[0]['usage'])
