import copy
import json
from pathlib import Path
import tempfile
import unittest
from real_video.studio_roles import ROLES
from real_video.studio_swarm import plan_issues, brief_issues, run_swarm


def team():
    roles=[r[0] for r in ROLES if r[0]!='sound-agents']+['sound-agents']*3
    return dict(summary='Test team',workers=[dict(id=f'w{i}',roleId=role,name=f'Worker {i}',partId='scene',task='Scoped task',dependsOn=[]) for i,role in enumerate(roles)])


class FakeLLM:
    model_id='fake-fixture-not-a-provider'
    def __init__(self, plan):self.plan=plan;self.usage=[]
    def chat(self,messages,**kwargs):
        self.usage.append({'test':True})
        return json.dumps(self.plan if len(self.usage)==1 else dict(summary='Concrete brief only',requirements=['Verified dynamic reconstruction'],region=None))


class SwarmTests(unittest.TestCase):
    def test_contract_counts_and_all_roles(self):
        self.assertEqual(plan_issues(team()),[])
        p=team();p['workers'].pop();self.assertTrue(plan_issues(p))
        p=team();p['workers'][1]['roleId']='invented';self.assertTrue(plan_issues(p))
    def test_cycle_unknown_duplicate_and_overflow(self):
        for mutate in [lambda p:p['workers'][0].update(dependsOn=['w1']),lambda p:p['workers'][1].update(id='w0'),lambda p:p['workers'][0].update(partId='../escape')]:
            p=team();mutate(p);self.assertTrue(plan_issues(p))
        p=team();p['workers']*=2;self.assertTrue(plan_issues(p))
    def test_region_finite_bounds(self):
        b=dict(summary='brief',requirements=[],region=[0,.1,1,.9]);self.assertEqual(brief_issues(b),[])
        for box in [[0,0,float('nan'),1],[0,0,1,2],[1,0,0,1],[True,0,1,1]]:
            self.assertTrue(brief_issues(dict(b,region=box)))
    def test_actual_lifecycle_three_sounds_and_no_geometry(self):
        calls=[];snapshots=[]
        def sound(prompt,out):
            calls.append(prompt);out.mkdir();(out/'sound.mp3').write_bytes(b'test-not-real-audio')
            return dict(output='sound.mp3',review='Test fixture only')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            from real_video.agentvideo_bridge import load_upstream
            upstream=load_upstream(Path(__file__).resolve().parents[3]/'work/agentvideo-reference')
            result=run_swarm('Test',root/'out',root=root,progress=lambda **s:snapshots.append(copy.deepcopy(s)),client=FakeLLM(team()),sound_tool=sound,upstream=upstream)
            self.assertEqual(len(calls),3);self.assertEqual(result['completed'],16)
            self.assertFalse(result['sceneModified']);self.assertFalse(result['gaussianOwnershipVerified'])
            self.assertEqual(result['soundLayers'],3)
            self.assertTrue(any(any(w['status']=='running' for w in s.get('workers',[])) for s in snapshots))
            self.assertTrue(all(w.get('finishedAt') and w['seconds']>=0 for w in result['workers']))
            previous=dict(result,id='previous-fixture')
            before=len(calls)
            unused=FakeLLM(team())
            resumed=run_swarm('Test',root/'resumed',root=root,progress=lambda **s:None,client=unused,sound_tool=sound,upstream=upstream,previous=previous)
            self.assertEqual(len(calls),before);self.assertEqual(unused.usage,[])
            self.assertTrue(all(w['reusedFrom']=='previous-fixture' for w in resumed['workers']))
    def test_failed_dependency_no_paid_retry(self):
        p=team();p['workers'][-1]['dependsOn']=[p['workers'][-2]['id']]
        calls=[]
        def sound(prompt,out):calls.append(prompt);raise RuntimeError('provider body must stay private')
        with tempfile.TemporaryDirectory() as tmp:
            from real_video.agentvideo_bridge import load_upstream
            upstream=load_upstream(Path(__file__).resolve().parents[3]/'work/agentvideo-reference')
            result=run_swarm('Test',Path(tmp)/'out',root=Path(tmp),progress=lambda **s:None,client=FakeLLM(p),sound_tool=sound,upstream=upstream)
            self.assertEqual(len(calls),1);self.assertEqual(result['workers'][-1]['status'],'blocked')
            self.assertNotIn('provider body',json.dumps(result))

    def test_resume_rechecks_verifier_after_changed_sound_outcome(self):
        p=team()
        p['workers']=sorted(p['workers'],key=lambda w:w['roleId']=='verifiers')
        calls=[]
        def sound(prompt,out):
            calls.append(prompt);out.mkdir();(out/'sound.mp3').write_bytes(b'test-not-real-audio')
            return dict(output='sound.mp3',review='Offline fixture')
        with tempfile.TemporaryDirectory() as tmp:
            from real_video.agentvideo_bridge import load_upstream
            upstream=load_upstream(Path(__file__).resolve().parents[3]/'work/agentvideo-reference')
            root=Path(tmp)
            previous=run_swarm('Fixture',root/'first',root=root,progress=lambda **s:None,client=FakeLLM(p),sound_tool=sound,upstream=upstream)
            previous['id']='previous'
            failed=next(w for w in previous['workers'] if w['roleId']=='sound-agents')
            failed.update(status='failed',execution='not_started',error='auth_failed')
            client=FakeLLM(p);client.usage=[{'fixture_skip_director':True}]
            resumed=run_swarm('Fixture',root/'resume',root=root,progress=lambda **s:None,client=client,sound_tool=sound,upstream=upstream,previous=previous)
            self.assertEqual(len(calls),4)  # Only one unfinished sound retried.
            self.assertEqual(len(client.usage),2)  # One fresh reviewer, no director.
            self.assertNotIn('reusedFrom',next(w for w in resumed['workers'] if w['roleId']=='verifiers'))
            self.assertTrue(all(w.get('reusedFrom')=='previous' for w in resumed['workers'] if w['roleId'] not in ('verifiers','sound-agents')))


if __name__=='__main__':unittest.main()
