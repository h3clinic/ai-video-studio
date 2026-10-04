import copy
import tempfile
import unittest
from pathlib import Path
from real_video.studio_agent_tasks import validate_decision
from real_video.studio_roles import role_state
from real_video.studio_backend import Studio


class AgentTaskTests(unittest.TestCase):
    def decision(self):
        return dict(summary='Diagnostic',blockers=[],operations=[dict(kind='recolour',agent_id='field-agents',part_id='apple',expected_revision=0,rgb=[.8,.1,.1],strength=.2)],motion=None)

    def test_typed_recolour(self):
        self.assertEqual(validate_decision(self.decision(),'field-agents','apple')['operations'][0]['strength'],.2)

    def test_no_cross_part_or_unknown_fields(self):
        for key,value in (('part_id','donkey'),('agent_id','vector-agents'),('expected_revision',3),('shell','pwd')):
            data=self.decision();data['operations'][0][key]=value
            with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')

    def test_no_ungrounded_texture(self):
        data=self.decision();data['operations'][0]=dict(kind='texture_paint',agent_id='field-agents',part_id='apple',expected_revision=0,texture_id='invented',strength=1.)
        with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')

    def test_malformed_and_oversized_motion(self):
        data=self.decision();data['operations']=[None]
        with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')
        data=self.decision();data['operations'][0]=dict(kind='rigid_transform',agent_id='field-agents',part_id='apple',expected_revision=0,translation=[5000,0,0])
        with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')

    def test_noncommuting_motion_rejected(self):
        data=self.decision();data['operations']=[]
        data['motion']=dict(duration_seconds=5,fps=8,velocity=[0,0,0],acceleration=[0,0,0],angular_velocity=[1,0,0],angular_acceleration=[0,1,0],pivot=[0,0,0])
        with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')

    def test_blockers_no_partial_execution(self):
        data=self.decision();data['blockers']=['Missing scene correspondence']
        with self.assertRaises(ValueError):validate_decision(data,'field-agents','apple')

    def test_role_status_from_actual_job(self):
        agents=role_state([{'id':'p'}],[dict(projectId='p',agentId='vector-agents',partId='apple',status='blocked',stage='No GPU')],[])
        self.assertEqual(len(agents),14)
        vector=next(a for a in agents if a['id']=='vector-agents')
        self.assertEqual(vector['partIds'],['apple']);self.assertEqual(vector['status'],'blocked')

    def test_backend_scope_no_project_creation_on_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            studio=Studio(Path(temp));studio.project('archived','Old',True)
            for data in (dict(kind='plan',projectId='archived',prompt='edit'),dict(kind='agent_task',projectId='new',agentId='not-a-role',prompt='edit'),dict(kind='plan',projectId='new',prompt='edit',code='x')):
                with self.assertRaises(ValueError):studio.submit(data)
            self.assertEqual(len(studio.projects),1)


if __name__=='__main__':unittest.main()
