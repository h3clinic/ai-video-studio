import tempfile
import unittest
from pathlib import Path
from real_video.studio_backend import Studio
from real_video.studio_planner import issues

class ProjectTests(unittest.TestCase):
    def test_archive_and_rename_preserve_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            s=Studio(Path(temp));s.project('cat','Cat');s.project('cat','New title')
            self.assertEqual(s.state()['projects'][0]['title'],'New title')
            s.project('cat','New title',True)
            self.assertEqual(s.state()['projects'],[])
            s.project('cat','New title',False)
            self.assertEqual(len(s.state()['projects']),1)

    def test_plans_scoped_no_apple_default(self):
        with tempfile.TemporaryDirectory() as temp:
            s=Studio(Path(temp));self.assertEqual(s.state()['parts'],[])
            s.jobs=[dict(id='a',kind='plan',status='completed',projectId='cat',plan={'parts':[dict(id='tail',label='Tail',agentId='part:tail',protected=False)]})]
            p=s.state()['parts'][0];self.assertEqual(p['projectId'],'cat');self.assertFalse(p['bindingVerified'])

    def test_plan_schema_no_executable_ids(self):
        valid=dict(summary='Plan only',parts=[dict(id='tail',label='Tail',task='Retain shape',protected=True)])
        self.assertEqual(issues(valid),[])
        valid['parts'][0]['gaussian_ids']=[123]
        self.assertTrue(issues(valid))

if __name__=='__main__':unittest.main()
