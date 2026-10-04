import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from real_video.studio_backend import Studio, append_job_event
from real_video.elevenlabs_client import SoundError
from real_video.runway_settings import CredentialStore


class StudioSoundEventsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.studio=Studio(Path(self.temp.name))

    def submit(self):
        with patch('threading.Thread.start'):
            self.studio.submit(dict(kind='agent_task',projectId='p',agentId='sound-agents',prompt='A short apple crunch',partId='apple'))
        return self.studio.jobs[-1]

    def test_missing_owner_key_blocks_without_network(self):
        job=self.submit()
        with patch.object(CredentialStore,'configured',return_value=False),patch('real_video.elevenlabs_client.generate_sound') as call:
            self.studio.run(job)
            call.assert_not_called()
        self.assertEqual(job['status'],'blocked')
        self.assertEqual(job['error'],'credential_missing')
        self.assertEqual(job['queuedAt'],job['createdAt'])
        self.assertLessEqual(job['queuedAt'],job['startedAt'])
        self.assertLessEqual(job['startedAt'],job['finishedAt'])
        self.assertEqual([e['sequence'] for e in job['events']],list(range(1,len(job['events'])+1)))
        self.assertNotIn('A short apple',json.dumps(job['events']))

    def test_sound_registers_audio_not_video_and_no_geometry_plan(self):
        job=self.submit()
        def generated(prompt,output):
            output.mkdir();(output/'sound.mp3').write_bytes(b'dummy_audio')
            return dict(output='sound.mp3',requests=1,billing_characters=30,review='Unreviewed audio')
        with patch.object(CredentialStore,'configured',return_value=True),patch('real_video.elevenlabs_client.generate_sound',side_effect=generated),patch('real_video.studio_agent_tasks.plan_agent_task') as planner:
            self.studio.run(job)
            planner.assert_not_called()
        self.assertEqual(job['status'],'completed')
        self.assertEqual(job['outputKind'],'audio')
        state=self.studio.state()
        self.assertEqual([m['kind'] for m in state['media']],['audio'])
        self.assertTrue(state['media'][0]['url'].startswith('gaussian-media://artifact/audio_'))
        self.assertEqual(set(state['credentials']['elevenlabs']),{'configured'})

    def test_provider_failure_only_public_code_in_job(self):
        job=self.submit()
        with patch.object(CredentialStore,'configured',return_value=True),patch('real_video.elevenlabs_client.generate_sound',side_effect=SoundError('auth_failed')):
            self.studio.run(job)
        self.assertEqual(job['status'],'failed');self.assertEqual(job['error'],'auth_failed')
        self.assertIsNone(job['output'])

    def test_events_bound_and_do_not_copy_model_blockers(self):
        job=dict(status='blocked',stage='sensitive model/provider response',agentId='sound-agents')
        for _ in range(220):append_job_event(job)
        self.assertEqual(len(job['events']),200)
        self.assertEqual(job['events'][0]['sequence'],21)
        self.assertEqual(job['events'][-1]['sequence'],220)
        self.assertNotIn('sensitive',json.dumps(job['events']))

    def test_restart_records_observed_interruption_not_inferred_start(self):
        self.studio.jobs=[dict(id='legacy',kind='plan',status='running')];self.studio.persist()
        again=Studio(self.studio.root);job=again.jobs[0]
        self.assertNotIn('startedAt',job)
        self.assertIn('finishedAt',job)
        self.assertEqual(job['events'][-1]['status'],'interrupted')

    def test_opaque_elevenlabs_key_roundtrip_and_control_char_rejection(self):
        store=CredentialStore('elevenlabs');store.path=Path(self.temp.name)/'dummy.dpapi'
        key='dummy_'+'x'*50
        store.save(key)
        self.assertEqual(store.load_for_api(),key)
        self.assertNotIn(key.encode(),store.path.read_bytes())
        for invalid in ('too short',key+'\r\nHeader: fail',key+' ',None):
            with self.assertRaises(ValueError):store.save(invalid)


if __name__=='__main__':unittest.main()
