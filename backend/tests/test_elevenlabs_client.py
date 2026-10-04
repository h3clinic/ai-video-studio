import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from real_video.elevenlabs_client import ENDPOINT, MAX_BYTES, SoundError, generate_sound, mp3_duration, _supervised_sound


def fixture_mp3(frames=30):
    # Syntactic MPEG frames only. Unit fixture, never user-facing generated sound.
    header=bytes.fromhex('fffb9000')
    return (header+bytes(417-4))*frames


class SoundClientTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)/'sound'
        self.store=Mock();self.store.configured.return_value=True
        self.store.load_for_api.return_value='dummy_private_key_12345'
        self.response=Mock(status_code=200,headers={'Content-Type':'audio/mpeg','character-cost':'50'})
        self.response.iter_content.return_value=[fixture_mp3()]
        self.session=Mock();self.session.post.return_value=self.response

    def call(self, **kwargs):
        return generate_sound('Dry apple crunch, no speech',self.out,session=self.session,store=self.store,**kwargs)

    def test_fixed_endpoint_bounded_request_audio_metadata_no_secrets(self):
        result=self.call()
        self.session.post.assert_called_once()
        args=self.session.post.call_args
        self.assertEqual(args.args,(ENDPOINT,))
        self.assertFalse(args.kwargs['allow_redirects'])
        self.assertEqual(args.kwargs['timeout'],(10,60))
        self.assertEqual(args.kwargs['params'],{'output_format':'mp3_44100_128'})
        self.assertLessEqual(args.kwargs['json']['duration_seconds'],4.8)
        self.assertFalse(self.session.trust_env)
        self.assertEqual((self.out/'sound.mp3').read_bytes(),fixture_mp3())
        stored=(self.out/'sound.json').read_text()
        self.assertNotIn('dummy_private_key',stored)
        self.assertNotIn('Dry apple',stored)
        self.assertEqual(json.loads(stored)['billing_characters'],50)
        self.assertFalse(result['video_modified']);self.assertFalse(result['synchronized_to_video'])
        self.assertIsNone(result['cost_usd'])
        self.response.close.assert_called_once()

    def test_missing_credential_no_request(self):
        self.store.configured.return_value=False
        with self.assertRaisesRegex(SoundError,'credential_missing'):self.call()
        self.session.post.assert_not_called()

    def test_no_retry_or_error_body_on_provider_failure(self):
        self.response.status_code=401
        self.response.text='dummy_private_key_12345'
        with self.assertRaisesRegex(SoundError,'^auth_failed$'):self.call()
        self.session.post.assert_called_once()
        self.response.iter_content.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_transport_error_redacted(self):
        self.session.post.side_effect=requests.ConnectionError('secret header dummy_private_key_12345')
        with self.assertRaisesRegex(SoundError,'^transport_failed$'):self.call()
        self.session.post.assert_called_once()

    def test_redirect_never_followed(self):
        self.response.status_code=302
        with self.assertRaisesRegex(SoundError,'provider_rejected'):self.call()
        self.session.post.assert_called_once()

    def test_invalid_audio_rejected(self):
        self.response.iter_content.return_value=[b'<html>success</html>']
        with self.assertRaises(SoundError):self.call()
        self.assertFalse(self.out.exists())

    def test_over_limit_even_without_content_length(self):
        self.response.iter_content.return_value=[bytes(MAX_BYTES),b'x']
        with self.assertRaisesRegex(SoundError,'audio_byte_limit'):self.call()

    def test_declared_size_rejects_before_download(self):
        self.response.headers['Content-Length']=str(MAX_BYTES+1)
        with self.assertRaisesRegex(SoundError,'audio_byte_limit'):self.call()
        self.response.iter_content.assert_not_called()

    def test_duration_includes_encoder_padding_and_caps_actual_frames(self):
        self.assertAlmostEqual(mp3_duration(fixture_mp3(30)),30*1152/44100)
        with self.assertRaisesRegex(SoundError,'audio_duration_limit'):mp3_duration(fixture_mp3(192))

    def test_bad_inputs_and_existing_output_never_spend(self):
        for duration in (True,0,float('nan'),5,100):
            with self.assertRaises(ValueError):self.call(duration_seconds=duration)
        self.out.mkdir()
        with self.assertRaises(ValueError):self.call()
        self.session.post.assert_not_called()

    def test_slow_drip_deadline_terminates_worker_without_retry(self):
        context=Mock();receiver=Mock();sender=Mock();worker=Mock()
        context.Pipe.return_value=(receiver,sender);context.Process.return_value=worker
        receiver.poll.return_value=False
        worker.is_alive.side_effect=[True,False]
        with self.assertRaisesRegex(SoundError,'^request_deadline_ambiguous$'):
            _supervised_sound('Short crunch',self.out,4.8,context=context)
        receiver.poll.assert_called_once_with(75)
        context.Process.assert_called_once();worker.start.assert_called_once()
        worker.terminate.assert_called_once();worker.join.assert_called_once_with(timeout=5)
        worker.kill.assert_not_called()
        receiver.recv_bytes.assert_not_called()
        self.assertNotIn('dummy_private_key',str(context.Process.call_args))

    def test_worker_deadline_cleanup_kills_if_termination_not_observed(self):
        context=Mock();receiver=Mock();sender=Mock();worker=Mock()
        context.Pipe.return_value=(receiver,sender);context.Process.return_value=worker
        receiver.poll.return_value=False;worker.is_alive.return_value=True
        with self.assertRaisesRegex(SoundError,'request_deadline_ambiguous'):
            _supervised_sound('Short crunch',self.out,4.8,context=context)
        worker.kill.assert_called_once()
        self.assertEqual(worker.join.call_count,2)

    def test_successful_worker_returns_only_bounded_metadata(self):
        context=Mock();receiver=Mock();sender=Mock();worker=Mock()
        context.Pipe.return_value=(receiver,sender);context.Process.return_value=worker
        receiver.poll.return_value=True;worker.is_alive.return_value=False
        receiver.recv_bytes.return_value=b'{"result":{"output":"sound.mp3"}}'
        self.assertEqual(_supervised_sound('Short crunch',self.out,4.8,context=context),{'output':'sound.mp3'})
        receiver.recv_bytes.assert_called_once_with(16384)
        worker.terminate.assert_not_called()

    def test_real_spawn_missing_dummy_store_never_reaches_provider(self):
        # A fresh empty credential root is inherited by the spawned process.
        # The child exits at configured()==False before creating an HTTP session.
        with patch.dict('os.environ',{'LOCALAPPDATA':self.temp.name}):
            with self.assertRaisesRegex(SoundError,'^credential_missing$'):
                _supervised_sound('Offline missing-key spawn diagnostic',self.out,4.8)
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
