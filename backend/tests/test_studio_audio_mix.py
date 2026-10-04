import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import imageio_ffmpeg
from real_video.studio_audio_mix import validate_gains, confined_file, mix_scene_audio
from real_video.studio_backend import Studio


class AudioMixTests(unittest.TestCase):
    def test_invalid_gains(self):
        for values in ([1,1],[1,1,True],[1,1,float('nan')],[1,1,1.1],None):
            with self.assertRaises(ValueError):validate_gains(values)
        self.assertEqual(validate_gains([0,.4,1]),[0,.4,1])

    def test_confined_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'root';root.mkdir();(Path(tmp)/'outside').write_text('test')
            with self.assertRaises(ValueError):confined_file(root,'../outside')
            with self.assertRaises(ValueError):confined_file(root,'missing.mp3')

    def test_submit_checks_project_and_completed_tracks_before_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            studio=Studio(Path(tmp));studio.project('one','Fixture')
            studio.jobs=[dict(id='a'*32,kind='agent_swarm',projectId='one',status='completed',workers=[])]
            with patch('threading.Thread') as thread:
                for project in ('one','wrong'):
                    with self.assertRaises(ValueError):studio.submit(dict(kind='sound_mix',projectId=project,sourceJobId='a'*32,gains=[.2,.5,.5]))
                thread.assert_not_called()

    def test_real_bounded_mux_preserves_all_decoded_video_frames(self):
        # Tiny offline test fixture, never a generated demonstration.
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        def ff(*args):
            return subprocess.run([ffmpeg,'-hide_banner','-loglevel','error',*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20).stdout
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4';track=root/'fixture.mp3'
            ff('-f','lavfi','-i','color=c=blue:s=64x64:r=8:d=1','-an','-c:v','libx264','-pix_fmt','yuv420p',str(source))
            ff('-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=0.8','-c:a','libmp3lame','-b:a','128k',str(track))
            digest=hashlib.sha256(track.read_bytes()).hexdigest()
            workers=[dict(id=f'sound{i}',name=f'Test {i}',partId=f'audio{i}',roleId='sound-agents',status='completed',execution='elevenlabs_sound',output='fixture.mp3',sound={'sha256':digest}) for i in range(3)]
            result=mix_scene_audio(dict(id='fixture',kind='agent_swarm',workers=workers),root/'mix',root=root,gains=[.2,.3,.4],source_video='source.mp4')
            def frames(path):
                data=ff('-i',str(path),'-map','0:v:0','-f','framemd5','-').decode()
                return [line.split(',')[-1].strip() for line in data.splitlines() if line and not line.startswith('#')]
            self.assertEqual(frames(source),frames(root/'mix'/result['output']))
            self.assertEqual(len(frames(source)),8)
            self.assertEqual(result['new_api_calls'],0)
            self.assertFalse(result['visuals_modified']);self.assertFalse(result['gaussians_modified'])
            self.assertGreater(len(ff('-i',str(root/'mix'/result['output']),'-map','0:a:0','-f','s16le','-')),0)
            self.assertTrue((root/'mix'/'mix_report.json').is_file())
            self.assertNotIn('api_key',json.dumps(result))
            workers[0]['sound']['sha256']='incorrect'
            with self.assertRaises(ValueError):mix_scene_audio(dict(id='fixture',kind='agent_swarm',workers=workers),root/'bad',root=root,gains=[.2,.3,.4],source_video='source.mp4')
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
