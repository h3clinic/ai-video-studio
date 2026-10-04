import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from real_video.studio_backend import Studio, handler
from cloud.runpod_client import RunPodError


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.studio=Studio(Path(self.temp.name))

    def test_restart_marks_interrupted(self):
        self.studio.jobs=[dict(id='a',status='running',kind='gemini_reference')]
        self.studio.persist()
        again=Studio(Path(self.temp.name))
        self.assertEqual(again.jobs[0]['status'],'interrupted')

    def test_no_fake_bound_parts_or_media(self):
        state=self.studio.state()
        self.assertTrue(all(p['gaussianCount'] is None for p in state['parts']))
        self.assertTrue(all(not p['bindingVerified'] for p in state['parts']))
        self.assertEqual(state['media'],[])

    def test_no_arbitrary_gpu_job(self):
        with self.assertRaises(ValueError):self.studio.submit({'kind':'shell'})
        with self.assertRaises(ValueError):self.studio.submit({'kind':'remote_preflight'})

    def test_reject_ssrf(self):
        with self.assertRaises(ValueError):self.studio.connect({'podId':'abcdefghijklm','baseUrl':'http://localhost:80'})

    def test_auth_failure_not_stale_running(self):
        self.studio.remote={'status':'RUNNING','podId':'abcdefghijklm'}
        with patch('real_video.studio_backend.pod_request',side_effect=RunPodError('auth_failed',401)):
            with self.assertRaises(RunPodError):self.studio.provider('abcdefghijklm')
        remote=self.studio.state()['remote']
        self.assertEqual(remote['status'],'auth_failed')
        self.assertEqual(remote['providerHttpStatus'],401)
        self.assertNotIn('baseUrl',remote)

    def test_rejected_worker_cannot_spend_even_with_approval(self):
        with patch.dict('os.environ',{'GAUSSIAN_APPLE_APPROVED':'1'}),patch('subprocess.run') as remote:
            with self.assertRaises(ValueError):self.studio.submit({'kind':'apple_experiment'})
            remote.assert_not_called()
        self.assertFalse(self.studio.state()['capabilities']['appleReplacement']['ready'])

    def test_retained_rejected_job_also_blocked(self):
        job={'id':'historic','kind':'apple_experiment'}
        self.studio.jobs=[job]
        with patch('subprocess.run') as remote:
            self.studio.run(job)
            remote.assert_not_called()
        self.assertEqual(job['status'],'blocked')

    def test_http_auth_and_no_key_readback(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.studio,'x'*40,0))
        port=server.server_port
        server.RequestHandlerClass=handler(self.studio,'x'*40,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base=f'http://127.0.0.1:{port}'
            with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+'/api/state')
            req=urllib.request.Request(base+'/api/state',headers={'Authorization':'Bearer '+'x'*40})
            with urllib.request.urlopen(req) as response:data=json.load(response)
            self.assertTrue(all(set(v)=={'configured'} for v in data['credentials'].values()))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
