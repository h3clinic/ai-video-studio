import http.client
import json
import os
import threading
import tempfile
from pathlib import Path
import unittest
from http.server import ThreadingHTTPServer
from real_video.runway_settings import handler, protect, CredentialStore


class FakeStore:
    key=None
    def configured(self): return self.key is not None
    def save(self,key): self.key=key


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.store=FakeStore()
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.store,'unused','token'))
        self.origin=f'http://127.0.0.1:{self.server.server_port}'
        self.server.RequestHandlerClass=handler(self.store,self.origin,'token')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,body=None,**override):
        client=http.client.HTTPConnection('127.0.0.1',self.server.server_port)
        headers={'Origin':self.origin,'X-Settings-Token':'token','Content-Type':'application/json'}
        headers.update(override)
        client.request('POST' if body is not None else 'GET',path,body,headers)
        response=client.getresponse(); result=(response.status,response.read(),dict(response.getheaders()))
        client.close();return result
    def test_save_never_echoes_and_status_only_boolean(self):
        key='key_'+'a'*128
        code,body,headers=self.request('/api/runway/key',json.dumps({'key':key}))
        self.assertEqual(code,200);self.assertNotIn(key.encode(),body)
        self.assertEqual(self.store.key,key)
        self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertEqual(json.loads(self.request('/api/runway/status')[1]),{'configured':True})
    def test_csrf_host_invalid_key_and_arbitrary_paths_rejected(self):
        body=json.dumps({'key':'key_'+'b'*128})
        for headers in ({'Origin':'https://evil.example'},{'X-Settings-Token':''},{'Host':'evil.example'}):
            self.assertEqual(self.request('/api/runway/key',body,**headers)[0],403)
        self.assertEqual(self.request('/api/runway/key','{}')[0],400)
        self.assertEqual(self.request('/../../runway.dpapi')[0],404)
        self.assertIsNone(self.store.key)
    def test_masked_form_and_csp(self):
        code,body,headers=self.request('/')
        self.assertEqual(code,200);self.assertIn(b'type="password"',body)
        self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])
    @unittest.skipUnless(os.name=='nt','Windows only')
    def test_dpapi_encryption_not_plaintext(self):
        dummy=b'not-a-real-api-key'
        encrypted=protect(dummy)
        self.assertNotIn(dummy,encrypted);self.assertGreater(len(encrypted),len(dummy))

    @unittest.skipUnless(os.name=='nt','Windows only')
    def test_dpapi_roundtrip_only_dummy_credential(self):
        with tempfile.TemporaryDirectory() as folder:
            store=CredentialStore()
            store.path=Path(folder)/'dummy.dpapi'
            key='key_'+'c'*128
            store.save(key)
            self.assertEqual(store.load_for_api(),key)
            self.assertNotIn(key.encode(),store.path.read_bytes())


if __name__=='__main__': unittest.main()
