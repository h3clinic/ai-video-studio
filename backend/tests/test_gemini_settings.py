import json
import tempfile
from pathlib import Path
from real_video.runway_settings import handler,CredentialStore
from tests.test_runway_settings import SettingsTests


class GeminiSettingsTests(SettingsTests):
    def setUp(self):
        super().setUp()
        self.server.RequestHandlerClass=handler(self.store,self.origin,'token',provider='gemini')
    def test_save_never_echoes_and_status_only_boolean(self):
        key='AIza'+'a'*35
        code,body,headers=self.request('/api/gemini/key',json.dumps({'key':key}))
        self.assertEqual(code,200);self.assertNotIn(key.encode(),body)
        self.assertEqual(self.store.key,key)
        self.assertEqual(json.loads(self.request('/api/gemini/status')[1]),{'configured':True})
        self.assertEqual(self.request('/api/runway/status')[0],404)
    def test_csrf_host_invalid_key_and_arbitrary_paths_rejected(self):
        body=json.dumps({'key':'AIza'+'b'*35})
        for headers in ({'Origin':'https://evil.example'},{'X-Settings-Token':''},{'Host':'evil.example'}):
            self.assertEqual(self.request('/api/gemini/key',body,**headers)[0],403)
        self.assertEqual(self.request('/api/gemini/key','{}')[0],400)
        self.assertEqual(self.request('/../../gemini.dpapi')[0],404)
        self.assertIsNone(self.store.key)
    def test_dpapi_roundtrip_only_dummy_credential(self):
        with tempfile.TemporaryDirectory() as folder:
            store=CredentialStore('gemini');store.path=Path(folder)/'dummy.dpapi'
            key='AIza'+'c'*35;store.save(key)
            self.assertEqual(store.load_for_api(),key);self.assertNotIn(key.encode(),store.path.read_bytes())
