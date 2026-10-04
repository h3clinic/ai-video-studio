import json
from real_video.runway_settings import handler
from tests.test_runway_settings import SettingsTests


class ElevenLabsSettingsTests(SettingsTests):
    def setUp(self):
        super().setUp()
        self.server.RequestHandlerClass=handler(self.store,self.origin,'token',provider='elevenlabs')

    def test_save_never_echoes_and_status_only_boolean(self):
        key='dummy_elevenlabs_key_not_real'
        code,body,headers=self.request('/api/elevenlabs/key',json.dumps({'key':key}))
        self.assertEqual(code,200)
        self.assertNotIn(key.encode(),body)
        self.assertEqual(self.store.key,key)
        self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertEqual(json.loads(self.request('/api/elevenlabs/status')[1]),{'configured':True})
        self.assertEqual(self.request('/api/runway/status')[0],404)

    def test_csrf_host_invalid_key_and_arbitrary_paths_rejected(self):
        body=json.dumps({'key':'dummy_elevenlabs_key_not_real'})
        for headers in ({'Origin':'https://evil.example'},{'X-Settings-Token':''},{'Host':'evil.example'}):
            self.assertEqual(self.request('/api/elevenlabs/key',body,**headers)[0],403)
        code,response,_=self.request('/api/elevenlabs/key','{}')
        self.assertEqual(code,400)
        self.assertNotIn(b'Google',response)
        self.assertEqual(self.request('/../../elevenlabs.dpapi')[0],404)
        self.assertIsNone(self.store.key)

    def test_masked_form_and_csp(self):
        super().test_masked_form_and_csp()
        _,body,_=self.request('/')
        self.assertIn(b'Connect ElevenLabs',body)
        self.assertIn(b'/api/elevenlabs/key',body)
        self.assertIn(b'maxlength="512"',body)
        self.assertNotIn(b'Runway',body)
