import unittest
from unittest.mock import patch,MagicMock
from cloud.runpod_control import classify_provider_error
from cloud import runpod_control

class ProviderErrors(unittest.TestCase):
    def test_capacity(self):
        self.assertEqual(classify_provider_error(b'There are not enough free GPUs on the host'), 'gpu_capacity_unavailable')
    def test_json(self):
        self.assertEqual(classify_provider_error(b'Unexpected end of JSON input'), 'invalid_request_body')
    def test_secret_not_echoed(self):
        self.assertEqual(classify_provider_error(b'{"token":"rpa_fake_secret", "error":"unknown"}'), 'unspecified_provider_error')
    def test_host(self):
        self.assertEqual(classify_provider_error(b'host offline'), 'host_unavailable')
    def test_bodyless_start_has_no_json_content_type(self):
        opener=MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value=b'{}'
        with patch.object(runpod_control,'load_key',return_value='test'), patch.object(runpod_control.urllib.request,'build_opener',return_value=opener):
            runpod_control.api('POST','/start')
        request=opener.open.call_args.args[0]
        self.assertIsNone(request.get_header('Content-type'))
        self.assertIsNone(request.data)
    def test_patch_has_json_content_type(self):
        opener=MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value=b'{}'
        with patch.object(runpod_control,'load_key',return_value='test'), patch.object(runpod_control.urllib.request,'build_opener',return_value=opener):
            runpod_control.api('PATCH',payload={'name':'fixture'})
        self.assertEqual(opener.open.call_args.args[0].get_header('Content-type'),'application/json')

if __name__=='__main__': unittest.main()
