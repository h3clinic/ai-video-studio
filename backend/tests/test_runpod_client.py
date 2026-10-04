import io
import json
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from cloud.runpod_client import pod_request, RunPodError
from cloud import runpod_control


class RunPodClientTests(unittest.TestCase):
    def request_error(self, status, body=b''):
        with patch('cloud.runpod_client.load_key',return_value='secret-never-print'), patch('urllib.request.build_opener') as factory:
            factory.return_value.open.side_effect=HTTPError('https://rest.runpod.io/v1/pods',status,'secret-never-print',{},io.BytesIO(body))
            with self.assertRaises(RunPodError) as caught: pod_request()
            self.assertNotIn('secret-never-print',str(caught.exception))
            return caught.exception

    def test_401_empty_body(self):
        error=self.request_error(401)
        self.assertEqual(error.code,'auth_failed')
        self.assertEqual(error.public()['providerHttpStatus'],401)

    def test_capacity_is_not_auth(self):
        self.assertEqual(self.request_error(500,b'not enough free GPU secret-never-print').code,'gpu_capacity_unavailable')
        self.assertEqual(self.request_error(403).code,'permission_denied')
        self.assertEqual(self.request_error(404).code,'not_found')

    def test_no_raw_provider_payload(self):
        self.assertEqual(self.request_error(500,b'secret-never-print').code,'provider_error')

    def test_expected_origin_header_and_no_retry(self):
        with patch('cloud.runpod_client.load_key',return_value='secret-never-print'), patch('urllib.request.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value=b'[]'
            self.assertEqual(pod_request(),[])
            request=factory.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url,'https://rest.runpod.io/v1/pods')
            self.assertEqual(request.get_header('Authorization'),'Bearer secret-never-print')
            self.assertEqual(factory.return_value.open.call_count,1)

    def test_identity_rejected(self):
        with patch('cloud.runpod_client.load_key',return_value='secret'),patch('urllib.request.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value=b'{"id":"wrong"}'
            with self.assertRaises(RunPodError) as caught:pod_request('abcdefghijklm')
            self.assertEqual(caught.exception.code,'invalid_response')

    def test_invalid_path_never_loads_key(self):
        with patch('cloud.runpod_client.load_key') as key:
            with self.assertRaises(ValueError):pod_request('../secrets')
            key.assert_not_called()

    def test_one_owner_store_no_auth_fallback(self):
        with patch('cloud.runpod_control.os.name','nt'),patch('real_video.runway_settings.CredentialStore') as store,patch('cloud.runpod_control.load_temporary_key') as temp:
            store.return_value.configured.return_value=True
            store.return_value.load_for_api.side_effect=RuntimeError('decrypt failed')
            with self.assertRaises(RuntimeError):runpod_control.load_key()
            temp.assert_not_called()


if __name__=='__main__':unittest.main()
