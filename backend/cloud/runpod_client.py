"""Fixed-origin RunPod control API with redacted, actionable errors.

Authentication, GPU capacity, and model-quality failures are different stages.
Nothing in this module retries, creates Pods, purchases credit or enables keys.
"""
import json
import re
import urllib.error
import urllib.request
from cloud.runpod_control import load_key, NoRedirect, classify_provider_error


MESSAGES = {
    'auth_failed': 'RunPod rejected the saved key (HTTP 401). Check whether that key is disabled or revoked; adding GPU credit will not fix authentication.',
    'permission_denied': 'RunPod denied this operation (HTTP 403). Check the saved key permissions; no automatic permission expansion was attempted.',
    'not_found': 'RunPod could not find the requested Pod (HTTP 404). The saved Pod ID may no longer exist.',
    'rate_limited': 'RunPod rate-limited the request (HTTP 429). No automatic retry was made.',
    'gpu_capacity_unavailable': 'RunPod reported unavailable GPU capacity. This is separate from model execution or authentication.',
    'insufficient_balance': 'RunPod reported insufficient credit. No top-up was attempted.',
    'provider_error': 'RunPod returned a provider error. No automatic retry was made.',
    'invalid_response': 'RunPod returned an invalid or oversized response.',
    'connection_failed': 'Could not reach the RunPod control API. No automatic retry was made.',
    'credential_unavailable': 'The saved RunPod credential is unavailable or could not be decrypted.',
}


class RunPodError(RuntimeError):
    def __init__(self, code, http_status=None):
        self.code = code if code in MESSAGES else 'provider_error'
        self.http_status = http_status
        super().__init__(MESSAGES[self.code])

    def public(self):
        return {'errorCode': self.code, 'error': str(self), 'providerHttpStatus': self.http_status}


def pod_request(pod=None, *, stop=False):
    if pod is not None and (not isinstance(pod, str) or not re.fullmatch(r'[a-z0-9]{10,32}', pod)):
        raise ValueError('Invalid Pod ID')
    if stop and pod is None:
        raise ValueError('A Pod ID is required to stop')
    try:
        key = load_key()
    except (OSError, ValueError, RuntimeError, KeyError):
        raise RunPodError('credential_unavailable') from None
    path = 'pods' + ('/' + pod if pod is not None else '') + ('/stop' if stop else '')
    request = urllib.request.Request('https://rest.runpod.io/v1/' + path,
        method='POST' if stop else 'GET',
        headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'GaussianStudio/1'})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
            raw = response.read(2 * 2**20 + 1)
        if len(raw) > 2 * 2**20:
            raise RunPodError('invalid_response')
        result = json.loads(raw) if raw else {}
        if not stop:
            if pod is None and not isinstance(result, list):
                raise RunPodError('invalid_response')
            if pod is not None and (not isinstance(result, dict) or result.get('id') != pod):
                raise RunPodError('invalid_response')
        return result
    except urllib.error.HTTPError as error:
        code = {401:'auth_failed', 403:'permission_denied', 404:'not_found', 429:'rate_limited'}.get(error.code)
        if code is None:
            category = classify_provider_error(error.read(65536))
            code = category if category in ('gpu_capacity_unavailable','insufficient_balance') else 'provider_error'
        raise RunPodError(code, error.code) from None
    except (OSError, urllib.error.URLError):
        raise RunPodError('connection_failed') from None
    except (ValueError, UnicodeError):
        raise RunPodError('invalid_response') from None
