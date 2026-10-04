"""Narrow control for the authorized Pod; secrets never enter logs or arguments.

Watchdog runs in its own process, not the chat/browser. It is NOT a guaranteed
provider-side spending cap: host shutdown or API/network failure can prevent stop.
No purchasing, top-ups, Pod creation, termination, or automatic restart.
"""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

POD = os.environ.get('GAUSSIAN_RUNPOD_TARGET','wdq34cebh6111k')
if POD not in ('wdq34cebh6111k','k5494yeo0hf0b7'):
    raise ValueError('Pod outside approved existing experiment targets')
STORE = Path(os.environ.get('LOCALAPPDATA', '.'))/'GaussianVectorVideo'/'RunpodTemporary'


def seal():
    from real_video.runway_settings import protect
    incoming = STORE/'incoming.key'
    raw = incoming.read_bytes().strip()
    if not re.fullmatch(rb'rpa_[A-Za-z0-9_-]+', raw): raise ValueError('Invalid key format')
    target = STORE/'shutdown.dpapi'
    with target.open('xb') as stream: stream.write(protect(raw))
    incoming.unlink()  # Remove only the approved one-use plaintext credential.
    return {'sealed':True, 'plaintext_removed':True}


def load_key():
    """Shared owner credential selection; never silently try a second account."""
    if os.name == 'nt':
        from real_video.runway_settings import CredentialStore
        store = CredentialStore('runpod')
        if store.configured():
            return store.load_for_api()
    return load_temporary_key()


def credential_configured():
    if os.name != 'nt':
        return bool(os.environ.get('RUNPOD_SHUTDOWN_KEY'))
    from real_video.runway_settings import CredentialStore
    return CredentialStore('runpod').configured() or (STORE/'shutdown.dpapi').is_file()


def load_temporary_key():
    if os.name != 'nt':
        value = os.environ['RUNPOD_SHUTDOWN_KEY']
        if not re.fullmatch(r'rpa_[A-Za-z0-9_-]+', value): raise ValueError('Invalid key')
        return value
    from real_video.runway_settings import Blob
    from ctypes import wintypes
    encrypted = (STORE/'shutdown.dpapi').read_bytes()
    raw = ctypes.create_string_buffer(encrypted)
    incoming = Blob(len(encrypted),ctypes.cast(raw,ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    api = ctypes.WinDLL('crypt32',use_last_error=True)
    api.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,
        ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    api.CryptUnprotectData.restype = wintypes.BOOL
    if not api.CryptUnprotectData(ctypes.byref(incoming),None,None,None,None,1,ctypes.byref(outgoing)):
        raise RuntimeError('Credential decryption failed')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    try:
        value=ctypes.string_at(outgoing.data,outgoing.size).decode('ascii')
        if not re.fullmatch(r'rpa_[A-Za-z0-9_-]+',value): raise ValueError('Invalid stored key')
        return value
    finally:
        ctypes.memset(outgoing.data,0,outgoing.size);kernel.LocalFree(outgoing.data)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None


def classify_provider_error(body):
    """Return only fixed diagnostic labels, never echo provider payload/secrets."""
    text=body[:65536].decode('utf-8',errors='replace').lower()
    patterns=(
        ('gpu_capacity_unavailable',('not enough free gpu','insufficient gpu','gpu unavailable','no available gpu')),
        ('host_unavailable',('host offline','machine offline','host unavailable')),
        ('insufficient_balance',('insufficient balance','insufficient funds')),
        ('invalid_request_body',('unexpected end of json','invalid json','empty json','body is required')),
        ('permission_denied',('permission denied','unauthorized','forbidden')),
    )
    for category,phrases in patterns:
        if any(phrase in text for phrase in phrases): return category
    return 'unspecified_provider_error'


def api(method='GET',suffix='',payload=None):
    if (method,suffix) not in (('GET',''),('PATCH',''),('POST','/stop'),('POST','/start')):
        raise ValueError('Unsupported control operation')
    url=f'https://rest.runpod.io/v1/pods/{POD}{suffix}'
    headers={'Authorization':'Bearer '+load_key(), 'User-Agent':'GaussianVideoBudgetGuard/1'}
    if payload is not None: headers['Content-Type']='application/json'
    req=urllib.request.Request(url,method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(req,timeout=15) as response:
            body=response.read(1024*1024)
        result=json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        category=classify_provider_error(error.read(65536))
        raise RuntimeError(f'RunPod API HTTP {error.code}: {category}') from None
    except (OSError,ValueError):
        raise RuntimeError('RunPod API connection/response failed') from None
    return result


def request(action='status'):
    if action not in ('status','stop'): raise ValueError('Only status/stop supported')
    result=api('GET' if action=='status' else 'POST','' if action=='status' else '/stop')
    if action=='stop': return {'stop_request_accepted':True, 'pod':POD}
    if result.get('id') != POD: raise RuntimeError('Unexpected Pod identity')
    return {k:result.get(k) for k in ('id','name','desiredStatus','costPerHr','vcpuCount','memoryInGb')}


@contextmanager
def awake():
    if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try: yield
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def watch(seconds,out):
    if not 1 <= seconds <= 2400: raise ValueError('1..2400 second guard required')
    out.mkdir(parents=True,exist_ok=False)
    events=out/'events.jsonl'
    def record(**value):
        with events.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**value))+'\n')
    # Verify authenticated status BEFORE marking guard armed. No automatic start.
    status=request();start=time.monotonic();deadline=time.time()+seconds
    record(event='armed',pid=os.getpid(),pod=POD,seconds=seconds,deadline_epoch=deadline,initial_status=status)
    with awake():
        while time.monotonic()-start < seconds and time.time() < deadline:
            time.sleep(min(1,max(0,seconds-(time.monotonic()-start))))
        record(event='deadline_reached')
        for attempt in range(120):
            try:
                result=request('stop');record(event='stop_response',attempt=attempt,**result)
                status=request()
                if status['desiredStatus']=='EXITED':
                    record(event='stopped_verified',status=status);return
            except Exception as error:
                # Exceptions above are redacted; do not record arbitrary response bodies.
                record(event='stop_retry',attempt=attempt,error_type=type(error).__name__)
            time.sleep(5)
        record(event='FAILED_STOP_REQUIRES_ATTENTION')
        raise RuntimeError('Automatic stop could not be verified')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('seal','status','stop','watch'))
    parser.add_argument('--seconds',type=int);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.action=='seal': result=seal()
    elif args.action=='watch':
        if args.seconds is None or args.out is None: parser.error('watch requires seconds and out')
        watch(args.seconds,args.out);result={'watch_finished':True}
    else: result=request(args.action)
    print(json.dumps(result))
