"""Loopback-only credential settings. No Runway calls or generation on save.

Windows DPAPI binds the encrypted credential to the current Windows user.
The file is outside the repository. Never serve/decrypt credentials over HTTP.
"""
import argparse
import ctypes
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile

KEY_PATTERNS = {'runway': r'key_[0-9a-f]{128}', 'gemini': r'AIza[A-Za-z0-9_-]{35}',
                'runpod': r'rpa_[A-Za-z0-9_-]{16,256}',
                # ElevenLabs keys are opaque; do not require a particular prefix.
                'elevenlabs': r'[A-Za-z0-9_-]{16,512}'}


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def protect(value):
    if os.name != 'nt':
        raise RuntimeError('Windows DPAPI required')
    raw = ctypes.create_string_buffer(value)
    incoming = Blob(len(value), ctypes.cast(raw, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    api = ctypes.WinDLL('crypt32', use_last_error=True)
    api.CryptProtectData.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    api.CryptProtectData.restype = wintypes.BOOL
    if not api.CryptProtectData(ctypes.byref(incoming), 'Gaussian video provider credential',
                               None, None, None, 1, ctypes.byref(outgoing)):
        raise RuntimeError('Credential encryption failed')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        kernel.LocalFree(outgoing.data)


class CredentialStore:
    def __init__(self, provider='runway'):
        if provider not in KEY_PATTERNS: raise ValueError('Unsupported credential provider')
        self.provider=provider
        self.path = Path(os.environ['LOCALAPPDATA'])/'GaussianVectorVideo'/f'{provider}.dpapi'

    def configured(self):
        return self.path.is_file()

    def load_for_api(self):
        """Server-side only. Never expose this result in an HTTP settings response."""
        if os.name != 'nt':
            raise RuntimeError('Windows DPAPI required')
        encrypted=self.path.read_bytes()
        raw=ctypes.create_string_buffer(encrypted)
        incoming=Blob(len(encrypted),ctypes.cast(raw,ctypes.POINTER(ctypes.c_ubyte)))
        outgoing=Blob()
        api=ctypes.WinDLL('crypt32',use_last_error=True)
        api.CryptUnprotectData.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        api.CryptUnprotectData.restype=wintypes.BOOL
        if not api.CryptUnprotectData(ctypes.byref(incoming),None,None,None,None,1,ctypes.byref(outgoing)):
            raise RuntimeError('Credential decryption failed')
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.LocalFree.argtypes=[ctypes.c_void_p]
        kernel.LocalFree.restype=ctypes.c_void_p
        try:
            key=ctypes.string_at(outgoing.data,outgoing.size).decode('ascii')
            if not re.fullmatch(KEY_PATTERNS[self.provider],key):
                raise RuntimeError('Stored credential format invalid')
            return key
        finally:
            ctypes.memset(outgoing.data,0,outgoing.size)
            kernel.LocalFree(outgoing.data)

    def save(self, key):
        if not isinstance(key,str) or not re.fullmatch(KEY_PATTERNS[self.provider],key):
            raise ValueError('Invalid credential format')
        encrypted = protect(key.encode('ascii'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.provider+'-', suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encrypted)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)


PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gaussian Video · Runway settings</title>
<style nonce="TOKEN">body{background:#111923;color:#eef3f9;font:16px system-ui;margin:0;padding:32px}main{max-width:620px;margin:5vh auto;background:#1b2838;padding:32px;border-radius:16px}h1{font-size:26px}p{line-height:1.6;color:#c8d5e5}label{display:block;margin-top:24px}input{box-sizing:border-box;width:100%;margin:10px 0 20px;padding:14px;background:#101822;color:white;border:1px solid #7891aa;border-radius:6px}button{padding:12px 20px;background:#81d8bb;color:#10221d;border:0;border-radius:6px;font-weight:700;cursor:pointer}a{color:#81d8bb}#status{min-height:48px}</style>
<main><a href="http://127.0.0.1:8768/">← Gaussian scene</a>
<h1>Connect Runway</h1><p>Paste your API key below. It is encrypted for your Windows account and stored outside the project. It is never returned to this page or written to logs.</p>
<form id="form"><label for="key">Runway API key</label><input id="key" type="password" autocomplete="off" spellcheck="false" placeholder="key_…" required maxlength="132"><button id="save">Save key locally</button></form>
<p id="status" role="status" aria-live="polite"></p>
<p>Saving does not validate credits, upload your cat, or start a paid generation. Runway integration is not yet connected to the video pipeline.</p></main>
<script nonce="TOKEN">
const status=document.getElementById('status'), field=document.getElementById('key'), button=document.getElementById('save');
fetch('/api/runway/status').then(r=>r.json()).then(s=>{status.textContent=s.configured?'A key is stored. Saving replaces it. API validity has not been checked.':'No key stored yet.'}).catch(()=>{status.textContent='Settings server unavailable.'});
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;
let key=field.value.trim();field.value='';
try{const response=await fetch('/api/runway/key',{method:'POST',headers:{'Content-Type':'application/json','X-Settings-Token':'TOKEN'},body:JSON.stringify({key})});key='';const data=await response.json();status.textContent=data.message;}catch{status.textContent='Could not save. Check that the local server is running.';}finally{key='';button.disabled=false;}});
</script></html>'''


def handler(store, origin, token, provider='runway'):
    if provider not in KEY_PATTERNS: raise ValueError('Unsupported provider')
    page=PAGE
    if provider=='gemini':
        page=page.replace('Runway','Gemini').replace('/api/runway/','/api/gemini/').replace('key_…','AIza…')
        page=page.replace('maxlength="132"','maxlength="39"')
        page=page.replace('Saving does not validate credits, upload your cat, or start a paid generation. Gemini integration is not yet connected to the video pipeline.',
            'Saving makes no Google API call and uploads no media. Gemini will propose structured edit plans; it cannot directly delete Gaussians or execute code. API billing is separate from a Gemini app subscription. Only selected scene descriptions and preview images will be sent when an analysis is explicitly run.')
    elif provider=='elevenlabs':
        page=page.replace('Runway','ElevenLabs').replace('/api/runway/','/api/elevenlabs/').replace('key_…','Paste ElevenLabs API key')
        page=page.replace('maxlength="132"','maxlength="512"')
        page=page.replace('Saving does not validate credits, upload your cat, or start a paid generation. ElevenLabs integration is not yet connected to the video pipeline.',
            'Saves directly to AI Video Studio’s encrypted key store. No API call or paid generation starts when you save.')
    class SettingsHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log request bodies, headers, credentials, or URLs.

        def reply(self, code, data, html=False):
            body = data.encode() if html else json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'none'; connect-src 'self'; "
                f"script-src 'nonce-{token}'; style-src 'nonce-{token}'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def local(self):
            return self.headers.get('Host') == origin.removeprefix('http://')

        def do_GET(self):
            if not self.local():
                return self.reply(403, {'message':'Invalid host'})
            if self.path == '/':
                return self.reply(200, page.replace('TOKEN', token), html=True)
            if self.path == f'/api/{provider}/status':
                return self.reply(200, {'configured':store.configured()})
            self.reply(404, {'message':'Not found'})

        def do_POST(self):
            if (not self.local() or self.headers.get('Origin') != origin
                    or not secrets.compare_digest(self.headers.get('X-Settings-Token',''),token)):
                return self.reply(403, {'message':'Request rejected; reload the local settings page.'})
            if self.path != f'/api/{provider}/key':
                return self.reply(404, {'message':'Not found'})
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 1024 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError()
                payload = json.loads(self.rfile.read(size))
                key = payload.get('key')
                if not isinstance(key,str) or not re.fullmatch(KEY_PATTERNS[provider],key):
                    raise ValueError()
            except (ValueError, AttributeError, UnicodeError):
                message={'runway':'Expected key_ followed by 128 hexadecimal characters.',
                    'gemini':'Expected a 39-character Google API key beginning AIza.'}.get(provider,'Invalid API key format.')
                return self.reply(400, {'message':message+' Nothing saved.'})
            try:
                store.save(key)
            except Exception:
                return self.reply(500, {'message':'Secure storage failed. Key was not saved.'})
            self.reply(200, {'message':'Key encrypted and saved locally. No generation started; API validity not checked.'})
    return SettingsHandler


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8779)
    parser.add_argument('--provider',choices=('runway','gemini','elevenlabs'),default='runway')
    args=parser.parse_args()
    origin=f'http://127.0.0.1:{args.port}'
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(CredentialStore(args.provider),origin,secrets.token_urlsafe(32),provider=args.provider))
    server.timeout=10
    print(f'{args.provider} settings: {origin}',flush=True)
    server.serve_forever()
