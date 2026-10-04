"""Gemini password field; local encrypted save only, never auto-spend or upload."""
import secrets
from http.server import ThreadingHTTPServer
from .runway_settings import CredentialStore,handler

if __name__=='__main__':
    origin='http://127.0.0.1:8780'
    server=ThreadingHTTPServer(('127.0.0.1',8780),
        handler(CredentialStore('gemini'),origin,secrets.token_urlsafe(32),provider='gemini'))
    print('Gemini settings: '+origin,flush=True)
    server.serve_forever()
