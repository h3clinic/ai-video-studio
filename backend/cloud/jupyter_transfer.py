"""Authenticated transport for our owned RunPod Jupyter server.

No CLI accepts credentials. Tokens enter through the constructor or environment;
HTTP Authorization headers only, TLS verified, redirects disabled. This module
does not start/stop Pods or provide a billing watchdog. No remote work occurs on
import. Execution is explicit and uses a newly owned, finally-deleted kernel.

Protocol references:
https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html
https://jupyter-server.readthedocs.io/en/latest/operators/security.html
https://jupyter-server.readthedocs.io/en/latest/developers/websocket-protocols.html
https://jupyter-client.readthedocs.io/en/latest/messaging.html
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import time
from urllib.parse import quote, urlsplit, urlunsplit
import uuid

OWNED_HOST = 'wdq34cebh6111k-8888.proxy.runpod.net'
MAX_FILE_BYTES = 8 * 2**20
MAX_RESPONSE_BYTES = 16 * 2**20


class TransportError(RuntimeError):
    def __init__(self, message, *, http_status=None, operation=None):
        super().__init__(message)
        self.http_status = http_status
        self.operation = operation


def remote_path(value):
    if (not isinstance(value, str) or not value or value.startswith('/')
            or any(c in value for c in ('\\', '%', '\x00', '\r', '\n'))
            or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError('Expected a nonempty relative Jupyter contents path')
    return value


class JupyterClient:
    def __init__(self, base_url, *, token=None, token_env='JUPYTER_TOKEN',
                 known_tokens=(), allowed_host=OWNED_HOST, session=None,
                 websocket_factory=None, request_timeout=30):
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'https' or parsed.hostname != allowed_host
                or parsed.port not in (None, 443) or parsed.username or parsed.password
                or parsed.query or parsed.fragment or '..' in parsed.path.split('/')):
            raise ValueError('Expected the approved HTTPS server origin, without credentials or query')
        token = token if token is not None else os.environ.get(token_env)
        if not isinstance(token, str) or not token or any(c in token for c in '\r\n'):
            raise ValueError('A nonempty Jupyter token is required internally or through environment')
        if not isinstance(request_timeout, (int, float)) or not 0 < request_timeout <= 60:
            raise ValueError('Request timeout must be within 60 seconds')
        self.base_url = base_url.rstrip('/')
        self._token = token
        self._secrets = tuple(sorted({token, *(v for v in known_tokens if isinstance(v, str) and v)}, key=len, reverse=True))
        self._timeout = request_timeout
        self.request_history = []
        self._websocket_factory = websocket_factory
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False  # Do not leak headers into implicit proxies/netrc.
        self._session = session

    def _redact(self, value):
        value = str(value)
        for secret in self._secrets:
            value = value.replace(secret, '[REDACTED]')
        return value

    def _api(self, method, endpoint, *, payload=None, params=None, missing_ok=False):
        response = None
        try:
            response = self._session.request(method, self.base_url+'/api/'+endpoint,
                headers={'Authorization': 'token '+self._token, 'Accept':'application/json'}, json=payload, params=params,
                timeout=self._timeout, verify=True, allow_redirects=False, stream=True)
            # Fixed structural metadata only: no host, path, headers, body,
            # query value or kernel identifier can reach the diagnostic log.
            self.request_history.append(dict(method=method,resource=endpoint.split('/',1)[0],
                path_depth=len(endpoint.split('/'))-1,http_status=response.status_code,
                content=params.get('content') if isinstance(params,dict) else None))
            self.request_history=self.request_history[-40:]
            if not 200 <= response.status_code < 300:
                # A proxy HTML 404 is NOT evidence that a remote file is absent.
                # Contents callers confirm absence against a parent listing.
                if response.status_code == 404 and missing_ok:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=4096):
                        body.extend(chunk)
                        if len(body)>65536:break
                    try:
                        missing = json.loads(body) if len(body)<=65536 else None
                    except (ValueError,UnicodeError):
                        missing = None
                    if isinstance(missing,dict) and isinstance(missing.get('message'),str):
                        return None
                # Never include server bodies, redirect targets or request headers.
                operation=method+' '+endpoint.split('/',1)[0]
                raise TransportError(f'Jupyter HTTP status {response.status_code} ({operation})',
                    http_status=response.status_code,operation=operation)
            if response.status_code == 204:
                return {}
            data = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise TransportError('Jupyter response exceeded transfer limit')
            return json.loads(data)
        except TransportError:
            raise
        except Exception:
            raise TransportError('Jupyter request or response validation failed') from None
        finally:
            if response is not None:
                response.close()

    def verify_service(self):
        """Authenticated read-only readiness; RUNNING alone is not readiness."""
        root=self._api('GET','contents',params={'content':0})
        specs=self._api('GET','kernelspecs')
        if not isinstance(root,dict) or root.get('type')!='directory':
            raise TransportError('Jupyter contents root is not ready')
        if not isinstance(specs,dict) or 'python3' not in specs.get('kernelspecs',{}):
            raise TransportError('Jupyter Python kernel specification is not ready')
        return {'authenticated_contents_ready':True,'python3_kernelspec_ready':True}

    def _contents_model(self, path, *, params, missing_ok=False, writable_parent=False):
        """A 404 alone never authorizes creating/overwriting remote contents.

        Jupyter Server v2.17.0 ContentsHandler._finish_error writes a plain
        missing-path sentence despite its application/json response type. See
        https://github.com/jupyter-server/jupyter_server/blob/v2.17.0/jupyter_server/services/contents/handlers.py
        We accept neither that sentence nor an opaque proxy page as absence:
        a bounded authenticated parent listing must independently prove it.
        No POST/PUT/kernel retry is introduced here.
        """
        path=remote_path(path)
        try:
            model=self._api('GET','contents/'+quote(path,safe='/'),params=params,missing_ok=missing_ok)
            if model is not None:return model
        except TransportError as error:
            if not missing_ok or error.http_status!=404:raise
        if not missing_ok:raise TransportError('Jupyter contents model missing')
        parent,_,name=path.rpartition('/')
        endpoint='contents'+('/'+quote(parent,safe='/') if parent else '')
        listing=self._parent_listing(endpoint)
        if (not isinstance(listing,dict) or listing.get('type')!='directory'
                or listing.get('path')!=parent or not isinstance(listing.get('content'),list)
                or len(listing['content'])>10000):
            raise TransportError('Cannot confirm absent contents from parent directory')
        if writable_parent and listing.get('writable') is not True:
            raise TransportError('Upload parent is not explicitly writable')
        names=set()
        for entry in listing['content']:
            entry_name=entry.get('name') if isinstance(entry,dict) else None
            if (not isinstance(entry_name,str) or not entry_name or entry_name in ('.','..')
                    or '/' in entry_name or '\\' in entry_name or '\x00' in entry_name
                    or entry.get('path')!=(parent+'/' if parent else '')+entry_name
                    or entry.get('type') not in ('file','directory','notebook') or entry_name in names):
                raise TransportError('Invalid parent directory contents model')
            names.add(entry_name)
        if name in names:
            raise TransportError('Requested contents exists but could not be read; no overwrite authorized')
        return None

    def _parent_listing(self, endpoint):
        """Retry only an idempotent directory read, never infer absence from 404.

        Proxy/readiness failures have been observed immediately after a valid
        authenticated check. The last failure is still terminal; none of these
        responses authorizes PUT without a verified writable parent listing.
        """
        for attempt in range(3):
            try:return self._api('GET',endpoint,params={'content':1})
            except TransportError as error:
                if error.http_status not in (404,502,503,504) or attempt==2:raise
                time.sleep(attempt+1)

    def read_bytes(self, path, *, missing_ok=False):
        path = remote_path(path)
        model = self._contents_model(path,
            params={'type': 'file', 'format': 'base64', 'content': 1}, missing_ok=missing_ok)
        if model is None:
            return None
        return self._file_bytes(model)

    def _file_bytes(self, model):
        try:
            if model['type'] != 'file' or model['format'] != 'base64':
                raise ValueError('Wrong contents model')
            data = base64.b64decode(''.join(model['content'].split()), validate=True)
            if len(data) > MAX_FILE_BYTES or model.get('size') not in (None, len(data)):
                raise ValueError('Size mismatch')
            return data
        except Exception:
            raise TransportError('Jupyter file format/size verification failed') from None

    def ensure_directory(self, path):
        path = remote_path(path)
        current = ''
        for part in path.split('/'):
            current = part if not current else current+'/'+part
            endpoint = 'contents/'+quote(current, safe='/')
            model = self._contents_model(current,params={'content':0},missing_ok=True,writable_parent=True)
            if model is None:
                model = self._api('PUT', endpoint, payload={'type': 'directory'})
            if model.get('type') != 'directory':
                raise TransportError('Upload directory conflicts with an existing file')

    def put_bytes(self, path, data, *, expected_sha256=None, replace_corrupt=False):
        path = remote_path(path)
        if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES:
            raise ValueError('Upload payload must be bytes up to 8 MiB')
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError('Local upload digest mismatch')
        model = self._contents_model(path,params={'type':'file','format':'base64','content':1},
            missing_ok=True,writable_parent=True)
        old = self._file_bytes(model) if model is not None else None
        if old == data:
            return dict(path=path, bytes=len(data), sha256=digest, reused=True)
        if old is not None and not replace_corrupt:
            raise FileExistsError('Refusing to overwrite different remote contents')
        self._api('PUT', 'contents/'+quote(path, safe='/'), payload={
            'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode('ascii')})
        observed = self.read_bytes(path)
        if len(observed) != len(data) or hashlib.sha256(observed).hexdigest() != digest:
            raise TransportError('Uploaded file failed readback SHA-256/size verification')
        return dict(path=path, bytes=len(data), sha256=digest, reused=False)

    def upload_parts(self, directory, remote_directory, manifest_sha256):
        """Resume a pinned chunk_transfer manifest, including the 13-part bundle.

        Only numbered files authorized by that exact local manifest can be
        repaired. No assembly/extraction/execution is performed here.
        """
        from cloud.chunk_transfer import load_manifest, inspect, no_links
        directory = no_links(directory)
        manifest = load_manifest(directory, manifest_sha256)
        if not inspect(directory, manifest_sha256)['ready']:
            raise ValueError('Local transfer parts are incomplete or corrupt')
        remote_directory = remote_path(remote_directory)
        self.ensure_directory(remote_directory)
        # Pin directory ownership before repairing any numbered file; do not
        # mutate a directory already committed to a different manifest.
        self.put_bytes(remote_directory+'/manifest.json', no_links(directory/'manifest.json').read_bytes(),
                       expected_sha256=manifest_sha256)
        results = []
        for part in manifest['parts']:
            data = no_links(directory/part['name']).read_bytes()
            if len(data) != part['bytes']:
                raise ValueError('Local part changed after validation')
            results.append(self.put_bytes(remote_directory+'/'+part['name'], data,
                expected_sha256=part['sha256'], replace_corrupt=True))
        self.put_bytes(remote_directory+'/manifest.json', no_links(directory/'manifest.json').read_bytes(),
                       expected_sha256=manifest_sha256)
        return dict(parts=len(results), reused=sum(p['reused'] for p in results),
            uploaded_bytes=sum(p['bytes'] for p in results if not p['reused']),
            manifest_sha256=manifest_sha256, whole_file_sha256=manifest['sha256'],
            verified_parts=results, assembled=False)

    def execute(self, code, logfile, *, timeout_seconds=60, kernel_path=None):
        """Run explicit caller code; wait for correlated reply AND IOPub idle.

        Timeout/connection loss does not prove a subprocess stopped. The caller
        must retain its independent Pod billing watchdog. This method attempts
        to delete only the fresh kernel it created, even after failure.
        """
        if not isinstance(code, str) or not code or any(s in code for s in self._secrets):
            raise ValueError('Code must be nonempty and must not embed authentication tokens')
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 600:
            raise ValueError('Execution timeout must be within 600 seconds')
        factory = self._websocket_factory
        if factory is None:
            try:
                from websocket import create_connection, enableTrace
            except ImportError:
                raise TransportError('Install websocket-client in the project environment before execution') from None
            enableTrace(False)  # Handshake tracing would include auth headers.
            factory = create_connection
        payload = {'name': 'python3'}
        if kernel_path is not None:
            payload['path'] = remote_path(kernel_path)
        # Never overwrite existing evidence. Do this before creating a kernel.
        logfile = Path(logfile)
        log = logfile.open('x', encoding='utf-8')
        kernel_id = None
        socket = None
        reply = None
        idle = False
        cleanup_ok = False
        started = time.monotonic()
        msg_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
        # Delay a suffix so secrets split across separate stream messages are
        # redacted before any fragment reaches disk.
        pending = ''
        keep = max(len(s) for s in self._secrets)-1
        def output(value, final=False):
            nonlocal pending
            pending += str(value)
            pending = self._redact(pending)
            cut = len(pending) if final else max(0, len(pending)-keep)
            log.write(pending[:cut]); log.flush(); pending = pending[cut:]
        try:
            model = self._api('POST', 'kernels', payload=payload)
            kernel_id = model.get('id')
            if not isinstance(kernel_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', kernel_id):
                raise TransportError('Invalid new kernel identity')
            parsed = urlsplit(self.base_url)
            url = urlunsplit(('wss', parsed.netloc, parsed.path+'/api/kernels/'+kernel_id+'/channels',
                             'session_id='+session_id, ''))
            # websocket-client otherwise follows redirects with custom headers.
            socket = factory(url, header={'Authorization': 'token '+self._token},
                origin='https://'+parsed.netloc, timeout=min(self._timeout, timeout_seconds),
                sslopt={'cert_reqs': ssl.CERT_REQUIRED, 'check_hostname': True}, redirect_limit=0,
                http_no_proxy=[parsed.hostname])
            # websocket-client 1.9 can return a socket for a redirect when its
            # redirect budget is zero. Refuse that response before sending code.
            if getattr(getattr(socket, 'handshake_response', None), 'status', None) != 101:
                raise TransportError('WebSocket upgrade failed or was redirected')
            socket.send(json.dumps(dict(header=dict(msg_id=msg_id, username='gaussian-evaluator',
                session=session_id, date=datetime.now(timezone.utc).isoformat(),
                msg_type='execute_request', version='5.3'), parent_header={}, metadata={},
                channel='shell', buffers=[], content=dict(code=code, silent=False,
                    store_history=False, user_expressions={}, allow_stdin=False, stop_on_error=True))))
            while reply is None or not idle:
                remaining = timeout_seconds-(time.monotonic()-started)
                if remaining <= 0:
                    raise TransportError('Kernel execution deadline expired; external watchdog still required')
                socket.settimeout(min(remaining, 30))
                try:
                    raw = socket.recv()
                except Exception as error:
                    if isinstance(error, TimeoutError) or type(error).__name__ == 'WebSocketTimeoutException':
                        continue
                    raise
                if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_RESPONSE_BYTES:
                    raise TransportError('Unexpected or oversized kernel message')
                message = json.loads(raw)
                if message.get('parent_header', {}).get('msg_id') != msg_id:
                    continue
                kind = message.get('header', {}).get('msg_type', message.get('msg_type'))
                content = message.get('content', {})
                if kind == 'stream':
                    output(content.get('text', ''))
                elif kind in ('execute_result', 'display_data'):
                    output(content.get('data', {}).get('text/plain', '')+'\n')
                elif kind == 'error':
                    output('\n'.join(content.get('traceback', []))+'\n')
                elif kind == 'execute_reply':
                    reply = content.get('status')
                    if reply not in ('ok', 'error', 'aborted'):
                        raise TransportError('Invalid execution status')
                elif kind == 'status' and content.get('execution_state') == 'idle':
                    idle = True
            if reply != 'ok':
                raise TransportError('Remote Python execution failed; sanitized output retained')
        except TransportError:
            raise
        except Exception:
            raise TransportError('Kernel transport failed; remote completion is unconfirmed') from None
        finally:
            if socket is not None:
                try: socket.close()
                except Exception: pass
            if kernel_id and re.fullmatch(r'[A-Za-z0-9_-]+', kernel_id):
                try:
                    self._api('DELETE', 'kernels/'+kernel_id)
                    cleanup_ok = True
                except Exception:
                    output('\nKernel cleanup was not confirmed; external watchdog required.\n')
            output('', final=True)
            log.close()
        if not cleanup_ok:
            raise TransportError('Execution completed but kernel cleanup was not confirmed')
        return dict(status=reply, matched_reply_and_idle=True, kernel_deleted=True,
                    seconds=time.monotonic()-started, logfile=str(logfile))
