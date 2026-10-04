import base64
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from urllib.parse import unquote

from cloud.jupyter_transfer import JupyterClient, TransportError, OWNED_HOST, remote_path
from cloud.chunk_transfer import split


class Response:
    def __init__(self, status, value=None):
        self.status_code, self.value = status, value
    def iter_content(self, chunk_size):
        yield json.dumps(self.value).encode()
    def close(self): pass


class Session:
    def __init__(self):
        self.files, self.dirs, self.calls = {}, {'','parts'}, []
        self.status = None
        self.get_overrides = {}
        self.readonly_dirs = set()
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.status: return Response(self.status)
        endpoint = unquote(url.split('/api/', 1)[1])
        if endpoint == 'kernels': return Response(201, {'id': 'kernel-1'})
        if endpoint == 'kernels/kernel-1': return Response(204)
        path = '' if endpoint=='contents' else endpoint.removeprefix('contents/')
        if method=='GET' and path in self.get_overrides:return self.get_overrides[path]
        if method == 'PUT':
            data = kwargs['json']
            if data['type'] == 'directory': self.dirs.add(path)
            else: self.files[path] = base64.b64decode(data['content'])
        if path in self.dirs:
            prefix=path+'/' if path else ''
            children=[]
            for child in sorted((self.dirs|set(self.files))-{path}):
                if child.startswith(prefix) and '/' not in child[len(prefix):]:
                    children.append(dict(name=child[len(prefix):],path=child,type='directory' if child in self.dirs else 'file'))
            return Response(200, dict(type='directory',path=path,writable=path not in self.readonly_dirs,content=children))
        if path not in self.files: return Response(404,{'message':'No such file','reason':None})
        content = self.files[path]
        return Response(200, dict(type='file', format='base64', size=len(content),
                                  content=base64.b64encode(content).decode()))


class Socket:
    def __init__(self, events=None):
        self.events = events or [('stream', {'text':'hello\n'}),
            ('execute_reply', {'status':'ok'}), ('status', {'execution_state':'idle'})]
        self.sent = None; self.closed = False
        self.handshake_response = SimpleNamespace(status=101)
    def send(self, value): self.sent = json.loads(value)
    def settimeout(self, value): self.timeout = value
    def recv(self):
        kind, content = self.events.pop(0)
        return json.dumps(dict(header={'msg_type':kind}, content=content,
            parent_header={'msg_id':self.sent['header']['msg_id']}))
    def close(self): self.closed = True


class JupyterTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = Session(); self.socket = Socket(); self.connection = None
        def connect(url, **kwargs):
            self.connection = (url, kwargs)
            return self.socket
        self.client = JupyterClient('https://'+OWNED_HOST, token='SECRET-12345',
            session=self.session, websocket_factory=connect)

    def test_rejects_wrong_origin_unsafe_url_and_path(self):
        for url in ('http://'+OWNED_HOST, 'https://evil.example',
                    'https://'+OWNED_HOST+'?token=x', 'https://user:pw@'+OWNED_HOST):
            with self.assertRaises(ValueError): JupyterClient(url, token='x')
        for path in ('/absolute', '../parent', 'x/../y', 'x%2fy', 'x\\y', 'x//y'):
            with self.assertRaises(ValueError): remote_path(path)

    def test_roundtrip_headers_tls_no_redirects(self):
        report = self.client.put_bytes('parts/a.bin', b'abc')
        self.assertFalse(report['reused'])
        self.assertEqual(self.client.read_bytes('parts/a.bin'), b'abc')
        for _, url, kwargs in self.session.calls:
            self.assertNotIn('SECRET', url)
            self.assertEqual(kwargs['headers']['Authorization'], 'token SECRET-12345')
            self.assertEqual(kwargs['headers']['Accept'],'application/json')
            self.assertTrue(kwargs['verify']); self.assertFalse(kwargs['allow_redirects'])

    def test_redirect_rejected_without_body_or_location(self):
        self.session.status = 302
        with self.assertRaisesRegex(TransportError, '302') as error:
            self.client.read_bytes('x')
        self.assertNotIn('SECRET', str(error.exception))
        self.assertEqual(len(self.session.calls), 1)

    def test_proxy_404_does_not_authorize_a_write(self):
        self.session.status=404
        with patch('cloud.jupyter_transfer.time.sleep'),self.assertRaises(TransportError) as error:
            self.client.put_bytes('missing/file',b'unchanged')
        self.assertEqual(error.exception.http_status,404)
        self.assertEqual(error.exception.operation,'GET contents')
        self.assertEqual(len(self.session.calls),4)
        self.assertTrue(all(call[0]=='GET' for call in self.session.calls))

    def test_transient_parent_404_rechecks_read_but_puts_only_once(self):
        original=self.session.request
        attempts=[]
        def request(method,url,**kwargs):
            if url.endswith('/contents/parts') and method=='GET':
                attempts.append(method)
                if len(attempts)<3:return Response(404)
            return original(method,url,**kwargs)
        self.session.request=request
        with patch('cloud.jupyter_transfer.time.sleep') as sleep:
            self.client.put_bytes('parts/new.bin',b'abc')
        self.assertEqual(len(attempts),3)
        self.assertEqual(sleep.call_count,2)
        self.assertEqual(sum(method=='PUT' for method,_,_ in self.session.calls),1)
        self.assertEqual(self.session.files['parts/new.bin'],b'abc')
        history=json.dumps(self.client.request_history)
        for value in ('SECRET','parts','new.bin',OWNED_HOST):self.assertNotIn(value,history)

    def test_plain_jupyter_missing_response_uses_parent_listing_before_create(self):
        class PlainMissing(Response):
            def iter_content(self,chunk_size):
                yield b"file or directory 'newdir' does not exist"
        self.session.get_overrides['newdir']=PlainMissing(404)
        self.client.ensure_directory('newdir')
        self.assertIn('newdir',self.session.dirs)
        self.assertEqual([call[0] for call in self.session.calls],['GET','GET','PUT'])
        self.assertTrue(self.session.calls[1][1].endswith('/api/contents'))
        self.assertEqual(self.session.calls[1][2]['params'],{'content':1})

    def test_opaque_404_only_allows_upload_after_writable_parent_absence(self):
        self.session.get_overrides['parts/new.bin']=Response(404)
        # Allow successful readback after the single upload; no request retry.
        original=self.session.request
        def request(method,url,**kwargs):
            result=original(method,url,**kwargs)
            if method=='PUT':self.session.get_overrides.pop('parts/new.bin',None)
            return result
        self.session.request=request
        self.client.put_bytes('parts/new.bin',b'new')
        self.assertEqual(self.session.files['parts/new.bin'],b'new')
        self.assertEqual([call[0] for call in self.session.calls],['GET','GET','PUT','GET'])

    def test_parent_existing_entry_prevents_overwrite_after_false_404(self):
        self.session.files['parts/old.bin']=b'preserved'
        self.session.get_overrides['parts/old.bin']=Response(404)
        with self.assertRaisesRegex(TransportError,'exists but could not be read'):
            self.client.put_bytes('parts/old.bin',b'new',replace_corrupt=True)
        self.assertEqual(self.session.files['parts/old.bin'],b'preserved')
        self.assertTrue(all(call[0]=='GET' for call in self.session.calls))

    def test_nonwritable_or_unverifiable_parent_prevents_creation(self):
        self.session.readonly_dirs.add('parts')
        with self.assertRaisesRegex(TransportError,'not explicitly writable'):
            self.client.put_bytes('parts/new.bin',b'new')
        self.assertTrue(all(call[0]=='GET' for call in self.session.calls))
        self.session.calls.clear();self.session.readonly_dirs.clear()
        self.session.get_overrides['parts']=Response(200,{'type':'directory','path':'wrong','writable':True,'content':[]})
        with self.assertRaisesRegex(TransportError,'Cannot confirm absent'):
            self.client.put_bytes('parts/new.bin',b'new')
        self.assertTrue(all(call[0]=='GET' for call in self.session.calls))

    def test_immutable_conflict_and_explicit_chunk_repair(self):
        self.session.files['x'] = b'old'
        with self.assertRaises(FileExistsError): self.client.put_bytes('x', b'new')
        self.assertEqual(self.session.files['x'], b'old')
        self.client.put_bytes('x', b'new', replace_corrupt=True)
        self.assertEqual(self.session.files['x'], b'new')

    def test_local_digest_mismatch_before_requests(self):
        with self.assertRaises(ValueError):
            self.client.put_bytes('x', b'abc', expected_sha256='0'*64)
        self.assertEqual(self.session.calls, [])

    def test_thirteen_part_resume_skips_verified(self):
        source = self.root/'input'; source.write_bytes(bytes(range(130)))
        parts = self.root/'parts'; info = split(source, parts, chunk_bytes=10)
        report = self.client.upload_parts(parts, 'transfer/run1', info['manifest_sha256'])
        self.assertEqual(report['parts'], 13)
        self.assertEqual(report['uploaded_bytes'], 130)
        self.session.calls.clear()
        del self.session.files['transfer/run1/part-00004.bin']
        self.session.files['transfer/run1/part-00008.bin'] = b'corrupt!!!'
        report = self.client.upload_parts(parts, 'transfer/run1', info['manifest_sha256'])
        self.assertEqual(report['reused'], 11)
        self.assertEqual(report['uploaded_bytes'], 20)
        self.assertEqual(sum(method == 'PUT' for method,_,_ in self.session.calls), 2)

    def test_different_manifest_cannot_mutate_old_parts(self):
        source = self.root/'input'; source.write_bytes(b'abcd')
        parts = self.root/'parts'; info = split(source, parts, chunk_bytes=2)
        self.session.dirs.add('transfer')
        self.session.files['transfer/manifest.json'] = b'old manifest'
        self.session.files['transfer/part-00000.bin'] = b'old data'
        with self.assertRaises(FileExistsError):
            self.client.upload_parts(parts, 'transfer', info['manifest_sha256'])
        self.assertEqual(self.session.files['transfer/part-00000.bin'], b'old data')

    def test_kernel_reply_and_idle_tls_cleanup(self):
        report = self.client.execute('print(1)', self.root/'run.log')
        self.assertTrue(report['matched_reply_and_idle']); self.assertTrue(report['kernel_deleted'])
        self.assertEqual((self.root/'run.log').read_text(), 'hello\n')
        url, args = self.connection
        self.assertTrue(url.startswith('wss://')); self.assertNotIn('SECRET', url)
        self.assertEqual(args['redirect_limit'], 0)
        self.assertEqual(args['sslopt']['cert_reqs'], ssl.CERT_REQUIRED)
        self.assertTrue(args['sslopt']['check_hostname'])
        self.assertFalse(self.socket.sent['content']['store_history'])
        self.assertFalse(self.socket.sent['content']['allow_stdin'])
        self.assertEqual(self.session.calls[-1][0], 'DELETE')

    def test_split_secret_stream_redacted(self):
        self.socket.events = [('stream', {'text':'before SECRET-'}), ('stream', {'text':'12345 after\n'}),
            ('status', {'execution_state':'idle'}), ('execute_reply', {'status':'ok'})]
        self.client.execute('print(1)', self.root/'run.log')
        log = (self.root/'run.log').read_text()
        self.assertEqual(log, 'before [REDACTED] after\n')
        self.assertNotIn('SECRET', log)

    def test_remote_failure_cleanup_and_sanitized_log(self):
        self.socket.events = [('error', {'traceback':['bad SECRET-12345']}),
            ('execute_reply', {'status':'error'}), ('status', {'execution_state':'idle'})]
        with self.assertRaisesRegex(TransportError, 'execution failed'):
            self.client.execute('raise ValueError()', self.root/'run.log')
        self.assertTrue(self.socket.closed); self.assertEqual(self.session.calls[-1][0], 'DELETE')
        self.assertNotIn('SECRET', (self.root/'run.log').read_text())

    def test_websocket_redirect_does_not_send_code(self):
        self.socket.handshake_response.status = 302
        with self.assertRaisesRegex(TransportError, 'redirected'):
            self.client.execute('print(1)', self.root/'run.log')
        self.assertIsNone(self.socket.sent)
        self.assertTrue(self.socket.closed)
        self.assertEqual(self.session.calls[-1][0], 'DELETE')

    def test_disconnect_cleanup_no_success(self):
        self.socket.events = [('status', {'execution_state':'idle'})]
        with self.assertRaisesRegex(TransportError, 'completion is unconfirmed'):
            self.client.execute('print(1)', self.root/'run.log')
        self.assertEqual(self.session.calls[-1][0], 'DELETE')

    def test_no_secret_code_or_existing_log(self):
        with self.assertRaises(ValueError):
            self.client.execute("print('SECRET-12345')", self.root/'run.log')
        (self.root/'run.log').write_text('existing')
        with self.assertRaises(FileExistsError): self.client.execute('print(1)', self.root/'run.log')
        self.assertEqual(self.session.calls, [])


if __name__ == '__main__': unittest.main()
