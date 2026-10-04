"""Mock HTTPS streams only: no live downloads, model execution or GPU."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from real_video.resumable_model_download import (
    DownloadVerificationError, ResumeRejectedError, download_verified,
)


URL = 'https://huggingface.co/Official/model/resolve/pinned/weights.bin'
DATA = b'correct immutable model bytes'
SHA = hashlib.sha256(DATA).hexdigest()


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error


def response(status, body, headers=None, error=None):
    return httpx.Response(status, headers=headers, stream=Chunks([body] if isinstance(body, bytes) else body, error))


class ResumableModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name)/'model.bin'
        self.partial = self.target.with_name('model.bin.partial')
        self.metadata = self.target.with_name('model.bin.partial.json')
        # Exercise the same bounded buffering logic with tiny test fixtures.
        chunk_size = patch('real_video.resumable_model_download.READ_CHUNK_BYTES', 7)
        chunk_size.start()
        self.addCleanup(chunk_size.stop)

    def call(self, handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        with patch('real_video.resumable_model_download.httpx.Client', return_value=client) as factory:
            answer = download_verified(kwargs.get('url', URL), self.target, kwargs.get('size', len(DATA)), kwargs.get('sha256', SHA), range_bytes=kwargs.get('range_bytes', 16*2**20))
        self.assertTrue(factory.call_args.kwargs['verify'])
        self.assertFalse(factory.call_args.kwargs['follow_redirects'])
        return answer

    def interrupt(self, prefix=DATA[:7]):
        with self.assertRaises(httpx.ReadError):
            self.call(lambda request: response(206, prefix, {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}', 'Content-Length': str(len(DATA))}, httpx.ReadError('disconnect')))
        self.assertFalse(self.target.exists())
        self.assertEqual(self.partial.read_bytes(), prefix)
        self.assertTrue(self.metadata.exists())

    def test_complete_verified_promote_and_offline_cache_hit(self):
        path = self.call(lambda request: response(206, DATA, {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}', 'Content-Length':str(len(DATA))}))
        self.assertEqual(path, self.target)
        self.assertEqual(path.read_bytes(), DATA)
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.metadata.exists())
        with patch('real_video.resumable_model_download.httpx.Client', side_effect=AssertionError('network not needed')):
            self.assertEqual(download_verified(URL, self.target, len(DATA), SHA), self.target)

    def test_interrupted_prefix_is_resumed_with_exact_range(self):
        self.interrupt()
        def resumed(request):
            self.assertEqual(request.headers['Range'], f'bytes=7-{len(DATA)-1}')
            self.assertEqual(request.headers['Accept-Encoding'], 'identity')
            return response(206, DATA[7:], {'Content-Range':f'bytes 7-{len(DATA)-1}/{len(DATA)}', 'Content-Length':str(len(DATA)-7)})
        self.assertEqual(self.call(resumed).read_bytes(), DATA)

    def test_ignored_range_preserves_old_prefix_without_appending(self):
        self.interrupt()
        with self.assertRaises(ResumeRejectedError):
            self.call(lambda request: response(200, DATA))
        self.assertEqual(self.partial.read_bytes(), DATA[:7])
        self.assertFalse(self.target.exists())

    def test_bad_content_ranges_lengths_and_encoding_never_mutate_prefix(self):
        self.interrupt()
        variants = [({}, 206), ({'Content-Range':'bytes 0-3/4'}, 206),
            ({'Content-Range':f'bytes 8-{len(DATA)-1}/{len(DATA)}'}, 206),
            ({'Content-Range':f'bytes 7-{len(DATA)-1}/*'}, 206),
            ({'Content-Range':f'bytes 7-{len(DATA)-1}/{len(DATA)}', 'Content-Length':'2'}, 206),
            ({'Content-Encoding':'gzip'}, 206), ({}, 416)]
        for headers, status in variants:
            with self.subTest(headers=headers, status=status):
                with self.assertRaises(ResumeRejectedError):
                    self.call(lambda request: response(status, DATA[7:], headers))
                self.assertEqual(self.partial.read_bytes(), DATA[:7])

    def test_hash_failure_retains_complete_file_and_never_promotes(self):
        incorrect = b'X'*len(DATA)
        with self.assertRaises(DownloadVerificationError):
            self.call(lambda request: response(206, incorrect, {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}'}))
        self.assertEqual(self.partial.read_bytes(), incorrect)
        self.assertFalse(self.target.exists())
        with patch('real_video.resumable_model_download.httpx.Client', side_effect=AssertionError('should verify offline')):
            with self.assertRaises(DownloadVerificationError):
                download_verified(URL, self.target, len(DATA), SHA)
        self.assertEqual(self.partial.read_bytes(), incorrect)

    def test_full_valid_partial_after_promotion_interruption_promotes_offline(self):
        with patch('real_video.resumable_model_download.os.link', side_effect=OSError('interrupted publish')):
            with self.assertRaises(OSError):
                self.call(lambda request: response(206, DATA, {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}'}))
        self.assertEqual(self.partial.read_bytes(), DATA)
        with patch('real_video.resumable_model_download.httpx.Client', side_effect=AssertionError('no network')):
            result = download_verified(URL, self.target, len(DATA), SHA)
        self.assertEqual(result.read_bytes(), DATA)

    def test_short_body_saved_but_wrong_short_range_rejected_before_append(self):
        with self.assertRaises(DownloadVerificationError):
            self.call(lambda request: response(206, DATA[:7], {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}', 'Content-Length':str(len(DATA))}))
        self.assertEqual(self.partial.read_bytes(), DATA[:7])
        with self.assertRaises(ResumeRejectedError):
            self.call(lambda request: response(206, DATA[7:12], {'Content-Range':f'bytes 7-11/{len(DATA)}'}))
        self.assertEqual(self.partial.read_bytes(), DATA[:7])

    def test_source_binding_changes_and_unbound_partial_rejected(self):
        self.interrupt()
        for kwargs in ({'url':URL+'?different=1'}, {'sha256':'a'*64}, {'size':len(DATA)+1}):
            with patch('real_video.resumable_model_download.httpx.Client', side_effect=AssertionError('no network')):
                with self.assertRaises(ResumeRejectedError):
                    download_verified(kwargs.get('url', URL), self.target, kwargs.get('size', len(DATA)), kwargs.get('sha256', SHA))
        self.metadata.unlink()
        with self.assertRaises(ResumeRejectedError):
            download_verified(URL, self.target, len(DATA), SHA)
        self.assertEqual(self.partial.read_bytes(), DATA[:7])

    def test_https_redirect_range_preserved_and_http_downgrade_rejected(self):
        self.interrupt()
        def handler(request):
            self.assertEqual(request.headers['Range'], f'bytes=7-{len(DATA)-1}')
            if request.url.host == 'huggingface.co':
                return response(302, b'', {'Location':'https://cdn.hf.co/signed-weight'})
            return response(206, DATA[7:], {'Content-Range':f'bytes 7-{len(DATA)-1}/{len(DATA)}'})
        self.assertEqual(self.call(handler).read_bytes(), DATA)
        self.target.unlink()
        with self.assertRaises(ValueError):
            self.call(lambda request: response(302, b'', {'Location':'http://cdn.hf.co/weight'}))
        self.assertFalse(self.target.exists())

    def test_existing_different_destination_never_overwritten(self):
        self.target.write_bytes(b'user data')
        with self.assertRaises(DownloadVerificationError):
            download_verified(URL, self.target, len(DATA), SHA)
        self.assertEqual(self.target.read_bytes(), b'user data')

    def test_destination_created_during_transfer_prevents_overwrite(self):
        def handler(request):
            self.target.write_bytes(b'external data')
            return response(206, DATA, {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}'})
        with self.assertRaises(FileExistsError):
            self.call(handler)
        self.assertEqual(self.target.read_bytes(), b'external data')
        self.assertEqual(self.partial.read_bytes(), DATA)

    def test_invalid_contract_rejected_without_network(self):
        for url, size, sha in [('http://example.org/a',len(DATA),SHA), ('https://u:p@example.org/a',len(DATA),SHA),
                (URL,0,SHA), (URL,True,SHA), (URL,len(DATA),'bad')]:
            with self.assertRaises(ValueError):
                download_verified(url, self.target, size, sha)

    def test_explicit_bounded_ranges_exact_end_and_progress_between_windows(self):
        requested = []
        def handler(request):
            start, end = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
            requested.append((start, end))
            self.assertLessEqual(end-start+1, 8)
            if start:
                self.assertEqual(self.partial.read_bytes(), DATA[:start])
            return response(206, DATA[start:end+1], {'Content-Range':f'bytes {start}-{end}/{len(DATA)}'})
        self.assertEqual(self.call(handler, range_bytes=8).read_bytes(), DATA)
        self.assertEqual(requested, [(0, 7), (8, 15), (16, 23), (24, len(DATA)-1)])

    def test_first_request_also_rejects_ignored_bounded_range(self):
        with self.assertRaises(ResumeRejectedError):
            self.call(lambda request: response(200, DATA))
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.target.exists())

    def test_buffered_subchunk_on_disconnect_loses_only_unconfirmed_tail(self):
        with self.assertRaises(httpx.ReadError):
            self.call(lambda request: response(206, DATA[:10], {'Content-Range':f'bytes 0-{len(DATA)-1}/{len(DATA)}'}, httpx.ReadError('disconnect')))
        self.assertEqual(self.partial.read_bytes(), DATA[:7])
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
