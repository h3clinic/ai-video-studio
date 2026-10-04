import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cloud.chunk_transfer import split, inspect, assemble, sha256, validate_manifest, load_manifest


class ChunkTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.zip'
        self.source.write_bytes(bytes(range(251)) * 20)
        self.parts = self.root / 'parts'
        self.info = split(self.source, self.parts, 1024)
        self.pin = self.info['manifest_sha256']
        self.out = self.root / 'assembled.zip'

    def test_round_trip(self):
        self.assertEqual(self.info['parts'], 5)
        self.assertTrue(inspect(self.parts, self.pin)['ready'])
        assemble(self.parts, self.out, self.pin)
        self.assertEqual(self.out.read_bytes(), self.source.read_bytes())
        self.assertEqual(list(self.root.glob('*.partial')), [])

    def test_completed_assembly_idempotent(self):
        assemble(self.parts, self.out, self.pin)
        self.assertTrue(assemble(self.parts, self.out, self.pin)['reused'])

    def test_resume_only_missing(self):
        part = self.parts / 'part-00002.bin'
        saved = part.read_bytes(); part.unlink()
        before = sha256(self.parts / 'part-00000.bin')
        state = inspect(self.parts, self.pin)
        self.assertEqual(state['missing'], ['part-00002.bin'])
        self.assertEqual(state['remaining_upload_bytes'], 1024)
        with self.assertRaises(ValueError): assemble(self.parts, self.out, self.pin)
        self.assertFalse(self.out.exists())
        part.write_bytes(saved)
        assemble(self.parts, self.out, self.pin)
        self.assertEqual(before, sha256(self.parts / 'part-00000.bin'))

    def test_corrupt_same_length_detected(self):
        (self.parts / 'part-00001.bin').write_bytes(b'x'*1024)
        self.assertEqual(inspect(self.parts, self.pin)['corrupt'], ['part-00001.bin'])
        with self.assertRaises(ValueError): assemble(self.parts, self.out, self.pin)

    def test_truncated_part(self):
        (self.parts / 'part-00000.bin').write_bytes(b'x')
        self.assertFalse(inspect(self.parts, self.pin)['ready'])

    def test_output_not_replaced(self):
        self.out.write_bytes(b'previous results')
        with self.assertRaises(FileExistsError): assemble(self.parts, self.out, self.pin)
        self.assertEqual(self.out.read_bytes(), b'previous results')

    def test_pinned_manifest_required(self):
        for bad in ('', '0'*64, '../manifest'):
            with self.subTest(bad=bad), self.assertRaises(ValueError): inspect(self.parts, bad)

    def test_manifest_tampering(self):
        with (self.parts/'manifest.json').open('a') as stream: stream.write(' ')
        with self.assertRaises(ValueError): inspect(self.parts, self.pin)

    def test_manifest_uses_same_bytes_for_hash_and_parse(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('must not reopen')):
            self.assertEqual(load_manifest(self.parts, self.pin)['bytes'], 5020)

    def test_manifest_size_is_bounded(self):
        (self.parts/'manifest.json').write_bytes(b' ' * (4 * 2**20 + 1))
        with self.assertRaisesRegex(ValueError, 'Oversized'):
            load_manifest(self.parts, self.pin)

    def test_manifest_schema(self):
        original = json.loads((self.parts/'manifest.json').read_text())
        mutations = [lambda m: m.update(bytes=True), lambda m: m.update(bytes=0),
            lambda m: m.update(chunk_bytes=9*2**20), lambda m: m.update(sha256='bad'),
            lambda m: m['parts'][0].update(name='../escape'),
            lambda m: m['parts'][0].update(name='C:\\escape'),
            lambda m: m['parts'][0].update(bytes=False),
            lambda m: m['parts'].reverse(), lambda m: m['parts'].pop()]
        for mutate in mutations:
            m = json.loads(json.dumps(original)); mutate(m)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): validate_manifest(m)

    def test_whole_file_digest_checked(self):
        path = self.parts/'manifest.json'
        m = json.loads(path.read_text()); m['sha256'] = '0'*64
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError): assemble(self.parts, self.out, sha256(path))
        self.assertFalse(self.out.exists())
        self.assertEqual(len(list(self.root.glob('*.partial'))), 1)

    def test_split_no_clobber(self):
        with self.assertRaises(FileExistsError): split(self.source, self.parts)

    def test_split_limits(self):
        for size in (0, -1, 9*2**20, True):
            with self.subTest(size=size), self.assertRaises(ValueError): split(self.source, self.root/'bad', size)
        self.source.write_bytes(b'')
        with self.assertRaises(ValueError): split(self.source, self.root/'bad')

    def test_symlink_rejected(self):
        link = self.root/'linked'
        try: link.symlink_to(self.parts, target_is_directory=True)
        except OSError: self.skipTest('OS does not allow unprivileged symlinks')
        with self.assertRaises(ValueError): inspect(link, self.pin)


if __name__ == '__main__': unittest.main()
