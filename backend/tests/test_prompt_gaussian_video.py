"""CPU-only provenance and source-input guard tests for fresh generation."""
import builtins
import io
from pathlib import Path
import tempfile
import unittest

import torch

from real_video.prompt_gaussian_video import inference_input_guard, validate_text_packet
from real_video.wan_baseline import MODEL, REVISION, PROMPT, NEGATIVE


class TextPacketValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

    def packet(self):
        return dict(positive=self.embedding, negative=self.embedding,
                    config=dict(prompt=PROMPT, negative_prompt=NEGATIVE,
                                model=MODEL, revision=REVISION, seed=421001))

    def test_exact_text_packet_passes_and_old_seed_is_not_a_scene_input(self):
        packet = self.packet()
        validate_text_packet(packet, PROMPT, NEGATIVE)
        self.assertEqual(packet['config']['seed'], 421001)

    def test_each_provenance_field_must_match_exactly(self):
        for key in ('prompt', 'negative_prompt', 'model', 'revision'):
            with self.subTest(key=key):
                packet = self.packet()
                packet['config'][key] += ' changed'
                with self.assertRaises(ValueError):
                    validate_text_packet(packet, PROMPT, NEGATIVE)

    def test_requested_prompt_and_negative_are_not_silently_ignored(self):
        for prompt, negative in ((PROMPT + ' ', NEGATIVE), (PROMPT, NEGATIVE + ' ')):
            with self.subTest(prompt=prompt, negative=negative):
                with self.assertRaises(ValueError):
                    validate_text_packet(self.packet(), prompt, negative)

    def test_invalid_shapes_and_non_tensor_rejected_for_both_embeddings(self):
        for name in ('positive', 'negative'):
            for shape in ((512, 4096), (1, 511, 4096), (1, 512, 4095), (2, 512, 4096)):
                with self.subTest(name=name, shape=shape):
                    packet = self.packet()
                    packet[name] = torch.empty(shape)
                    with self.assertRaises(ValueError):
                        validate_text_packet(packet, PROMPT, NEGATIVE)
            packet = self.packet()
            packet[name] = 'not a tensor'
            with self.assertRaises(ValueError):
                validate_text_packet(packet, PROMPT, NEGATIVE)

    def test_nonfinite_and_integer_embeddings_rejected(self):
        for name in ('positive', 'negative'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(name=name, value=value):
                    packet = self.packet()
                    packet[name] = self.embedding.clone()
                    packet[name][0, 0, 0] = value
                    with self.assertRaises(ValueError):
                        validate_text_packet(packet, PROMPT, NEGATIVE)
            packet = self.packet()
            packet[name] = torch.zeros_like(self.embedding, dtype=torch.int32)
            with self.assertRaises(ValueError):
                validate_text_packet(packet, PROMPT, NEGATIVE)


class InferenceInputGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_dir = self.root / 'model'
        self.model_dir.mkdir()
        self.embeddings = self.root / 'text.pt'
        self.decoder = self.root / 'decoder.pt'
        torch.save({'kind': 'text'}, self.embeddings)
        torch.save({'kind': 'decoder'}, self.decoder)

    def guard(self):
        return inference_input_guard(self.model_dir, self.embeddings, self.decoder)

    def test_named_text_and_decoder_packets_can_be_loaded(self):
        with self.guard() as observed:
            self.assertEqual(torch.load(self.embeddings, weights_only=True)['kind'], 'text')
            self.assertEqual(torch.load(self.decoder, weights_only=True)['kind'], 'decoder')
        self.assertEqual(set(observed), {str(self.embeddings.resolve()), str(self.decoder.resolve())})

    def test_media_and_prior_tensor_reads_are_rejected_by_builtin_open(self):
        suffixes = ('.mp4', '.avi', '.mov', '.mkv', '.png', '.jpg', '.jpeg',
                    '.gif', '.webp', '.npy', '.npz', '.pt', '.pth', '.JPG')
        for suffix in suffixes:
            path = self.root / ('source' + suffix)
            path.write_bytes(b'not a real media file')
            with self.subTest(suffix=suffix), self.guard():
                with self.assertRaises(AssertionError):
                    with builtins.open(path, 'rb'):
                        pass

    def test_pathlib_io_and_bytes_paths_are_also_guarded(self):
        path = self.root / 'source.png'
        path.write_bytes(b'pixels')
        with self.guard():
            with self.assertRaises(AssertionError):
                path.read_bytes()
            with self.assertRaises(AssertionError):
                with io.open(path, 'rb'):
                    pass
            with self.assertRaises(AssertionError):
                with builtins.open(bytes(path), 'rb'):
                    pass

    def test_read_write_modes_do_not_bypass_source_read_guard(self):
        path = self.root / 'source.mp4'
        path.write_bytes(b'pixels')
        for mode in ('r+b', 'a+b', 'w+b'):
            with self.subTest(mode=mode), self.guard():
                with self.assertRaises(AssertionError):
                    with builtins.open(path, mode):
                        pass

    def test_output_only_writes_remain_allowed(self):
        with self.guard():
            for suffix in ('.mp4', '.png', '.pt'):
                with builtins.open(self.root / ('output' + suffix), 'wb') as stream:
                    stream.write(b'output')

    def test_only_exact_tensor_paths_are_allowed_even_under_model_dir(self):
        paths = (self.root / 'elsewhere' / 'text.pt', self.model_dir / 'prior.pt')
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'prior': True}, path)
            with self.subTest(path=path), self.guard():
                with self.assertRaises(AssertionError):
                    torch.load(path, weights_only=True)

    def test_anonymous_tensor_stream_is_rejected(self):
        stream = io.BytesIO(self.embeddings.read_bytes())
        with self.guard(), self.assertRaises(AssertionError):
            torch.load(stream, weights_only=True)

    def test_pretrained_weight_paths_are_restricted_to_model_directory(self):
        for suffix in ('.safetensors', '.bin'):
            inside = self.model_dir / ('weights' + suffix)
            outside = self.root / ('weights' + suffix)
            sibling = self.root / ('model-other') / ('weights' + suffix)
            sibling.parent.mkdir(exist_ok=True)
            for path in (inside, outside, sibling):
                path.write_bytes(b'weights')
            with self.subTest(suffix=suffix), self.guard() as observed:
                self.assertEqual(inside.read_bytes(), b'weights')
                for forbidden in (outside, sibling):
                    with self.assertRaises(AssertionError):
                        forbidden.read_bytes()
                self.assertIn(str(inside.resolve()), observed)

    def test_guard_is_restored_after_exception(self):
        original_open, original_io, original_load = builtins.open, io.open, torch.load
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with self.guard():
                raise RuntimeError('sentinel')
        self.assertIs(builtins.open, original_open)
        self.assertIs(io.open, original_io)
        self.assertIs(torch.load, original_load)


if __name__ == '__main__':
    unittest.main()
