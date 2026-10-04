"""CPU algebra/state tests; no learned writeback or visual quality claims."""
import tempfile
import unittest
from pathlib import Path

import torch

from real_video.gaussian_latent_memory import GaussianLatentMemory


IDS = torch.tensor([29, 7, 61])
DIGEST = 'a'*64


def cache():
    return dict(pixel=torch.tensor([0, 1]), ids=torch.tensor([1, 0]), weight=torch.tensor([1., 1.]))


class GaussianLatentMemoryTests(unittest.TestCase):
    def memory(self):
        return GaussianLatentMemory(IDS, 2, DIGEST, initial_features=torch.tensor([[1., 2.], [3., 4.], [8., 9.]]))

    def test_original_row_order_and_same_cache_replay(self):
        memory = self.memory()
        result = memory.read(cache(), 1, 2, ids=IDS)
        torch.testing.assert_close(result['features'], torch.tensor([[[3., 1.]], [[4., 2.]]]))
        latent = torch.tensor([[[7., 5.]], [[8., 6.]]])
        memory.write(latent, cache(), ids=IDS, confidence=torch.ones(1, 2))
        self.assertTrue(torch.equal(memory.read(cache(), 1, 2, ids=IDS)['features'], latent))
        with self.assertRaises(ValueError):
            memory.read(cache(), 1, 2, ids=IDS.flip(0))

    def test_occlusion_and_zero_confidence_preserve_identity_features(self):
        memory = self.memory()
        before = memory.snapshot()
        memory.write(torch.ones(2, 1, 2)*77, cache(), ids=IDS, confidence=torch.tensor([[1., 0.]]))
        after = memory.snapshot()
        self.assertTrue(torch.equal(after['features'][[0, 2]], before['features'][[0, 2]]))
        self.assertTrue(torch.equal(after['ids'], before['ids']))
        self.assertEqual(after['observation_mass'][2], 0)

    def test_weighted_sum_read_and_weighted_evidence_write(self):
        memory = self.memory()
        mixed = dict(pixel=torch.tensor([0, 0, 1]), ids=torch.tensor([0, 1, 1]), weight=torch.tensor([.25, .5, 1.]))
        recalled = memory.read(mixed, 1, 2, ids=IDS)
        torch.testing.assert_close(recalled['features'][:, 0, 0], torch.tensor([1.75, 2.5]))
        self.assertEqual(float(recalled['coverage'][0, 0]), .75)
        latent = torch.tensor([[[2., 8.]], [[4., 10.]]])
        memory.write(latent, mixed, ids=IDS, confidence=torch.tensor([[1., .5]]))
        torch.testing.assert_close(memory.snapshot()['features'][1], torch.tensor([5., 7.]))

    def test_bounded_allocation_and_mass(self):
        memory = GaussianLatentMemory(IDS, 2, DIGEST, max_observation_mass=3.)
        before = memory.memory_bytes()
        pointers = [memory._features.data_ptr(), memory._mass.data_ptr(), memory._ids.data_ptr()]
        for _ in range(100):
            memory.write(torch.ones(2, 1, 2), cache(), ids=IDS, confidence=torch.ones(1, 2))
        self.assertEqual(memory.memory_bytes(), before)
        self.assertEqual([memory._features.data_ptr(), memory._mass.data_ptr(), memory._ids.data_ptr()], pointers)
        self.assertLessEqual(float(memory.snapshot()['observation_mass'].max()), 3.)
        self.assertEqual(before['frame_history_bytes'], 0)

    def test_invalid_confidence_cache_and_missing_confidence_rejected(self):
        memory = self.memory()
        before = memory.snapshot()
        for confidence in (torch.ones(1, 2)*-1, torch.ones(1, 2)*2, torch.ones(1, 2)*float('nan'), torch.ones(2, 1)):
            with self.assertRaises(ValueError):
                memory.write(torch.ones(2, 1, 2), cache(), ids=IDS, confidence=confidence)
        with self.assertRaises(TypeError):
            memory.write(torch.ones(2, 1, 2), cache(), ids=IDS)
        invalid = cache()
        invalid['ids'] = torch.tensor([99, 0])
        with self.assertRaises(ValueError):
            memory.read(invalid, 1, 2, ids=IDS)
        self.assertTrue(torch.equal(memory.snapshot()['features'], before['features']))
        self.assertEqual(memory.writes, 0)

    def test_explicit_validity_and_empty_cache(self):
        memory = self.memory()
        before = memory.snapshot()
        memory.write(torch.zeros(2, 1, 2), cache(), ids=IDS, confidence=torch.ones(1, 2), validity=torch.zeros(1, 2, dtype=torch.bool))
        self.assertTrue(torch.equal(memory.snapshot()['features'], before['features']))
        empty = dict(pixel=torch.empty(0, dtype=torch.long), ids=torch.empty(0, dtype=torch.long), weight=torch.empty(0))
        result = memory.read(empty, 1, 2, ids=IDS)
        self.assertEqual(result['features'].count_nonzero(), 0)
        self.assertEqual(result['coverage'].count_nonzero(), 0)

    def test_snapshot_exact_resume_and_hash_binding(self):
        first, second = self.memory(), self.memory()
        latent = torch.tensor([[[1., 4.]], [[2., 5.]]])
        first.write(latent, cache(), ids=IDS, confidence=torch.ones(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latent_state.pt'
            first.save(path)
            second.load(path)
        for memory in (first, second):
            memory.write(latent+1, cache(), ids=IDS, confidence=torch.ones(1, 2)*.5)
        for key, value in first.snapshot().items():
            actual = second.snapshot()[key]
            self.assertTrue(torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual)
        wrong = GaussianLatentMemory(IDS, 2, 'b'*64)
        with self.assertRaises(ValueError):
            wrong.restore(first.snapshot())
        snapshot = first.snapshot()
        snapshot['features'].zero_()
        self.assertFalse(torch.equal(snapshot['features'], first.snapshot()['features']))


if __name__ == '__main__':
    unittest.main()
