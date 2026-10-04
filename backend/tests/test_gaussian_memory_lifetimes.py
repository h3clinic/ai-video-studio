"""Lifetime consistency tests, not evidence of video generation quality."""
import unittest
import torch
from real_video.gaussian_latent_memory import GaussianLatentMemory, bind_alpha_cache
from real_video.latent_track_memory import build_anchor_memory_condition, planar_gaussian_cache
from real_video.residual_gaussian_writer import write_residual

DIGEST = 'a' * 64


class MemoryLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([7, 9])
        self.gen = torch.zeros(2, dtype=torch.long)
        self.raw = dict(pixel=torch.arange(2), ids=torch.arange(2), weight=torch.ones(2))
        self.bank = GaussianLatentMemory(self.ids, 1, DIGEST, generations=self.gen,
            initial_features=torch.tensor([[3.], [8.]]), initial_mass=1.)

    def cache(self, **kwargs):
        args = dict(ids=self.ids, generations=self.gen, asset_digest=DIGEST, height=1, width=2)
        args.update(kwargs)
        return bind_alpha_cache(self.raw, **args)

    def assertSnapshotEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key], key)

    def test_retirement_clears_only_replaced_material(self):
        before = self.bank.memory_bytes()
        self.assertEqual(self.bank.advance_generations(torch.tensor([1, 0]), ids=self.ids)['retired_slots'], 1)
        snap = self.bank.snapshot()
        self.assertEqual(snap['features'].flatten().tolist(), [0., 8.])
        self.assertEqual(snap['observation_mass'].tolist(), [0., 1.])
        self.assertEqual(snap['writes'], 0)
        for epoch in range(2, 20):
            self.bank.advance_generations(torch.tensor([epoch, 0]), ids=self.ids)
        self.assertEqual(before, self.bank.memory_bytes())
        self.assertEqual(before['generation_bytes'], 16)
        self.assertEqual(before['frame_history_bytes'], 0)

    def test_wrong_binding_fails_without_mutation(self):
        invalid = [self.raw, self.cache(ids=self.ids.flip(0)), self.cache(generations=torch.ones(2, dtype=torch.long)),
                   self.cache(asset_digest='b'*64), self.cache(height=2)]
        before = self.bank.snapshot()
        for cache in invalid:
            with self.subTest(cache=cache):
                with self.assertRaises(ValueError):
                    self.bank.read(cache, 1, 2, ids=self.ids)
                with self.assertRaises(ValueError):
                    self.bank.write(torch.ones(1, 1, 2), cache, ids=self.ids, confidence=torch.ones(1, 2))
                with self.assertRaises(ValueError):
                    write_residual(self.bank, torch.ones(1, 1, 2), cache, ids=self.ids, confidence=torch.ones(1, 2))
                self.assertSnapshotEqual(before, self.bank.snapshot())

    def test_old_cache_stale_after_retirement(self):
        cache = self.cache()
        self.bank.advance_generations(torch.tensor([1, 0]), ids=self.ids)
        with self.assertRaises(ValueError):
            self.bank.read(cache, 1, 2, ids=self.ids)
        out = self.bank.read(self.cache(generations=torch.tensor([1, 0])), 1, 2, ids=self.ids)
        self.assertEqual(out['features'].flatten().tolist(), [0., 8.])

    def test_metadata_is_copied(self):
        cache = self.cache()
        self.ids.zero_()
        self.gen.fill_(99)
        self.assertEqual(cache['identity_binding']['ids'].tolist(), [7, 9])
        self.assertEqual(cache['identity_binding']['generations'].tolist(), [0, 0])

    def test_invalid_generation_updates_atomic(self):
        self.bank.advance_generations(torch.ones(2, dtype=torch.long), ids=self.ids)
        before = self.bank.snapshot()
        for value in (torch.tensor([0, 1]), torch.tensor([-1, 1]), torch.ones(2), torch.ones(3, dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.bank.advance_generations(value, ids=self.ids)
            self.assertSnapshotEqual(before, self.bank.snapshot())

    def test_restore_resumes_and_rejects_schema_downgrade(self):
        original = self.bank.snapshot()
        self.bank.advance_generations(torch.ones(2, dtype=torch.long), ids=self.ids)
        self.bank.restore(original)
        self.assertSnapshotEqual(original, self.bank.snapshot())
        invalid = dict(original, generations=torch.tensor([-1, 0]))
        with self.assertRaises(ValueError):
            self.bank.restore(invalid)
        self.assertSnapshotEqual(original, self.bank.snapshot())
        legacy = GaussianLatentMemory(self.ids, 1, DIGEST)
        with self.assertRaises(ValueError):
            legacy.restore(original)
        with self.assertRaises(ValueError):
            self.bank.restore(legacy.snapshot())
        with self.assertRaises(ValueError):
            legacy.read(self.cache(), 1, 2, ids=self.ids)

    def test_residual_writer_preserves_lifetime(self):
        write_residual(self.bank, torch.tensor([[[4., 9.]]]), self.cache(), ids=self.ids, confidence=torch.ones(1, 2))
        self.assertTrue(torch.equal(self.bank.snapshot()['generations'], self.gen))
        self.assertGreater(self.bank.snapshot()['features'][0, 0], 3.)

    def test_empty_planar_cache_still_bound(self):
        args = (torch.zeros(2, 2), torch.zeros(2), self.ids, 1, 2)
        cache = planar_gaussian_cache(*args, generations=self.gen, asset_digest=DIGEST)
        self.assertEqual(len(cache['weight']), 0)
        self.bank.read(cache, 1, 2, ids=self.ids)
        with self.assertRaises(ValueError):
            planar_gaussian_cache(*args, generations=self.gen)

    def build(self, generations=None, visible=None):
        return build_anchor_memory_condition(torch.full((1, 2, 2), 3.),
            torch.zeros(9, 2, 2), torch.ones(9, 2) if visible is None else visible,
            asset_digest=DIGEST, ids=self.ids, return_snapshot=True, track_generations=generations)

    def test_anchor_replacement_cannot_recall_previous_appearance(self):
        gen = torch.ones(9, 2, dtype=torch.long)
        gen[0] = 0
        strict, metrics, snapshot = self.build(gen)
        legacy, _, _ = self.build()
        self.assertEqual(int(strict[:, :, 1:].count_nonzero()), 0)
        self.assertGreater(int(legacy[:, :, 1:].count_nonzero()), 0)
        self.assertTrue(metrics['material_lifetimes_checked'])
        self.assertEqual(snapshot['generations'].tolist(), [0, 0])
        self.assertGreater(float(snapshot['features'].sum()), 0.)
        self.assertEqual(snapshot['generations'].shape, (2,))

    def test_occlusion_does_not_retire_and_later_changes_are_local(self):
        gen = torch.zeros(9, 2, dtype=torch.long)
        visible = torch.ones(9, 2)
        visible[1:5] = 0
        result, _, _ = self.build(gen, visible)
        self.assertEqual(int(result[:, :, 1].count_nonzero()), 0)
        self.assertGreater(int(result[:, :, 2].count_nonzero()), 0)
        gen[5:] = 1
        changed, _, _ = self.build(gen, visible)
        self.assertTrue(torch.equal(changed[:, :, :2], result[:, :, :2]))
        self.assertEqual(int(changed[:, :, 2].count_nonzero()), 0)

    def test_nonmonotonic_history_rejected(self):
        gen = torch.zeros(9, 2, dtype=torch.long)
        gen[2] = 1
        with self.assertRaises(ValueError):
            self.build(gen)


if __name__ == '__main__':
    unittest.main()
