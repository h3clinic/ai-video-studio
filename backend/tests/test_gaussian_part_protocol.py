"""Part routing/coordination mechanics, not generated-video quality evidence."""
import unittest

import torch

from real_video.gaussian_latent_memory import GaussianLatentMemory, bind_alpha_cache
from real_video.gaussian_part_protocol import GaussianPartProtocol


class GaussianPartProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([40, 10, 90])
        self.gen = torch.tensor([0, 2, 0])
        self.parts = torch.tensor([7, 2, 7])
        self.bank = GaussianLatentMemory(self.ids, 2, 'a'*64, generations=self.gen,
            initial_features=torch.tensor([[2., 4.], [8., 6.], [3., 5.]]), initial_mass=1.)
        self.protocol = GaussianPartProtocol(self.bank, ids=self.ids, generations=self.gen, part_ids=self.parts)
        # Pixel0 has front alpha=.5 and back alpha=.8 => global weights .5,.4.
        self.raw = dict(pixel=torch.tensor([0, 0, 1]), ids=torch.tensor([0, 1, 2]),
                        weight=torch.tensor([.5, .4, .25]))
        self.cache = bind_alpha_cache(self.raw, ids=self.ids, generations=self.gen,
                                      asset_digest='a'*64, height=1, width=2)

    def proposal(self, ids=(40,), part=7, task='move'):
        return self.protocol.proposal(part, torch.tensor(ids), torch.ones(len(ids), 3), task_id=task)

    def test_partition_sums_to_global_read_and_keeps_occlusion(self):
        result = self.protocol.read_parts(self.cache, 1, 2)
        full = self.bank.read(self.cache, 1, 2, ids=self.ids)
        torch.testing.assert_close(result['features'].sum(0), full['features'])
        torch.testing.assert_close(result['coverage'].sum(0), full['coverage'])
        self.assertEqual(result['part_ids'].tolist(), [2, 7])
        self.assertAlmostEqual(float(result['coverage'][0, 0, 0]), .4)
        self.assertNotAlmostEqual(float(result['coverage'][0, 0, 0]), .8)

    def test_empty_occluded_parts_present(self):
        empty = bind_alpha_cache(dict(pixel=torch.empty(0,dtype=torch.long),
            ids=torch.empty(0,dtype=torch.long), weight=torch.empty(0)), ids=self.ids,
            generations=self.gen, asset_digest='a'*64, height=1, width=2)
        result = self.protocol.read_parts(empty, 1, 2)
        self.assertEqual(tuple(result['features'].shape), (2, 2, 1, 2))
        self.assertEqual(int(result['features'].count_nonzero()), 0)
        self.assertEqual(int(result['coverage'].count_nonzero()), 0)

    def test_bad_complete_cache_cannot_be_hidden_by_part_mask(self):
        for cache in (dict(self.cache, ids=torch.tensor([0, 1, 999])),
                      dict(self.cache, weight=torch.tensor([.9, .9, .25])), self.raw):
            with self.assertRaises(ValueError):
                self.protocol.read_parts(cache, 1, 2)

    def test_stale_cache_and_protocol_rejected(self):
        stale = bind_alpha_cache(self.raw, ids=self.ids, generations=self.gen+1,
                                 asset_digest='a'*64, height=1, width=2)
        with self.assertRaises(ValueError):
            self.protocol.read_parts(stale, 1, 2)
        proposal = self.proposal()
        self.bank.advance_generations(self.gen+1, ids=self.ids)
        with self.assertRaises(ValueError):
            self.protocol.read_parts(self.cache, 1, 2)
        with self.assertRaises(ValueError):
            self.protocol.validate_proposals([proposal])

    def test_metadata_copied_and_legacy_bank_rejected(self):
        self.ids.fill_(0); self.gen.fill_(99); self.parts.fill_(55)
        self.assertEqual(self.protocol.read_parts(self.cache, 1, 2)['part_ids'].tolist(), [2, 7])
        legacy = GaussianLatentMemory(torch.arange(2), 1, 'a'*64)
        with self.assertRaises(ValueError):
            GaussianPartProtocol(legacy, ids=torch.arange(2), generations=torch.zeros(2,dtype=torch.long),
                                 part_ids=torch.zeros(2,dtype=torch.long))

    def test_invalid_assignment_and_lifetime_fail_at_construction(self):
        for parts in (torch.tensor([-1,2,7]), torch.ones(3), torch.ones(2,dtype=torch.long)):
            with self.assertRaises(ValueError):
                GaussianPartProtocol(self.bank, ids=self.ids, generations=self.gen, part_ids=parts)
        with self.assertRaises(ValueError):
            GaussianPartProtocol(self.bank, ids=self.ids.flip(0), generations=self.gen, part_ids=self.parts)
        with self.assertRaises(ValueError):
            GaussianPartProtocol(self.bank, ids=self.ids, generations=self.gen+1, part_ids=self.parts)

    def test_bound_cache_asset_dimensions_and_row_order_cannot_change(self):
        for change in (dict(asset_digest='b'*64), dict(height=2), dict(ids=self.ids.flip(0))):
            binding = dict(self.cache['identity_binding'], **change)
            with self.assertRaises(ValueError):
                self.protocol.read_parts(dict(self.cache, identity_binding=binding), 1, 2)

    def test_nonmonotonic_external_ids_resolve_original_rows(self):
        result = self.protocol.validate_proposals([self.proposal(ids=(90,40))])[0]
        self.assertEqual(result['original_rows'].tolist(), [2, 0])

    def test_unknown_crosspart_duplicate_ids_rejected(self):
        for ids, part in (((99,), 7), ((10,), 7), ((40,40), 7), ((40,), 100), ((), 7)):
            with self.subTest(ids=ids, part=part), self.assertRaises(ValueError):
                self.proposal(ids=ids, part=part)

    def test_proposal_asset_generation_and_task_validation(self):
        proposal = self.proposal()
        invalid = [dict(proposal, asset_digest='b'*64), dict(proposal, generations=self.gen+1),
                   dict(proposal, schema='unknown'), dict(proposal, task_id=''),
                   dict(proposal, unknown='instruction')]
        for item in invalid:
            with self.assertRaises(ValueError):
                self.protocol.validate_proposals([item])
        with self.assertRaises(ValueError):
            self.protocol.validate_proposals([proposal, proposal])

    def test_conflicts_fail_closed_and_explicit_acknowledgement_only(self):
        first, second = self.proposal(), self.proposal(task='refine')
        before = self.bank.snapshot()
        with self.assertRaises(ValueError):
            self.protocol.validate_proposals([first, second])
        with self.assertRaises(ValueError):
            self.protocol.validate_proposals([first, second], approved_overlap_ids=torch.tensor([90]))
        with self.assertRaises(ValueError):
            self.protocol.validate_proposals([first, second], approved_overlap_ids=torch.tensor([999]))
        self.assertEqual(len(self.protocol.validate_proposals([first, second],
                             approved_overlap_ids=torch.tensor([40]))), 2)
        self.assertTrue(torch.equal(before['features'], self.bank.snapshot()['features']))
        self.assertEqual(self.bank.writes, 0)

    def test_distinct_parts_are_not_screen_overlap_conflicts(self):
        validated = self.protocol.validate_proposals([self.proposal(), self.proposal(ids=(10,),part=2,task='background')])
        self.assertEqual(len(validated), 2)

    def test_controls_retain_gradient_and_reject_invalid_values(self):
        controls = torch.tensor([[1.,2.,3.]], requires_grad=True)
        item = self.protocol.proposal(7, torch.tensor([40]), controls, task_id='learned_control')
        result = self.protocol.validate_proposals([item])[0]
        self.assertIs(result['controls'], controls)
        result['controls'].square().sum().backward()
        torch.testing.assert_close(controls.grad, controls.detach()*2)
        for invalid in (torch.ones(1,3,dtype=torch.long), torch.ones(2,3),
                        torch.tensor([[float('nan')]]), torch.ones(1,0)):
            with self.assertRaises(ValueError):
                self.protocol.proposal(7, torch.tensor([40]), invalid, task_id='bad')


if __name__ == '__main__':
    unittest.main()
