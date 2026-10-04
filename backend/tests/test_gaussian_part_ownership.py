import unittest
from dataclasses import replace
from real_video.gaussian_part_ownership import (
    PartBinding, OwnershipManifest, PartEdit, apply_transactions, replacement_scope,
    inherit_densification)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.state = {i: {'position': [0., 0., 1.], 'colour': [.5, .5, .5],
                          'opacity': 1., 'covariance': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
                      for i in range(4)}
        self.apple = PartBinding('apple', (0,), binding_verified=True, agent_id='skin')
        self.slice = PartBinding('slice', (1,), 'detached_from', 'apple',
                                 binding_verified=True, agent_id='slice_agent')
        self.donkey = PartBinding('donkey', (2,), protected=True, binding_verified=True)
        self.manifest = OwnershipManifest(0, (self.apple, self.slice, self.donkey))

    def edit(self, **kwargs):
        args = dict(agent_id='skin', part_id='apple', expected_revision=0,
                    updates={0: {'colour': [1., .1, .1]}})
        args.update(kwargs)
        return PartEdit(**args)

    def test_atomic_scoped_edits_preserve_other_parts(self):
        second = self.edit(agent_id='slice_agent', part_id='slice', updates={1: {'opacity': .8}})
        out, manifest = apply_transactions(self.state, self.manifest, [self.edit(), second])
        self.assertEqual(out[0]['colour'], (1., .1, .1))
        self.assertEqual(out[2]['colour'], tuple(self.state[2]['colour']))
        self.assertEqual(manifest.revision, 1)
        self.assertEqual(self.state[0]['colour'], [.5, .5, .5])
        with self.assertRaises(TypeError):
            out[0]['colour'] = (0., 0., 0.)

    def test_stale_and_overlapping_batches_fail(self):
        for edits in ([self.edit(expected_revision=1)], [self.edit(), self.edit()]):
            with self.assertRaises(ValueError):
                apply_transactions(self.state, self.manifest, edits)

    def test_unbound_protected_foreign_and_wrong_agent_fail(self):
        for edit in (self.edit(part_id='donkey', updates={2: {'opacity': 0.}}),
                     self.edit(updates={1: {'opacity': 0.}}),
                     self.edit(agent_id='other'), self.edit(part_id='absent')):
            with self.assertRaises(ValueError):
                apply_transactions(self.state, self.manifest, [edit])
        for apple in (replace(self.apple, binding_verified=False), replace(self.apple, gaussian_ids=())):
            with self.assertRaises(ValueError):
                apply_transactions(self.state, OwnershipManifest(0, (apple,)), [self.edit()])

    def test_invalid_attributes_reject_whole_batch(self):
        for attrs in ({'position': [float('nan'), 0., 1.]}, {'opacity': 2},
                      {'colour': [1., 0.]}, {'covariance': [[1, 0, 0], [0, 0, 0], [0, 0, 1]]},
                      {'covariance': [[1, 2, 0], [0, 1, 0], [0, 0, 1]]}, {'id': 10}):
            with self.assertRaises(ValueError):
                apply_transactions(self.state, self.manifest, [self.edit(updates={0: attrs})])
        self.assertEqual(self.state[0]['colour'], [.5, .5, .5])

    def test_manifest_overlap_missing_parent_cycle(self):
        for parts in ((self.apple, replace(self.slice, gaussian_ids=(0,))),
                      (self.slice,), (replace(self.apple, parent_id='apple'),)):
            with self.assertRaises(ValueError):
                OwnershipManifest(0, parts)

    def test_explicit_relation_scope_not_theme_equivalence(self):
        peel = PartBinding('orange_peel', (3,))
        m = OwnershipManifest(0, self.manifest.parts + (peel,))
        self.assertEqual(replacement_scope(m, 'apple'), ('apple', 'slice'))
        self.assertEqual(replacement_scope(m, 'orange_peel'), ('orange_peel',))

    def test_field_permissions_and_input_aliasing(self):
        colours = [1., 0., 0.]
        edit = self.edit(updates={0: {'colour': colours}})
        colours[0] = 0.
        out, _ = apply_transactions(self.state, self.manifest, [edit])
        self.assertEqual(out[0]['colour'][0], 1.)
        limited = replace(self.apple, allowed_fields=frozenset({'position'}))
        with self.assertRaises(ValueError):
            apply_transactions(self.state, OwnershipManifest(0, (limited,)), [edit])

    def test_absent_bound_state_fails(self):
        with self.assertRaises(ValueError):
            apply_transactions({}, self.manifest, [self.edit()])

    def densify(self, **changes):
        args=dict(agent_id='skin',part_id='apple',expected_revision=0,
                  existing_ids=set(self.state)|{9},child_to_parent={10:0,11:0},lineage_verified=True)
        args.update(changes)
        return inherit_densification(self.manifest,**args)

    def test_densification_preserves_ids_owner_and_untouched_parts(self):
        out,lineage=self.densify()
        self.assertEqual(out.parts[0].gaussian_ids,(0,10,11))
        self.assertEqual(out.parts[0].agent_id,'skin')
        self.assertEqual(out.parts[1:],self.manifest.parts[1:])
        self.assertEqual(self.manifest.parts[0].gaussian_ids,(0,))
        self.assertEqual(out.revision,1)
        self.assertEqual(lineage[0]['parent_id'],0)
        with self.assertRaises(TypeError):lineage[0]['part_id']='donkey'

    def test_densification_rejects_collisions_and_guessed_ownership(self):
        for change in [dict(child_to_parent={0:0}),dict(child_to_parent={9:0}),
                       dict(child_to_parent={10:2}),dict(child_to_parent={True:0}),
                       dict(agent_id='unknown'),dict(part_id='donkey'),dict(expected_revision=1),
                       dict(lineage_verified=False),dict(existing_ids={0})]:
            with self.assertRaises(ValueError):self.densify(**change)


if __name__ == '__main__':
    unittest.main()
