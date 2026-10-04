"""Structural success cannot silently approve anatomical readiness."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from real_video.asset_readiness import (
    REVIEW_DIMENSIONS, audit_asset, lower_mesh_components,
    require_motion_ready, review_template,
)


class AssetReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset_path = self.root / 'asset.pt'
        self.rig_path = self.root / 'rig.pt'
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = torch.tensor([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
        self.asset = dict(position=vertices[faces].mean(1), covariance=torch.eye(3).repeat(4,1,1)*.01,
                          colour=torch.ones(4,3)*.5, opacity=torch.ones(4)*.9, ids=torch.arange(4),
                          face_id=torch.arange(4), barycentric=torch.ones(4,3)/3,
                          mesh_vertices=vertices, mesh_faces=faces)
        self.rig = dict(joints=torch.tensor([[0.,.3,0.],[.5,.3,0.]]), parents=[-1,0],
                        vertex_weights=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]))
        self.save()
        self.evidence = [self.root / 'front.png', self.root / 'oblique.png']
        # Bytes only: the gate hashes evidence; tests do not claim these are images.
        for i, p in enumerate(self.evidence):
            p.write_bytes(bytes([i]))

    def save(self):
        torch.save(self.asset, self.asset_path)
        torch.save(self.rig, self.rig_path)

    def review(self):
        review = review_template(self.asset_path, rig_path=self.rig_path, evidence_paths=self.evidence)
        review['reviewer'] = {'kind': 'agent', 'name': 'unit-test fixture, not a real review'}
        for entry, view in zip(review['evidence'], ('front','oblique')):
            entry.update(view=view, observation='Synthetic assertion for gate testing only.')
        for dim in REVIEW_DIMENSIONS:
            review['dimensions'][dim].update(status='pass', observation='Fixture assertion.',
                                             evidence_ids=['view_0','view_1'])
        return review

    def audit(self, review=None):
        return audit_asset(self.asset_path, semantic_review=review, rig_path=self.rig_path)

    def test_renderable_does_not_approve_anatomy(self):
        report = self.audit()
        self.assertTrue(report['decision']['renderable'])
        self.assertTrue(report['decision']['surface_bound'])
        self.assertFalse(report['decision']['motion_jobs_allowed'])
        with self.assertRaises(RuntimeError):
            require_motion_ready(report)

    def test_complete_review_is_required(self):
        report = self.audit(self.review())
        self.assertTrue(report['decision']['motion_jobs_allowed'])
        self.assertTrue(require_motion_ready(report))

    def test_explicit_failure_or_uncertainty_blocks(self):
        for status in ('fail','uncertain',None):
            review = self.review()
            review['dimensions']['anatomical_parts']['status'] = status
            self.assertFalse(self.audit(review)['decision']['motion_jobs_allowed'])

    def test_stale_geometry_hash_blocks_previous_review(self):
        review = self.review()
        self.asset['colour'][0,0] = .6
        self.save()
        self.assertFalse(self.audit(review)['decision']['motion_jobs_allowed'])

    def test_guard_rehashes_after_report_generation(self):
        report = self.audit(self.review())
        self.rig['joints'][1,0] = .6
        self.save()
        with self.assertRaises(RuntimeError):
            require_motion_ready(report)

    def test_changed_view_blocks_guard(self):
        report = self.audit(self.review())
        self.evidence[0].write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            require_motion_ready(report)

    def test_removed_input_is_a_block_not_uncaught_io_error(self):
        report = self.audit(self.review())
        self.rig_path.unlink()
        with self.assertRaises(RuntimeError):
            require_motion_ready(report)

    def test_changed_implementation_binding_blocks(self):
        report = self.audit(self.review())
        report['implementation']['sha256'] = '0' * 64
        with self.assertRaises(RuntimeError):
            require_motion_ready(report)

    def test_duplicate_ids_not_renderable(self):
        self.asset['ids'][1] = 0
        self.save()
        self.assertFalse(self.audit()['decision']['renderable'])

    def test_nonfinite_position_not_renderable(self):
        self.asset['position'][0,0] = float('nan')
        self.save()
        self.assertFalse(self.audit()['decision']['renderable'])

    def test_nonpositive_covariance_not_renderable(self):
        self.asset['covariance'][0,0,0] = -.1
        self.save()
        self.assertFalse(self.audit()['decision']['renderable'])

    def test_broken_binding_blocks_motion(self):
        self.asset['barycentric'][0] = torch.tensor([.8,.8,-.6])
        self.save()
        report = self.audit(self.review())
        self.assertTrue(report['decision']['renderable'])
        self.assertFalse(report['decision']['surface_bound'])
        self.assertFalse(report['decision']['motion_jobs_allowed'])

    def test_changed_mesh_without_rebinding_blocks(self):
        self.asset['mesh_vertices'][0,0] = .2
        self.save()
        self.assertFalse(self.audit(self.review())['decision']['surface_bound'])

    def test_bad_rig_weights_block(self):
        self.rig['vertex_weights'][0] = torch.tensor([-.1,1.1])
        self.save()
        self.assertFalse(self.audit(self.review())['decision']['motion_jobs_allowed'])

    def test_missing_rig_blocks_even_review_pass(self):
        report = audit_asset(self.asset_path, semantic_review=self.review())
        self.assertFalse(report['decision']['motion_jobs_allowed'])

    def test_false_oblique_or_duplicate_view_cannot_pass(self):
        review = self.review()
        review['evidence'][1] = copy.deepcopy(review['evidence'][0])
        review['evidence'][1].update(id='view_1', view='oblique')
        self.assertFalse(self.audit(review)['decision']['motion_jobs_allowed'])

    def test_components_are_diagnostic_only(self):
        vertices = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                                 [5.,0.,0.],[6.,0.,0.],[5.,1.,0.]])
        result = lower_mesh_components(vertices, torch.tensor([[0,1,2],[3,4,5]]))
        self.assertEqual(result['bands'][0]['component_count'], 2)
        self.assertIsNone(result['semantic_limb_count'])
        self.assertIsNone(result['anatomical_pass'])

    def test_forged_minimal_approval_does_not_pass(self):
        with self.assertRaises(RuntimeError):
            require_motion_ready({'schema_version':1, 'decision':{'motion_jobs_allowed':True}})


if __name__ == '__main__':
    unittest.main()
