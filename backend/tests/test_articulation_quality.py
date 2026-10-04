"""Visual gate mechanics only: fixtures are not real quality assessments."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from real_video.articulation_quality import DIMENSIONS, create_review, decide_review


class ArticulationQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'candidate.mp4'
        self.video.write_bytes(b'unit test video bytes, not actual media')
        self.frames = []
        for index in (0, 4, 8):
            path = self.root / f'frame_{index:02d}.png'
            path.write_bytes(f'unit test frame {index}'.encode())
            self.frames.append({'path': path, 'frame_index': index})

    def complete_review(self):
        review = create_review(self.video, self.frames)
        review['claim_type'] = 'conditional_motion_generation'
        review['reviewer'] = {'kind': 'agent', 'name': 'unit-test fixture', 'observed_video': True}
        review['provenance'] = dict.fromkeys(review['provenance'], False)
        for frame in review['frames']:
            frame['observation'] = 'Test-only observation placeholder.'
        for dimension in DIMENSIONS:
            review['dimensions'][dimension] = {
                'status': 'pass', 'observation': 'Test-only judgment placeholder.',
                'evidence_frame_ids': [frame['id'] for frame in review['frames']],
            }
        return review

    def decide(self, review):
        return decide_review(review, self.video)

    def test_template_is_pending_and_supplies_no_judgment(self):
        review = create_review(self.video, [f['path'] for f in self.frames])
        self.assertIsNone(review['frames'][0]['frame_index'])
        self.assertIsNone(review['dimensions']['paw_placement']['status'])
        self.assertEqual(self.decide(review)['status'], 'pending')
        self.assertFalse(self.decide(review)['accepted'])

    def test_missing_review_is_pending(self):
        for review in (None, [], '', 123):
            self.assertEqual(self.decide(review)['status'], 'pending')

    def test_complete_evidence_can_pass_mechanics_without_mutation(self):
        review = self.complete_review()
        before = deepcopy(review)
        decision = self.decide(review)
        self.assertTrue(decision['accepted'])
        self.assertEqual(decision['verified_frame_count'], 3)
        self.assertEqual(review, before)

    def test_any_failed_or_uncertain_dimension_blocks_acceptance(self):
        for dimension in DIMENSIONS:
            for status in ('fail', 'uncertain'):
                with self.subTest(dimension=dimension, status=status):
                    review = self.complete_review()
                    review['dimensions'][dimension]['status'] = status
                    review['numeric_metrics'] = {'score': 1000000, 'accepted': True}
                    review['accepted'] = True
                    decision = self.decide(review)
                    self.assertFalse(decision['accepted'])
                    self.assertFalse(decision['visual_pass'])
                    self.assertEqual(decision['status'], 'rejected')

    def test_missing_dimension_or_observation_is_not_approval(self):
        for key in ('status', 'observation', 'evidence_frame_ids'):
            review = self.complete_review()
            del review['dimensions']['paw_placement'][key]
            self.assertFalse(self.decide(review)['accepted'])
        review = self.complete_review()
        del review['dimensions']['body_shape']
        self.assertFalse(self.decide(review)['accepted'])

    def test_changed_video_rejects_stale_review(self):
        review = self.complete_review()
        self.video.write_bytes(b'a different video')
        decision = self.decide(review)
        self.assertEqual(decision['status'], 'rejected')
        self.assertFalse(decision['accepted'])

    def test_different_candidate_path_rejects_same_bytes(self):
        review = self.complete_review()
        alternate = self.root / 'other.mp4'
        alternate.write_bytes(self.video.read_bytes())
        self.assertFalse(decide_review(review, alternate)['accepted'])

    def test_modified_or_missing_frame_rejects(self):
        review = self.complete_review()
        self.frames[0]['path'].write_bytes(b'changed')
        self.assertEqual(self.decide(review)['status'], 'rejected')
        self.frames[0]['path'].unlink()
        self.assertFalse(self.decide(review)['accepted'])

    def test_requires_three_temporally_distinct_files(self):
        for field in ('id', 'path', 'frame_index'):
            review = self.complete_review()
            review['frames'][1][field] = review['frames'][0][field]
            self.assertFalse(self.decide(review)['accepted'])
        review = self.complete_review()
        review['frames'].pop()
        self.assertFalse(self.decide(review)['accepted'])

    def test_requires_three_observed_and_cited_frames(self):
        review = self.complete_review()
        review['frames'][0]['observation'] = '  '
        self.assertFalse(self.decide(review)['accepted'])
        review = self.complete_review()
        for entry in review['dimensions'].values():
            entry['evidence_frame_ids'] = ['frame_000']
        self.assertFalse(self.decide(review)['accepted'])

    def test_source_video_hash_must_match(self):
        review = self.complete_review()
        review['frames'][0]['source_video_sha256'] = 'wrong'
        self.assertFalse(self.decide(review)['accepted'])

    def test_future_conditioning_and_test_leakage_block_generation(self):
        for flag in ('future_conditioned', 'test_used_for_training', 'test_used_for_selection'):
            review = self.complete_review()
            review['provenance'][flag] = True
            decision = self.decide(review)
            self.assertTrue(decision['visual_pass'])
            self.assertFalse(decision['claim_supported'])
            self.assertFalse(decision['accepted'])

    def test_future_conditioning_can_be_explicitly_labelled_reconstruction(self):
        review = self.complete_review()
        review['claim_type'] = 'reconstruction'
        review['provenance']['future_conditioned'] = True
        self.assertTrue(self.decide(review)['accepted'])

    def test_missing_or_nonboolean_provenance_fails_closed(self):
        for value in (None, 'false', 0, 1):
            review = self.complete_review()
            review['provenance']['future_conditioned'] = value
            self.assertFalse(self.decide(review)['accepted'])

    def test_explicit_reviewer_attestation_is_required(self):
        for key, value in (('kind', 'automatic_score'), ('name', ''), ('observed_video', False)):
            review = self.complete_review()
            review['reviewer'][key] = value
            self.assertFalse(self.decide(review)['accepted'])

    def test_malformed_sections_fail_closed_without_crashing(self):
        for key in ('video', 'reviewer', 'frames', 'dimensions', 'provenance'):
            review = self.complete_review()
            review[key] = None
            self.assertFalse(self.decide(review)['accepted'])


if __name__ == '__main__':
    unittest.main()
