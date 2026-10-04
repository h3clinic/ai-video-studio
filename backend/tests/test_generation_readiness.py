"""Contract mechanics; these fixture attestations are NOT scientific evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from real_video.articulation_quality import create_review
from real_video.generation_readiness import (
    NATIVE_FLAGS, NATIVE_STAGE, REQUIRED_CHECKS, classify_mechanism,
    current_manifest, evaluate_readiness,
)


class GenerationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, value):
        data = json.dumps(value).encode() if isinstance(value, dict) else value
        path = self.root/name
        path.write_bytes(data)
        return {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}

    def fixture(self):
        manifest = dict(schema_version=1, requested_stage=NATIVE_STAGE,
            mechanism=dict(NATIVE_FLAGS, motion_source='learned_gaussian', output_domain='gaussian_state'),
            artifacts={}, checks={})
        artifacts = manifest['artifacts']
        for role in set(role for roles in REQUIRED_CHECKS.values() for role in roles):
            artifacts[role] = self.put(role+'.bin', ('fixture only: '+role).encode())
        frames = []
        for i in range(3):
            descriptor = self.put(f'frame{i}.bin', str(i).encode())
            frames.append({'path': self.root/descriptor['path'], 'frame_index': i})
        review = create_review(self.root/artifacts['video']['path'], frames)
        review['claim_type'] = 'conditional_motion_generation'
        review['reviewer'] = {'kind': 'agent', 'name': 'unit-test fixture', 'observed_video': True}
        review['provenance'] = dict.fromkeys(review['provenance'], False)
        for frame in review['frames']:
            frame['observation'] = 'Fixture, not an observed real video.'
        for item in review['dimensions'].values():
            item.update(status='pass', observation='Test-only statement.',
                        evidence_frame_ids=[frame['id'] for frame in review['frames']])
        artifacts['visual_review'] = self.put('review.json', review)
        for check, roles in REQUIRED_CHECKS.items():
            manifest['checks'][check] = self.put(check+'.json', dict(schema_version=1,
                check_id=check, status='pass', evaluator={'kind': 'instrumented_test', 'name': 'fixture'},
                observation='Test of contract mechanics only.',
                subjects={role: artifacts[role]['sha256'] for role in roles}))
        return manifest

    def decide(self, manifest):
        return evaluate_readiness(manifest, self.root)

    def test_empty_manifest_fails_closed(self):
        for value in (None, {}, [], 'claim'):
            result = self.decide(value)
            self.assertFalse(result['ready'])
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(set(result['checks']), set(REQUIRED_CHECKS))

    def test_complete_fixture_exercises_mechanics_not_real_quality(self):
        manifest = self.fixture()
        before = deepcopy(manifest)
        result = self.decide(manifest)
        self.assertTrue(result['ready'], result)
        self.assertEqual(result['classification'], NATIVE_STAGE)
        self.assertEqual(manifest, before)

    def test_skeletal_wan_replay_scripted_are_not_native(self):
        for source in ('pretrained_skeletal', 'wan_rgb', 'recorded', 'scripted', 'none'):
            manifest = self.fixture()
            manifest['mechanism']['motion_source'] = source
            result = self.decide(manifest)
            self.assertFalse(result['ready'])
            self.assertIn('mechanism_mismatch', [x['code'] for x in result['blockers']])

    def test_native_claim_requires_every_explicit_boolean(self):
        for flag, expected in NATIVE_FLAGS.items():
            manifest = self.fixture()
            manifest['mechanism'][flag] = not expected
            self.assertFalse(self.decide(manifest)['ready'])
            manifest['mechanism'][flag] = int(expected)
            self.assertFalse(self.decide(manifest)['ready'])

    def test_untrained_writer_is_not_closed_loop(self):
        manifest = self.fixture()
        manifest['mechanism'].update(motion_source='wan_rgb', trained_state_writer=False)
        self.assertEqual(self.decide(manifest)['classification'], 'wan_rgb_gaussian_conditioning')
        self.assertFalse(self.decide(manifest)['ready'])

    def test_stale_source_and_checkpoint_fail(self):
        for role in ('generator_code', 'generator_checkpoint', 'canonical_state', 'video'):
            manifest = self.fixture()
            (self.root/manifest['artifacts'][role]['path']).write_bytes(b'changed')
            self.assertFalse(self.decide(manifest)['ready'])

    def test_stale_or_missing_check_fails(self):
        for check in REQUIRED_CHECKS:
            manifest = self.fixture()
            (self.root/manifest['checks'][check]['path']).write_bytes(b'{"status":"pass"}')
            self.assertFalse(self.decide(manifest)['ready'])
            manifest['checks'].pop(check)
            self.assertFalse(self.decide(manifest)['ready'])

    def test_claim_of_pass_without_subject_hashes_fails(self):
        manifest = self.fixture()
        for check in REQUIRED_CHECKS:
            path = self.root/manifest['checks'][check]['path']
            evidence = json.loads(path.read_text())
            evidence['subjects'] = {}
            manifest['checks'][check] = self.put(path.name, evidence)
        self.assertFalse(self.decide(manifest)['ready'])

    def test_additional_character_must_be_in_identity_evidence(self):
        manifest = self.fixture()
        manifest['artifacts']['canonical_state_dog'] = self.put('dog.bin', b'dog')
        result = self.decide(manifest)
        self.assertFalse(result['ready'])
        self.assertFalse(result['checks']['persistent_identity']['passed'])

    def test_visual_failure_cannot_be_overridden_by_pass_check(self):
        manifest = self.fixture()
        review = json.loads((self.root/manifest['artifacts']['visual_review']['path']).read_text())
        review['dimensions']['body_shape']['status'] = 'fail'
        manifest['artifacts']['visual_review'] = self.put('review.json', review)
        path = self.root/manifest['checks']['visual_quality']['path']
        check = json.loads(path.read_text())
        check['subjects']['visual_review'] = manifest['artifacts']['visual_review']['sha256']
        manifest['checks']['visual_quality'] = self.put(path.name, check)
        self.assertFalse(self.decide(manifest)['ready'])

    def test_path_escape_and_malformed_role_fail(self):
        manifest = self.fixture()
        manifest['artifacts']['video']['path'] = '../not-inside.bin'
        manifest['artifacts'][2] = manifest['artifacts']['generator_code']
        self.assertFalse(self.decide(manifest)['ready'])

    def test_empty_current_project_never_invents_readiness(self):
        manifest = current_manifest(self.root)
        self.assertEqual(manifest['artifacts'], {})
        self.assertEqual(manifest['checks'], {})
        self.assertFalse(self.decide(manifest)['ready'])
        self.assertEqual(classify_mechanism(manifest['mechanism']), 'unverified')


if __name__ == '__main__':
    unittest.main()
