"""Coordinator fixtures only: no real anatomy, model or visual approval claims.

Subprocess execution and promotion validators are mocked where indicated so
these tests exercise queue contracts without GPU work or external services.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from real_video.gaussian_program import (
    Program, evidence, input_evidence, manifest_evidence, promotion_decision, run_cycle,
    worker, write_json,
)


class GaussianProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_path = self.root / 'fixture_asset.bin'
        self.input_path.write_bytes(b'unit-test-only source; not a model')
        self.program = self.open_program()

    def open_program(self):
        program = Program(self.root, 'state')
        self.addCleanup(program.db.close)
        return program

    def payload(self, tag='one', extra_inputs=()):
        return dict(inputs=[evidence(self.root, self.input_path), *extra_inputs],
                    issue_key='test.'+tag, issue_title='Unit-test fixture '+tag,
                    next_action='Fixture only: repair source; no real quality attestation')

    def enqueue(self, tag='one', resource='cpu', payload=None):
        payload = payload if payload is not None else self.payload(tag)
        ident = self.program.enqueue('geometry', 'asset_preflight', payload, resource)
        return ident, payload

    def report(self, payload, ready=False, name='report.json', **extra):
        path = self.root / name
        write_json(path, dict(ready=ready, program_input_evidence=payload['inputs'],
                              fixture='Tests only, not a model approval', **extra))
        return path

    def row(self, ident):
        return dict(self.program.db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone())

    def test_sqlite_persistence_and_dedup_across_instances(self):
        ident, payload = self.enqueue()
        reopened = self.open_program()
        self.assertEqual(ident, reopened.enqueue('geometry', 'asset_preflight', payload))
        self.assertEqual(len(reopened.status()['jobs']), 1)
        self.assertEqual(reopened.claim()['id'], ident)
        self.assertIsNone(self.program.claim())

    def test_stale_input_before_claim_is_never_executed(self):
        ident, _ = self.enqueue()
        self.input_path.write_bytes(b'changed fixture')
        self.assertIsNone(self.program.claim())
        self.assertEqual(self.row(ident)['status'], 'stale')
        self.assertEqual(self.row(ident)['attempt'], 0)

    def test_stale_input_during_execution_prevents_finish(self):
        ident, payload = self.enqueue()
        self.program.claim()
        report = self.report(payload, ready=True)
        self.input_path.write_bytes(b'changed during unit-test worker')
        self.program.finish(ident, report)
        self.assertEqual(self.row(ident)['status'], 'stale')
        self.assertIsNone(self.row(ident)['report'])
        self.assertIsNone(self.program.status()['accepted_native_candidate'])

    def test_report_must_bind_exact_job_inputs(self):
        ident, payload = self.enqueue()
        self.program.claim()
        report = self.root / 'wrong_report.json'
        write_json(report, dict(ready=True, program_input_evidence=[]))
        with self.assertRaises(ValueError):
            self.program.finish(ident, report)
        self.assertEqual(self.row(ident)['status'], 'running')
        self.program.finish(ident, error='test harness caught unbound report')
        self.assertEqual(self.row(ident)['status'], 'failed')

    def test_failure_needs_explicit_retry_and_two_attempt_cap(self):
        ident, _ = self.enqueue()
        self.assertEqual(self.program.claim()['attempt'], 1)
        self.program.finish(ident, error='test execution error')
        self.assertIsNone(self.program.claim())
        self.program.retry(ident)
        self.assertEqual(self.program.claim()['attempt'], 2)
        self.program.finish(ident, error='second test execution error')
        with self.assertRaises(ValueError):
            self.program.retry(ident)
        self.assertIsNone(self.program.claim())

    def test_rejected_quality_is_completed_not_retryable(self):
        ident, payload = self.enqueue()
        self.program.claim()
        self.program.finish(ident, self.report(payload, ready=False))
        status = self.program.status()
        self.assertEqual(self.row(ident)['status'], 'completed')
        self.assertEqual(len(status['open_issues']), 1)
        self.assertIsNone(status['accepted_native_candidate'])
        with self.assertRaises(ValueError):
            self.program.retry(ident)
        self.assertEqual(self.program.enqueue('geometry', 'asset_preflight', payload), ident)
        self.assertIsNone(self.program.claim())

    def test_ready_worker_still_does_not_promote_final_video(self):
        ident, payload = self.enqueue()
        self.program.claim()
        self.program.finish(ident, self.report(payload, ready=True))
        self.assertIsNone(self.program.status()['accepted_native_candidate'])

    def test_live_owner_recovery_does_not_steal(self):
        ident, _ = self.enqueue()
        self.program.claim()
        reopened = self.open_program()
        self.assertEqual(reopened.recover(), [])
        self.assertEqual(self.row(ident)['status'], 'running')
        self.assertIsNone(reopened.claim())

    def test_dead_owner_becomes_interrupted_not_queued(self):
        ident, _ = self.enqueue()
        self.program.claim()
        with patch('psutil.Process', side_effect=psutil.NoSuchProcess(999999)), patch('psutil.pid_exists', return_value=False):
            self.assertEqual(self.program.recover(), [ident])
        self.assertEqual(self.row(ident)['status'], 'interrupted')
        self.assertIsNone(self.program.claim())
        self.program.retry(ident)
        self.assertEqual(self.row(ident)['status'], 'queued')

    def test_reused_pid_is_not_treated_as_live_owner(self):
        ident, _ = self.enqueue()
        self.program.claim()
        original = self.row(ident)['owner_created']
        with patch('psutil.Process', return_value=SimpleNamespace(create_time=lambda: original+50)):
            self.assertEqual(self.program.recover(), [ident])

    def test_access_denied_live_pid_is_not_stolen(self):
        ident, _ = self.enqueue()
        self.program.claim()
        with patch('psutil.Process', side_effect=psutil.AccessDenied(os.getpid())), patch('psutil.pid_exists', return_value=True):
            self.assertEqual(self.program.recover(), [])
        self.assertEqual(self.row(ident)['status'], 'running')

    def test_paused_queue_does_not_claim(self):
        ident, _ = self.enqueue()
        self.program.db.execute("UPDATE control SET value='false' WHERE key='enabled'")
        self.assertIsNone(self.program.claim())
        self.assertFalse(self.program.status()['enabled'])
        self.assertEqual(self.row(ident)['attempt'], 0)

    def test_gpu_claims_are_excluded_unless_allowed_and_serialized(self):
        first, _ = self.enqueue('gpu1', resource='gpu')
        second, _ = self.enqueue('gpu2', resource='gpu')
        self.assertIsNone(self.program.claim())
        self.assertEqual(self.program.claim(allow_gpu=True)['id'], first)
        other = self.open_program()
        self.assertIsNone(other.claim(allow_gpu=True))
        self.program.finish(first, error='unit-test completion; no GPU was used')
        self.assertEqual(other.claim(allow_gpu=True)['id'], second)

    def test_status_tracks_seeded_nested_view_evidence(self):
        view = self.root / 'view.bin'
        view.write_bytes(b'fixture, not visual evidence')
        nested = {'dimensions': {'fixture': {'views': [evidence(self.root,view)]}}}
        extra = input_evidence(self.root, nested)
        ident, payload = self.enqueue(payload=self.payload(extra_inputs=extra))
        self.program.claim()
        self.program.finish(ident, self.report(payload, semantic_review=nested))
        before = self.program.status()
        self.assertTrue(before['jobs'][0]['evidence_current'])
        self.assertTrue(before['open_issues'][0]['evidence_current'])
        view.write_bytes(b'changed evidence')
        after = self.program.status()
        self.assertFalse(after['jobs'][0]['evidence_current'])
        self.assertFalse(after['open_issues'][0]['evidence_current'])

    def test_status_rejects_changed_report_file(self):
        ident, payload = self.enqueue()
        self.program.claim()
        report = self.report(payload)
        self.program.finish(ident, report)
        report.write_text('{}', encoding='utf-8')
        self.assertFalse(self.program.status()['jobs'][0]['evidence_current'])

    def manifest_fixture(self):
        frame = self.root/'manifest_frame.bin'
        frame.write_bytes(b'unit-test view, not real visual evidence')
        check = self.root/'manifest_check.json'
        write_json(check, {'fixture':'Not a real model check', 'frame':evidence(self.root,frame)})
        manifest = {'artifacts':{'canonical_state':evidence(self.root,self.input_path)},
                    'checks':{'visual_quality':evidence(self.root,check)}}
        return manifest, check, frame

    def test_manifest_check_change_invalidates_completed_job(self):
        manifest, check, _ = self.manifest_fixture()
        pinned = manifest_evidence(self.root,manifest)
        self.assertIn(evidence(self.root,check), pinned)
        ident, payload = self.enqueue(payload=self.payload(extra_inputs=pinned))
        self.program.claim()
        self.program.finish(ident,self.report(payload,ready=False))
        self.assertTrue(self.program.status()['jobs'][0]['evidence_current'])
        check.write_text('{}',encoding='utf-8')
        status = self.program.status()
        self.assertFalse(status['jobs'][0]['evidence_current'])
        self.assertFalse(status['open_issues'][0]['evidence_current'])

    def test_manifest_nested_frame_change_invalidates_completed_job(self):
        manifest, _, frame = self.manifest_fixture()
        pinned = manifest_evidence(self.root,manifest)
        self.assertIn(evidence(self.root,frame), pinned)
        ident, payload = self.enqueue(payload=self.payload(extra_inputs=pinned))
        self.program.claim()
        self.program.finish(ident,self.report(payload,ready=False))
        frame.write_bytes(b'changed unit-test view')
        status = self.program.status()
        self.assertFalse(status['jobs'][0]['evidence_current'])
        self.assertFalse(status['open_issues'][0]['evidence_current'])

    def test_manifest_invalid_declared_hash_is_never_blessed(self):
        manifest, check, frame = self.manifest_fixture()
        original = manifest['checks']['visual_quality']['sha256']
        check.write_text('{"changed":true}',encoding='utf-8')
        pinned = manifest_evidence(self.root,manifest)
        self.assertEqual(manifest['checks']['visual_quality']['sha256'],original)
        self.assertNotIn(evidence(self.root,check),pinned)
        # Do not traverse a check whose own declared digest no longer matches.
        self.assertNotIn(evidence(self.root,frame),pinned)
        from real_video.generation_readiness import evaluate_readiness
        decision = evaluate_readiness(manifest,root=self.root)
        self.assertFalse(decision['ready'])

    def test_experiment_ledger_validates_evidence_and_deduplicates(self):
        output = self.root/'experiment_output.bin'
        output.write_bytes(b'unit-test failed output, not a video')
        record = dict(role='geometry', outcome='rejected', issue_key='test.geometry',
                      hypothesis='Unit-test hypothesis only', observation='Fixture rejected',
                      next_action='Fixture only: repair geometry before retry',
                      inputs=[evidence(self.root,self.input_path)], outputs=[evidence(self.root,output)])
        fingerprint = self.program.record_experiment(record)
        reopened = self.open_program()
        self.assertEqual(reopened.record_experiment(record),fingerprint)
        self.assertEqual(len(reopened.status()['experiments']),1)
        self.assertIsNone(reopened.status()['accepted_native_candidate'])
        invalid = dict(record,outcome='accepted')
        with self.assertRaises(ValueError):
            self.program.record_experiment(invalid)
        output.write_bytes(b'changed fixture output')
        with self.assertRaises(ValueError):
            self.program.record_experiment(record)

    def test_nested_evidence_rejects_missing_and_outside_files(self):
        with self.assertRaises(ValueError):
            input_evidence(self.root, {'path':'missing', 'sha256':'0'*64})
        with self.assertRaises(ValueError):
            input_evidence(self.root, {'path':'../outside', 'sha256':'0'*64})

    def test_empty_unchanged_cycle_does_no_work(self):
        with patch('real_video.gaussian_program.subprocess.run') as launch:
            result = run_cycle(self.program, max_jobs=4, budget_seconds=10)
            result2 = run_cycle(self.program, max_jobs=4, budget_seconds=10)
        launch.assert_not_called()
        self.assertEqual(result['jobs'], result2['jobs'])

    def test_unchanged_rejected_cycle_does_not_restart_worker(self):
        ident, payload = self.enqueue()
        self.program.claim()
        self.program.finish(ident, self.report(payload))
        self.program.enqueue('geometry','asset_preflight',payload)
        with patch('real_video.gaussian_program.subprocess.run') as launch:
            run_cycle(self.program, max_jobs=4, budget_seconds=10)
        launch.assert_not_called()
        self.assertEqual(self.row(ident)['attempt'], 1)

    def test_cycle_success_means_evaluated_not_promoted(self):
        ident, _ = self.enqueue()
        def fake_run(command, **kwargs):
            incoming = Path(command[command.index('--input')+1])
            outgoing = Path(command[command.index('--output')+1])
            payload = json.loads(incoming.read_text())
            write_json(outgoing, {'ready':False, 'program_input_evidence':payload['inputs']})
        with patch('real_video.gaussian_program.subprocess.run', side_effect=fake_run) as launch:
            result = run_cycle(self.program, max_jobs=4, budget_seconds=10)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(self.row(ident)['status'], 'completed')
        self.assertEqual(len(result['open_issues']), 1)
        self.assertIsNone(result['accepted_native_candidate'])

    def test_cycle_directory_failure_records_failed_job(self):
        ident, _ = self.enqueue()
        collision = self.program.state/'jobs'/ident/'attempt_1'
        collision.mkdir(parents=True)
        with patch('real_video.gaussian_program.subprocess.run') as launch:
            run_cycle(self.program, max_jobs=1, budget_seconds=10)
        launch.assert_not_called()
        self.assertEqual(self.row(ident)['status'], 'failed')

    def test_asset_worker_passes_rig_review_and_reads_actual_decision(self):
        for name in ('asset','mesh','rig','review'):
            (self.root/name).write_bytes(b'unit-test-only')
        payload = dict(self.payload(), asset='asset', mesh='mesh', rig='rig', review='review')
        with patch('real_video.asset_readiness.audit_asset', return_value={'decision':{'motion_jobs_allowed':False}}) as audit:
            worker('asset_preflight', payload, 'worker_report.json', root=self.root)
        self.assertEqual(audit.call_args.kwargs['rig_path'], self.root/'rig')
        self.assertEqual(audit.call_args.kwargs['semantic_review'], self.root/'review')
        report = json.loads((self.root/'worker_report.json').read_text())
        self.assertIs(report['ready'], False)
        self.assertEqual(report['program_input_evidence'], payload['inputs'])

    def promotion_fixture(self):
        objects = {}
        reports = {}
        for name in ('cat','dog'):
            asset = self.root/f'{name}.bin'
            asset.write_bytes(name.encode())
            objects[name] = {'asset':evidence(self.root,asset), 'mesh':evidence(self.root,asset),
                             'rig':evidence(self.root,asset)}
            path = self.root/f'{name}_readiness.json'
            write_json(path, {'inputs':objects[name], 'fixture':'Mocked prerequisite, not approval'})
            reports[name] = evidence(self.root,path)
        (self.root/'video.bin').write_bytes(b'fixture, not a real video')
        write_json(self.root/'visual_review.json', {'fixture':'Mocked review, not approval'})
        manifest = {'artifacts':{
            'video':evidence(self.root,'video.bin'),
            'visual_review':evidence(self.root,'visual_review.json'),
            'canonical_state':objects['cat']['asset'],
            'canonical_state_dog':objects['dog']['asset']}}
        write_json(self.root/'manifest.json', manifest)
        return objects, reports, evidence(self.root,'manifest.json')

    def promoted(self, objects, reports, manifest):
        # Isolate coordinator binding from semantic model/visual validators.
        with patch('real_video.asset_readiness.require_motion_ready', return_value=True), \
             patch('real_video.generation_readiness.evaluate_readiness', return_value={'ready':True}), \
             patch('real_video.articulation_quality.decide_review', return_value={'accepted':True}):
            return promotion_decision(self.root, reports, manifest, 'visual_review.json', 'video.bin', objects)

    def test_promotion_fixture_is_accepted_only_under_mocked_validators(self):
        self.assertTrue(self.promoted(*self.promotion_fixture())['accepted'])

    def test_promotion_rejects_missing_object_report(self):
        objects, reports, manifest = self.promotion_fixture()
        del reports['dog']
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])

    def test_promotion_rejects_wrong_object_or_duplicate_cat_report(self):
        objects, reports, manifest = self.promotion_fixture()
        reports['dog'] = reports['cat']
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])

    def test_promotion_rejects_unbound_canonical_animal(self):
        objects, reports, manifest = self.promotion_fixture()
        doc = json.loads((self.root/'manifest.json').read_text())
        del doc['artifacts']['canonical_state_dog']
        write_json(self.root/'manifest.json', doc)
        manifest = evidence(self.root,'manifest.json')
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])

    def test_promotion_rejects_stale_report(self):
        objects, reports, manifest = self.promotion_fixture()
        (self.root/'dog_readiness.json').write_text('{}', encoding='utf-8')
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])

    def test_promotion_rejects_malformed_manifest_without_crashing(self):
        objects, reports, _ = self.promotion_fixture()
        (self.root/'manifest.json').write_text('not JSON', encoding='utf-8')
        manifest = evidence(self.root,'manifest.json')
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])

    def test_promotion_rejects_missing_video_without_crashing(self):
        objects, reports, manifest = self.promotion_fixture()
        (self.root/'video.bin').unlink()
        self.assertFalse(self.promoted(objects, reports, manifest)['accepted'])


if __name__ == '__main__':
    unittest.main()
