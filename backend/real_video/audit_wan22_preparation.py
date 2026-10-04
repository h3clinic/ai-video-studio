"""Record tested preparation separately from model execution and visual success."""
import argparse
import importlib
import io
import json
from pathlib import Path
import unittest

from .checkpoint_io import digest
from .gaussian_program import Program, ROOT, evidence, write_json
from .prepare_wan22 import MODEL, REPO, REVISION


def run(out):
    if out.exists():
        raise FileExistsError('Immutable audit output required')
    out.mkdir(parents=True)
    modules = ['tests.test_wan22_gaussian_anchor','tests.test_wan22_runner',
               'tests.test_latent_track_memory','tests.test_gaussian_latent_memory',
               'tests.test_resumable_model_download',
               'tests.test_wan22_download_supervisor', 'tests.test_wan22_download_transport',
               'tests.test_guarded_worker']
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    extra = []
    identities = importlib.import_module('tests.test_identity_conditioning')
    for name in sorted(n for n in vars(identities) if n.startswith('test_')):
        getattr(identities,name)()
        extra.append(name)
    files = ['real_video/prepare_wan22.py','real_video/sample_wan22_memory.py',
        'real_video/wan22_gaussian_anchor.py','real_video/identity_conditioning.py',
        'real_video/resumable_model_download.py','research/sweep_2026-10-03_stronger_backbone.json',
        'research/sweep_2026-10-03_native_anchor_execution.json', 'real_video/guarded_worker.py']
    download = json.loads((MODEL/'download_manifest.json').read_text())
    attempts = [dict(path=str(p), result=json.loads(p.read_text()),
        worker_log=(p.parent/'worker.log').read_text(encoding='utf-8', errors='replace')
            if (p.parent/'worker.log').exists() else None)
        for p in sorted((MODEL/'attempts').glob('*/result.json'))]
    report = dict(model=REPO,revision=REVISION,scope='CPU-tested stronger-backbone integration; not a generated video result',
        tests=dict(unittest_count=result.testsRun,extra_function_tests=len(extra),
            all_passed=result.wasSuccessful(),log=log.getvalue(),identity_tests=extra),
        code=[evidence(ROOT,p) for p in files],download_snapshot=download,download_attempts=attempts,
        verified_download_bytes=sum(row['size'] for row in download['files']),
        retained_range_partials=[dict(path=str(p.relative_to(MODEL)),size=p.stat().st_size)
            for p in MODEL.rglob('*.partial')],
        earlier_download_failure=dict(received_bytes_before_disconnect=637173760,
            expected_first_shard_bytes=4978254344,error='Peer closed response, then DNS getaddrinfo failed (11001)',
            observed_from='Completed unified execution session 89674; no completed model shard retained'),
        model_inference_run=False,new_video_created=False,quality_accepted=False,
        native_gaussian_generation_achieved=False,compute_savings_established=False,
        next_action='Inspect the latest supervised Xet attempt and preserved range partials; never duplicate an active worker. '
            'Continue only a supported changed transport hypothesis or measured transfer progress. '
            'Then one idle-GPU, RAM/battery-guarded native I2V comparison capped at600 seconds; '
            'review exact MP4 future frames before any quality claim. No old adapter or future input.')
    write_json(out/'report.json',report)
    program=Program()
    try:
        outputs=[evidence(ROOT,out/'report.json')]
        program.record_experiment(dict(role='generation',issue_key='generation.latent_appearance',
            hypothesis='A stronger pretrained native I2V backbone can preserve source appearance through a real serialized Gaussian anchor better than the rejected small T2V reader',
            inputs=report['code'],outputs=outputs,outcome='partial' if result.wasSuccessful() else 'failed_execution',
            observation='Native48-channel bank/conditioning, content controls and complete-file verification are CPU-tested. '
                'Independent download supervision and official Xet staging now preserve earlier range partials. '
                'Download status and every attempt outcome are snapshotted in the report; model quality remains untested.',
            next_action=report['next_action']))
        program.issue('generation.latent_appearance','generation','Stronger native-I2V Gaussian-anchor comparison awaiting verified model execution',
            outputs+report['code'],report['next_action'])
        program.export()
    finally:
        program.db.close()
    print(json.dumps(dict(tests=result.testsRun+len(extra),passed=result.wasSuccessful(),
                         download_status=download['status'],model_inference_run=False)))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
