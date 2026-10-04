"""Record a completed bounded trial without claiming model/quality success.

This CLI performs no provider calls and requires the Studio backend closed so
it cannot overwrite concurrent user jobs. All media/checkpoints are preserved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
from real_video.gaussian_program import Program, evidence
from real_video.studio_backend import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def completed_report(report):
    return (isinstance(report,dict)
        and report.get('status') in ('blocked_no_safe_insertion','completed_partial_review_required')
        and type(report.get('output_frames')) is int and report['output_frames']>0
        and isinstance(report.get('output_sha256'),str))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_id');args=parser.parse_args()
    if not re.fullmatch(r'apple-observed-parts-20261004-v[1-9][0-9]*',args.run_id):
        raise ValueError('Expected an explicit owned repair run')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8790))==0:
            raise RuntimeError('Close Studio before importing a completed trial')
    directory=ROOT/'artifacts/cloud'/args.run_id
    control=json.loads((directory/'experiment/control.json').read_text())
    final=json.loads((directory/'final_status.json').read_text())
    if final.get('desiredStatus')!='EXITED':raise ValueError('Verify GPU stopped before recording')
    output=directory/'experiment/results/output'
    report_path=output/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else None
    remote_job=control.get('remote_job') or control.get('remote') or {}
    experiment_attempted='experiment' in remote_job.get('stages',{})
    review_path=directory/'visual_review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else None
    now=datetime.now(timezone.utc).isoformat()
    worker_completed=completed_report(report)
    if not worker_completed:
        observation=('Remote experiment started but failed before its completion report: '+str(remote_job.get('error','No worker report'))
            if experiment_attempted else 'Remote setup failed before model execution: '+control.get('error','No worker report'))
        if review:observation+=' '+str(review.get('summary',''))
        outcome='failed_execution';status='failed';stage=observation
        next_action=('Repair the exact worker exception and add a regression test; do not repeat unchanged input.'
            if experiment_attempted else 'Resolve the logged transport failure before another unchanged model attempt.')
    else:
        observation=(f"Remote worker processed {report.get('output_frames')} observed frames; "
            f"safe replacement in {report.get('replaced_frames')} frames. "
            +str(report.get('unresolved',''))+' '+str((review or {}).get('summary','No visual review; cannot pass.')))
        outcome='partial';status='blocked';stage='Partial repair evaluated; full apple replacement not accepted'
        next_action='Use preserved labeled masks to resolve semantic ambiguity, fit distinct apple-slice/peel/bite assets, then validate temporal contact. No unchanged rejected rerun.'
    # The immutable bundle contains the exact executed code, unlike working
    # source files which can already contain a fix when a failed run is logged.
    inputs=[directory/'experiment/inputs.zip','research/sweep_2026-10-04_observed_part_repair.json']
    outputs=[directory/'experiment/control.json',directory/'final_status.json',directory/'experiment/inputs.zip']
    if report_path.exists():outputs.append(report_path)
    if review_path.exists():outputs.append(review_path)
    for name in ('job.json','experiment.log'):
        candidate=directory/'experiment/results'/name
        if candidate.exists():outputs.append(candidate)
    program=Program(ROOT)
    try:
        fingerprint=program.record_experiment(dict(role='rendering',issue_key='rendering.observed_part_replacement',
            hypothesis='Named instance masks plus observed depth placement and rendered coverage gates avoid static floating replacements and accidental piece erasure.',
            inputs=[evidence(ROOT,p) for p in inputs],outputs=[evidence(ROOT,p) for p in outputs],
            outcome=outcome,observation=observation,next_action=next_action))
        program.export()
    finally:program.db.close()
    job_path=ROOT/'artifacts/studio/jobs.json'
    jobs=json.loads(job_path.read_text())
    job_id=hashlib.sha256(args.run_id.encode()).hexdigest()[:32]
    if not any(job['id']==job_id for job in jobs):
        # Keep old failures as history; access restoration is a separate event.
        jobs.append(dict(id=job_id,kind='apple_experiment',projectId='apple-experiment',
            prompt='Replace orange-related parts using observed Gaussian part evidence; preserve unsupported source parts.',
            status=status,stage=stage,createdAt=now,finishedAt=now,error=control.get('error'),output=None,
            reply=observation,review='Not accepted: incomplete scene replacement.',
            evidence=str((report_path if report else directory/'experiment/control.json').relative_to(ROOT)),
            remotePodId=final['id'],workerRevision=control.get('worker_revision'),
            execution=dict(remote=True,workerStarted=experiment_attempted,workerCompleted=worker_completed,
                modelExecuted=True if worker_completed else None if experiment_attempted else False,
                newVideoGeneration=False,weightsChanged=False)))
    else:
        # Correct a live report previously mistaken for completion, retaining
        # its old status in audit history instead of silently erasing it.
        job=next(job for job in jobs if job['id']==job_id)
        if job.get('status')!=status or job.get('stage')!=stage:
            job.setdefault('statusCorrections',[]).append(dict(at=now,status=job.get('status'),stage=job.get('stage')))
            job.update(status=status,stage=stage,reply=observation)
        job['execution']=dict(remote=True,workerStarted=experiment_attempted,workerCompleted=worker_completed,
            modelExecuted=True if worker_completed else None,newVideoGeneration=False,weightsChanged=False)
    atomic_json(job_path,jobs)
    print(json.dumps({'recorded':args.run_id,'outcome':outcome,'fingerprint':fingerprint,'pod_stopped':True}))


if __name__=='__main__':main()
