"""Durable coordinator for Gaussian-video development, not a video model.

Typed local workers produce evidence; specialist coding/research agents consume
the persisted repair backlog. No shell/LLM command is executed from a report.
Jobs are bounded, source-hash keyed, restartable and serialized for the GPU.
Unchanged rejected candidates are never retried automatically or promoted.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = Path('artifacts/real_video/program')
CONFIG = Path('real_video/program_config.json')
ROLES = ('rendering', 'geometry', 'generation', 'evaluation')
KINDS = ('asset_preflight', 'generation_preflight', 'visual_review')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def local_path(root, value):
    root = Path(root).resolve()
    result = (root/Path(value)).resolve()
    if not result.is_relative_to(root): raise ValueError('Path is outside this project')
    return result


def evidence(root, path):
    path = local_path(root, path)
    return dict(path=path.relative_to(Path(root).resolve()).as_posix(), sha256=sha256(path))


def verify_evidence(root, entry):
    try:
        return isinstance(entry, dict) and sha256(local_path(root, entry['path'])) == entry['sha256']
    except (KeyError, ValueError, OSError, TypeError): return False


def input_evidence(root, value, strict=True):
    """Collect nested file bindings (reviews include independently hashed views)."""
    found = []
    def visit(item):
        if isinstance(item, dict):
            if 'path' in item and 'sha256' in item:
                if not verify_evidence(root, item):
                    if strict: raise ValueError('Missing or changed nested evidence')
                    return
                found.append(evidence(root, item['path']))
            else:
                for child in item.values(): visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(value)
    return list({canonical(entry): entry for entry in found}.values())


def manifest_evidence(root, manifest):
    """Pin check documents AND their visual dependencies, not just model files.

    Invalid bindings are left in the manifest for the worker to reject. Never
    bless their current bytes by replacing a declared hash with a new one.
    """
    entries=input_evidence(root,manifest,strict=False)
    for entry in list(entries):
        path=local_path(root,entry['path'])
        if path.suffix.lower()=='.json':
            try: entries += input_evidence(root,json.loads(path.read_text(encoding='utf-8')),strict=False)
            except (OSError,ValueError,TypeError): pass
    return list({canonical(entry):entry for entry in entries}.values())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with pending.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


class Program:
    def __init__(self, root=ROOT, state=STATE):
        self.root = Path(root).resolve()
        self.state = local_path(self.root, state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.state/'program.sqlite3', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, role TEXT NOT NULL,
          kind TEXT NOT NULL, payload TEXT NOT NULL, resource TEXT NOT NULL,
          status TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
          owner_pid INTEGER, owner_created REAL, started REAL, report TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
          job_id TEXT, event TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS issues (
          key TEXT PRIMARY KEY, role TEXT NOT NULL, title TEXT NOT NULL,
          status TEXT NOT NULL, evidence TEXT NOT NULL, next_action TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS experiments (
          fingerprint TEXT PRIMARY KEY, record TEXT NOT NULL);
        INSERT OR IGNORE INTO control VALUES ('enabled','true');
        ''')

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); raise

    def event(self, job, event, data):
        self.db.execute('INSERT INTO events(timestamp,job_id,event,data) VALUES(?,?,?,?)',
                        (time.time(), job, event, canonical(data)))

    def enqueue(self, role, kind, payload, resource='cpu'):
        if role not in ROLES or kind not in KINDS or resource not in ('cpu', 'gpu'):
            raise ValueError('Unknown typed worker or role/resource')
        if any(not isinstance(payload.get(key),str) or not payload[key].strip()
               for key in ('issue_key','issue_title','next_action')):
            raise ValueError('Every job needs an owned repair issue and next action')
        inputs = payload.get('inputs', [])
        if not inputs or not all(verify_evidence(self.root, item) for item in inputs):
            raise ValueError('Job requires verified input/code fingerprints')
        fingerprint = hashlib.sha256(canonical([role, kind, payload, resource]).encode()).hexdigest()
        with self.transaction():
            previous = self.db.execute('SELECT id FROM jobs WHERE fingerprint=?',(fingerprint,)).fetchone()
            if previous: return previous['id']
            ident = uuid.uuid4().hex[:16]
            self.db.execute('INSERT INTO jobs(id,fingerprint,role,kind,payload,resource,status) VALUES(?,?,?,?,?,?,?)',
                (ident,fingerprint,role,kind,canonical(payload),resource,'queued'))
            self.event(ident,'queued',dict(role=role,kind=kind))
        return ident

    def claim(self, allow_gpu=False):
        import psutil
        with self.transaction():
            if self.db.execute("SELECT value FROM control WHERE key='enabled'").fetchone()[0] != 'true': return None
            jobs = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid").fetchall()
            for job in jobs:
                if job['resource']=='gpu':
                    if not allow_gpu: continue
                    if self.db.execute("SELECT 1 FROM jobs WHERE resource='gpu' AND status='running'").fetchone(): continue
                payload = json.loads(job['payload'])
                if not all(verify_evidence(self.root, x) for x in payload['inputs']):
                    self.db.execute("UPDATE jobs SET status='stale' WHERE id=?",(job['id'],))
                    self.event(job['id'],'stale',{'reason':'Input changed before execution'})
                    continue
                process = psutil.Process()
                self.db.execute("UPDATE jobs SET status='running',attempt=attempt+1,owner_pid=?,owner_created=?,started=? WHERE id=?",
                    (os.getpid(),process.create_time(),time.time(),job['id']))
                self.event(job['id'],'claimed',{'owner_pid':os.getpid()})
                return dict(self.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone())
        return None

    def recover(self):
        """Never steal a live worker's slot. Dead jobs need a deliberate retry."""
        import psutil
        recovered=[]
        with self.transaction():
            for row in self.db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
                try:
                    live=abs(psutil.Process(row['owner_pid']).create_time()-row['owner_created'])<.01
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    live=psutil.pid_exists(row['owner_pid'])
                if not live:
                    self.db.execute("UPDATE jobs SET status='interrupted',error='Owner process exited; explicit retry required' WHERE id=?",(row['id'],))
                    self.event(row['id'],'interrupted',{'automatic_rerun':False})
                    recovered.append(row['id'])
        return recovered

    def retry(self, ident):
        with self.transaction():
            row=self.db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
            if row is None or row['status'] not in ('failed','interrupted') or row['attempt']>=2:
                raise ValueError('Only failed/interrupted execution gets at most two attempts; rejected quality is not retryable')
            self.db.execute("UPDATE jobs SET status='queued' WHERE id=?",(ident,))
            self.event(ident,'explicit_retry',{})

    def issue(self, key, role, title, artifacts, next_action):
        if role not in ROLES: raise ValueError('Invalid issue owner')
        if not all(verify_evidence(self.root,x) for x in artifacts): raise ValueError('Stale issue evidence')
        old=self.db.execute('SELECT * FROM issues WHERE key=?',(key,)).fetchone()
        values=(key,role,title,'open',canonical(artifacts),next_action)
        if old and tuple(old)==values: return
        self.db.execute('INSERT INTO issues VALUES(?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET role=excluded.role,title=excluded.title,status=excluded.status,evidence=excluded.evidence,next_action=excluded.next_action',values)
        self.event(None,'issue_updated',{'key':key,'role':role})

    def finish(self, ident, report_path=None, error=None):
        with self.transaction():
            row=self.db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
            if row is None or row['status']!='running' or row['owner_pid']!=os.getpid():
                raise ValueError('Only the current owner may finish a running job')
            if error:
                self.db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(str(error),ident))
                self.event(ident,'execution_failed',{'error':str(error)}); return
            payload=json.loads(row['payload'])
            if not all(verify_evidence(self.root,x) for x in payload['inputs']):
                self.db.execute("UPDATE jobs SET status='stale',error='Inputs changed during execution' WHERE id=?",(ident,))
                self.event(ident,'stale',{'reason':'Inputs changed during execution'}); return
            entry=evidence(self.root,report_path)
            report=json.loads(local_path(self.root,report_path).read_text(encoding='utf-8'))
            if not isinstance(report,dict) or report.get('program_input_evidence')!=payload['inputs']:
                raise ValueError('Worker report is not bound to this job')
            # A worker run can complete while its subject fails. Never conflate.
            self.db.execute("UPDATE jobs SET status='completed',report=?,error=NULL WHERE id=?",(canonical(entry),ident))
            self.event(ident,'evaluated',{'report':entry,'ready':report.get('ready') is True})
            if report.get('ready') is not True:
                self.issue(payload['issue_key'],row['role'],payload['issue_title'],[entry]+payload['inputs'],payload['next_action'])
            else:
                # Readiness does NOT grant final video promotion.
                self.db.execute("UPDATE issues SET status='resolved' WHERE key=?",(payload['issue_key'],))

    def status(self):
        jobs=[dict(x) for x in self.db.execute('SELECT id,role,kind,resource,status,attempt,report,error,payload FROM jobs ORDER BY rowid')]
        for job in jobs:
            inputs=json.loads(job.pop('payload'))['inputs']
            job['report']=json.loads(job['report']) if job['report'] else None
            job['evidence_current']=all(verify_evidence(self.root,x) for x in inputs)
            if job['report']: job['evidence_current'] &= verify_evidence(self.root,job['report'])
        issues=[dict(x) for x in self.db.execute("SELECT * FROM issues WHERE status='open' ORDER BY role,key")]
        for issue in issues:
            issue['evidence']=json.loads(issue['evidence'])
            issue['evidence_current']=all(verify_evidence(self.root,x) for x in issue['evidence'])
        return dict(schema_version=1, coordinator='durable local jobs + externally scheduled specialist agents',
            enabled=self.db.execute("SELECT value FROM control WHERE key='enabled'").fetchone()[0]=='true',
            accepted_native_candidate=None, jobs=jobs, open_issues=issues,
            experiments=[json.loads(x[0]) for x in self.db.execute('SELECT record FROM experiments ORDER BY rowid')],
            execution_note='The queue runs typed checks. Research/code changes require the active Codex subagents or configured heartbeat. It is not itself a generative model.')

    def record_experiment(self, record):
        """Preserve outcomes from bounded specialist work, including failures."""
        if record.get('role') not in ROLES or record.get('outcome') not in ('rejected','partial','accepted_mechanism','failed_execution'):
            raise ValueError('Explicit role and scoped outcome required')
        if any(not isinstance(record.get(k),str) or not record[k].strip()
               for k in ('issue_key','hypothesis','observation','next_action')):
            raise ValueError('Experiment needs rationale, observation and next action')
        for key in ('inputs','outputs'):
            if not record.get(key) or not all(verify_evidence(self.root,e) for e in record[key]):
                raise ValueError('Experiment requires current input/output hashes')
        fingerprint=hashlib.sha256(canonical(record).encode()).hexdigest()
        with self.transaction():
            if self.db.execute('SELECT 1 FROM experiments WHERE fingerprint=?',(fingerprint,)).fetchone(): return fingerprint
            self.db.execute('INSERT INTO experiments VALUES(?,?)',(fingerprint,canonical(record)))
            self.event(None,'experiment_recorded',{'fingerprint':fingerprint,'outcome':record['outcome']})
        return fingerprint

    def export(self):
        status=self.status()
        write_json(self.state/'status.json',status)
        events=[dict(x) for x in self.db.execute('SELECT * FROM events ORDER BY sequence')]
        for event in events: event['data']=json.loads(event['data'])
        write_json(self.state/'events.json',events)
        return status


def seed(program, config_path=CONFIG):
    config=json.loads(local_path(program.root,config_path).read_text(encoding='utf-8'))
    if config.get('schema_version')!=1: raise ValueError('Unsupported program configuration')
    common=[evidence(program.root,'real_video/gaussian_program.py'),evidence(program.root,config_path)]
    names=[a['id'] for a in config['assets']]
    if len(names)!=len(set(names)): raise ValueError('Duplicate object identity in program configuration')
    for descriptor in config['assets']:
        name,asset,mesh=descriptor['id'],descriptor['asset'],descriptor.get('mesh')
        inputs=common+[evidence(program.root,'real_video/asset_readiness.py'),evidence(program.root,asset)]
        if mesh: inputs.append(evidence(program.root,mesh))
        rig,review=descriptor['rig'],descriptor['review']
        inputs += [evidence(program.root,rig),evidence(program.root,review)]
        inputs += input_evidence(program.root,json.loads(local_path(program.root,review).read_text(encoding='utf-8')))
        program.enqueue('geometry','asset_preflight',dict(asset=asset,mesh=mesh,rig=rig,review=review,inputs=inputs,
            issue_key=f'geometry.{name}',issue_title=f'{name}: anatomical geometry and rig readiness',
            next_action='Repair or replace the malformed source geometry; fit a neutral anatomical rig and collect independent multiview/contact evidence before another motion candidate. Do not merely increase Gaussian count or replay another gait.'))
    from .generation_readiness import current_manifest
    manifest_path=config.get('generation_manifest')
    manifest=(json.loads(local_path(program.root,manifest_path).read_text(encoding='utf-8'))
              if manifest_path else current_manifest(program.root))
    inputs=common+[evidence(program.root,'real_video/generation_readiness.py')]
    if manifest_path: inputs.append(evidence(program.root,manifest_path))
    # Bind each worker cycle to the actual model implementation and checkpoints
    # listed in its manifest, not only an easily changed description of them.
    inputs += manifest_evidence(program.root,manifest)
    program.enqueue('generation','generation_preflight',dict(manifest=manifest,inputs=inputs,
        issue_key='generation.native',issue_title='Trained causal Gaussian-state generation is not established',
        next_action='Implement/train a geometry-aware Gaussian state writer and a matching generator read path, with corruption-aware memory training and held-out rollout tests. Keep pretrained skeletal and Wan RGB baselines separately labeled. Start with the smallest real-data CUDA experiment and record exact weights/data hashes.'))
    for candidate in config.get('video_candidates',[]):
        review,video=candidate['review'],candidate['video']
        inputs=common+[evidence(program.root,review),evidence(program.root,video),evidence(program.root,'real_video/articulation_quality.py')]
        inputs += input_evidence(program.root,json.loads(local_path(program.root,review).read_text(encoding='utf-8')))
        program.enqueue('evaluation','visual_review',dict(review=review,video=video,inputs=inputs,
            issue_key='evaluation.'+candidate['id'],issue_title='Candidate action video requires anatomy/contact review',
            next_action='Require a new hash-bound independent review of a materially repaired candidate; inspect separated poses, paws, bending, tearing, body shape and oblique views. Do not rewrite the existing rejection into a pass.'))
    for issue in config.get('repair_issues',[]):
        program.issue(issue['key'],issue['role'],issue['title'],
            [evidence(program.root,p) for p in issue['evidence_paths']],issue['next_action'])


def worker(kind, payload, output, root=ROOT):
    root=Path(root).resolve()
    if kind not in KINDS: raise ValueError('Worker not in allowlist')
    if not all(verify_evidence(root,x) for x in payload['inputs']): raise ValueError('Worker source inputs changed')
    if kind=='asset_preflight':
        from .asset_readiness import audit_asset
        report=audit_asset(local_path(root,payload['asset']),
            mesh_path=local_path(root,payload['mesh']) if payload.get('mesh') else None,
            rig_path=local_path(root,payload['rig']) if payload.get('rig') else None,
            semantic_review=local_path(root,payload['review']) if payload.get('review') else None)
        report['ready']=report['decision']['motion_jobs_allowed'] is True
    elif kind=='generation_preflight':
        from .generation_readiness import evaluate_readiness
        report=evaluate_readiness(payload['manifest'],root=root)
    else:
        from .articulation_quality import decide_review
        review=json.loads(local_path(root,payload['review']).read_text(encoding='utf-8'))
        report=decide_review(review,local_path(root,payload['video']))
        report['ready']=report.get('accepted') is True
    report['program_input_evidence']=payload['inputs']
    write_json(local_path(root,output),report)


def run_cycle(program, max_jobs=4, budget_seconds=300):
    if not 1<=max_jobs<=8 or not 5<=budget_seconds<=600: raise ValueError('Bounded cycle required')
    program.recover()
    deadline=time.monotonic()+budget_seconds
    completed=[]
    while len(completed)<max_jobs and deadline-time.monotonic()>5:
        job=program.claim(allow_gpu=False)
        if job is None: break
        directory=program.state/'jobs'/job['id']/f"attempt_{job['attempt']}"
        try:
            directory.mkdir(parents=True,exist_ok=False)
            payload=json.loads(job['payload'])
            write_json(directory/'input.json',payload)
            output=directory/'report.json'
            command=[sys.executable,'-m','real_video.gaussian_program','worker','--kind',job['kind'],
                '--input',str(directory/'input.json'),'--output',str(output)]
            with (directory/'worker.log').open('x',encoding='utf-8') as log:
                subprocess.run(command,cwd=program.root,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(120,deadline-time.monotonic()),check=True)
            program.finish(job['id'],output)
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            program.finish(job['id'],error=repr(error))
        completed.append(job['id'])
        program.export()
    return program.export()


def promotion_decision(root, geometry_reports, generation_manifest, review_path, video_path, expected_objects):
    """Re-evaluate evidence, never trust a prewritten 'accepted' decision file."""
    from .articulation_quality import decide_review
    from .generation_readiness import evaluate_readiness
    from .asset_readiness import require_motion_ready
    blockers=[]
    if not expected_objects or set(geometry_reports)!=set(expected_objects):
        blockers.append('Every candidate object needs its own readiness report')
    for object_id,entry in geometry_reports.items():
        if not verify_evidence(root,entry): blockers.append('Stale/missing geometry report'); continue
        try:
            report=json.loads(local_path(root,entry['path']).read_text(encoding='utf-8'))
            require_motion_ready(report)
            if report['inputs']!=expected_objects.get(object_id):
                blockers.append(f'Geometry report is not bound to candidate object {object_id}')
        except (OSError,KeyError,TypeError,ValueError,RuntimeError): blockers.append('Animal geometry gate rejected')
    if not verify_evidence(root,generation_manifest): blockers.append('Stale/missing generation manifest')
    else:
        try:
            manifest=json.loads(local_path(root,generation_manifest['path']).read_text(encoding='utf-8'))
            if not evaluate_readiness(manifest,root=root).get('ready'): blockers.append('Native generation gate rejected')
            artifacts=manifest.get('artifacts',{})
            if artifacts.get('video')!=evidence(root,video_path) or artifacts.get('visual_review')!=evidence(root,review_path):
                blockers.append('Generation evidence is for a different video/review')
            canonical_hashes={item.get('sha256') for name,item in artifacts.items() if name.startswith('canonical_state')}
            if any(obj.get('asset',{}).get('sha256') not in canonical_hashes for obj in expected_objects.values()):
                blockers.append('Candidate animals are not bound to the generator')
        except (OSError,KeyError,TypeError,ValueError,AttributeError):
            blockers.append('Malformed or stale native generation evidence')
    try:
        review=json.loads(local_path(root,review_path).read_text(encoding='utf-8'))
        decision=decide_review(review,local_path(root,video_path))
    except (OSError,KeyError,TypeError,ValueError):
        decision={'accepted':False,'reasons':['Missing or malformed video review']}
    if not decision['accepted']: blockers.append('Independent video quality gate rejected')
    return dict(accepted=not blockers,blockers=blockers,visual=decision)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('init','cycle','status','recover','pause','resume','retry','worker','record'))
    parser.add_argument('--kind',choices=KINDS); parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path); parser.add_argument('--job')
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--max-jobs',type=int,default=4); parser.add_argument('--budget-seconds',type=float,default=300)
    args=parser.parse_args()
    if args.command=='worker':
        worker(args.kind,json.loads(local_path(ROOT,args.input).read_text(encoding='utf-8')),args.output); return
    program=Program()
    try:
        if args.command in ('init','cycle'): seed(program,args.config)
        if args.command=='cycle':
            from .checkpoint_io import keep_windows_awake
            with keep_windows_awake(): result=run_cycle(program,args.max_jobs,args.budget_seconds)
        else:
            if args.command=='recover': program.recover()
            if args.command=='retry': program.retry(args.job)
            if args.command=='record': program.record_experiment(json.loads(local_path(ROOT,args.input).read_text(encoding='utf-8')))
            if args.command in ('pause','resume'):
                with program.transaction():
                    program.db.execute("UPDATE control SET value=? WHERE key='enabled'",('true' if args.command=='resume' else 'false',))
                    program.event(None,args.command,{})
            result=program.export()
        print(json.dumps(result,indent=2))
    finally: program.db.close()


if __name__=='__main__': main()
