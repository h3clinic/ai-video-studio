"""One bounded remote experiment, preserving prior results and model caches."""
import json
import hashlib
from cloud import run_object_edit as base
from cloud.runpod_control import awake

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-provider-start',action='store_true')
    args=parser.parse_args()
    base.OUT=base.ROOT/'artifacts/cloud/object_edit_semantic_v1'
    if args.retry_provider_start:
        previous=json.loads((base.OUT/'control.json').read_text())
        if previous.get('error')!='RunPod API HTTP 500' or previous.get('provider_status',{}).get('desiredStatus')!='EXITED':
            raise RuntimeError('Retry only allowed after verified stopped provider-start failure')
        base.OUT=base.ROOT/'artifacts/cloud/object_edit_semantic_v1_start_retry'
    base.FILES=base.FILES+['cloud/object_edit_semantic_worker.py','real_video/local_object_repair.py','research/prior_manifest.json']
    # All new remote paths are isolated. Reuse source Gaussian state read-only.
    base.JOB=base.JOB.replace('/workspace/object_edit','/workspace/object_edit_semantic_v1')
    base.JOB=base.JOB.replace("str(p/'model_cache')","'/workspace/object_edit/model_cache'")
    base.JOB=base.JOB.replace("'cloud/object_edit_worker.py'","'cloud/object_edit_semantic_worker.py'")
    # Base controller path references are isolated through a generated function,
    # not monkeypatched filesystem operations. No changes to old artifact files.
    import inspect
    source=inspect.getsource(base.main).replace('/workspace/object_edit','/workspace/object_edit_semantic_v1').replace('workspace/object_edit/','workspace/object_edit_semantic_v1/').replace("'workspace/object_edit'","'workspace/object_edit_semantic_v1'")
    source=source.replace("'balance_before_usd':11.25","'balance_before_usd':10.89")
    source=source.replace("bundle.writestr('job.py',JOB)","bundle.writestr('job.py',JOB)\n        bundle.writestr('prior_manifest.json',(ROOT/'research/prior_manifest.json').read_bytes())")
    namespace=vars(base).copy()
    exec(compile(source,'<isolated_semantic_controller>','exec'),namespace)
    with awake(): namespace['main']()

if __name__=='__main__': main()
