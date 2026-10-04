"""Second validation-only loop: conservative weight retention after regressions.

Blend independently with original weights for both learned-loss candidate and
matched-training control. No cat results or final-test clips select weights.
"""
import json
from pathlib import Path
import torch
from .checkpoint_io import load_verified,save_inference_checkpoint,keep_windows_awake,digest
from .train_coherent_motion import BASE,DATA,evaluate
from .learned_motion import LearnedGaussianMotion
from .motion_coherence import passes_gate
from .audit_coherent_motion import moving_audit,compare

ROOT=Path('artifacts/real_video/coherent_motion/v2')
OUT=ROOT/'retention'


def blend(original,updated,alpha):
    if not 0<=alpha<=1: raise ValueError('alpha must be in [0,1]')
    if original.keys()!=updated.keys(): raise ValueError('Checkpoint keys differ')
    result={}
    for key in original:
        if original[key].shape!=updated[key].shape: raise ValueError('Checkpoint shapes differ')
        result[key]=original[key]+alpha*(updated[key]-original[key])
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve retention loop')
    protocol=dict(alphas=[.25,.5,.75],hidden=32,selection='All original primary and secondary gates versus original and equally blended matched-training control; minimum edge MSE among eligible.',
                  cause='Full update passed aggregate gate but failed per-clip regression veto (14/57).',
                  limits='Adaptive validation experiment, not untouched test. Weight interpolation is not semantic interpolation or anatomy.',
                  cat_used=False,new_training=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4)
    with keep_windows_awake():
        initial=load_verified(BASE); candidate=load_verified(ROOT/'model.pt')
        control=load_verified(ROOT/'lambda_0/checkpoints/step_00001000.pt')
        data=load_verified(DATA); val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
        baseline=json.loads((ROOT/'baseline.json').read_text())['learned']
        original_audit=json.loads((ROOT/'independent_audit.json').read_text())['audited']['original']
        rows=[]; selected=None
        for alpha in protocol['alphas']:
            scores={}; audits={}; checkpoints={}
            for name,source in [('candidate',candidate),('control',control)]:
                ckpt=dict(model=blend(initial['model'],source['model'],alpha),config=dict(hidden=32))
                model=LearnedGaussianMotion().cuda().eval(); model.load_state_dict(ckpt['model'])
                scores[name]=evaluate(model,data,val); del model
                audits[name]=moving_audit(ckpt,data,val); checkpoints[name]=ckpt
            primary={name:passes_gate(scores['candidate']['means'],ref['means']) for name,ref in [('original',baseline),('control',scores['control'])]}
            secondary={name:compare(audits['candidate'],ref) for name,ref in [('original',original_audit),('control',audits['control'])]}
            eligible=all(g['passed'] for g in [*primary.values(),*secondary.values()])
            row=dict(alpha=alpha,scores=scores,audits=audits,primary=primary,secondary=secondary,eligible=eligible)
            rows.append(row)
            checkpoint=dict(**checkpoints['candidate'],alpha=alpha,base_sha256=digest(BASE),update_sha256=digest(ROOT/'model.pt'),
                            eligible=eligible,visually_accepted=False)
            sha=save_inference_checkpoint(checkpoint,OUT/f'alpha_{alpha:g}.pt'); row['sha256']=sha
            if eligible and (selected is None or row['scores']['candidate']['means']['edge_mse_pixels64']<selected['scores']['candidate']['means']['edge_mse_pixels64']): selected=row
            (OUT/'history.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(dict(alpha=alpha,means=scores['candidate']['means'],primary=primary,secondary=secondary,eligible=eligible)),flush=True)
        if selected:
            checkpoint=load_verified(OUT/f"alpha_{selected['alpha']:g}.pt")
            save_inference_checkpoint(checkpoint,OUT/'model.pt')
        (OUT/'results.json').write_text(json.dumps(dict(selected_alpha=selected['alpha'] if selected else None,
            status='validation_gates_passed_pending_visual' if selected else 'all_retention_candidates_rejected',
            test_clips=0,cat_used=False,wan_weights_changed=False),indent=2))


if __name__=='__main__': main()
