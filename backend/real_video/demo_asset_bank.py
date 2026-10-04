"""Lossless split-bank export and independently attached learned-motion audit."""
import json
from pathlib import Path
import time
import torch

from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .gaussian_asset_bank import bank_from_points,make_motion,GaussianAssetBank,tensor_bytes
from .learned_motion import LearnedGaussianMotion
from .demo_learned_motion import transfer_displacement
from .dense_gaussian_seed import FixedGeometrySplat

ROOT=Path('artifacts/real_video/asset_bank/v1')
SEED=Path('artifacts/real_video/learned_motion/v1/dense_seed/seed_1px.pt')
MODEL=Path('artifacts/real_video/learned_motion/v1/model.pt')


@torch.no_grad()
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'audit.json').exists(): raise FileExistsError('Preserve exported experiment')
    torch.set_num_threads(4)
    with keep_windows_awake():
        seed=load_verified(SEED); weights=load_verified(MODEL)
        bank=bank_from_points(seed['points'],source_sha256=digest(SEED))
        save_inference_checkpoint(bank,ROOT/'appearance_geometry_bank.pt')
        before=digest(ROOT/'appearance_geometry_bank.pt')
        # Same already-trained predictor; no Wan weights are changed here.
        model=LearnedGaussianMotion(weights['config']['hidden']).eval().cuda(); model.load_state_dict(weights['model'])
        observed=seed['observed_coarse'].cuda()
        state=model.initialize(observed[:,:,0],observed[:,:,1],observed[:,:,2],dt=.5)
        state=model.step(state,dt=.5)
        delta=transfer_displacement(state['fields'][:,:2]-observed[:,:2,2],seed['sample_uv'].cuda()).cpu()
        motion=make_motion(bank,bank['ids'],delta,time_seconds=1/16)
        assert 'colour' not in motion and 'appearance' not in motion
        save_inference_checkpoint(motion,ROOT/'motion_frame_0001.pt')
        # Drop source predictor/seed before independently reloading the split files.
        direct=seed['points'].clone(); direct[:,:2]+=delta
        radius=seed['radius']; del model,state,observed,seed,weights,delta,motion,bank
        bank=load_verified(ROOT/'appearance_geometry_bank.pt'); motion=load_verified(ROOT/'motion_frame_0001.pt')
        runtime=GaussianAssetBank(bank,'cuda')
        zero=make_motion(bank,bank['ids'],torch.zeros_like(motion['displacement']))
        identity=runtime.attach(zero)
        assert torch.equal(identity.cpu(),runtime.reference.cpu())
        attached=runtime.attach(motion)
        assert torch.equal(attached.cpu(),direct)
        permutation=torch.randperm(len(motion['ids']),generator=torch.Generator().manual_seed(430301))
        shuffled=dict(motion,ids=motion['ids'][permutation],displacement=motion['displacement'][permutation])
        assert torch.equal(runtime.attach(shuffled),attached)
        assert torch.equal(attached[:,6:9],runtime.reference[:,6:9])
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(10): runtime.attach(shuffled)
        torch.cuda.synchronize(); attach_ms=(time.perf_counter()-start)*100
        # Render both independent paths, same geometry/support/colours.
        a,_=FixedGeometrySplat(attached,radius=radius).render((attached[:,6:9]+1)/2)
        direct=direct.cuda(); b,_=FixedGeometrySplat(direct,radius=radius).render((direct[:,6:9]+1)/2)
        error=(a-b).abs().max().item(); assert error<1e-6
        assert before==digest(ROOT/'appearance_geometry_bank.pt')
        report=dict(points=len(runtime.ids),zero_motion_exact=True,permutation_exact=True,
                    same_predictions_exact_points=True,same_predictions_max_rgb_error=error,
                    appearance_exactly_unchanged=True,bank_file_hash_unchanged=True,
                    bank_sha256=before,motion_sha256=digest(ROOT/'motion_frame_0001.pt'),
                    bank_tensor_bytes=tensor_bytes(bank),appearance_tensor_bytes=tensor_bytes(bank['appearance']),
                    geometry_tensor_bytes=tensor_bytes(bank['geometry']),id_tensor_bytes=tensor_bytes(bank['ids']),
                    motion_tensor_bytes=tensor_bytes(motion),runtime_bank_and_lookup_bytes=runtime.storage_bytes(),
                    original_points_tensor_bytes=direct.numel()*direct.element_size(),
                    mean_attach_ms_10_calls_including_validation_and_upload=attach_ms,
                    wan_weights_changed=False,new_training=False,new_video_quality_claim=False,
                    limits='Same existing learned deformation, different storage/interface. IDs are asset-local correspondence, not anatomy/depth. Explicit IDs/lookup add memory; no compression or tearing fix established.')
        (ROOT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
