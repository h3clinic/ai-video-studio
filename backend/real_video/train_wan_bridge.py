"""Bounded real-video pilot for direct normalized Wan latent -> Gaussian fields.

Separate from unchanged Wan RGB baseline. Never trains on its generated clip.
Defaults to 80 training / 20 validation clips from existing source-group splits;
explicit expanded mode retains all train/validation clips without using test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint, save_training_checkpoint
from .wan_baseline import REVISION, MODEL
from .wan_gaussian import WanGaussianDecoder, render_gaussian_video, splat_frame

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent.parent/'work'


def selected_entries(manifest, expanded=False):
    result = []
    for split, count in [('train', 8), ('validation', 2)]:
        for label in manifest['labels']:
            items = [e for e in manifest['entries'] if e['split'] == split and e['label'] == label]
            items.sort(key=lambda e: hashlib.sha256(('wan-bridge-v1:'+e['file']).encode()).hexdigest())
            if len(items) < count:
                raise ValueError(f'Insufficient {split} clips for {label}')
            result.extend(items if expanded else items[:count])
    return result


def read_clip(path, size=128, frames=9):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total < 1:
            raise ValueError(f'Invalid source {path}')
        stride = max(1, round(fps/8))
        maximum = max(0, total - 1 - (frames-1)*stride)
        offset = int(hashlib.sha256(('wan-bridge-window:'+path.name).encode()).hexdigest()[:8], 16) % (maximum+1)
        indices = np.minimum(offset+np.arange(frames)*stride, total-1)
        images = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Failed to decode {path} at {index}')
            h, w = frame.shape[:2]
            side = min(h, w)
            crop = frame[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
            images.append(cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        return torch.from_numpy(np.stack(images)), dict(frame_indices=indices.tolist(), source_fps=fps)
    finally:
        cap.release()


def prepare(args, out):
    cache = cache_path(args)
    if cache.exists():
        data = load_verified(cache)
        if data['videos'].shape[2:4] != (args.size, args.size):
            raise ValueError('Cached spatial dimensions mismatch')
        return data
    manifest = json.loads((ROOT/'artifacts/real_video/data_protocol.json').read_text())
    entries = selected_entries(manifest, expanded=args.all_clips)
    assert not ({e['group'] for e in entries if e['split']=='train'} &
                {e['group'] for e in entries if e['split']=='validation'})
    from diffusers import AutoencoderKLWan
    start = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(WORK/'wan21_13b/vae', torch_dtype=torch.float32,
                                          local_files_only=True).eval().requires_grad_(False).to('cuda')
    mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
    codes, videos, metadata, vae_errors = [], [], [], []
    with torch.inference_mode():
        for index, entry in enumerate(entries):
            path = WORK/'real_video/clips'/entry['file']
            if digest(path) != entry['sha256']:
                raise ValueError(f'Source checksum mismatch: {path.name}')
            frames, window = read_clip(path, size=args.size)
            image = frames.permute(3, 0, 1, 2).unsqueeze(0).to('cuda', dtype=torch.float32)/127.5 - 1
            posterior = vae.encode(image).latent_dist.mode()
            normalized = ((posterior-mean)/std).cpu()
            if entry['split'] == 'validation':
                decoded = ((vae.decode(posterior, return_dict=False)[0].clamp(-1, 1)+1)/2)
                error = ((decoded-(image+1)/2)**2).mean().item()
                vae_errors.append(dict(file=entry['file'], mse=error, psnr=float(-10*np.log10(max(error, 1e-12)))))
            codes.append(normalized[0]); videos.append(frames); metadata.append(dict(entry, **window))
            if (index+1) % 10 == 0:
                print(json.dumps(dict(stage='prepare_wan_training_codes', completed=index+1, total=len(entries))), flush=True)
    data = dict(latents=torch.stack(codes), videos=torch.stack(videos), metadata=metadata,
                model=MODEL, revision=REVISION, source_protocol_sha256=digest(ROOT/'artifacts/real_video/data_protocol.json'),
                vae_validation=vae_errors, prepare_seconds=time.perf_counter()-start,
                policy=f'Expanded={args.all_clips}; source-group split retained; 9 frames at ~8fps; {args.size}-square centre crop; posterior mode')
    save_inference_checkpoint(data, cache)
    del vae, mean, std, image, posterior
    torch.cuda.empty_cache()
    return data


def cache_path(args):
    if not args.all_clips and args.size == 128:
        return WORK/'real_video/wan_bridge_v1.pt'
    return WORK/f'real_video/wan_bridge_{"all" if args.all_clips else "pilot"}_{args.size}.pt'


@torch.no_grad()
def evaluate(model, data, indices):
    model.eval()
    values = []
    for index in indices:
        raw = model(data['latents'][index:index+1].to('cuda'))
        size = data['videos'].shape[2]
        video = render_gaussian_video(raw, size, size)
        target = data['videos'][index:index+1].permute(0, 1, 4, 2, 3).to('cuda').float()/255
        mse = (video-target).square().mean().item()
        values.append(dict(file=data['metadata'][index]['file'], mse=mse, psnr=float(-10*np.log10(max(mse, 1e-12)))))
    return dict(mean_mse=float(np.mean([e['mse'] for e in values])),
                mean_psnr=float(np.mean([e['psnr'] for e in values])), cases=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/wan_bridge/v1'))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--size', type=int, choices=[128, 256], default=128)
    parser.add_argument('--all-clips', action='store_true')
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--eval-every', type=int, default=250)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'protocol.json').exists():
        raise FileExistsError('Pilot already started; preserve it and use a new output directory')
    manifest = json.loads((ROOT/'artifacts/real_video/data_protocol.json').read_text())
    entries = selected_entries(manifest, expanded=args.all_clips)
    protocol = dict(steps=args.steps, seed=429301, learning_rate=args.lr, batch=2, frames_per_clip_per_step=2,
                    width=64, train_clips=sum(e['split']=='train' for e in entries),
                    validation_clips=sum(e['split']=='validation' for e in entries), size=args.size, frames=9,
                    warm_start_sha256=digest(args.warm_start) if args.warm_start else None,
                    training_data_path=str(cache_path(args).resolve()),
                    loss='pixel MSE + 0.1 * horizontal/vertical finite-difference L1',
                    selection=f'lowest validation mean MSE every {args.eval_every} steps, including initial step 0',
                    limits='Small reconstruction-trained decoder pilot, not matched-quality speedup or new foundation-model training',
                    generated_cat_not_used_for_training_or_selection=True, test_split_untouched=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed']); np.random.seed(protocol['seed'])
    with keep_windows_awake():
        data = prepare(args, args.out)
        # Weight initialization must not depend on whether preparation was cached.
        torch.manual_seed(protocol['seed']); np.random.seed(protocol['seed'])
        train = [i for i,e in enumerate(data['metadata']) if e['split']=='train']
        validation = [i for i,e in enumerate(data['metadata']) if e['split']=='validation']
        model = WanGaussianDecoder(width=64).to('cuda')
        if args.warm_start:
            model.load_state_dict(load_verified(args.warm_start)['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
        initial = evaluate(model, data, validation)
        history = [dict(step=0, validation=initial)]
        best = initial['mean_mse']
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        def save(step, is_best):
            save_training_checkpoint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step, config=protocol, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                validation=history[-1]['validation']), args.out, step, is_best)

        save(0, True)
        for step in range(1, args.steps+1):
            model.train()
            ids = [train[j] for j in torch.randint(len(train), (2,)).tolist()]
            times = torch.randperm(9)[:2].tolist()
            raw = model(data['latents'][ids].to('cuda'))
            losses = []
            for t in times:
                image, _ = splat_frame(raw[:, :, t], args.size, args.size)
                target = data['videos'][ids, t].permute(0, 3, 1, 2).to('cuda').float()/255
                gradient = F.l1_loss(image[:, :, 1:]-image[:, :, :-1], target[:, :, 1:]-target[:, :, :-1])
                gradient += F.l1_loss(image[:, :, :, 1:]-image[:, :, :, :-1], target[:, :, :, 1:]-target[:, :, :, :-1])
                losses.append(F.mse_loss(image, target)+.1*gradient)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite bridge training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if step % 50 == 0:
                progress = dict(stage='train_gaussian_bridge', step=step, total=args.steps,
                                loss=loss.item(), elapsed_seconds=time.perf_counter()-start)
                (args.out/'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
                print(json.dumps(progress), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                score = evaluate(model, data, validation)
                improved = score['mean_mse'] < best
                best = min(best, score['mean_mse'])
                history.append(dict(step=step, train_loss=loss.item(), validation=score))
                save(step, improved)
                (args.out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
                print(json.dumps(dict(stage='validate_gaussian_bridge', step=step, mean_psnr=score['mean_psnr'], improved=improved)), flush=True)
        elapsed = time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated()
        checkpoint = load_verified(args.out/'best.pt')
        exported = dict(model=checkpoint['model'], config=protocol, selected_step=checkpoint['step'],
                        validation=checkpoint['validation'], model_revision=REVISION,
                        training_data_sha256=digest(cache_path(args)),
                        status=('trained_small_real_video_decoder_pilot_not_quality_matched' if checkpoint['step']
                                else 'untrained_initialization_selected_no_improvement'))
        sha = save_inference_checkpoint(exported, args.out/'gaussian_decoder.pt')
        report = dict(selected_step=checkpoint['step'], initial=initial,
                      selected_validation=checkpoint['validation'], original_vae_validation=data['vae_validation'],
                      training_seconds=elapsed, training_peak_cuda_allocated_bytes=peak,
                      prepare_seconds=data['prepare_seconds'], checkpoint_sha256=sha,
                      parameter_count=sum(p.numel() for p in model.parameters()), protocol=protocol)
        (args.out/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(stage='complete_bridge_training', selected_step=checkpoint['step'],
                              mean_validation_psnr=checkpoint['validation']['mean_psnr'])), flush=True)


if __name__ == '__main__':
    main()
