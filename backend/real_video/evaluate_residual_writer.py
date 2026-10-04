"""CPU-only development ablation on recorded first-frame Wan latents, not video generation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .gaussian_latent_memory import GaussianLatentMemory
from .latent_track_memory import planar_gaussian_cache
from .residual_gaussian_writer import write_residual
from .wan22_gaussian_anchor import _grid, _gradient_energy


def run(source, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = torch.load(source, map_location='cpu', weights_only=True)['latent']
    z = data[0, :, 0].float().contiguous()
    c, h, w = z.shape
    if (c, h, w) != (48, 16, 28):
        raise ValueError('Expected saved development native48 first-frame latent')
    ids = torch.arange(h*w)
    cache = planar_gaussian_cache(_grid(h,w), torch.ones(h*w), ids, h, w, sigma=.35)
    q = torch.ones(h,w)

    def read(bank):
        return bank.read(cache,h,w,ids=ids)['features']

    def metrics(value):
        error = (value.double()-z.double()).square().mean().sqrt()
        return dict(rmse=float(error), relative_rmse=float(error/z.double().square().mean().sqrt()),
                    gradient_energy_ratio=_gradient_energy(value.double())/_gradient_energy(z.double()))

    report = dict(scope='Development fixed planar geometry appearance fitting, no video generation',
                  source=str(source.resolve()), source_sha256=digest, frames_used=[0],
                  shape=list(z.shape), gaussian_count=len(ids), cuda_used=False,
                  generator_weights_trained=False, geometry_updated=False, cases={})
    for method in ('weighted', 'residual'):
        bank = GaussianLatentMemory(ids,c,digest)
        start = time.perf_counter()
        if method == 'weighted':
            details = bank.write(z,cache,ids=ids,confidence=q)
        else:
            details = write_residual(bank,z,cache,ids=ids,confidence=q,regularization=.001,
                                     max_iterations=24,trust_radius=16.)
        seconds = time.perf_counter()-start
        initial = read(bank).clone()
        snapshot = bank.snapshot()
        torch.save(snapshot,out/(method+'_bank.pt'))
        restored = GaussianLatentMemory(ids,c,digest)
        restored.restore(torch.load(out/(method+'_bank.pt'),weights_only=True))
        assert torch.equal(read(restored),initial)
        for _ in range(20):
            current = read(bank)
            if method == 'weighted':
                bank.write(current,cache,ids=ids,confidence=q)
            else:
                write_residual(bank,current,cache,ids=ids,confidence=q)
        report['cases'][method] = dict(fit_seconds=seconds, fit=metrics(initial),
            after_20_self_reads=metrics(read(bank)),
            self_read_drift_max=float((read(bank)-initial).abs().max()),
            writes_before=snapshot['writes'],writes_after=bank.writes,
            memory_bytes=bank.memory_bytes(),fit_diagnostics=details,restore_exact=True)
    # Noise sensitivity is a limitation check, not a claim of denoising.
    noise = torch.randn(z.shape,generator=torch.Generator().manual_seed(173))*.01
    noisy = GaussianLatentMemory(ids,c,digest)
    write_residual(noisy,z+noise,cache,ids=ids,confidence=q,regularization=.001,trust_radius=16.)
    clean = torch.load(out/'residual_bank.pt',weights_only=True)['features']
    report['noise_sensitivity'] = dict(input_noise_rms=float(noise.square().mean().sqrt()),
        feature_response_rms=float((noisy.snapshot()['features']-clean).square().mean().sqrt()),
        recalled_noisy_fit=metrics(read(noisy)))
    report['limits'] = ('One observed development frame; supplied planar geometry and full confidence. '
        'No decoded visual quality, unseen-view geometry, trained network, motion, compression or Wan speedup established. '
        'Self-read is only an idempotence stress test, not permission to treat generated pixels as observations.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.out)
