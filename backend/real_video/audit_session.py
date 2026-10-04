"""Check the reusable session's inputs, numerical equivalence and saved videos."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import torch
from .checkpoint_io import load_verified,keep_windows_awake
from .session import GaussianVideoSession
from .latent import GaussianCodec,inference_decoder
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields_chunked


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True,help='Existing completed session output directory')
    args=parser.parse_args(); torch.set_num_threads(2)
    allowed=args.checkpoint.resolve(); original_load=torch.load; reads=[]
    def restricted(path,*positional,**kwargs):
        if not isinstance(path,(str,Path)) or Path(path).resolve()!=allowed: raise AssertionError('Unexpected input checkpoint/data')
        if kwargs.get('weights_only') is not True: raise AssertionError('Unsafe load')
        reads.append(str(allowed)); return original_load(path,*positional,**kwargs)
    with patch('torch.load',side_effect=restricted),patch.object(GaussianCodec,'encode',side_effect=AssertionError('Analysis encoder called')):
        with GaussianVideoSession(args.checkpoint) as session:
            actual=[session.generate(session.labels[i],3010000+i) for i in (0,1)]
            repeated=session.generate(session.labels[0],3010000)
            torch.testing.assert_close(actual[0]['fields'],repeated['fields'],rtol=0,atol=0)
        assert session.closed
    if len(reads)!=1: raise AssertionError('Checkpoint must be loaded once per session')
    packet=load_verified(args.checkpoint)
    prior=GaussianVideoDenoiser(**packet['config']).cuda().eval(); prior.load_state_dict(packet['model'])
    decoder=inference_decoder(packet['codec_config'],packet['codec']).cuda()
    comparisons=[]
    for i,result in enumerate(actual):
        labels=torch.tensor([i],device='cuda')
        z=sample(prior,labels,3010000+i,shape=tuple(packet['latent_shape']),distilled_guidance=packet.get('distilled_guidance'))
        z=(z*packet['latent_std'].cuda()+packet['latent_mean'].cuda()).half().float()
        with torch.autocast('cuda',dtype=torch.bfloat16): fields=decoder.decode(z).float()
        fields=fields*packet['std'].cuda()+packet['mean'].cuda()
        video=render_fields_chunked(fields,frame_chunk=1)
        error=dict(seed=3010000+i,field_max_error=(fields[0].cpu()-result['fields']).abs().max().item(),
                   pixel_max_error=(video[0].cpu()-result['video']).abs().max().item())
        if max(error['field_max_error'],error['pixel_max_error'])>2e-5: raise AssertionError('Session changed output')
        comparisons.append(error)
    records=json.loads((args.out/'session.json').read_text(encoding='utf-8'))['records']
    sizes=[]
    for record in records:
        saved=load_verified(args.out/record['gaussian_packet'])
        if saved['seed']!=record['seed'] or saved['label']!=record['label']: raise AssertionError('Packet provenance mismatch')
        if saved['fields'].shape!=(9,8,32,32) or saved['latent'].shape!=(16,4,8,8): raise AssertionError('Wrong Gaussian shape')
        reader=cv2.VideoCapture(str(args.out/record['video'])); count=0
        while True:
            ok,frame=reader.read()
            if not ok: break
            if frame.shape!=(64,64,3): raise AssertionError('Wrong video size')
            count+=1
        reader.release()
        if count!=8: raise AssertionError('Wrong frame count')
        sizes.append(count)
    report=dict(passed=True,checkpoint_reads_per_session=len(reads),permitted_tensor_inputs=reads,
                analysis_encoder_forbidden=True,repeat_after_intervening_call_exact=True,
                independent_eager_comparisons=comparisons,verified_packets=len(records),verified_videos=len(sizes),
                limitation='torch.load/encoder audit, not an operating-system sandbox; independent clips, not long-video recurrence.')
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with keep_windows_awake(): main()
