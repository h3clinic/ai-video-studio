"""Exercise the real latent-generation CLI with tensor loads restricted to its checkpoint."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from .latent import GaussianCodec
from . import latent_generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cuda-graph',action='store_true')
    parser.add_argument('--precast-weights',action='store_true')
    args=parser.parse_args()
    allowed=args.checkpoint.resolve(); reads=[]
    original_load=torch.load
    def restricted_load(path,*positional,**kwargs):
        if not isinstance(path,(str,Path)) or Path(path).resolve()!=allowed:
            raise AssertionError(f'Unexpected tensor input: {path}')
        if kwargs.get('weights_only') is not True: raise AssertionError('Unsafe tensor loading')
        reads.append(str(Path(path).resolve()))
        return original_load(path,*positional,**kwargs)
    argv=['latent_generate',str(args.checkpoint),'--label','BabyCrawling','--seed','12345','--out',str(args.out)]
    if args.cuda_graph: argv.append('--cuda-graph')
    if args.precast_weights: argv.append('--precast-weights')
    with patch.object(sys,'argv',argv),patch('torch.load',side_effect=restricted_load),\
         patch.object(GaussianCodec,'encode',side_effect=AssertionError('Codec encoder invoked during generation')):
        latent_generate.main()
    result=dict(passed=True,tensor_inputs=reads,codec_encoder_forbidden=True,
                limitation='Audits torch.load input paths and encoder invocation; not a sandbox for every possible filesystem API.')
    (args.out/'inference_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
