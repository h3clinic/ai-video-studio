"""Export verified inference weights without optimizer, duplicate train weights or RNG state."""
import argparse
import json
from pathlib import Path
from .checkpoint_io import load_verified,save_inference_checkpoint,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=load_verified(args.checkpoint)
    keep={'codec','codec_config','model','config','mean','std','labels','latent_mean','latent_std','latent_shape',
          'step','validation','scope','completed_updates','validation_reconstruction_psnr_db'}
    exported={k:v for k,v in checkpoint.items() if k in keep}
    exported['training_checkpoint_sha256']=digest(args.checkpoint)
    checksum=save_inference_checkpoint(exported,args.out)
    print(json.dumps(dict(path=str(args.out),sha256=checksum,bytes=args.out.stat().st_size,
                         removed='Optimizer, duplicate non-EMA training weights and RNG state; keep original checkpoint to resume training'),indent=2))


if __name__=='__main__': main()
