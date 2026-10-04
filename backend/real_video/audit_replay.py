"""Run the source-free Gaussian replay CLI with video reads forbidden."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from . import replay_fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coefficients',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    allowed=args.coefficients.resolve(); reads=[]; original=torch.load
    def restricted(path,*positional,**kwargs):
        # Atomic checkpoint validation reads its own freshly written .pending file.
        resolved=Path(path).resolve()
        validation=resolved.parent==args.out.resolve() and resolved.name.endswith('.pending')
        if resolved!=allowed and not validation: raise AssertionError('Unexpected tensor input')
        if kwargs.get('weights_only') is not True: raise AssertionError('Unsafe tensor input')
        reads.append(str(resolved)); return original(path,*positional,**kwargs)
    argv=['replay_fit','--replay',str(args.coefficients),'--render-only','--out',str(args.out)]
    with patch.object(sys,'argv',argv),patch('torch.load',side_effect=restricted),\
         patch.object(replay_fit,'read_video',side_effect=AssertionError('Source video read')),\
         patch.object(replay_fit,'fit',side_effect=AssertionError('Fit invoked during replay')):
        replay_fit.main()
    result=dict(passed=True,source_video_read_forbidden=True,fit_forbidden=True,tensor_reads=reads,
                limitation='Checks instrumented read/fit paths; not an OS-level filesystem sandbox.')
    (args.out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
