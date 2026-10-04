"""Explicit, bounded remote visual planning trial; never mutates Gaussian state."""
import hashlib
import json
import time
import argparse
from pathlib import Path
from real_video.gemini_edit_planner import propose_with_saved_key
from real_video.gaussian_program import Program, evidence, ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic-retry', action='store_true')
    parser.add_argument('--compact-schema', action='store_true')
    args=parser.parse_args()
    out = ROOT / ('artifacts/cloud/gemini_review_diagnostic_v1' if args.diagnostic_retry else 'artifacts/cloud/gemini_review_v1')
    if args.compact_schema:out=ROOT/'artifacts/cloud/gemini_review_compact_v1'
    out.mkdir(exist_ok=False)  # Never silently repeat a billed experiment.
    source = ROOT / 'artifacts/cloud/object_edit_remote_v2/results/output'
    inventory = {
        name: dict(parent=None, relation='unrelated', protected=protected,
                   binding_verified=False, revision=0)
        for name, protected in [('orange', False), ('apple', True),
                                ('donkey', True), ('bowl', True), ('background', True)]
    }
    instruction = (
        'Review replacing the orange and all its related residual parts with an apple. '
        'Each image is a comparison: LEFT is the original Gaussian reconstruction; '
        'RIGHT is the current apple edit. Assess leftovers on the RIGHT against the LEFT. '
        'Preserve the donkey, bowl, environment and replacement apple. '
        'The inventory names are provisional human labels, NOT verified segmentation. '
        'Describe unbound parts in unknown_parts with frame IDs and image locations. '
        'Inspect visually; do not assume a part exists just because the system lists it. '
        'No edits or new video generation are authorized by this planning call.'
    )
    frames = []
    paths = []
    for i in (0, 20, 39):
        path = source / f'comparison_{i:03d}.png'
        data = path.read_bytes()
        frames.append(dict(index=i, mime='image/png', bytes=data))
        paths.append(dict(index=i, path=str(path.relative_to(ROOT)),
                          sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    inputs = dict(model='gemini-3.5-flash-lite', instruction=instruction,
                  inventory=inventory, target='orange', frames=paths,
                  limits=dict(calls=1, max_output_tokens=4096, timeout_seconds=60),
                  research=dict(source='https://arxiv.org/html/2311.14521v2',
                      locator='Equation 6 and section 4.3',
                      finding='Semantic labels require opacity/transmittance-weighted projection; masked removal still requires boundary repair.',
                      limitation='A language proposal is not a verified dense mask or temporal binding.'))
    (out / 'inputs.json').write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    started = time.perf_counter()
    try:
        result = propose_with_saved_key(model=inputs['model'], instruction=instruction,
                                        target='orange', inventory=inventory, frames=frames)
        outcome = 'partial'
        observation = 'Live Gemini structured visual proposal received; execution remains blocked pending verified Gaussian bindings and residual masks.'
    except RuntimeError as error:
        result = dict(error=str(error), execution_performed=False)
        outcome = 'failed_execution'
        observation = str(error)
    result['elapsed_seconds'] = time.perf_counter() - started
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    program = Program()
    try:
        program.record_experiment(dict(role='evaluation', issue_key='rendering.related_part_elimination',
            outcome=outcome, hypothesis='A bounded remote visual planner can identify residual object parts beyond a fixed fruit ROI while preserving receivers.',
            observation=observation, next_action='Verify proposed parts with temporal segmentation and Gaussian ownership before any removal; no complete visual fix claimed.',
            inputs=[evidence(ROOT, out / 'inputs.json')], outputs=[evidence(ROOT, out / 'result.json')]))
        program.export()
    finally:
        program.db.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
