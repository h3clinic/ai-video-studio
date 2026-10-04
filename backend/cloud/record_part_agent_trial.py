"""Record bounded part-agent development; never promote an image to a video."""
import argparse
import json
from pathlib import Path
from real_video.gaussian_program import Program, ROOT, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = Path(args.report).resolve()
    data = json.loads(report.read_text(encoding='utf-8'))
    program = Program()
    try:
        program.record_experiment(dict(
            role='rendering', issue_key='rendering.part_owned_apple_detail', outcome='partial',
            hypothesis='Explicit part ownership plus separate apple material references avoids one red replacement blob and unrelated scene writes.',
            observation='Original AgentVideo protocol and atomic Gaussian ownership implemented and tested. Image API outcome is recorded in the attached report. No new accepted video, 3D fitting or temporal identity verified in this trial.',
            next_action='Verify generated part references, fit real 3D apple/slice/peel assets remotely, bind reviewed Gaussian IDs, then evaluate contact and citrus-residue removal across the clip.',
            inputs=[evidence(ROOT, ROOT / p) for p in (
                'real_video/agentvideo_parts.py', 'real_video/gaussian_part_ownership.py',
                'research/sweep_2026-10-04_part_agents.json')],
            outputs=[evidence(ROOT, report)]))
        program.export()
    finally:
        program.db.close()
    print(json.dumps({'recorded': True, 'video_accepted': False}))


if __name__ == '__main__':
    main()
