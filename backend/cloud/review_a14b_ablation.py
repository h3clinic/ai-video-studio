"""Extract hash-bound review frames from the completed frozen ablation."""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/cloud/a14b_remote_eval_v2/results/evaluation'
OUT = RUN.parent.parent / 'review'
INDICES = (0, 8, 16, 24, 32)


def main():
    OUT.mkdir(exist_ok=False)
    report = json.loads((RUN / 'report.json').read_text())
    records = []
    for seed in report['seeds']:
        board = Image.new('RGB', (416 * 5, 266 * 3), 'white')
        draw = ImageDraw.Draw(board)
        for row, branch in enumerate(report['branches']):
            name = f'{branch}_{seed}'
            path = RUN / report['videos'][name]['file']
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != report['videos'][name]['sha256']:
                raise ValueError('Video hash mismatch')
            reader = imageio.get_reader(path)
            frames = []
            try:
                for col, index in enumerate(INDICES):
                    frame = Image.fromarray(reader.get_data(index))
                    target = OUT / f'{name}_{index:03d}.png'
                    frame.save(target)
                    frames.append(dict(path=str(target), frame_index=index))
                    board.paste(frame.resize((416, 240)), (col * 416, row * 266 + 26))
                    draw.text((col * 416 + 5, row * 266 + 6), f'{branch} {index / 16:.1f}s', fill='black')
            finally:
                reader.close()
            records.append(dict(name=name, video=str(path), sha256=actual, frames=frames))
        board.save(OUT / f'comparison_{seed}.jpg')
    (OUT / 'extraction.json').write_text(json.dumps(records, indent=2))
    print(str(OUT))


if __name__ == '__main__':
    main()
