"""Resume only the verified suspended Wan run, recording the timing interruption."""
import json
from pathlib import Path
import time
import psutil


def main():
    directory = Path('artifacts/real_video/wan_baseline/cat_seed_421001')
    process = psutil.Process(5572)
    if abs(process.create_time() - 1790718413.4741056) > .01 or 'real_video.wan_baseline' not in process.cmdline():
        raise RuntimeError('Owned process identity does not match; do not resume a reused PID')
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged and battery.percent < 20:
        raise RuntimeError('Battery is still too low for this restart')
    if process.status() != psutil.STATUS_STOPPED:
        raise RuntimeError('Expected suspended process')
    record = dict(resumed_at_unix=time.time(), pid=process.pid, created=process.create_time(),
                  battery=battery._asdict() if battery else None,
                  progress_before_resume=json.loads((directory/'progress.json').read_text()),
                  timing_warning='Denoise elapsed time includes a long suspension. Not an inference speed benchmark.')
    with (directory/'resume.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    process.resume()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
