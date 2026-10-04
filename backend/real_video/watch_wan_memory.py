"""Bounded low-RAM watchdog for our own active Wan runner, never other tasks."""
import argparse
import json
from pathlib import Path
import time
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    if 'real_video.wan_baseline' not in process.cmdline():
        raise ValueError('Target is not our Wan baseline runner')
    created = process.create_time()
    low_count = 0
    record = dict(pid=args.pid, created=created, min_available_bytes=None, peak_process_rss_bytes=0,
                  threshold_bytes=2*1024**3, stopped_for_low_memory=False)
    try:
        for _ in range(3600):  # Two-hour bound, not a recurring monitor.
            if not process.is_running() or process.create_time() != created:
                break
            available = psutil.virtual_memory().available
            record['min_available_bytes'] = min(record['min_available_bytes'] or available, available)
            record['peak_process_rss_bytes'] = max(record['peak_process_rss_bytes'], process.memory_info().rss)
            low_count = low_count + 1 if available < record['threshold_bytes'] else 0
            if low_count >= 3:
                process.terminate()
                record['stopped_for_low_memory'] = True
                break
            args.out.write_text(json.dumps(record, indent=2), encoding='utf-8')
            time.sleep(2)
    except psutil.NoSuchProcess:
        pass
    finally:
        args.out.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
