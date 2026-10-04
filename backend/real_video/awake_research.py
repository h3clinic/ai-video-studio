"""Temporary, bounded sleep inhibitor; Ctrl-C releases the Windows request."""
import time
from .checkpoint_io import keep_windows_awake

if __name__ == '__main__':
    with keep_windows_awake():
        time.sleep(7200)
