#!/usr/bin/env python3
"""Stop the full person-range pipeline (launch + nodes)."""

from __future__ import annotations

import os
import signal
import subprocess
import time


MATCH_SUBSTR = (
    'person_range_fusion person_range.launch.py',
    'run_person_range.sh',
    '/install/foxglove_bridge/lib/foxglove_bridge/foxglove_bridge',
    '/install/person_range_fusion/lib/person_range_fusion/',
    '/install/yolo_person_car/lib/yolo_person_car/detector',
    '/install/camera_ros/lib/camera_ros/camera_node',
    'stereo_sgbm_light_container',
    'stereo_left_up_tf',
    'person_range_image_view',
    'person_range_rqt_graph',
    'person_range_rqt_image_view',
    'person_range_view',
)


def cmdline_of(pid: int) -> str:
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            raw = fh.read()
        return raw.replace(b'\x00', b' ').decode('utf-8', errors='replace')
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ''


def matching_pids(self_pid: int) -> list[int]:
    found: list[int] = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        cmd = cmdline_of(pid)
        if not cmd:
            continue
        if any(token in cmd for token in MATCH_SUBSTR):
            found.append(pid)
    return sorted(found)


def signal_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
            print(f'  {sig.name} -> pid {pid}')
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f'  cannot signal pid {pid}: {exc}')


def main() -> int:
    # Optional systemd unit
    subprocess.run(
        ['systemctl', '--user', 'stop', 'person-range-combined.service'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    self_pid = os.getpid()
    pids = matching_pids(self_pid)
    if not pids:
        print('No person-range processes found.')
    else:
        print(f'Sending SIGTERM to {len(pids)} process(es)...')
        signal_pids(pids, signal.SIGTERM)
        time.sleep(2.0)
        leftover = matching_pids(self_pid)
        if leftover:
            print(f'Sending SIGKILL to {len(leftover)} leftover process(es)...')
            signal_pids(leftover, signal.SIGKILL)
            time.sleep(0.5)

    still = matching_pids(self_pid)
    port = subprocess.run(
        ['ss', '-ltn'],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    port_open = ':8765' in port
    print('=== shutdown status ===')
    print(f'pipeline processes left: {len(still)}')
    for pid in still:
        print(f'  still running: {pid} {cmdline_of(pid)[:120]}')
    print(f'foxglove :8765: {"LISTEN" if port_open else "free"}')
    return 1 if still or port_open else 0


if __name__ == '__main__':
    raise SystemExit(main())
