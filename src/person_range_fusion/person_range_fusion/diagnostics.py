"""Pure diagnostics helpers for Raspberry Pi person-range pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue


def _status_level(value: object) -> int:
    """Normalize diagnostic level constants (int or single-byte) to int."""
    if isinstance(value, (bytes, bytearray)):
        return int(value[0])
    return int(value)


def _level_field(level: int) -> bytes:
    """ROS diagnostic_msgs expects level as a single-byte value."""
    return bytes([int(level) & 0xFF])


THROTTLE_BITS: dict[int, tuple[str, int]] = {
    0: ('under_voltage_now', _status_level(DiagnosticStatus.ERROR)),
    1: ('freq_capped_now', _status_level(DiagnosticStatus.WARN)),
    2: ('throttled_now', _status_level(DiagnosticStatus.ERROR)),
    3: ('soft_temp_limit_now', _status_level(DiagnosticStatus.WARN)),
    16: ('under_voltage_occurred', _status_level(DiagnosticStatus.WARN)),
    17: ('freq_capped_occurred', _status_level(DiagnosticStatus.WARN)),
    18: ('throttled_occurred', _status_level(DiagnosticStatus.WARN)),
    19: ('soft_temp_limit_occurred', _status_level(DiagnosticStatus.WARN)),
}

_TEMP_RE = re.compile(r"temp=([0-9.]+)'?C")


def parse_throttled_hex(raw: str) -> int:
    """Parse `vcgencmd get_throttled` output into an integer bitmask."""
    text = raw.strip().lower()
    if text.startswith('throttled='):
        text = text.split('=', 1)[1]
    if text.startswith('0x'):
        return int(text, 16)
    return int(text, 0)


def decode_throttled(raw: str) -> DiagnosticStatus:
    """Map a throttled hex string to a DiagnosticStatus."""
    value = parse_throttled_hex(raw)
    status = DiagnosticStatus()
    status.name = 'power'
    status.hardware_id = 'raspberry_pi_5'
    active: list[str] = []
    level = _status_level(DiagnosticStatus.OK)
    for bit, (name, bit_level) in THROTTLE_BITS.items():
        is_set = bool(value & (1 << bit))
        status.values.append(KeyValue(key=name, value='true' if is_set else 'false'))
        if is_set:
            active.append(name)
            level = max(level, bit_level)
    status.level = _level_field(level)
    status.message = ', '.join(active) if active else 'ok'
    return status


def parse_temperature_c(raw: str) -> float:
    """Parse `vcgencmd measure_temp` output into Celsius."""
    match = _TEMP_RE.search(raw.strip())
    if not match:
        raise ValueError(f'unrecognized temperature string: {raw!r}')
    return float(match.group(1))


def temperature_status(
    temp_c: float,
    *,
    warn_c: float = 70.0,
    error_c: float = 80.0,
) -> DiagnosticStatus:
    """Build thermal DiagnosticStatus from a Celsius reading."""
    if warn_c >= error_c:
        raise ValueError('expected warn_c < error_c')
    status = DiagnosticStatus()
    status.name = 'thermal'
    status.hardware_id = 'raspberry_pi_5'
    status.values = [KeyValue(key='temp_c', value=f'{temp_c:.1f}')]
    if temp_c >= error_c:
        status.level = _level_field(_status_level(DiagnosticStatus.ERROR))
        status.message = f'{temp_c:.1f}C >= error {error_c:.1f}C'
    elif temp_c >= warn_c:
        status.level = _level_field(_status_level(DiagnosticStatus.WARN))
        status.message = f'{temp_c:.1f}C >= warn {warn_c:.1f}C'
    else:
        status.level = _level_field(_status_level(DiagnosticStatus.OK))
        status.message = f'{temp_c:.1f}C ok'
    return status


@dataclass(frozen=True)
class ProcessSnapshot:
    """One process resource sample."""

    label: str
    pid: int
    cpu_percent: float
    rss_mb: float


def process_status(
    snapshot: ProcessSnapshot,
    *,
    rss_warn_mb: float,
) -> DiagnosticStatus:
    """Build per-process DiagnosticStatus from a resource snapshot."""
    status = DiagnosticStatus()
    status.name = f'process/{snapshot.label}'
    status.hardware_id = f'pid:{snapshot.pid}'
    status.values = [
        KeyValue(key='pid', value=str(snapshot.pid)),
        KeyValue(key='cpu_percent', value=f'{snapshot.cpu_percent:.1f}'),
        KeyValue(key='rss_mb', value=f'{snapshot.rss_mb:.1f}'),
        KeyValue(key='rss_warn_mb', value=f'{rss_warn_mb:.1f}'),
    ]
    if snapshot.rss_mb >= rss_warn_mb:
        status.level = _level_field(_status_level(DiagnosticStatus.WARN))
        status.message = (
            f'{snapshot.label} RSS {snapshot.rss_mb:.0f}MB >= warn {rss_warn_mb:.0f}MB'
        )
    else:
        status.level = _level_field(_status_level(DiagnosticStatus.OK))
        status.message = (
            f'{snapshot.label} cpu={snapshot.cpu_percent:.0f}% '
            f'rss={snapshot.rss_mb:.0f}MB'
        )
    return status
