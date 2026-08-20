from diagnostic_msgs.msg import DiagnosticStatus

from person_range_fusion.diagnostics import _status_level
from person_range_fusion.diagnostics import decode_throttled
from person_range_fusion.diagnostics import parse_temperature_c
from person_range_fusion.diagnostics import parse_throttled_hex
from person_range_fusion.diagnostics import process_status
from person_range_fusion.diagnostics import ProcessSnapshot
from person_range_fusion.diagnostics import temperature_status


def test_parse_throttled_hex_variants() -> None:
    assert parse_throttled_hex('0x0') == 0
    assert parse_throttled_hex('throttled=0x50000') == 0x50000
    assert parse_throttled_hex('0x50005') == 0x50005


def test_decode_throttled_ok() -> None:
    status = decode_throttled('throttled=0x0')
    assert _status_level(status.level) == _status_level(DiagnosticStatus.OK)
    assert status.message == 'ok'
    assert status.name == 'power'


def test_decode_throttled_history_under_voltage_and_throttling() -> None:
    status = decode_throttled('0x50000')
    assert _status_level(status.level) == _status_level(DiagnosticStatus.WARN)
    assert 'under_voltage_occurred' in status.message
    assert 'throttled_occurred' in status.message
    by_key = {item.key: item.value for item in status.values}
    assert by_key['under_voltage_now'] == 'false'
    assert by_key['under_voltage_occurred'] == 'true'
    assert by_key['throttled_occurred'] == 'true'


def test_decode_throttled_active_under_voltage_is_error() -> None:
    # bit0 + history bits
    status = decode_throttled('0x50005')
    assert _status_level(status.level) == _status_level(DiagnosticStatus.ERROR)
    assert 'under_voltage_now' in status.message


def test_parse_temperature() -> None:
    assert parse_temperature_c("temp=48.3'C") == 48.3
    assert parse_temperature_c('temp=70.0C') == 70.0


def test_temperature_status_levels() -> None:
    assert _status_level(temperature_status(48.0).level) == _status_level(
        DiagnosticStatus.OK
    )
    assert _status_level(temperature_status(72.0).level) == _status_level(
        DiagnosticStatus.WARN
    )
    assert _status_level(temperature_status(81.0).level) == _status_level(
        DiagnosticStatus.ERROR
    )


def test_process_status_rss_warn() -> None:
    ok = process_status(
        ProcessSnapshot('yolo_detector', 1, 10.0, 800.0),
        rss_warn_mb=1000.0,
    )
    warn = process_status(
        ProcessSnapshot('yolo_detector', 1, 10.0, 1100.0),
        rss_warn_mb=1000.0,
    )
    assert _status_level(ok.level) == _status_level(DiagnosticStatus.OK)
    assert _status_level(warn.level) == _status_level(DiagnosticStatus.WARN)
