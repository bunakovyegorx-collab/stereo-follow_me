"""Lightweight Pi diagnostics publisher for host-side Foxglove monitoring."""

from __future__ import annotations

import subprocess
from typing import Callable

import psutil
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from person_range_fusion.diagnostics import ProcessSnapshot
from person_range_fusion.diagnostics import _level_field
from person_range_fusion.diagnostics import _status_level
from person_range_fusion.diagnostics import decode_throttled
from person_range_fusion.diagnostics import parse_temperature_c
from person_range_fusion.diagnostics import process_status
from person_range_fusion.diagnostics import temperature_status


# Match patterns against process cmdline substrings.
DEFAULT_PROCESS_MATCHERS: dict[str, str] = {
    'sgbm_container': 'stereo_sgbm_light_container',
    'yolo_detector': 'yolo_person_car/detector',
    'fusion': 'person_range_fusion/fusion',
}


def sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def run_vcgencmd(args: list[str]) -> str:
    result = subprocess.run(
        ['vcgencmd', *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=1.0,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or 'vcgencmd failed').strip()
        raise RuntimeError(detail)
    return result.stdout.strip()


class PiDiagnosticsNode(Node):
    """Publish power/thermal/process diagnostics once per second."""

    def __init__(
        self,
        *,
        vcgencmd: Callable[[list[str]], str] = run_vcgencmd,
        process_matchers: dict[str, str] | None = None,
    ) -> None:
        super().__init__('pi_diagnostics')
        self._vcgencmd = vcgencmd
        self.declare_parameter('period_sec', 1.0)
        self.declare_parameter('temp_warn_c', 70.0)
        self.declare_parameter('temp_error_c', 80.0)
        self.declare_parameter('yolo_rss_warn_mb', 1000.0)
        self.declare_parameter('fusion_rss_warn_mb', 400.0)
        self.declare_parameter('sgbm_rss_warn_mb', 300.0)

        period = float(self.get_parameter('period_sec').value)
        if period <= 0.0:
            raise ValueError('period_sec must be positive')
        self._temp_warn = float(self.get_parameter('temp_warn_c').value)
        self._temp_error = float(self.get_parameter('temp_error_c').value)
        self._rss_warn = {
            'yolo_detector': float(self.get_parameter('yolo_rss_warn_mb').value),
            'fusion': float(self.get_parameter('fusion_rss_warn_mb').value),
            'sgbm_container': float(self.get_parameter('sgbm_rss_warn_mb').value),
        }
        self._matchers = process_matchers or dict(DEFAULT_PROCESS_MATCHERS)
        self._cpu_primed = False

        self._pub = self.create_publisher(
            DiagnosticArray, 'diagnostics', sensor_qos(),
        )
        self.create_timer(period, self._tick)
        self.get_logger().info(
            'pi diagnostics ready: period=%.1fs temp_warn=%.0f temp_error=%.0f'
            % (period, self._temp_warn, self._temp_error)
        )

    def _tick(self) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(self._power_status())
        array.status.append(self._thermal_status())
        array.status.extend(self._process_statuses())
        self._pub.publish(array)

    def _power_status(self) -> DiagnosticStatus:
        try:
            raw = self._vcgencmd(['get_throttled'])
            return decode_throttled(raw)
        except Exception as exc:  # noqa: BLE001 - keep diagnostics alive
            status = DiagnosticStatus()
            status.name = 'power'
            status.hardware_id = 'raspberry_pi_5'
            status.level = _level_field(_status_level(DiagnosticStatus.ERROR))
            status.message = f'vcgencmd get_throttled failed: {exc}'
            return status

    def _thermal_status(self) -> DiagnosticStatus:
        try:
            raw = self._vcgencmd(['measure_temp'])
            temp_c = parse_temperature_c(raw)
            return temperature_status(
                temp_c, warn_c=self._temp_warn, error_c=self._temp_error,
            )
        except Exception as exc:  # noqa: BLE001
            status = DiagnosticStatus()
            status.name = 'thermal'
            status.hardware_id = 'raspberry_pi_5'
            status.level = _level_field(_status_level(DiagnosticStatus.ERROR))
            status.message = f'temperature read failed: {exc}'
            return status

    def _process_statuses(self) -> list[DiagnosticStatus]:
        # First call to cpu_percent often returns 0.0; prime once.
        if not self._cpu_primed:
            for proc in psutil.process_iter(['pid']):
                try:
                    proc.cpu_percent(interval=None)
                except (psutil.Error, TypeError):
                    pass
            self._cpu_primed = True

        found: dict[str, ProcessSnapshot] = {}
        for proc in psutil.process_iter(['pid', 'cmdline', 'memory_info']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline') or [])
                if not cmdline:
                    continue
                for label, needle in self._matchers.items():
                    if needle in cmdline and label not in found:
                        rss_mb = float(proc.info['memory_info'].rss) / (1024.0 * 1024.0)
                        found[label] = ProcessSnapshot(
                            label=label,
                            pid=int(proc.info['pid']),
                            cpu_percent=float(proc.cpu_percent(interval=None)),
                            rss_mb=rss_mb,
                        )
            except (psutil.Error, TypeError, KeyError, AttributeError):
                continue

        statuses: list[DiagnosticStatus] = []
        for label in self._matchers:
            if label not in found:
                missing = DiagnosticStatus()
                missing.name = f'process/{label}'
                missing.hardware_id = 'missing'
                missing.level = _level_field(_status_level(DiagnosticStatus.WARN))
                missing.message = f'{label} not running'
                missing.values = [KeyValue(key='matcher', value=self._matchers[label])]
                statuses.append(missing)
                continue
            statuses.append(
                process_status(
                    found[label],
                    rss_warn_mb=self._rss_warn.get(label, 1024.0),
                )
            )
        return statuses


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PiDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
