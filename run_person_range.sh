#!/usr/bin/env bash
# One-shot start of the full person-range pipeline on Raspberry Pi:
# stereo cameras → depth → YOLO (person) → fusion → foxglove_bridge + rqt_image_view
#
# Usage:
#   ./run_person_range.sh
#   ./run_person_range.sh visualization:=true
#   ./run_person_range.sh rqt_image_view:=false
#
# Stop: Ctrl+C  (or: systemctl --user stop person-range-combined.service)
#
# Autostart on boot (user systemd + linger):
#   ./scripts/install_person_range_autostart.sh
#   systemctl --user status person-range-combined.service

set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ERROR: ROS 2 Jazzy not found at /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi
if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "ERROR: workspace not built. Run: cd $WS && colcon build --symlink-install" >&2
  exit 1
fi

# ROS/ament setup scripts read optional unset vars (e.g. AMENT_PREFIX_PATH).
# Under systemd the env is clean, so set -u must be off while sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

cd "$WS"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# Avoid two concurrent full-stack launches. When THIS process is the
# person-range-combined unit (PERSON_RANGE_SYSTEMD=1), do not stop ourselves.
if [[ "${PERSON_RANGE_SYSTEMD:-}" != "1" ]]; then
  systemctl --user stop person-range-combined.service 2>/dev/null || true
fi

# foxglove_bridge binds 0.0.0.0:8765 → all interfaces.
# wlan0 can be either a Pi-hosted hotspot (AP mode, short name "qqqq" via
# dnsmasq-shared.d) or a regular Wi-Fi client joined to some SSID (managed
# mode) — detect which, so the printed URL is never misleading.
# mDNS: qqqq.local (any same-L2 network with Avahi/Bonjour).
short_host="$(hostname -s 2>/dev/null || hostname)"
mdns_host="${short_host}.local"
wlan0_mode="$(iw dev wlan0 info 2>/dev/null | awk '/type/{print $2}')"
wlan0_ssid="$(nmcli -t -f DEVICE,CONNECTION dev status 2>/dev/null | awk -F: '$1=="wlan0"{print $2}')"
foxglove_urls=()
while read -r iface addr; do
  [[ -z "${addr}" || "${iface}" == "lo" ]] && continue
  label="${iface}"
  if [[ "${iface}" == "wlan0" ]]; then
    if [[ "${wlan0_mode}" == "AP" ]]; then
      label="wlan0/hotspot"
    else
      label="wlan0/wifi-client${wlan0_ssid:+ (${wlan0_ssid})}"
    fi
  fi
  foxglove_urls+=("${label}  ws://${addr}:8765")
done < <(ip -4 -o addr show scope global 2>/dev/null | awk '{gsub(/\/.*/, "", $4); print $2, $4}')

echo "=== person-range pipeline start $(date -Is) ==="
echo "  workspace: $WS"
echo "  DISPLAY:   $DISPLAY"
echo "  foxglove:  bind 0.0.0.0:8765  (all networks)"
if [[ "${wlan0_mode}" == "AP" ]]; then
  echo "             short   ws://${short_host}:8765   (hotspot DNS only)"
fi
echo "             mDNS    ws://${mdns_host}:8765"
if ((${#foxglove_urls[@]})); then
  for line in "${foxglove_urls[@]}"; do
    echo "             ${line}"
  done
else
  echo "             (no IPv4 yet — check eth0/wlan0)"
fi
echo "  (Ctrl+C to stop)"
echo

exec ros2 launch person_range_fusion person_range.launch.py "$@"
