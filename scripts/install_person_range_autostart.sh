#!/usr/bin/env bash
# Install + enable full person-range pipeline as a user systemd service on boot.
#
# Usage:
#   ./scripts/install_person_range_autostart.sh
#   ./scripts/install_person_range_autostart.sh --no-start   # enable only, do not restart now
#
# Requires: linger for this user (enabled here via loginctl, may ask sudo).

set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$WS/systemd/person-range-combined.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DST="$UNIT_DIR/person-range-combined.service"
DO_START=1

for arg in "$@"; do
  case "$arg" in
    --no-start) DO_START=0 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$UNIT_SRC" ]]; then
  echo "ERROR: missing unit file: $UNIT_SRC" >&2
  exit 1
fi
if [[ ! -x "$WS/run_person_range.sh" ]]; then
  chmod +x "$WS/run_person_range.sh"
fi
if [[ ! -f /opt/ros/jazzy/setup.bash || ! -f "$WS/install/setup.bash" ]]; then
  echo "ERROR: ROS/workspace not ready. Build first: colcon build --symlink-install" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
cp -f "$UNIT_SRC" "$UNIT_DST"
echo "Installed: $UNIT_DST"

# Linger: user systemd starts at boot without login (needed for headless reboot).
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)" != "yes" ]]; then
  echo "Enabling linger for user $USER (starts user services at boot)..."
  if loginctl enable-linger "$USER" 2>/dev/null; then
    :
  else
    sudo loginctl enable-linger "$USER"
  fi
fi
echo "Linger: $(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo '?')"

systemctl --user daemon-reload
systemctl --user enable person-range-combined.service
echo "Enabled: person-range-combined.service (WantedBy=default.target)"

if [[ "$DO_START" -eq 1 ]]; then
  # Stop any manual launch so the service owns the stack.
  if [[ -x "$WS/scripts/stop_person_range.py" ]]; then
    python3 "$WS/scripts/stop_person_range.py" 2>/dev/null || true
  fi
  systemctl --user restart person-range-combined.service
  sleep 2
  systemctl --user --no-pager --full status person-range-combined.service || true
  echo
  echo "Autostart active. Logs: journalctl --user -u person-range-combined -f"
else
  echo "Enabled only (not started). Start with:"
  echo "  systemctl --user start person-range-combined.service"
fi

echo
echo "Disable autostart:"
echo "  systemctl --user disable --now person-range-combined.service"
echo "  # optional: loginctl disable-linger $USER"
