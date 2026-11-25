#!/usr/bin/env bash
set -euo pipefail
CHANNEL="${1:-can0}"
sudo ip link set "$CHANNEL" down 2>/dev/null || true
sudo ip link set "$CHANNEL" type can bitrate 1000000
sudo ip link set "$CHANNEL" up
ip -details link show "$CHANNEL"
