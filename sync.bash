#!/bin/bash
set -euo pipefail

HOST="172.27.35.101"
USER="jaewoong"
PORT="22"

EXCLUDES=(
    ".git/"
    ".mypy_cache/"
    ".DS_Store"
    "Metadata/"
)
EXCLUDE_ARGS=()
for e in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=("--exclude=${e}")
done

WATCH_DIRS=(
    "Program"
)
REMOTE_DIR="/home/Live/jaewoong/JeongLab_Onboarding/"

on_interrupt() {
    exit 0
}
trap 'on_interrupt' INT

sync_dir() {
    echo "[SYNC] $1 at $(date '+%F %T')"
    time rsync --checksum --archive --times --recursive --compress --progress --port "${PORT}" "${EXCLUDE_ARGS[@]}" "$(realpath .)"/"$1" "${USER}@${HOST}:$2"
}

clear
for i in "${!WATCH_DIRS[@]}"; do
    sync_dir "${WATCH_DIRS[$i]}" "${REMOTE_DIR}"
done

while read -r _; do
    clear
    for i in "${!WATCH_DIRS[@]}"; do
        sync_dir "${WATCH_DIRS[$i]}" "${REMOTE_DIR}"
    done
done < <(fswatch --one-per-batch "${WATCH_DIRS[@]}")
