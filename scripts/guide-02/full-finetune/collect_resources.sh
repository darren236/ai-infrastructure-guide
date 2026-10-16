#!/usr/bin/env bash

# Record scheduler-node, GPU, and named-container memory every five seconds.
set -euo pipefail

CONTAINER_NAME="${1:?Usage: collect_resources.sh CONTAINER_NAME OUTPUT.csv [INTERVAL_SECONDS]}"
OUTPUT_PATH="${2:?Usage: collect_resources.sh CONTAINER_NAME OUTPUT.csv [INTERVAL_SECONDS]}"
INTERVAL_SECONDS="${3:-5}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "timestamp_utc,gpu_util_percent,gpu_memory_used_mib,gpu_memory_total_mib,container_memory_usage,container_memory_limit,container_memory_percent,host_memory_available_kib,host_memory_total_kib" > "$OUTPUT_PATH"

for _ in $(seq 1 120); do
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || {
  echo "Container $CONTAINER_NAME did not appear within 120 seconds" >&2
  exit 1
}

while docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; do
  gpu_values="$(nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  container_values="$(docker stats --no-stream \
    --format '{{.MemUsage}}|{{.MemPerc}}' "$CONTAINER_NAME" 2>/dev/null || true)"

  if [[ -n "$container_values" ]]; then
    IFS=',' read -r gpu_util gpu_used gpu_total <<< "$gpu_values"
    IFS='|' read -r memory_pair memory_percent <<< "$container_values"
    memory_usage="${memory_pair%% / *}"
    memory_limit="${memory_pair##* / }"
    host_available="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    host_total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$gpu_util" "$gpu_used" "$gpu_total" \
      "$memory_usage" "$memory_limit" "${memory_percent%%%}" \
      "$host_available" "$host_total" >> "$OUTPUT_PATH"
  fi

  sleep "$INTERVAL_SECONDS"
done
