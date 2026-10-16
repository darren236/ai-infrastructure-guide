#!/usr/bin/env bash
#SBATCH --job-name=nemotron-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=02:00:00
#SBATCH --output=nemotron-train-%j.out

set -euo pipefail

MODE="${MODE:-pilot}"
WORK="${WORK:-$HOME/work/nemotron-poc}"
TELEMETRY_DIR="$WORK/results/full_finetune/telemetry"

export MODE WORK
export CONTAINER_NAME="nemotron-${MODE}-$SLURM_JOB_ID"

bash "$WORK/check_prerequisites.sh"

mkdir -p "$TELEMETRY_DIR"
telemetry_path="$TELEMETRY_DIR/${MODE}-${SLURM_JOB_ID}.csv"

bash "$WORK/collect_resources.sh" \
  "$CONTAINER_NAME" "$telemetry_path" 5 &
monitor_pid=$!

set +e
bash "$WORK/run_training.sh" "$MODE"
training_status=$?
wait "$monitor_pid"
monitor_status=$?
set -e

echo "training_exit_status=$training_status"
echo "telemetry_exit_status=$monitor_status"
echo "telemetry=$telemetry_path"
if [[ "$training_status" -ne 0 ]]; then
  exit "$training_status"
fi
exit "$monitor_status"
