#!/usr/bin/env bash
#SBATCH --job-name=nemotron-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=02:00:00
#SBATCH --output=nemotron-eval-%j.out

set -euo pipefail

: "${MODE:?Export MODE=streaming or MODE=offline}"
: "${RUN_NAME:?Export RUN_NAME}"
: "${MANIFEST_HOST:?Export MANIFEST_HOST}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR}"

WORK="${WORK:-$HOME/work/nemotron-poc}"
MODEL_HOST="${MODEL_HOST:-pretrained}"
export WORK
export CONTAINER_NAME="nemotron-eval-$SLURM_JOB_ID"

bash "$WORK/check_prerequisites.sh"
bash "$WORK/run_evaluation.sh" \
  "$MODE" "$RUN_NAME" "$MANIFEST_HOST" "$OUTPUT_DIR" "$MODEL_HOST"
