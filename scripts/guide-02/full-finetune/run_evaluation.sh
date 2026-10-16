#!/usr/bin/env bash

# Run one model/manifest pair. Submit separate Slurm jobs for pretrained,
# POC v1, and the full-run candidate so each result has clear provenance.
set -euo pipefail

MODE="${1:?Usage: run_evaluation.sh streaming|offline RUN_NAME MANIFEST_HOST OUTPUT_DIR [MODEL_HOST]}"
RUN_NAME="${2:?Usage: run_evaluation.sh streaming|offline RUN_NAME MANIFEST_HOST OUTPUT_DIR [MODEL_HOST]}"
MANIFEST_HOST="${3:?Usage: run_evaluation.sh streaming|offline RUN_NAME MANIFEST_HOST OUTPUT_DIR [MODEL_HOST]}"
OUTPUT_DIR="${4:?Usage: run_evaluation.sh streaming|offline RUN_NAME MANIFEST_HOST OUTPUT_DIR [MODEL_HOST]}"
MODEL_HOST="${5:-pretrained}"

[[ "$MODE" == "streaming" || "$MODE" == "offline" ]] || {
  echo "MODE must be streaming or offline" >&2
  exit 1
}

WORK="${WORK:-$HOME/work/nemotron-poc}"
SOURCE="${SOURCE:-$HOME/work/nemo-speech-src}"
DATA="${DATA:-$HOME/data/nsc}"
CACHE="${CACHE:-$HOME/hf-cache}"
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-speech:26.07.00}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-24g}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-eval-${SLURM_JOB_ID:-manual}}"
MODEL_RELATIVE="models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo"

case "$MANIFEST_HOST" in
  "$DATA"/*) MANIFEST_CONTAINER="/data/nsc/${MANIFEST_HOST#"$DATA"/}" ;;
  *) echo "Manifest must be under $DATA" >&2; exit 1 ;;
esac

if [[ "$MODEL_HOST" == "pretrained" ]]; then
  MODEL_CONTAINER="/cache/$MODEL_RELATIVE"
else
  case "$MODEL_HOST" in
    "$WORK"/*) MODEL_CONTAINER="/work/${MODEL_HOST#"$WORK"/}" ;;
    *) echo "Local model must be under $WORK" >&2; exit 1 ;;
  esac
  [[ -f "$MODEL_HOST" ]] || { echo "Missing model: $MODEL_HOST" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_DIR"

common_docker_args=(
  --rm --gpus all
  --name "$CONTAINER_NAME"
  --memory "$CONTAINER_MEMORY"
  --memory-swap "$CONTAINER_MEMORY"
  --shm-size=2g
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$SOURCE:/nemo-src:ro"
  -v "$DATA:/data/nsc:ro"
  -v "$CACHE:/cache:ro"
  -v "$WORK:/work:ro"
  -v "$OUTPUT_DIR:/output"
)

echo "mode=$MODE"
echo "run_name=$RUN_NAME"
echo "manifest=$MANIFEST_HOST"
echo "model=$MODEL_HOST"
echo "output=$OUTPUT_DIR"

if [[ "$MODE" == "streaming" ]]; then
  docker run "${common_docker_args[@]}" "$IMAGE" \
    /opt/venv/bin/python \
      /nemo-src/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
      model_path="$MODEL_CONTAINER" \
      dataset_manifest="$MANIFEST_CONTAINER" \
      output_path=/output \
      batch_size=1 \
      target_lang=en-US \
      strip_lang_tags=true \
      'att_context_size=[56,3]' \
      decoder_type=rnnt \
      compare_vs_offline=false \
      random_seed=42 \
      amp=false \
      compute_dtype=float32 \
    2>&1 | tee "$OUTPUT_DIR/${RUN_NAME}-streaming.console.log"
else
  docker run "${common_docker_args[@]}" "$IMAGE" \
    /opt/venv/bin/python /work/baseline_eval.py \
      "$MANIFEST_CONTAINER" \
      --model-path "$MODEL_CONTAINER" \
      --run-name "$RUN_NAME" \
      --output-dir /output \
    2>&1 | tee "$OUTPUT_DIR/${RUN_NAME}-offline.console.log"
fi
