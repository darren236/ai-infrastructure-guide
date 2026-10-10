#!/usr/bin/env bash

# Guide 02 — NVIDIA Nemotron 3.5 ASR Fine-Tuning
#
# Uses NVIDIA's official speech_to_text_finetune.py.
#
# Modes:
#   smoke : 2 optimizer steps
#   pilot : 100 optimizer steps; prepared, execution not yet confirmed
#   full  : configurable optimizer steps
#
# Usage:
#   bash run_nsc_train.sh smoke
#   bash run_nsc_train.sh pilot
#   bash run_nsc_train.sh full 600
#
# Reconstructed from the supplied command history, including both Hydra fixes.
# Pilot case matches the operator-supplied October 10 host launcher section.
# Not compared byte-for-byte with the Brev file or independently run on an L4.
# Original datasets, NVIDIA source, and cached model are mounted read-only.

# All modes use the same official restoration/training pipeline.
# Smoke limits validation to two batches; full uses a validation fraction of 1.0.
# Pilot checks every 50 training batches, using the full validation dataloader.
# Accumulation inherits 1 from the audited source YAML; confirm resolved config.
# The full 600-step example is illustrative and has not been executed.
set -euo pipefail

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    STEPS=2
    VAL_INTERVAL=2
    VAL_BATCHES=2
    LOG_INTERVAL=1
    ;;

  pilot)
    STEPS=100
    VAL_INTERVAL=50
    VAL_BATCHES=1.0
    LOG_INTERVAL=10
    ;;

  full)
    STEPS="${2:?Specify optimizer steps: full <steps>}"
    VAL_INTERVAL=1.0
    VAL_BATCHES=1.0
    LOG_INTERVAL=10
    ;;

  *)
    echo "Usage: bash run_nsc_train.sh smoke|pilot|full [steps]"
    exit 1
    ;;
esac

WORK="$HOME/work/nemotron-poc"
SOURCE="$HOME/work/nemo-speech-src"
DATA="$HOME/data/nsc"
CACHE="$HOME/hf-cache"

# Restore the original pretrained .nemo, rather than start a new architecture.
# This observed HF snapshot must exist under the mounted host cache.
# The path may need to be updated on a new node; it is not universally portable.
MODEL_PATH="/root/.cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo"

RUN_ID="${MODE}-${STEPS}-$(date -u +%Y%m%dT%H%M%SZ)"
# UTC run names keep outputs separate from earlier runs (second resolution).
OUTPUT="$WORK/results/official_finetune/$RUN_ID"

mkdir -p "$OUTPUT"

echo "=== EXPERIMENT ==="
echo "Mode       : $MODE"
echo "Steps      : $STEPS"
echo "Container  : nemo-speech:26.07.00"
echo "Output     : $OUTPUT"

# --gpus all exposes the L4. Source, data, and cache are read-only inputs;
# /results is writable and survives --rm through the persistent host mount.
# NVIDIA source and YAML stay unchanged; Hydra supplies experiment settings.
# "~model.optim.sched" deletes the scheduler key for constant LR 1e-5.
# Setting it to null retains a key that NeMo tries to modify and failed here.
# + adds a missing key; ++ adds or overrides. Keep +trainer.limit_val_batches.
# Checkpoints: $OUTPUT/nemotron_nsc/$RUN_ID/checkpoints/; console log: $OUTPUT.
# pipefail preserves a Docker failure even though output passes through tee.
docker run --rm --gpus all \
  -v "$SOURCE:/nemo-src:ro" \
  -v "$DATA:/data/nsc:ro" \
  -v "$CACHE:/root/.cache/huggingface:ro" \
  -v "$OUTPUT:/results" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /nemo-src/examples/asr/speech_to_text_finetune.py \
    --config-path=/nemo-src/examples/asr/conf/fastconformer/cache_aware_streaming \
    --config-name=fastconformer_transducer_bpe_streaming_prompt \
    "+init_from_nemo_model=$MODEL_PATH" \
    model.train_ds.manifest_filepath=/data/nsc/poc/nemo/train_300.jsonl \
    model.validation_ds.manifest_filepath=/data/nsc/poc/nemo/dev_50.jsonl \
    model.train_ds.is_tarred=false \
    model.train_ds.batch_duration=null \
    model.train_ds.quadratic_duration=null \
    +model.train_ds.batch_size=1 \
    model.train_ds.max_duration=39.99 \
    model.train_ds.num_workers=0 \
    model.train_ds.default_prompt_mode=langID \
    model.validation_ds.batch_size=1 \
    model.validation_ds.num_workers=0 \
    +model.validation_ds.default_prompt_mode=langID \
    model.optim.lr=1e-5 \
    '~model.optim.sched' \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.sync_batchnorm=false \
    trainer.precision=bf16-mixed \
    trainer.max_steps="$STEPS" \
    trainer.val_check_interval="$VAL_INTERVAL" \
    +trainer.limit_val_batches="$VAL_BATCHES" \
    trainer.log_every_n_steps="$LOG_INTERVAL" \
    exp_manager.exp_dir=/results \
    exp_manager.name=nemotron_nsc \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    "++exp_manager.version=$RUN_ID" \
    ++exp_manager.use_datetime_version=false \
  2>&1 | tee "$OUTPUT/console.log"
