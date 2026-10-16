#!/usr/bin/env bash

# Run either the independent 50-step resource pilot or one resumable segment
# of the 1,000-step full experiment. The full run always resumes the same ID.
set -euo pipefail

MODE="${1:-pilot}"
[[ "$MODE" == "pilot" || "$MODE" == "full" ]] || {
  echo "Usage: bash run_training.sh pilot|full" >&2
  exit 1
}

WORK="${WORK:-$HOME/work/nemotron-poc}"
SOURCE="${SOURCE:-$HOME/work/nemo-speech-src}"
DATA="${DATA:-$HOME/data/nsc}"
CACHE="${CACHE:-$HOME/hf-cache}"
IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-speech:26.07.00}"
PROFILE_ENV="${PROFILE_ENV:-$WORK/results/full_finetune/profile/batch-profile.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK/results/full_finetune}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-24g}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-${MODE}-${SLURM_JOB_ID:-manual}}"

[[ -f "$PROFILE_ENV" ]] || {
  echo "Missing $PROFILE_ENV; run the OOMptimizer profile job first" >&2
  exit 1
}

# shellcheck disable=SC1090
source "$PROFILE_ENV"
: "${BUCKET_DURATION_BINS:?profile has no BUCKET_DURATION_BINS}"
: "${BUCKET_BATCH_SIZE:?profile has no BUCKET_BATCH_SIZE}"

MODEL_RELATIVE="models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo"
MODEL_PATH="/cache/$MODEL_RELATIVE"

if [[ "$MODE" == "pilot" ]]; then
  MAX_STEPS=50
  VAL_INTERVAL=50
  EXPERIMENT_ID="${EXPERIMENT_ID:-resource-pilot-v2}"
  RESUME=false
  EARLY_STOPPING=false
  MAX_TIME_PER_RUN=null
else
  MAX_STEPS=1000
  VAL_INTERVAL=100
  EXPERIMENT_ID="${EXPERIMENT_ID:-full-v2}"
  RESUME=true
  EARLY_STOPPING=true
  MAX_TIME_PER_RUN="00:01:45:00"
fi

mkdir -p "$OUTPUT_ROOT"
RUN_DIR="$OUTPUT_ROOT/nemotron_singlish/$EXPERIMENT_ID"

echo "=== EXPERIMENT ==="
echo "mode=$MODE"
echo "experiment_id=$EXPERIMENT_ID"
echo "max_steps=$MAX_STEPS"
echo "validation_interval=$VAL_INTERVAL"
echo "resume=$RESUME"
echo "profile=$PROFILE_ENV"
echo "profile_sha256=$(sha256sum "$PROFILE_ENV" | awk '{print $1}')"
echo "run_dir=$RUN_DIR"

docker run --rm --gpus all \
  --name "$CONTAINER_NAME" \
  --memory "$CONTAINER_MEMORY" \
  --memory-swap "$CONTAINER_MEMORY" \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env TRAIN_SEED=42 \
  -v "$SOURCE:/nemo-src:ro" \
  -v "$DATA:/data/nsc:ro" \
  -v "$CACHE:/cache:ro" \
  -v "$WORK:/work:ro" \
  -v "$OUTPUT_ROOT:/results" \
  "$IMAGE" \
  /opt/venv/bin/python /work/run_seeded_finetune.py \
    --config-path=/nemo-src/examples/asr/conf/fastconformer/cache_aware_streaming \
    --config-name=fastconformer_transducer_bpe_streaming_prompt \
    "+init_from_nemo_model=$MODEL_PATH" \
    model.train_ds.manifest_filepath=/data/nsc/full/nemo/train_ready.jsonl \
    model.validation_ds.manifest_filepath=/data/nsc/full/nemo/dev_ready.jsonl \
    model.train_ds.is_tarred=false \
    +model.train_ds.batch_size=null \
    model.train_ds.batch_duration=null \
    model.train_ds.quadratic_duration=null \
    model.train_ds.num_buckets=null \
    "+model.train_ds.bucket_duration_bins=$BUCKET_DURATION_BINS" \
    "model.train_ds.bucket_batch_size=$BUCKET_BATCH_SIZE" \
    model.train_ds.max_duration=39.99 \
    model.train_ds.num_workers=2 \
    model.train_ds.default_prompt_mode=langID \
    model.validation_ds.batch_size=1 \
    model.validation_ds.num_workers=1 \
    +model.validation_ds.default_prompt_mode=langID \
    model.optim.name=adamw \
    model.optim.lr=1e-5 \
    "model.optim.betas=[0.9,0.98]" \
    model.optim.weight_decay=0.001 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=20 \
    model.optim.sched.warmup_ratio=null \
    model.optim.sched.min_lr=1e-6 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    trainer.sync_batchnorm=false \
    trainer.precision=bf16-true \
    trainer.max_epochs=-1 \
    trainer.max_steps="$MAX_STEPS" \
    trainer.val_check_interval="$VAL_INTERVAL" \
    trainer.limit_train_batches=1.0 \
    +trainer.limit_val_batches=1.0 \
    trainer.accumulate_grad_batches=1 \
    trainer.gradient_clip_val=0.5 \
    trainer.log_every_n_steps=10 \
    exp_manager.exp_dir=/results \
    exp_manager.name=nemotron_singlish \
    "++exp_manager.version=$EXPERIMENT_ID" \
    ++exp_manager.use_datetime_version=false \
    exp_manager.resume_if_exists="$RESUME" \
    exp_manager.resume_ignore_no_checkpoint=true \
    +exp_manager.resume_past_end=false \
    exp_manager.checkpoint_callback_params.monitor=val_wer \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.checkpoint_callback_params.save_top_k=3 \
    +exp_manager.checkpoint_callback_params.save_last=true \
    exp_manager.checkpoint_callback_params.always_save_nemo=true \
    "++exp_manager.checkpoint_callback_params.filename=nemotron-singlish-step{step}" \
    "+exp_manager.create_early_stopping_callback=$EARLY_STOPPING" \
    +exp_manager.early_stopping_callback_params.monitor=val_wer \
    +exp_manager.early_stopping_callback_params.mode=min \
    +exp_manager.early_stopping_callback_params.patience=3 \
    +exp_manager.early_stopping_callback_params.min_delta=0.001 \
    "+exp_manager.max_time_per_run=$MAX_TIME_PER_RUN" \
  2>&1 | tee -a "$OUTPUT_ROOT/${MODE}-${EXPERIMENT_ID}.console.log"

echo "Segment finished. Inspect the log and checkpoint directory before resubmitting."
