#!/usr/bin/env bash

# Run inside a Slurm GPU allocation. This is a read-only gate: it does not
# download data, alter the NeMo checkout, or create training artifacts.
set -euo pipefail

WORK="${WORK:-$HOME/work/nemotron-poc}"
SOURCE="${SOURCE:-$HOME/work/nemo-speech-src}"
DATA="${DATA:-$HOME/data/nsc}"
CACHE="${CACHE:-$HOME/hf-cache}"

TRAIN_MANIFEST="$DATA/full/nemo/train_ready.jsonl"
DEV_MANIFEST="$DATA/full/nemo/dev_ready.jsonl"
MODEL_RELATIVE="models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.nemo"
MODEL_PATH="$CACHE/$MODEL_RELATIVE"

EXPECTED_TRAIN_HASH="e80e8795a2c009e12496bd3fafeb3b4ba0569f738d6a7f4f323eaed64dba69fe"
EXPECTED_DEV_HASH="a5a84863e904b093c814ee9742ef8de30e5032f7cb9d24c665965744bcd8680d"
EXPECTED_TRAIN_RECORDS=2152
EXPECTED_DEV_RECORDS=1203
MIN_MEMORY_KIB=$((30 * 1024 * 1024))
MIN_DISK_GIB=50
MIN_CPUS=4

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

check_manifest() {
  local path="$1"
  local expected_records="$2"
  local expected_hash="$3"
  local actual_records actual_hash

  actual_records="$(wc -l < "$path" | tr -d ' ')"
  actual_hash="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual_records" == "$expected_records" ]] || \
    fail "$path has $actual_records records; expected $expected_records"
  [[ "$actual_hash" == "$expected_hash" ]] || \
    fail "$path hash is $actual_hash; expected $expected_hash"
}

[[ -n "${SLURM_JOB_ID:-}" ]] || \
  fail "run this check inside the Slurm GPU allocation"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || \
  fail "Slurm did not expose a GPU to this allocation"

require_file "$SOURCE/scripts/speech_recognition/estimate_duration_bins.py"
require_file "$SOURCE/scripts/speech_recognition/oomptimizer.py"
require_file "$SOURCE/examples/asr/speech_to_text_finetune.py"
require_file "$SOURCE/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml"
require_file "$TRAIN_MANIFEST"
require_file "$DEV_MANIFEST"
require_file "$MODEL_PATH"

source_tag="$(git -C "$SOURCE" describe --tags --exact-match 2>/dev/null || true)"
[[ "$source_tag" == "v3.0.0" ]] || \
  fail "NeMo Speech checkout is '$source_tag'; expected exact tag v3.0.0"

check_manifest "$TRAIN_MANIFEST" "$EXPECTED_TRAIN_RECORDS" "$EXPECTED_TRAIN_HASH"
check_manifest "$DEV_MANIFEST" "$EXPECTED_DEV_RECORDS" "$EXPECTED_DEV_HASH"

memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
[[ "$memory_kib" -ge "$MIN_MEMORY_KIB" ]] || \
  fail "node has less than 30 GiB usable RAM; provision a 32 GiB-or-larger node"

cpus="$(nproc)"
[[ "$cpus" -ge "$MIN_CPUS" ]] || \
  fail "allocation exposes $cpus CPUs; expected at least $MIN_CPUS"

disk_gib="$(df --output=avail -BG "$WORK" | tail -n 1 | tr -dc '0-9')"
[[ "$disk_gib" -ge "$MIN_DISK_GIB" ]] || \
  fail "$WORK has ${disk_gib} GiB free; expected at least ${MIN_DISK_GIB} GiB"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | xargs)"
[[ "$gpu_memory_mib" -ge 23000 ]] || \
  fail "GPU has ${gpu_memory_mib} MiB VRAM; expected an L4-class 24 GB device"

echo "Prerequisite check passed."
echo "job_id=$SLURM_JOB_ID"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "cpus=$cpus"
echo "ram_gib=$((memory_kib / 1024 / 1024))"
echo "free_disk_gib=$disk_gib"
echo "gpu=$gpu_name"
echo "gpu_memory_mib=$gpu_memory_mib"
echo "nemo_speech_tag=$source_tag"
echo "train_records=$EXPECTED_TRAIN_RECORDS"
echo "dev_records=$EXPECTED_DEV_RECORDS"
