#!/usr/bin/env bash
#SBATCH --job-name=nemotron-profile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=02:00:00
#SBATCH --output=nemotron-profile-%j.out

set -euo pipefail

WORK="${WORK:-$HOME/work/nemotron-poc}"
export WORK
export CONTAINER_NAME="nemotron-profile-$SLURM_JOB_ID"

bash "$WORK/check_prerequisites.sh"
bash "$WORK/profile_batches.sh"
