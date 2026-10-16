# Full-data Nemotron Singlish fine-tuning

This is the resource-gated follow-on to the historical 300-record / 100-step POC. It keeps NVIDIA NeMo Speech v3.0.0, container `nvcr.io/nvidia/nemo-speech:26.07.00`, the original `.nemo` checkpoint, and NVIDIA's `speech_to_text_finetune.py`. It does not use NeMo AutoModel and does not replace the earlier POC launcher.

The files are implemented but **not yet executed on the required 32 GiB-RAM node**. Do not publish result claims or copy generated artifacts into Git until the pilot, full run, development selection, and reused-test evaluation are complete.

## Why this design

| Decision | Setting | Reason |
| --- | --- | --- |
| Product role | Dedicated Singlish model | NSC-only adaptation is intentionally routed separately; the untouched base checkpoint remains available for general-English and multilingual traffic. |
| Framework | NeMo Speech 3.0 official trainer | This FastConformer-RNNT `.nemo` checkpoint has a documented NeMo Speech adaptation path. AutoModel does not provide a corresponding recipe. |
| Node | 1 L4 24 GB, at least 32 GiB RAM, 4 CPUs | Two 10 GiB-container attempts on the 12 GiB Slurm node were killed by the container memory cgroup. The larger node supplies responsible host headroom. |
| Data | 2,152 train / 4.50 h; 1,203 dev / 2.65 h | These are all decoder-compatible, duration-eligible recordings after the predeclared exclusions. Test data is not used during training or selection. |
| Batching | 12 data-derived duration bins plus OOMptimizer batch sizes | About 179 recordings per initial bin balances a small dataset with useful length grouping. Batch sizes come from the actual model/GPU, not manual `batch_duration` guesses. |
| OOMptimizer reserve | 85% of L4 VRAM, no DDP simulation | Training uses one GPU with `strategy=auto`; the reserve leaves space for runtime variability. The pilot still has to show at least 10% VRAM headroom. |
| Pilot | Independent 50-step run | It validates training, validation startup, loss, RAM, VRAM, and utilization. It is not resumed into the full run because its 50-step cosine scheduler has a different state. |
| Full stopping | `max_steps=1000`, validation every 100 steps, early-stop patience 3 | Step-based stopping is stable with dynamic batches. This allows ten validation opportunities and stops after three sub-0.001 improvements. |
| Optimizer | AdamW, `1e-5`, betas `[0.9,0.98]`, weight decay `0.001` | It preserves the recipe's AdamW regularization while using the conservative learning rate already shown to be stable in the POC. |
| Schedule | Cosine to `1e-6`, 20 warmup steps | Twenty steps is 2% of the 1,000-step ceiling and avoids an abrupt full-rate start. |
| Numeric/trainer | `bf16-true`, accumulation 1, clip `0.5`, seed 42 | L4 supports BF16; the remaining values keep optimizer steps interpretable and retain the recipe's clipping. |
| Data behavior | SpecAugment retained, `en-US` / `langID`, 39.99 s maximum | These preserve the Nemotron prompt-aware recipe and the already-audited input boundary. |
| Checkpoints | Best three by `val_wer` plus last | This supports recovery and development-only candidate comparison without using test results for selection. |
| Segmentation | 1h45 NeMo timer inside a 2h Slurm job | NeMo saves `last.ckpt` before the scheduler limit, and the next submission resumes the fixed `full-v2` experiment. |

The 24 GiB Docker RAM limit leaves 25% of a 32 GiB allocation outside the container. The resource pilot is the empirical gate: continue only when it completes, validation starts, training loss is finite, host-RAM headroom stays at or above 20%, and VRAM headroom stays at or above 10%.

## Inputs and exclusions

| Split | Eligible after `<unk>` removal | Decoder failures | Over 39.99 s | Ready | Ready duration | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train | 2,158 | 3 | 3 | 2,152 | 4.50 h | `e80e8795a2c009e12496bd3fafeb3b4ba0569f738d6a7f4f323eaed64dba69fe` |
| Development | 1,225 | 21 | 1 | 1,203 | 2.65 h | `a5a84863e904b093c814ee9742ef8de30e5032f7cb9d24c665965744bcd8680d` |

The reviewed per-record list is in [the exclusions table](../../../docs/evidence/guide-02-full-finetune-exclusions.csv). Raw audio and manifests remain outside Git.

## Files

| File | Purpose |
| --- | --- |
| `check_prerequisites.sh` | Fails closed unless it is inside a Slurm allocation with the pinned source, expected manifests/hashes, at least four CPUs, approximately 32 GiB RAM, 50 GiB free disk, and an L4-class GPU. |
| `profile_batches.sh` | Runs NeMo's duration-bin estimator and OOMptimizer, then writes `batch-profile.env`. |
| `nemotron-profile.sbatch` | Requests the profiling allocation. |
| `run_seeded_finetune.py` | Seeds the process and delegates to NVIDIA's unmodified official fine-tuning entrypoint. |
| `run_training.sh` | Runs the independent resource pilot or one resumable full-training segment. |
| `collect_resources.sh` | Samples GPU utilization/VRAM, container RAM, and host RAM every five seconds. |
| `nemotron-train.sbatch` | Requests the training allocation and couples training with resource sampling. |
| `summarize_resources.py` | Enforces the pilot RAM/VRAM headroom gate and creates the resource plot. |
| `export_training_curves.py` | Extracts TensorBoard scalars and creates training/validation curves by optimizer step. |
| `run_evaluation.sh` / `nemotron-eval.sbatch` | Runs one native 320 ms streaming or historical offline evaluation per job. |
| `compare_predictions.py` | Applies the fixed normalization, computes corpus WER, paired bootstrap intervals, the development acceptance gate, and a WER chart. |
| `freeze_candidate.py` | Copies and hashes a development-selected candidate before any reused-test run. |

## Execution order and shell context

Every `sbatch` command below is entered in the **ordinary SSH/login shell**, not inside an interactive `srun` shell. The scripts themselves run inside the scheduled allocation.

### 1. Stage the reviewed scripts

From a checkout on the compute host:

```bash
cd ~/ai-infrastructure-guide
cp scripts/guide-02/full-finetune/* ~/work/nemotron-poc/
cp scripts/guide-02/baseline_eval.py ~/work/nemotron-poc/
chmod +x ~/work/nemotron-poc/*.sh ~/work/nemotron-poc/*.sbatch
bash -n ~/work/nemotron-poc/*.sh ~/work/nemotron-poc/*.sbatch
```

This copy is a deployment step. Editing the repository does not automatically change `/home/ubuntu/work/nemotron-poc`.

### 2. Generate the batch profile

```bash
cd ~/work/nemotron-poc
sbatch nemotron-profile.sbatch
squeue --me
```

After completion, inspect `nemotron-profile-<job-id>.out` and `results/full_finetune/profile/batch-profile.env`. Do not hand-edit the generated profile.

### 3. Run the independent resource pilot

```bash
cd ~/work/nemotron-poc
sbatch --export=ALL,MODE=pilot nemotron-train.sbatch
```

After the job completes, summarize its telemetry inside the NeMo container (replace `<job-id>`):

```bash
docker run --rm \
  -v "$HOME/work/nemotron-poc:/work" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  /opt/venv/bin/python /work/summarize_resources.py \
    "/work/results/full_finetune/telemetry/pilot-<job-id>.csv" \
    --output-dir /work/results/full_finetune/pilot-report
```

Proceed only when the JSON reports `pilot_gate.passed: true`, the job exit status is zero, the log reaches validation, and loss remains finite.

### 4. Run resumable full segments

```bash
cd ~/work/nemotron-poc
sbatch --export=ALL,MODE=full nemotron-train.sbatch
```

Each later submission uses the same command and resumes `nemotron_singlish/full-v2`. Submit one segment at a time after verifying the prior job ended cleanly. Stop when NeMo reaches 1,000 steps or early stopping fires; do not resubmit a completed experiment.

### 5. Evaluate and freeze

Run separate jobs for pretrained, POC v1, and the new candidate on the development manifest. The primary condition is `MODE=streaming`; repeat with `MODE=offline` for continuity. `run_evaluation.sh` fixes native cache-aware streaming to `att_context_size=[56,3]`, which is 320 ms.

Compare the three development prediction files with `compare_predictions.py`, using:

```text
--dataset development
--condition streaming_320ms
--candidate full-v2
--must-beat pretrained
--must-beat poc-v1
```

Only if that gate passes, run `freeze_candidate.py`. Then evaluate the frozen copy on NSC test. Label that dataset `reused_nsc_test`: earlier test results are already known, so it is not a pristine confirmatory test and it cannot change the selected checkpoint or hyperparameters.

## Required final evidence

- resolved Hydra configuration and generated batch-profile hash;
- pilot resource summary/plot and finite loss/validation evidence;
- resumed segment logs, checkpoint list, selected `.nemo` hash, and training curves;
- development streaming and offline comparisons for pretrained, POC v1, and candidate;
- reused-test streaming and offline comparisons only after freezing;
- paired-bootstrap intervals, WER charts, exclusions table, and limitations;
- raw audio, checkpoints, caches, full logs, and raw predictions kept outside Git.

## Recorded failed capacity experiments

Both attempts ran in a Docker container capped at 10 GiB on the original 12 GiB Slurm node. The first used `batch_duration=40`; the second used `batch_duration=20`. Both reached five training steps and exited `137`. The kernel identified a memory-cgroup OOM killing Python; the first report showed approximately 9.83 GiB anonymous RSS plus file/shared memory. Because halving batch duration did not move the failure point, these are capacity-planning evidence, not justification for further parameter guessing on that node.
