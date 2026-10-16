# Guide 03 — Scheduling GPU Workloads with Slurm

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md) · [Progress](../../PROGRESS.md)

## Goal

Learn Slurm by using it: inspect a cluster, submit CPU and GPU work, understand why jobs run or wait, read logs and states, and schedule an existing application. The single-node setup is supporting context for the exercises.

This guide reuses the NVIDIA L4 Brev node from Guides 01 and 02. The learning environment is intentionally small enough to show the complete path from a submitted job to the GPU on one machine.

<a id="hands-on-agenda"></a>

## Core learning path

1. [Read cluster state](#1-read-cluster-state) — use `sinfo` and `scontrol` to understand partitions, nodes, resources, and state. **Verified.**
2. [Read the configuration](#2-read-the-configuration) — understand the distinct roles of `slurm.conf`, `gres.conf`, and `cgroup.conf`. **Verified; no change was required.**
3. [Submit a CPU batch job](#3-cpu-batch-job) — use `sbatch`, follow the printed job ID, inspect the queue, and read the output file. **Verified.**
4. [Run interactively](#4-interactive-srun) — use `srun` to obtain an allocation and distinguish login-shell work from scheduled work. **Verified.**
5. [Submit a GPU job](#5-gpu-allocation) — request the L4 as a generic resource and verify its scheduler allocation. **Verified.**
6. [Observe contention](#6-gpu-contention) — hold the only GPU with one job and inspect why a second GPU job remains pending. **Verified.**
7. [Schedule the existing workload](#7-scheduled-nemotron-workload) — move a small Nemotron/NeMo smoke workload behind Slurm without modifying the existing Guide 02 helper. **Verified.**

The path starts with state and configuration because those explain later behavior. It ends with scheduled Nemotron inference, completing the guide's stated learning outcome.

## Environment assumption

This guide assumes a working Slurm cluster. Prefer a managed cluster or supported vendor/distribution packages; build packages from source only when a specific compatibility requirement justifies the maintenance burden.

The recorded node used Slurm `26.05.5` with MUNGE, one controller/worker host, unified cgroup v2, and one L4. Its source-package installation and service-account repair were environment-specific setup work, not part of the SA learning path. Git history retains those notes.

## 1. Read cluster state

| Component | Recorded state |
| --- | --- |
| Host | Ubuntu 22.04, kernel `6.8.0-1069-gcp`, `amd64`, systemd 249 |
| Control plane | `slurmctld` on the same node |
| Compute plane | `slurmd` on the same node |
| Authentication | MUNGE active |
| Scheduler version | Slurm `26.05.5` |
| CPU topology | 4 logical CPUs: 1 socket, 2 cores, 2 threads per core |
| Memory | 15,990 MiB detected; 12,288 MiB advertised to Slurm to retain host headroom |
| GPU | One NVIDIA L4 registered as `gpu:nvidia_l4:1` |
| Partition | `gpu`, default, one node, two-hour limit |
| Resource policy | An individual job is capped at three CPUs; cgroup v2 resource constraints enabled |
| Existing workload | NeMo Speech `26.07.00` image and Guide 02 launcher preserved |
| Accounting | Not configured or validated |
| Container integration | Docker validated in earlier guides; multi-GPU-aware binding remains untested |

`slurmd -C` reported the host topology and found the L4 through NVIDIA autodetection:

```text
CPUs=4 SocketsPerBoard=1 CoresPerSocket=2 ThreadsPerCore=2 RealMemory=15990
Gres=gpu:nvidia_l4:1
```

After the service-account issue was corrected and the controller and worker were restarted, `sinfo` reported one available idle node in the default `gpu` partition. That established a working control path before job submission began.

## 2. Read the configuration

The three compact files answer different questions:

| File | Responsibility | Change it when |
| --- | --- | --- |
| `/etc/slurm/slurm.conf` | Defines the cluster, nodes, partitions, advertised CPU/memory/GRES, and scheduling policy. | Nodes, partitions, resource capacity, limits, plugins, or cluster policy change. |
| `/etc/slurm/gres.conf` | Connects the logical GPU resource to the physical NVIDIA device, normally with NVML autodetection or an explicit device mapping. | GPUs are added, removed, replaced, retyped, remapped, or divided into supported MIG devices. |
| `/etc/slurm/cgroup.conf` | Defines how Linux cgroups enforce allocated CPU, memory, swap, and device access. | The cgroup version or cluster-wide isolation and memory/device enforcement policy changes. |

Individual workload requirements do not belong in these files. They belong in `#SBATCH` directives or `srun` options. The active files already supported this environment, so no configuration change was required merely to submit the exercises.

## 3. CPU batch job

The first job used an ordinary batch script:

```bash
#!/bin/bash
#SBATCH --job-name=cpu-hello
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=%x-%j.out

set -euo pipefail

echo "job_id=$SLURM_JOB_ID"
echo "job_name=$SLURM_JOB_NAME"
echo "hostname=$(hostname)"
echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
echo "started=$(date --iso-8601=seconds)"
sleep 30
echo "finished=$(date --iso-8601=seconds)"
```

The normal user workflow was deliberately kept simple:

```bash
sbatch cpu-hello.sbatch
squeue -j <job-id>
cat cpu-hello-<job-id>.out
```

`sbatch` printed the job ID and returned immediately. While the job was active, `squeue` showed it running on the node. The output file reached the final timestamp after 30 seconds. A later `squeue --me` was empty because `squeue` shows active work and the job had completed.

The CPU job ran in the `gpu` partition without allocating the GPU. A partition selects eligible nodes; a GPU must still be requested explicitly. Persistent completed-job state and exit-code history are not claimed because accounting is not configured and the controller purged the short-lived record.

## 4. Interactive `srun`

The interactive exercise requested a shell instead of submitting a script:

```bash
srun \
  --job-name=interactive-cpu \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=512M \
  --time=00:05:00 \
  --pty bash
```

The prompt looked unchanged because the login and compute roles are on the same host. Inside the new shell, the direct context check was:

```bash
echo $SLURM_JOB_ID
squeue -j $SLURM_JOB_ID
```

`SLURM_JOB_ID` was unset in the original SSH shell and set in the scheduled shell; `squeue` reported the interactive job running. `exit` ended the shell and released its allocation.

Use interactive `srun` for short inspection and troubleshooting. Use `sbatch` for repeatable work that should continue without an attached terminal.

## 5. GPU allocation

The GPU exercise added one explicit resource request:

```bash
srun \
  --job-name=gpu-check \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  --gres=gpu:nvidia_l4:1 \
  --time=00:05:00 \
  --pty bash
```

Inside the interactive shell, `SLURMD_NODENAME` identified the node executing the current task. This avoids querying or assuming a fixed node name when other nodes exist. One scheduler-side command verified the allocation:

```bash
scontrol show node -d "$SLURMD_NODENAME"
```

The relevant observed field was:

```text
GresUsed=gpu:nvidia_l4:1(IDX:0)
```

This establishes that Slurm allocated the configured L4, device index `0`, to the job. Repeating `SLURM_JOB_GPUS`, `CUDA_VISIBLE_DEVICES`, `nvidia-smi -L`, and several node views is unnecessary for the same scheduling claim. `nvidia-smi` or an application-level CUDA test answers a different question: whether the device is visible, utilized, or functional for a workload. Those lower-layer checks were already performed in earlier guides.

## 6. GPU contention

The interactive GPU job remained open to hold the only L4. From a second ordinary SSH shell, a batch job requested the same resource:

```bash
#!/bin/bash
#SBATCH --job-name=gpu-wait
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=00:02:00
#SBATCH --output=%x-%j.out

echo "job_id=$SLURM_JOB_ID"
echo "started=$(date --iso-8601=seconds)"
sleep 10
echo "finished=$(date --iso-8601=seconds)"
```

The job was submitted normally:

```bash
sbatch gpu-wait.sbatch
squeue --me
```

The observed queue state was:

```text
JOBID PARTITION NAME      USER    ST TIME NODES NODELIST(REASON)
12    gpu       gpu-wait  ubuntu  PD 0:00 1     (Resources)
11    gpu       gpu-chec  ubuntu  R  1:47 1     <node>
```

Job `11` held the only configured L4. Job `12` requested the same GPU and remained pending with reason `Resources`. That queue state is sufficient evidence that Slurm prevented the second job from receiving the already allocated GPU; no repeated device inspection is required.

<a id="8-scheduled-nemotron-workload"></a>

## 7. Scheduled Nemotron workload

The final exercise scheduled the already validated one-record Nemotron inference path rather than repeating training and creating another large checkpoint set. The Slurm wrapper requested the resources; the existing Guide 02 helper, data, and cache remained unchanged:

```bash
#!/bin/bash
#SBATCH --job-name=nemotron-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

echo "job_id=$SLURM_JOB_ID"
echo "node=$SLURMD_NODENAME"
echo "started=$(date --iso-8601=seconds)"

docker run --rm --gpus all \
  -v "$HOME/hf-cache:/root/.cache/huggingface" \
  -v "$HOME/data/nsc:/data/nsc:ro" \
  -v "$HOME/work/nemotron-poc:/work:ro" \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python /work/baseline_smoke.py \
    /data/nsc/poc/nemo/dev_50.jsonl

echo "finished=$(date --iso-8601=seconds)"
```

The operator submitted the wrapper with ordinary `sbatch`. The job reached the existing helper's `Smoke test completed.` message and the wrapper's final timestamp at `2026-10-15T08:58:36+00:00`. This connects the scheduler allocation to Docker, NeMo, model restoration, and one L4 transcription without changing the inference helper or input data.

This is a functional single-node integration result. Rootful Docker still uses `--gpus all`; on this one-GPU host that reaches the allocated L4, but it does not establish scheduler-aware container GPU binding on a multi-GPU cluster. No additional training, accuracy, latency, throughput, or production-readiness claim is made.

## What we learned

- Scheduler-visible resources are configured abstractions, so node configuration and physical inventory answer different questions.
- Selecting a partition does not allocate a GPU; the workload must request the generic resource explicitly.
- Interactive and batch work share the same allocation model even though their user experience differs.
- A pending reason of `Resources` explained the observed single-GPU contention without repeated device inspection.
- The scheduled Nemotron run connected allocation to real application execution, which completed the seven-step core journey.

<a id="7-lifecycle-control--deferred"></a>

## Further learning

### Lifecycle control

Cancellation and deliberate timeout exercises were not executed. They remain useful for learning `scancel`, `CANCELLED`, and `TIMEOUT`, but are not required for this guide's core outcome.

### Accounting and multi-GPU container integration

Persistent accounting is unconfigured, so no completed-job history is claimed from `sacct`. Rootful Docker uses `--gpus all`; a multi-GPU environment should test scheduler-aware device binding and isolation rather than infer them from this single-GPU result.

## Command vocabulary

| Command | Plain meaning |
| --- | --- |
| `sinfo` | Summarize partitions and node state. |
| `scontrol show node -d "$SLURMD_NODENAME"` | From an interactive job, show configured and allocated resources on the task's current node, including detailed GRES use. |
| `sbatch job.sh` | Submit a script and return immediately with a job ID. |
| `squeue` | Show queued and running jobs, including pending reasons. |
| `srun ...` | Start a scheduled task, either inside an allocation or by requesting one. |
| `sacct -j <job-id>` | Show completed job state, elapsed time, and exit code once accounting is available. |
| `scancel <job-id>` | Cancel a pending or running job. |

## Boundaries

The core guide is complete through scheduled Nemotron inference. This is a functional single-node learning environment, not a production cluster design. It does not validate high availability, shared storage, multi-node networking, multi-user isolation, identity integration, fair-share policy, reservations, backfill behavior under sustained load, operational monitoring, or disaster recovery. The controller and worker share one failure domain, and local data and the root filesystem are on the same underlying volume.

The CPU batch, interactive allocation, GPU allocation, observed two-job contention, and scheduled Nemotron inference are operator-reported. GPU allocation is not GPU utilization, and the contention result covers only the configured single-GPU case. Persistent accounting, cancellation/timeout behavior, scheduler-aware container GPU binding, and production workload behavior remain unvalidated.

The existing Nemotron inference helper and training launcher remain unchanged. The scheduled smoke job reused the existing one-record inference path and did not run a new training experiment. Cancellation, timeout, persistent accounting, and multi-GPU-aware container integration remain optional exercises if deeper Slurm operations are the next priority.
