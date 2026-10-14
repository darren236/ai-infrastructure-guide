# Guide 04 — Single-node Slurm learning lab

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md) · [Progress](../../PROGRESS.md)

## Goal

Learn Slurm by using it: inspect a cluster, submit CPU and GPU work, understand why jobs run or wait, read logs and exit states, and troubleshoot controlled failures. The single-node setup is supporting context, not the main exercise.

This lab reuses the NVIDIA L4 Brev node from Guides 01 and 02. It is intentionally small enough to see the complete path from a submitted job to the GPU on one machine.

## Hands-on agenda

| Stage | Learning outcome | Current status |
| --- | --- | --- |
| 1. Read cluster state | Use `sinfo` and `scontrol` to understand partitions, nodes, resources, and state. | Initial `sinfo` check complete |
| 2. Read the configuration | Use Vim to inspect and edit the three small configuration files, understanding each directive instead of pasting an opaque block. | Next |
| 3. Submit a CPU batch job | Write an `sbatch` script, request CPU/RAM/time, and inspect output and exit state. | Pending |
| 4. Run interactively | Use `srun` to obtain an allocation and distinguish login-shell work from scheduled work. | Pending |
| 5. Submit a GPU job | Request the L4 as a generic resource and verify the allocation from inside the job. | Pending |
| 6. Observe contention | Hold the only GPU with one job and inspect why a second GPU job remains pending. | Pending |
| 7. Control the lifecycle | Cancel a job, trigger a time limit, and interpret pending reasons and failure/exit states. | Pending |
| 8. Schedule the existing workload | Move a small Nemotron/NeMo smoke workload behind Slurm without changing the preserved Guide 02 launcher. | Pending |

The agenda starts with state and configuration because those explain later behavior. It does not require building Slurm from source as a normal learning step.

## Installation strategy

For a normal learning or customer environment, prefer this order:

1. Use an existing managed Slurm cluster when available.
2. Otherwise use supported vendor or distribution packages whose version matches the operating system and required features.
3. Build packages from source only when a concrete compatibility requirement justifies the extra maintenance.

This lab used the third path as an environment-specific exception. Ubuntu 22.04 offered an older Slurm release for the chosen setup, while the node uses unified cgroup v2. Slurm `26.05.5` was therefore downloaded from SchedMD, its published SHA-256 was verified, and upstream Debian packages were built and installed. The package build completed with exit code `0`.

That work establishes the lab; it is not a recommended beginner installation ritual. Package building, dependency resolution, and service-account repair are summarized here so the rest of the guide can focus on using the scheduler.

## Lab architecture and observed checkpoint

| Component | Recorded state |
| --- | --- |
| Host | Ubuntu 22.04, kernel `6.8.0-1069-gcp`, `amd64`, systemd 249 |
| Control plane | `slurmctld` on the same node |
| Compute plane | `slurmd` on the same node |
| Authentication | MUNGE active |
| Scheduler version | Slurm `26.05.5`, installed from locally built upstream Debian packages |
| CPU topology | 4 logical CPUs: 1 socket, 2 cores, 2 threads per core |
| Memory | 15,990 MiB detected; 12,288 MiB advertised to Slurm to retain host headroom |
| GPU | One NVIDIA L4 registered as `gpu:nvidia_l4:1` |
| Partition | `gpu`, default, one node, two-hour limit |
| Resource policy | An individual job is capped at three CPUs; cgroup v2 resource constraints enabled |
| Existing workload | NeMo Speech `26.07.00` image and Guide 02 launcher preserved |
| Accounting | MariaDB and `slurmdbd` packages installed; Slurm accounting is not configured or validated |
| Container integration | Docker works from earlier guides; scheduler-aware container execution remains a later exercise |

The three deliberately small configuration files are:

- `/etc/slurm/slurm.conf` — cluster identity, controller, node resources, scheduling policy, and partition.
- `/etc/slurm/gres.conf` — maps the physical L4 device to Slurm's GPU generic resource.
- `/etc/slurm/cgroup.conf` — asks Slurm to constrain allocated CPU, memory, swap, and devices through cgroup v2.

The next exercise uses Vim to review these files and type any changes. This makes the meaning of the directives visible without turning configuration generation into the subject of the lab.

## Evidence recorded so far

`slurmd -C` reported the host topology and found the L4 through NVIDIA autodetection:

```text
CPUs=4 SocketsPerBoard=1 CoresPerSocket=2 ThreadsPerCore=2 RealMemory=15990
Gres=gpu:nvidia_l4:1
```

After the service-account issue was corrected and the controller and worker were restarted, `sinfo` reported one available idle node in the default `gpu` partition:

```text
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
gpu*         up    2:00:00      1   idle <single-node hostname>
```

This establishes that the controller can read the configuration, the worker has registered, and the partition can accept work. It does not yet establish successful CPU scheduling, GPU isolation, accounting, container execution under Slurm, or workload completion. No Slurm job has been submitted at this checkpoint.

## Troubleshooting lessons from setup

### Do not diagnose a cluster before it has a configuration

Before the configuration was installed, `scontrol` attempted configless discovery and failed its DNS SRV lookup; `slurmd` also failed to start. Those results described an incomplete setup, not a scheduler defect.

### A service name in configuration must resolve to a real account

The installed packages did not create the expected `slurm` service account in this environment. The first restart therefore failed with `Invalid user for SlurmUser slurm`. Creating a locked, non-login system account, assigning the controller state directory to it, and restarting `slurmctld` and `slurmd` resolved the problem.

The useful diagnostic chain was short:

1. Read the first concrete error.
2. Correct the missing prerequisite it named.
3. Restart the two relevant services.
4. Use one functional check, `sinfo`, to confirm the result.

Repeated status commands would not have added evidence after the node reached `idle`.

## Command vocabulary for the workload exercises

| Command | Plain meaning |
| --- | --- |
| `sinfo` | Summarize partitions and node state. |
| `scontrol show node` | Show the resources and detailed state Slurm knows for a node. |
| `sbatch job.sh` | Submit a script and return immediately with a job ID. |
| `squeue` | Show queued and running jobs, including pending reasons. |
| `srun ...` | Start a scheduled task, either inside an allocation or by requesting one. |
| `sacct -j <job-id>` | Show completed job state, elapsed time, and exit code once accounting is available. |
| `scancel <job-id>` | Cancel a pending or running job. |

Flags will be introduced only when an exercise needs them. Each command in this guide will explain what it does and what its flags mean.

## Boundaries

This is a functional single-node learning lab, not a production cluster design. It does not validate high availability, shared storage, multi-node networking, multi-user isolation, identity integration, fair-share policy, reservations, backfill behavior under sustained load, operational monitoring, or disaster recovery. The controller and worker share one failure domain, and local data and the root filesystem are on the same underlying volume.

The existing Nemotron launcher remains unchanged. Moving that workload behind Slurm will happen only after small CPU and GPU jobs make allocation behavior understandable.

## Next step

Open the installed configuration in Vim, review the active values against the architecture table, and then write the first small CPU batch script. The first workload milestone is complete only when `sbatch`, `squeue`, the output file, and the final state tell one consistent story.
