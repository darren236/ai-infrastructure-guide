# Lab 01 — GPU Node Validation

## Part 1: Brev connectivity, Linux host, PCIe, and NVIDIA driver

**Status:** Part 1 complete; CUDA userspace, containers, and frameworks remain unvalidated.

**Milestone recorded:** 2026-10-01. This is the documentation sync date.

**Evidence basis:** The lab operator's sanitized milestone summary of commands performed and observations. Raw terminal recordings and instance connection details are excluded.

### Objective and outcome

Establish working access to a cloud GPU node and validate the host and GPU driver before moving up to CUDA, containers, or training frameworks.

Connected to a Brev-managed VM after resolving an SSH access issue, identified the Linux environment, confirmed NVIDIA PCIe device enumeration, and successfully queried an NVIDIA L4 through the NVIDIA driver. These checks establish host visibility and driver communication. CUDA workload execution remains unvalidated.

### Environment

| Component | Reported value |
| --- | --- |
| Provisioning and access platform | Brev cloud GPU VM |
| Underlying cloud | GCP, as reported by the lab operator; the kernel flavor is consistent with that environment |
| GPU | NVIDIA L4, 24 GB class VRAM |
| GPU memory reported by `nvidia-smi` | 23034 MiB |
| Architecture | x86_64 |
| Operating system | Ubuntu 22.04.5 LTS (Jammy) |
| Linux kernel | `6.8.0-1069-gcp` |
| NVIDIA driver | `595.91.07` |
| Driver-reported CUDA compatibility level | `13.2` |
| Installed CUDA Toolkit and userspace runtime | Not yet validated |

### Validation workflow

The following summarizes the completed workflow and retains only commands relevant to reproducing the checks.

1. **Establish cloud access.** Validated the locally installed Brev CLI, authenticated to Brev, and selected the intended organization. Attempted access using `brev shell`.
2. **Recover connectivity.** Encountered an SSH connection/configuration issue. Running `brev refresh` triggered reauthentication and refreshed connection configuration; a subsequent `brev shell` connection succeeded. This records the observed recovery sequence without claiming an independently isolated root cause.
3. **Confirm the execution context.** On the remote VM, checked the machine, user, and working directory before continuing:

   ```bash
   hostname
   whoami
   pwd
   ```

   Instance-specific identity values are omitted from this write-up.

4. **Capture the session.** Started terminal recording with `script`. Recordings can include authentication and connection information and are excluded from Git.
5. **Inspect the GPU and host.** Ran the following commands on the remote VM:

   ```bash
   nvidia-smi
   uname -a
   cat /etc/os-release
   lspci | grep -i nvidia
   ```

   `nvidia-smi` identified the L4 and returned GPU telemetry. The host commands established the OS, kernel, and architecture. `lspci` confirmed that the OS enumerated an NVIDIA device on the PCIe bus.

### Observed GPU state

| Check | Observation at validation time |
| --- | --- |
| Driver detection | NVIDIA L4 detected successfully |
| GPU memory in use | 0 MiB |
| GPU utilization | 0% |
| Temperature | Approximately 35°C |
| Power draw / cap | Approximately 12 W / 72 W |
| Running GPU processes | None reported |
| Uncorrectable ECC errors | None reported |

These readings describe an idle snapshot. They do not establish sustained thermal behavior, performance under load, CUDA execution, or long-term hardware reliability. No benchmark, training run, or application deployment is claimed.

### Driver compatibility, runtime, and Toolkit

The `CUDA Version: 13.2` field in `nvidia-smi` describes the CUDA compatibility level supported by the installed driver. It does **not** establish that CUDA Toolkit 13.2 is installed on the host, identify an application's CUDA runtime, or demonstrate successful CUDA execution. NVIDIA documents this distinction in its [driver and CUDA matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).

| Layer | Role | Validation status |
| --- | --- | --- |
| NVIDIA driver | Provides GPU access; `nvidia-smi` reports driver-level compatibility and telemetry. | Driver communication validated |
| CUDA runtime / userspace | Supplies libraries used by CUDA applications; versions depend on the selected environment. | Not yet validated |
| CUDA Toolkit | Supplies development tools, including `nvcc`, and libraries for building CUDA software. | Not yet validated |

Application compatibility also depends on the hardware, required features, and applicable [CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Record the actual runtime and Toolkit separately in the next step.

### Infrastructure lessons

- **Verify context first.** `hostname`, `whoami`, and `pwd` help establish which machine, account, and directory a command is using.
- **Troubleshoot bottom-up.** Use the sequence `hardware → PCIe enumeration → kernel → NVIDIA driver → CUDA userspace → container runtime → framework → application` to isolate failures by layer.
- **Distinguish enumeration from driver access.** `lspci` shows PCIe device visibility; `nvidia-smi` demonstrates driver-mediated GPU communication. If enumeration succeeds but `nvidia-smi` fails, investigate the driver, kernel module, and device access before CUDA or PyTorch.
- **Validate the access plane independently.** A cloud instance marked running still requires a successful SSH connection. In this session, refreshing Brev's connection configuration and reauthenticating restored access.
- **Inspect the infrastructure beneath the access tool.** The `-gcp` kernel suffix is consistent with the GCP-specific Ubuntu kernel used in [Google Cloud Ubuntu images](https://docs.cloud.google.com/compute/docs/images/os-details). This is a clue about the underlying environment; a kernel flavor alone does not independently prove cloud provider tenancy.

### Completion boundaries and next step

| Area | Status |
| --- | --- |
| Brev provisioning and connectivity | Complete for this validation session |
| Linux host identification | Complete |
| NVIDIA PCIe device visibility | Complete |
| NVIDIA driver communication | Complete |
| CUDA userspace and Toolkit validation | Not yet complete |
| Docker validation | Not yet complete |
| NVIDIA Container Toolkit validation | Not yet complete |
| PyTorch validation | Not yet complete |
| NeMo validation | Not yet complete |

Next, inspect the available CUDA userspace and Toolkit, record which environment supplies each component, and distinguish their versions from the driver's compatibility report. CUDA execution must be validated separately before claiming readiness for containers or training.

Costs, storage configuration, and instance cleanup were not included in the supplied milestone evidence and remain undocumented.
