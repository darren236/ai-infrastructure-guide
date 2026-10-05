# Lab 01 — GPU Node Validation

## Part 1: Brev connectivity, Linux host, PCIe, and NVIDIA driver

**Status:** Part 1 complete; host Toolkit discovery and host/container architecture documented in Part 2. CUDA runtime execution, containers, and frameworks remain unvalidated.

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
| Host CUDA Toolkit | No installation found in PATH or conventional `/usr/local/cuda*` locations; see Part 2 |
| CUDA userspace runtime and workload execution | Not yet validated |

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
| CUDA Toolkit | Supplies development tools, including `nvcc`, and libraries for building CUDA software. | Host discovery completed for the checked paths; no installation found. CUDA Toolkit contents inside a container not yet inspected. |

Application compatibility also depends on the hardware, required features, and applicable [CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Record actual application runtime and Toolkit versions separately when inspecting the selected container environment.

### Infrastructure lessons

- **Verify context first.** `hostname`, `whoami`, and `pwd` help establish which machine, account, and directory a command is using.
- **Troubleshoot bottom-up.** Use the sequence `hardware → PCIe enumeration → kernel → NVIDIA driver → CUDA userspace → container runtime → framework → application` to isolate failures by layer.
- **Distinguish enumeration from driver access.** `lspci` shows PCIe device visibility; `nvidia-smi` demonstrates driver-mediated GPU communication. If enumeration succeeds but `nvidia-smi` fails, investigate the driver, kernel module, and device access before CUDA or PyTorch.
- **Validate the access plane independently.** A cloud instance marked running still requires a successful SSH connection. In this session, refreshing Brev's connection configuration and reauthenticating restored access.
- **Inspect the infrastructure beneath the access tool.** The `-gcp` kernel suffix is consistent with the GCP-specific Ubuntu kernel used in [Google Cloud Ubuntu images](https://docs.cloud.google.com/compute/docs/images/os-details). This is a clue about the underlying environment; a kernel flavor alone does not independently prove cloud provider tenancy.

## Part 2: CUDA host vs container architecture

**Milestone recorded:** 2026-10-05, the documentation sync date. The observations below come from the lab operator's sanitized milestone summary.

### Host Toolkit discovery

| Check | Reported observation | Interpretation |
| --- | --- | --- |
| `nvidia-smi` | Succeeded; driver `595.91.07`, `CUDA Version: 13.2` | GPU communication through the installed driver works; the CUDA field is a driver compatibility report. |
| `which nvcc` | Returned no path | The CUDA compiler was not available through the current shell's PATH. |
| `/usr/local/cuda/bin/nvcc` | File does not exist | The compiler was not present at this conventional location. |
| Conventional `/usr/local/cuda*` locations | No Toolkit installation found | No standard installation was discovered in the locations checked. |

**Operational conclusion:** The host has a working NVIDIA driver and GPU access, with no host CUDA Toolkit found by these checks. `nvidia-smi` reporting `13.2` does **not** mean CUDA Toolkit 13.2 is installed. It reports the driver's supported CUDA compatibility level, including support for applications built with Toolkits up to that level, subject to hardware and compatibility requirements.

The discovery scope matters: a missing `nvcc` and absent conventional paths do not establish that every possible installation location or environment has been searched. Toolkit installations can use alternate paths, and runtime libraries can exist without the compiler. NVIDIA's [Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) covers configurable Toolkit paths and PATH setup. CUDA runtime availability and execution have not been validated in this milestone.

### Common containerized deployment pattern

A common production pattern keeps the host relatively minimal, with Linux and the NVIDIA driver providing GPU access. CUDA userspace libraries, PyTorch/NeMo, and application dependencies are supplied inside containers. The host does not require a CUDA Toolkit installation for this pattern; NVIDIA explicitly documents that distinction in the [NVIDIA Container Toolkit project](https://github.com/NVIDIA/nvidia-container-toolkit).

The following diagram shows the intended architecture, rather than an already validated deployment:

```text
Host
├── Linux
├── NVIDIA driver
└── NVIDIA GPU

         ↑
NVIDIA Container Toolkit
         ↓

Container
├── CUDA runtime / toolkit
├── PyTorch / NeMo
└── Training application
```

NVIDIA Container Toolkit is a host-side integration component that enables a container runtime, such as Docker, to expose GPU devices and required host driver libraries to containers. It is separate from the CUDA Toolkit and its compiler. The container uses the host GPU and driver rather than replacing them. See NVIDIA's [container architecture overview](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html).

The container's full CUDA Toolkit is optional: runtime images provide libraries for execution, while development images add compilation tools such as `nvcc`. Select the image according to whether the workload needs to build CUDA code or run packaged software. NVIDIA describes this distinction in its [CUDA container image documentation](https://nvidia.github.io/container-wiki/toolkit/container-images.html).

### Troubleshooting takeaway

If `nvidia-smi` succeeds but `nvcc` is missing, do not assume the GPU node is broken. First determine whether the deployment is intended to be containerized and whether CUDA userspace is expected inside the container. For host-side compilation, inspect Toolkit installation and PATH configuration. For a containerized workload, validate the container runtime, NVIDIA Container Toolkit integration, and the selected image before claiming application readiness.

Docker and NVIDIA Container Toolkit have **not** been validated yet. No container launch, container CUDA runtime test, PyTorch run, or NeMo run is claimed by this architecture milestone.

## Completion boundaries and next step

| Area | Status |
| --- | --- |
| Brev provisioning and connectivity | Complete for this validation session |
| Linux host identification | Complete |
| NVIDIA PCIe device visibility | Complete |
| NVIDIA driver communication | Complete |
| Host CUDA Toolkit discovery | Complete for PATH and conventional locations; no installation found |
| Host/container architecture | Documented as the intended deployment pattern |
| CUDA runtime availability and execution | Not yet complete |
| CUDA Toolkit inspection inside a container | Not yet complete |
| Docker validation | Not yet complete |
| NVIDIA Container Toolkit validation | Not yet complete |
| PyTorch validation | Not yet complete |
| NeMo validation | Not yet complete |

Next, validate Docker and NVIDIA Container Toolkit integration. Then inspect the selected container's CUDA runtime and any development Toolkit, record their versions, and validate GPU access and CUDA execution inside the container before proceeding to PyTorch or NeMo.

Costs, storage configuration, and instance cleanup were not included in the supplied milestone evidence and remain undocumented.
