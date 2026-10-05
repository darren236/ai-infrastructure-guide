# Guide 01 — GPU Node Validation

## Part 1: Brev connectivity, Linux host, PCIe, and NVIDIA driver

**Status:** Parts 1–4 documented: host/driver validation, host Toolkit discovery, container GPU access, CUDA development toolchain inspection, and PyTorch CUDA availability/device identity checks complete. Actual GPU computation and real workloads remain unvalidated.

**Milestone recorded:** 2026-10-01. This is the documentation sync date.

**Evidence basis:** The operator's sanitized milestone summary of commands performed and observations. Raw terminal recordings and instance connection details are excluded.

### Objective and outcome

Establish working access to a cloud GPU node and validate the host and GPU driver before moving up to CUDA, containers, or training frameworks.

Connected to a Brev-managed VM after resolving an SSH access issue, identified the Linux environment, confirmed NVIDIA PCIe device enumeration, and successfully queried an NVIDIA L4 through the NVIDIA driver. These checks establish host visibility and driver communication. CUDA workload execution remains unvalidated.

### Environment

| Component | Reported value |
| --- | --- |
| Provisioning and access platform | Brev cloud GPU VM |
| Underlying cloud | GCP, as reported by the operator; the kernel flavor is consistent with that environment |
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

**Milestone recorded:** 2026-10-05, the documentation sync date. The observations below come from the operator's sanitized milestone summary.

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

At the Part 2 milestone, Docker and NVIDIA Container Toolkit had **not** yet been validated. That architecture milestone claimed no container launch, CUDA runtime test, PyTorch run, or NeMo run. Part 3 below records the subsequent container and toolchain checks.

## Part 3: Docker, NVIDIA container GPU path, and CUDA devel image

**Milestone recorded:** 2026-10-05, the documentation sync date. Evidence is the operator's sanitized summary; the maintainer did not rerun these checks on the remote VM.

### Docker and NVIDIA integration

Ran on the Brev GPU host:

```bash
which docker
docker --version
docker info | grep -i nvidia
```

| Check | Reported result |
| --- | --- |
| Docker executable | `/usr/bin/docker` |
| Docker version | `29.8.2` |
| NVIDIA CDI device names | Includes `nvidia.com/gpu=0` and `nvidia.com/gpu=all` |
| Registered runtimes | `nvidia runc io.containerd.runc.v2` |
| Default runtime | `nvidia` |

Docker is installed and operational, with NVIDIA container integration configured. The successful GPU query below validates the working container GPU path. CDI entries show available device configuration; this evidence does not independently isolate whether that particular launch used CDI or another NVIDIA integration mode. A Container Toolkit package version was not captured.

### GPU access from the base image

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

The container successfully exposed the NVIDIA L4. The first run pulled the image because it was not cached locally; this was expected behavior, not an error. This validates the base image launch and GPU visibility through the host driver, rather than execution of a CUDA kernel.

Inside the CUDA 13.0 image, `nvidia-smi` still reported `CUDA Version: 13.2`. That field reflects host driver compatibility, not the container's Toolkit version. Host driver and container Toolkit version numbers need not match; the driver must support the selected CUDA userspace and workload requirements. See NVIDIA's [CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### Development toolchain inspection

```bash
docker run --rm nvidia/cuda:13.0.0-base-ubuntu22.04 which nvcc
```

No path was returned in the base image. The host also still has no Toolkit found in the previously checked PATH and conventional locations. Neither result indicates a broken GPU path.

```bash
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu22.04 which nvcc
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu22.04 nvcc --version
```

The devel image returned `/usr/local/cuda/bin/nvcc`; its compiler reported CUDA compilation tools release `13.0`, version `V13.0.48`. These inspection commands did not request GPUs: they establish toolchain presence, not compilation or GPU execution.

| CUDA image role | Purpose | Evidence in this milestone |
| --- | --- | --- |
| `base` | Minimal CUDA components | Image launched; L4 visible through `nvidia-smi`; no `nvcc` path returned |
| `runtime` | Additional runtime libraries for executing packaged CUDA applications | Role documented; no runtime image tested |
| `devel` | Development tools, including headers and `nvcc` | Compiler path and version inspected; no program compiled yet |

These image roles are described in NVIDIA's [CUDA container image documentation](https://nvidia.github.io/container-wiki/toolkit/container-images.html).

### Host and container responsibilities

```text
Host
├── Ubuntu Linux
├── NVIDIA driver
└── NVIDIA L4

         ↑
NVIDIA container integration / CDI
         ↓

Container
├── CUDA userspace
├── optional CUDA development tools
├── PyTorch / NeMo (next workload layers; not yet validated)
└── application (not yet validated)
```

The host supplies the GPU and driver; the selected image supplies CUDA userspace and optional development tooling. NVIDIA integration makes host GPU access available inside the container. The observed runtime and CDI configuration supports this architecture; the framework and application layers are still planned.

### SA troubleshooting progression

| Check | Question it answers | Current status |
| --- | --- | --- |
| `nvidia-smi` | Can this environment query the GPU through the driver? | Validated on host and in base container |
| `nvcc --version` | Is the CUDA development toolchain present? | Validated in devel image |
| Tiny CUDA program | Can CUDA code compile and a kernel execute correctly? | Next step |
| PyTorch CUDA test | Can the ML framework use the GPU? | Pending |
| NeMo training | Can the real workload run? | Pending |

`--gpus all` requests GPU exposure; a successful container `nvidia-smi` confirms that path. `nvcc --version` checks the compiler. Neither proves CUDA kernel execution.

When a customer's framework workload fails, a minimal CUDA program provides a lower-layer test. A successful compile, kernel launch, synchronization, and result check helps narrow investigation toward framework/application code; a failure directs attention to the relevant compiler, CUDA, driver, or container layer. It does not establish that every workload requirement is satisfied.

## Part 4: PyTorch CUDA access through an NVIDIA NGC container

**Milestone recorded:** 2026-10-05, the documentation sync date. Results and startup-banner observations come from the operator's sanitized summary; the maintainer did not rerun the remote GPU checks.

### Interactive validation and observed results

On the GPU host, started the container and then entered Python inside it:

```bash
docker run --rm --gpus all -it nvcr.io/nvidia/pytorch:26.09-py3 bash
python
```

At the Python prompt:

```python
import torch
torch.__version__
torch.version.cuda
torch.cuda.is_available()
torch.cuda.get_device_name(0)
```

| Check | Reported result |
| --- | --- |
| PyTorch version | `2.14.0a0+b2c75dd062.nv26.09` |
| PyTorch CUDA build (`torch.version.cuda`) | `13.4` |
| CUDA availability | `True` |
| GPU identity at index 0 | `NVIDIA L4` |
| Container release banner | NVIDIA Release `26.09` |
| Compatibility banner | CUDA Forward Compatibility mode enabled; CUDA 13.4 compatibility components in container userspace |
| Host kernel driver | Remains `595.91.07` |

PyTorch successfully detected CUDA and identified the L4 through the containerized stack. This milestone validates framework CUDA availability and GPU identity; no GPU tensor calculation, CUDA kernel result, training run, or benchmark was reported.

### Validated access path and compatibility

```text
PyTorch
   ↓
CUDA userspace in container
   ↓
NVIDIA container integration
   ↓
host NVIDIA driver
   ↓
NVIDIA L4
```

PyTorch's CUDA build version is separate from the host driver's reported compatibility level. In this session the container reported forward-compatibility mode, while the host kernel driver remained unchanged. NVIDIA's [forward-compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) describes compatible userspace driver components operating with an older kernel driver on supported hardware/driver combinations. Versions need not be identical, but NVIDIA's platform, driver, and feature requirements still apply. The banner and availability results do not prove every workload or feature is supported.

### Two practical validation workflows

**Fast one-liner / copy-paste workflow — host shell:**

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/pytorch:26.09-py3 \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use for quick smoke tests, repeatable diagnostics, and remote support. This is a documented equivalent of the interactive checks; a separate execution was not reported. It prints the PyTorch version, availability, and GPU name. Add `print(torch.version.cuda)` when CUDA build details are needed.

**Interactive / no-copy-paste workflow:**

Start the container and Python using the commands above, then type the Python expressions individually. Use for customer consoles, manual troubleshooting, and step-by-step inspection. Run Docker on the host and the `torch` expressions inside the container's Python session. Exit Python with `exit()` and the container shell with `exit`; `--rm` removes the stopped container, while the downloaded image remains cached.

### Small troubleshooting lesson: spelling before infrastructure

`torch.cuda.is_availble()` produced an `AttributeError` because the attribute name was misspelled. The correct `torch.cuda.is_available()` returned `True`. Check the exact API name and error message before treating a user/application mistake as a driver, CUDA, or container failure.

### SA workflow: choose the test for the question

| Step | Minimum check | Customer question |
| --- | --- | --- |
| 1 | Host `nvidia-smi` | Does the node see the GPU through its driver? |
| 2 | Container-level `nvidia-smi` | Can the container access the GPU? |
| 3 | `torch.cuda.is_available()` and device identity | Does PyTorch detect usable CUDA and the expected GPU? |
| 4 | Actual customer workload | Does the real application run correctly? |
| 5 | Targeted deeper diagnostics | Is performance limited by NCCL, storage, CPU feed, networking, or another layer? |

An SA should not run every diagnostic every time. Use the minimum test that answers the current customer question, preferably in the customer's real execution environment. Follow failures toward the relevant layer; use deeper diagnostics when symptoms justify them. Availability and identity checks answer the access question, while a real GPU tensor computation is a useful final functional check before deciding whether to close Guide 01.

## Completion boundaries and next step

| Area | Status |
| --- | --- |
| Brev provisioning and connectivity | Complete for this validation session |
| Linux host identification | Complete |
| NVIDIA PCIe device visibility | Complete |
| NVIDIA driver communication | Complete |
| Host CUDA Toolkit discovery | Complete for PATH and conventional locations; no installation found |
| Host/container architecture | Documented as the intended deployment pattern |
| Actual CUDA kernel execution | Not yet complete |
| CUDA devel image / `nvcc` inspection | Complete: release 13.0, V13.0.48; compilation still pending |
| Docker installed and validated | Complete: reported version 29.8.2 |
| NVIDIA container runtime/integration | Complete for the tested GPU access path; package version not recorded |
| GPU passthrough into container | Complete: L4 visible with `--gpus all` |
| CUDA base image validation | Complete for launch and GPU visibility; kernel execution pending |
| PyTorch NGC container | Complete for startup and reported checks: `26.09-py3` |
| PyTorch CUDA availability | Complete: `True` |
| GPU identity visible from PyTorch | Complete: NVIDIA L4 |
| Interactive and one-liner workflows | Documented; interactive execution reported |
| Real GPU tensor computation | Not yet complete |
| Real training workload | Not yet complete |
| Distributed training / NCCL | Not yet complete |
| Performance benchmarking | Not yet complete |
| NeMo validation | Not yet complete |

Guide 01 remains in progress pending the completion decision. A recommended final functional check is a small GPU tensor computation in the same PyTorch container, with synchronization and result verification. A standalone CUDA compile/kernel test remains unperformed and can be used when that lower-layer question matters. Real training, NeMo, distributed training/NCCL, and performance benchmarking remain incomplete.

Costs, storage configuration, and instance cleanup were not included in the supplied milestone evidence and remain undocumented.
