# Guide 01 — GPU Node Validation

[Guide agendas](../../README.md#guides-and-agendas) · [Guide index](README.md)

## Agenda: validate the stack from GPU to application

Follow these eight layers in order. Each section states the question, the relevant check, and what the observed result establishes. Guide 01 covers infrastructure and framework access; Guide 02 continues into the Nemotron ASR application. This is a reading and troubleshooting agenda, rather than the chronological order in which every command was run.

| Layer | Check and current evidence |
| --- | --- |
| [1. Physical GPU](#1-physical-gpu) | NVIDIA L4, 24 GB class; remote inventory, not a physical inspection |
| [2. PCIe enumeration](#2-pcie-enumeration) | `lspci` enumerated the NVIDIA device |
| [3. Linux kernel](#3-linux-kernel) | `uname -a` identified kernel `6.8.0-1069-gcp` |
| [4. NVIDIA driver](#4-nvidia-driver) | Host `nvidia-smi` works; reported CUDA compatibility `13.2`, not an installed Toolkit version |
| [5. CUDA runtime / Toolkit](#5-cuda-runtime--toolkit) | No host `nvcc` found; container devel image reports Toolkit `13.0`; actual CUDA computation pending |
| [6. Container runtime](#6-container-runtime) | `docker info` and CUDA-container `docker run` validated GPU access |
| [7. PyTorch / NeMo](#7-pytorch--nemo) | PyTorch detects CUDA/L4; NeMo imports and places Nemotron on GPU in Guide 02 |
| [8. Application](#8-application) | Guide 02 prepares the ASR POC; actual transcription, training, and evaluation pending |

**Current position:** Hardware/host visibility, driver access, container toolchain inspection, and framework access are recorded. We are preparing the layer 8 application POC; successful GPU computation or ASR inference is not yet claimed. For customer troubleshooting, start with the minimum relevant check and go deeper only when needed.

**Evidence basis:** Operator-reported, sanitized milestones recorded on 2026-10-01 and 2026-10-05. NeMo results are linked from Guide 02. Raw recordings and instance connection details are excluded; no new remote checks were performed during this reorganization.

## Before starting: establish host access

Validated the local Brev CLI, authenticated, selected the intended organization, and connected using `brev shell`. An SSH configuration/access issue was resolved after `brev refresh` triggered reauthentication and refreshed the connection configuration. The underlying root cause was not independently isolated; an instance marked running does not prove SSH access is working.

Before running the checks on the remote node, confirm the execution context:

```bash
hostname
whoami
pwd
```

The operator started terminal recording with `script`. Recordings and instance-specific identity values are excluded from Git.

<a id="part-1-brev-connectivity-linux-host-pcie-and-nvidia-driver"></a>

## 1. Physical GPU

**Question:** What GPU is provisioned for this node?

The Brev-managed cloud VM was reported to have an **NVIDIA L4, 24 GB class VRAM**. The later driver query reported **23,034 MiB**. This is the physical GPU layer in the architecture, established here through remote inventory and subsequent enumeration/driver checks; no hands-on hardware inspection, stress test, or long-term reliability claim is made.

## 2. PCIe enumeration

**Question:** Can Linux enumerate the NVIDIA device on the PCIe bus?

```bash
lspci | grep -i nvidia
```

**Observed:** An NVIDIA device was enumerated. This establishes device visibility, independently of whether the NVIDIA driver or a CUDA application works. If `lspci` sees the GPU but `nvidia-smi` fails, investigate the driver/kernel/device-access layer before CUDA or PyTorch.

## 3. Linux kernel

**Question:** Which kernel, OS, and architecture are running on this node?

```bash
uname -a
cat /etc/os-release
```

| Check | Reported value |
| --- | --- |
| Linux kernel | `6.8.0-1069-gcp` |
| Operating system | Ubuntu 22.04.5 LTS (Jammy) |
| Architecture | x86_64 |
| Underlying cloud | GCP, as reported by the operator |

The `-gcp` suffix is consistent with the GCP-specific Ubuntu kernel in [Google Cloud Ubuntu images](https://docs.cloud.google.com/compute/docs/images/os-details). It is a clue, not independent proof of cloud tenancy. Kernel identification does not by itself establish that a GPU driver module works; the next layer checks communication through that driver.

## 4. NVIDIA driver

**Question:** Can the installed NVIDIA driver communicate with the GPU?

```bash
nvidia-smi
```

**Observed:** NVIDIA L4 detected, driver `595.91.07`, reported `CUDA Version: 13.2`.

### Supported CUDA version versus installed CUDA version

The `CUDA Version: 13.2` field in `nvidia-smi` describes the CUDA compatibility level supported by the installed driver. It does **not** establish that CUDA Toolkit 13.2 is installed on the host, identify an application's CUDA runtime, or demonstrate successful CUDA execution. NVIDIA documents this distinction in its [driver and CUDA matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).

Application compatibility also depends on the hardware, required features, and applicable [CUDA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Record actual application runtime and Toolkit versions separately inside the selected environment.

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

<a id="part-2-cuda-host-vs-container-architecture"></a>

## 5. CUDA runtime / Toolkit

**Question:** Where does CUDA userspace live, and is the development toolchain present?

In this guide's containerized setup, CUDA userspace and optional development tools are expected inside the image. The host needs the NVIDIA driver; a host Toolkit installation is not required. Missing host `nvcc` is consistent with that design. Other deployments may intentionally install a host Toolkit.

`nvcc --version` identifies the compiler/Toolkit, not every runtime library or successful kernel execution. Runtime-only images can run packaged CUDA software without including `nvcc`.

### Host Toolkit discovery

| Check | Reported observation | Interpretation |
| --- | --- | --- |
| `nvidia-smi` | Succeeded; driver `595.91.07`, `CUDA Version: 13.2` | GPU communication through the installed driver works; the CUDA field is a driver compatibility report. |
| `which nvcc` | Returned no path | The CUDA compiler was not available through the current shell's PATH. |
| `/usr/local/cuda/bin/nvcc` | File does not exist | The compiler was not present at this conventional location. |
| Conventional `/usr/local/cuda*` locations | No Toolkit installation found | No standard installation was discovered in the locations checked. |

**Operational conclusion:** The host has a working NVIDIA driver and GPU access, with no host CUDA Toolkit found by these checks. `nvidia-smi` reporting `13.2` does **not** mean CUDA Toolkit 13.2 is installed. It reports the driver's supported CUDA compatibility level, including support for applications built with Toolkits up to that level, subject to hardware and compatibility requirements.

The discovery scope matters: a missing `nvcc` and absent conventional paths do not establish that every possible installation location or environment has been searched. Toolkit installations can use alternate paths, and runtime libraries can exist without the compiler. NVIDIA's [Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) covers configurable Toolkit paths and PATH setup. These host discovery checks do not test CUDA runtime execution; the container and framework checks follow in layers 6 and 7.

### Container development toolchain

The commands below inspect image contents. They use Docker, which is validated in layer 6; this section first explains what the CUDA checks mean.

```bash
docker run --rm nvidia/cuda:13.0.0-base-ubuntu22.04 which nvcc
```

No path was returned in the base image. `which nvcc` also returns a nonzero exit status when the compiler is absent; this is expected for this image. The host likewise has no Toolkit found in the previously checked PATH and conventional locations. These compiler-discovery results do not indicate a broken GPU path.

```bash
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu22.04 which nvcc
docker run --rm nvidia/cuda:13.0.0-devel-ubuntu22.04 nvcc --version
```

The devel image returned `/usr/local/cuda/bin/nvcc`; its compiler reported CUDA compilation tools release `13.0`, version `V13.0.48`. These inspection commands did not request GPUs: they establish toolchain presence, not compilation or GPU execution.

| CUDA image role | Purpose | Recorded evidence |
| --- | --- | --- |
| `base` | Minimal CUDA components | Image launched; L4 visible through `nvidia-smi`; no `nvcc` path returned |
| `runtime` | Additional runtime libraries for executing packaged CUDA applications | Role documented; no runtime image tested |
| `devel` | Development tools, including headers and `nvcc` | Compiler path and version inspected; no program compiled yet |

These image roles are described in NVIDIA's [CUDA container image documentation](https://nvidia.github.io/container-wiki/toolkit/container-images.html).

### Host and container architecture

A common production pattern keeps the host relatively minimal, with Linux and the NVIDIA driver providing GPU access. CUDA userspace libraries, PyTorch/NeMo, and application dependencies are supplied inside containers. The host does not require a CUDA Toolkit installation for this pattern; NVIDIA explicitly documents that distinction in the [NVIDIA Container Toolkit project](https://github.com/NVIDIA/nvidia-container-toolkit).

The diagram separates host and container responsibilities. Container GPU access and framework/model loading have been validated; application execution remains pending.

```text
Host
├── Linux
├── NVIDIA driver
└── NVIDIA GPU

         ↑
NVIDIA Container Toolkit
         ↓

Container
├── CUDA userspace / optional Toolkit
├── PyTorch / NeMo
└── Application
```

NVIDIA Container Toolkit is a host-side integration component that enables a container runtime, such as Docker, to expose GPU devices and required host driver libraries to containers. It is separate from the CUDA Toolkit and its compiler. The container uses the host GPU and driver rather than replacing them. See NVIDIA's [container architecture overview](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html).

The container's full CUDA Toolkit is optional: runtime images provide libraries for execution, while development images add compilation tools such as `nvcc`. Select the image according to whether the workload needs to build CUDA code or run packaged software. NVIDIA describes this distinction in its [CUDA container image documentation](https://nvidia.github.io/container-wiki/toolkit/container-images.html).

### Troubleshooting takeaway

If `nvidia-smi` succeeds but `nvcc` is missing, do not assume the GPU node is broken. First determine whether the deployment is intended to be containerized and whether CUDA userspace is expected inside the container. For host-side compilation, inspect Toolkit installation and PATH configuration. For a containerized workload, validate the container runtime, NVIDIA Container Toolkit integration, and the selected image before claiming application readiness.

<a id="part-3-docker-nvidia-container-gpu-path-and-cuda-devel-image"></a>

## 6. Container runtime

**Question:** Can Docker start a container with access to the host GPU?

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

`--gpus all` requests GPU exposure; a successful container `nvidia-smi` confirms the access path. It does not demonstrate a CUDA kernel running. NVIDIA Container Toolkit integrates GPU devices and host driver libraries with the container runtime; it is distinct from the CUDA development Toolkit checked in layer 5.

<a id="part-4-pytorch-cuda-access-through-an-nvidia-ngc-container"></a>

## 7. PyTorch / NeMo

**Question:** Can the framework inside its container access CUDA and the expected GPU?

### PyTorch container checks

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

PyTorch successfully detected CUDA and identified the L4 through the containerized stack. These checks validate framework CUDA availability and GPU identity; no GPU tensor calculation, CUDA kernel result, training run, or benchmark was reported.

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

### NeMo Speech container checks

The operator also validated these checks in [Guide 02](02-nemotron-streaming-singapore-english.md#nemo-container-and-model-on-gpu--complete):

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo; print(nemo.__version__)"
```

**Observed:** NeMo `3.0.0`.

```bash
mkdir -p ~/hf-cache

docker run --rm --gpus all \
  -v ~/hf-cache:/root/.cache/huggingface \
  nvcr.io/nvidia/nemo-speech:26.07.00 \
  python -c "import nemo.collections.asr as n; m=n.models.ASRModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b'); m.cuda(); print(next(m.parameters()).device)"
```

**Observed:** Restored `EncDecRNNTBPEModelWithPrompt`; parameter device `cuda:0`. This validates ASR import, model loading, and GPU placement. The persistent Hugging Face cache lives on the host. It does not yet demonstrate an audio transcript or training run.

## 8. Application

**Question:** Does the real workload produce the expected result?

Framework availability and model placement lead to application validation. A GPU tensor computation with synchronization and a checked result would test actual computation; an ASR application must process audio and produce a transcript. Neither successful application result has been reported yet.

Continue in [Guide 02 — Adapting NVIDIA Nemotron 3.5 Streaming ASR to Singapore English](02-nemotron-streaming-singapore-english.md). It is intentionally a small POC/tutorial. Model loading, source checks, annotation policy, and generation/independent checks of 300 train / 50 validation utterances are complete. Transcript normalization is next; NeMo manifest conversion, baseline inference, training, checkpointing, and evaluation remain pending. A one-WAV inference reference is prepared but unexecuted.

When a customer's framework workload fails, a minimal CUDA program can isolate the lower CUDA/driver/container stack. A successful compile, kernel launch, synchronization, and result check narrows the investigation; it does not establish that every workload requirement is satisfied. A standalone CUDA compile/kernel test remains unperformed here.

### Choose the test for the customer question

| Minimum check | Customer question |
| --- | --- |
| Host `nvidia-smi` | Does the node see the GPU through its driver? |
| Container-level `nvidia-smi` | Can the container access the GPU? |
| `torch.cuda.is_available()` and device identity | Does PyTorch detect usable CUDA and the expected GPU? |
| Actual customer workload | Does the real application run correctly? |
| Targeted deeper diagnostics | Is performance limited by NCCL, storage, CPU feed, networking, or another layer? |

An SA should not run every diagnostic every time. Use the minimum test that answers the current customer question, preferably in the customer's real execution environment. Follow failures toward the relevant layer; use deeper diagnostics when symptoms justify them. Availability and identity checks answer the access question, while a real GPU tensor computation is a useful final functional check before deciding whether to close Guide 01.

## Completion boundaries and next step

Layers 1–7 have the specific access, inventory, and inspection evidence recorded above. They are not blanket claims that every feature or GPU workload works. Layer 8 is in progress through Guide 02 preparation; actual inference, training, true streaming, NCCL, and performance benchmarking remain unvalidated.

Costs, storage configuration, and instance cleanup were not included in the Guide 01 milestone evidence and remain undocumented. See [progress](../../PROGRESS.md) for the dated milestone history and [Guide 02](02-nemotron-streaming-singapore-english.md) for the next application step.
