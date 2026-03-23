# Native vLLM + ROCm Installation Guide

**Hardware:** AMD Radeon RX 7900 XTX (gfx1100, RDNA3), 24 GB VRAM  
**OS:** Ubuntu 24.04 LTS, kernel 6.14 (in-tree amdgpu driver)  
**Result:** vLLM 0.18.1rc1 + PyTorch 2.10.0 + ROCm 7.1.0, natively on host

---

## Why Native Instead of Docker

The official `rocm/vllm-dev:nightly` container bundles ROCm and vLLM together as a
frozen snapshot — updating one requires rebuilding the whole image. Native installation
allows updating ROCm (via apt) and vLLM (via git pull + rebuild) independently.

---

## Prerequisites

### ROCm (already installed on this machine)
ROCm 7.1.0 is globally installed at `/opt/rocm-7.1.0` via AMD's apt repository.
`rocminfo` and `rocm-smi` are available on the host `PATH`.

> **Do NOT install `amdgpu-dkms`.**  
> The in-tree amdgpu driver bundled with kernel 6.14 supports gfx1100 perfectly.  
> The out-of-tree DKMS package fails to compile against kernel 6.14  
> (see ROCm/ROCm GitHub issues #5074 and #5085).

### Python
Python 3.12.3 is installed at `/usr/bin/python3.12` — no upgrade needed.
vLLM supports Python 3.9–3.12.

### Build Tools
Install missing build tools (only `cmake` and `ninja` were missing):

```bash
sudo apt-get install -y cmake ninja-build build-essential pybind11-dev python3-dev
```

---

## Step 1 — Create the Python Virtual Environment

Create a dedicated venv outside the project folder so it can be reused globally:

```bash
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/vllm-rocm
source ~/.venvs/vllm-rocm/bin/activate
pip install --upgrade pip setuptools wheel
```

**To activate in any future terminal session:**
```bash
source ~/.venvs/vllm-rocm/bin/activate
```

---

## Step 2 — Install PyTorch for ROCm 7.1

```bash
source ~/.venvs/vllm-rocm/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
```

This installs:
- `torch 2.10.0+rocm7.1`
- `torchvision 0.25.0+rocm7.1`
- `triton-rocm 3.6.0` (bundled automatically — no separate Triton build needed)

**Verify GPU is visible:**
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.hip)"
# Expected: True  Radeon RX 7900 XTX  7.1.25424
```

---

## Step 3 — Install AMD SMI Python Bindings

The `/opt/rocm/share/amd_smi` directory is not writable, so copy it first:

```bash
cp -r /opt/rocm/share/amd_smi /tmp/amd_smi
pip install /tmp/amd_smi
```

This installs `amdsmi 26.1.0`, which gives vLLM access to GPU metrics.

---

## Step 4 — Clone vLLM and Install Dependencies

```bash
mkdir -p ~/src
git clone https://github.com/vllm-project/vllm ~/src/vllm
cd ~/src/vllm

# Core build dependencies
pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools-scm

# Full ROCm-specific requirements
pip install -r requirements/rocm.txt
```

---

## Step 5 — Build vLLM from Source

> **This step takes 15–30 minutes** the first time. Subsequent rebuilds after `git pull`
> are faster as the compile cache is reused.

```bash
cd ~/src/vllm
export PYTORCH_ROCM_ARCH="gfx1100"
python3 setup.py develop
```

**Important notes:**
- `python3 setup.py develop` is required — `pip install .` does NOT work for ROCm builds
- `PYTORCH_ROCM_ARCH="gfx1100"` targets the RX 7900 XTX specifically, speeding up compilation
- `develop` mode live-links the install to the source tree, so `git pull` + rebuild picks up changes immediately

---

## Step 6 — Verify the Installation

```bash
source ~/.venvs/vllm-rocm/bin/activate
python3 -c "
import vllm, torch
print('vLLM:', vllm.__version__)
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"
```

Expected output:
```
vLLM: 0.18.1rc1.dev22+geaf497862
PyTorch: 2.10.0+rocm7.1
GPU: Radeon RX 7900 XTX
VRAM: 24.0 GB
```

---

## Step 7 — Smoke Test with a Local Model

Run a quick inference test against the locally cached `facebook/opt-125m`:

```bash
source ~/.venvs/vllm-rocm/bin/activate

python3 -c "
from vllm import LLM, SamplingParams

model_path = '/home/marcolap/Schreibtisch/testVllm/models/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6'

llm = LLM(model=model_path)
params = SamplingParams(temperature=0.8, max_tokens=50)
outputs = llm.generate(['Hello, my name is', 'vLLM on ROCm is'], params)

for out in outputs:
    print(f'Prompt: {out.prompt!r}')
    print(f'Output: {out.outputs[0].text!r}')
"
```

---

## Updating vLLM

To update to the latest vLLM main branch:

```bash
source ~/.venvs/vllm-rocm/bin/activate
cd ~/src/vllm
git pull
pip install -r requirements/rocm.txt   # pick up any new deps
export PYTORCH_ROCM_ARCH="gfx1100"
python3 setup.py develop
```

To update PyTorch (if a newer rocm7.x wheel is released):

```bash
source ~/.venvs/vllm-rocm/bin/activate
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
# Then rebuild vLLM as above
```

---

## File Locations Summary

| Component | Location |
|---|---|
| ROCm | `/opt/rocm-7.1.0/` (system-wide, managed by apt) |
| Python venv | `~/.venvs/vllm-rocm/` |
| vLLM source | `~/src/vllm/` |
| Models cache | `~/Schreibtisch/testVllm/models/hub/` |
| torch compile cache | `~/.cache/vllm/torch_compile_cache/` |
