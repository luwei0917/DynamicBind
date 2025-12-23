# Minimal setup for DynamicBind (disk-friendly)

This guide creates a smaller conda environment that contains the essential packages for running inference while avoiding large optional packages and CUDA libraries. Use this if disk space is limited.

1) Create minimal environment from the provided file

```bash
conda env create -f environment-minimal.yml
conda activate dynamicbind-minimal
```

2) Install PyTorch (CPU-only, smaller) — choose one of the following:

CPU-only (recommended if no GPU):

```bash
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 cpuonly -c pytorch
```

GPU (if you have a compatible CUDA and want GPU support):

```bash
# Example for CUDA 11.7
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3) Install PyG (torch-geometric) wheels matching your installed `torch` version.
- If you installed CPU-only `torch`, install CPU-compatible PyG wheels or use pip fallback.
- For GPU/CUDA, follow instructions at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html or use the `https://data.pyg.org/whl/` links used in the original environment.

Example (pip wheel links may vary by torch version):

```bash
# For CUDA 11.7 (replace URLs with ones matching exact torch version if necessary):
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Recommended precise PyG installs for `torch==2.0.1` + `cu117`:
```bash
pip install torch-geometric==2.3.0 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
```

Installing NVIDIA drivers / CUDA (Ubuntu, brief):

1. Verify GPU and current driver:

```bash
nvidia-smi
```

2. If `nvidia-smi` missing or driver older than required, install driver + CUDA (example using NVIDIA package repos):

```bash
# Add NVIDIA package repo (example):
sudo apt update && sudo apt install -y wget gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt update
sudo apt -y install nvidia-driver-535 cuda-toolkit-11-7
sudo reboot
```

Notes:
- Exact driver package names and repo URLs depend on your OS version — check NVIDIA's official instructions for Ubuntu 24.04.
- On managed platforms (Colab, Codespaces) you may not be able to install drivers; use provider GPU options instead.

Notes on `torch-geometric` (PyG) for GPU:
- Use the PyG wheel index matching `torch` + CUDA, e.g. `https://data.pyg.org/whl/torch-2.0.1+cu117.html`.
- If pip wheel install fails, try `conda install -c conda-forge pyg-lib pyg torch-scatter -y` or consult the PyG installation guide.

Notes on `torch-geometric` (PyG):
- If pip wheel install fails, try `conda install -c conda-forge pyg-lib pyg` or follow the official PyG installation guide.
- PyG wheels must match the installed `torch` (CPU vs CUDA and major/minor versions).

4) Remaining lightweight Python packages
- The `environment-minimal.yml` already includes many common packages (numpy, scipy, pandas, matplotlib, etc.) and `fair-esm` and `e3nn` needed by the code.

5) Optional: Relaxation environment (if you plan to run relaxation locally)
- Relaxation requires `openmm`, `pdbfixer`, `openmmforcefields`, `openff-toolkit`, `ambertools` etc. To save disk, create a separate `relax` environment only when needed (see original README).

6) What was removed/trimmed compared to the original environment
- CUDA runtime libraries and many conda-forge system packages were avoided to reduce disk usage.
- Jupyter / dev tools and many pinned build packages were removed; add them back only if needed.

If you want, I can:
- pin exact versions for `torch` and `torch-geometric` that match your target hardware,
- generate CPU-specific PyG wheel links,
- or create a full `requirements.txt` instead of the conda manifest.
