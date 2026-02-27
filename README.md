# SPFF-UNet (SPCCT)

This repository provides training and evaluation code for **SPFF-UNet** and several baseline 3D segmentation models used for **voxel-wise multi-material classification** in **five-bin Spectral Photon-Counting CT (SPCCT)** phantom data (hydroxyapatite and iodine concentration classes, soft-tissue surrogates, and water).

## Dataset
The SPCCT phantom dataset used in this project is publicly available via IEEE DataPort:

- **DOI:** 10.21227/gbhn-nk95

> **Note:** The dataset files are **not** included in this GitHub repository.

---

## Repository structure
- `innovative3D/` — core package (datasets, models, utilities)
- `train.py` — training entry point
- `test.py` — evaluation entry point

---

## Configuration (important)
Key settings are defined in:

- `innovative3D/config.py`

### 1) Dataset path
By default, the code expects the dataset under (WSL/Linux example used by the authors):

```python
BASE_DIR = Path("/home/nadine/datasets/Fivedatasets")
```

Expected dataset subfolders:
```text
Fivedatasets/
  firstscan/
  filtered/
  filtered2/
  filtered3/
  filtered4/
```

### 2) Train/test split (scan-level)

The default split is scan-level (to avoid leakage across correlated slices):
```python
TRAIN_INDICES = [0, 1, 2, 4]
TEST_INDICES  = [3]
```

### 3) Checkpoints and logs
The default checkpoint root is:
```python
_PRIMARY_CKPT_DIR = BASE_DIR / "final checkpoints" / "trial"
```
You can override checkpoint and log locations using environment variables:
CHECKPOINT_DIR (where checkpoints are saved)
LOG_DIR (where logs are written; defaults to runs/ under the project)

```markdown
Example:

```bash
export CHECKPOINT_DIR=/home/<user>/spff_runs/checkpoints
export LOG_DIR=/home/<user>/spff_runs/logs
```

### Models / Variants
All model variants are registered in innovative3D/config.py under VARIANTS.

### Baselines
3DUNet
UNETR
R2UNet3D
SwinUNETR
ResUNet++

### Proposed model
SPFF-UNet

### Ablations / controls
E_SP_UNet
FG_SP_UNet
SP_UNet
PlainCore_UNet

### Seeds

Configured seeds:
```python
SEEDS = [42, 123, 999]
```
### Installation
### Option A (pip)
pip install -r requirements.txt

### Option B (conda example)
conda create -n spffunet python=3.10 -y
conda activate spffunet
pip install -r requirements.txt

Tip: You can generate requirements.txt from your working environment using:

pip freeze > requirements.txt
### Running training and evaluation
### 1) Select a model variant

Your config supports selecting a variant via environment variable:
export INNOVATIVE3D_VARIANT="SPFF-UNet"

To run a baseline, for example:

export INNOVATIVE3D_VARIANT="ResUNet++"
### 2) Train
python train.py
### 3) Evaluate / Test
python test.py

### Reproducibility notes
The manuscript reports results as mean ± SD across three seeds with a unified protocol:
same preprocessing and augmentations (including grid-puzzle)
early stopping on validation macro Dice
held-out scan for external testing (scan-level split)
Ensure your BASE_DIR and dataset folder names match the dataset structure you downloaded from IEEE DataPort.

### Baseline attribution
This repository includes independent implementations inspired by the cited papers (e.g., 3D U-Net, ResUNet++, R2U-Net, UNETR, Swin UNETR) used as baselines in our comparison. These implementations were written by the authors of this repository for research reproducibility. Please cite the original papers when using these architectures.

### Citation
If you use this code or dataset, please cite:
the associated manuscript (PLOS ONE submission / accepted paper), and
the dataset DOI: 10.21227/gbhn-nk95

### License
See the LICENSE file in this repository.