# MMEL-DTI main model

This directory contains a compact release of the MMEL-DTI main configuration.
It uses a GAT drug encoder, chemically informed atom ordering, native
bidirectional `mamba_ssm.Mamba2`, three protein branches (ESM-2, sequence and
structure), and ISF prediction. The legacy `nd_mamba2` implementation,
compatibility adapter, checkpoints, ESM model files, PDB files, plotting code,
and experiment scripts are intentionally excluded.

## Input files

The dataset is one complete CSV file with columns:

```text
drug_idx, protein_idx, SMILES, Target Sequence, Label
```

The two embedding files are precomputed dictionaries saved with
`torch.save`, keyed by `protein_idx`:

* ESM-2 embeddings: `[640]` per protein;
* structure embeddings: `[1024]` per protein.

They are external inputs and are not included in this repository.

## Install

Install CUDA-compatible PyTorch first, then:

```bash
pip install -r requirements.txt
```

The official Mamba-2 CUDA kernels require a CUDA device.

## Train

```bash
python -m mmdlti.train \
  --csv /path/to/DrugBank.csv \
  --esm-embeddings /path/to/esm_embeddings.pt \
  --structure-embeddings /path/to/structure_embeddings.pt \
  --output mmel_dti.pt
```

The model can also be imported directly:

```python
from mmdlti import MMELDTI, ModelConfig
model = MMELDTI(ModelConfig()).cuda()
```
