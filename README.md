# MMEL-DTI

This repository provides the implementation of MMEL-DTI described in the
accompanying paper. MMEL-DTI integrates graph attention, chemically informed
atom ordering, bidirectional Mamba-2, protein sequence and pretrained
representations, structural features, and ISF-based prediction.

## Requirements

```bash
pip install -r requirements.txt
```

The native Mamba-2 implementation requires a CUDA-enabled environment.

## Training

```bash
python -m MMEL_DTI.train \
  --csv DATA.csv \
  --esm-embeddings ESM_EMBEDDINGS.pt \
  --structure-embeddings STRUCTURE_EMBEDDINGS.pt \
  --output mmel_dti.pt
```

The repository contains the main model implementation only. Trained weights,
large pretrained model files, structural files, and experimental scripts are
not included.
