# MMEL-DTI

This repository provides the implementation of MMEL-DTI described in the
accompanying paper. MMEL-DTI integrates graph attention, chemically informed
atom ordering, bidirectional Mamba, protein sequence and pretrained
representations, structural features, and ISF-based prediction.

## Requirements

```bash
pip install -r requirements.txt
```

The Mamba implementation requires a CUDA-enabled environment.

## Training

```bash
python -m MMEL_DTI.train \
  --csv data/BioSNAP.csv \
  --splits data/BioSNAP_5fold_seed42.npz \
  --fold 1 \
  --esm-embeddings ESM_EMBEDDINGS.pt \
  --structure-embeddings STRUCTURE_EMBEDDINGS.pt \
  --gpu 0
```

The BioSNAP interactions and fixed five-fold indices are included. Trained
weights, pretrained model files, and PDB files are not included.
