"""Configuration for the released MMEL-DTI main model."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Input dimensions of the released main configuration.
    atom_feature_dim: int = 102
    bond_feature_dim: int = 7
    esm_dim: int = 640
    structure_dim: int = 1024

    # Drug encoder: GAT -> chemically informed ranking -> bidirectional Mamba.
    gat_hidden: int = 256
    gat_heads: int = 2
    drug_dim: int = 256

    # Native official Mamba-2.
    mamba_dim: int = 128
    mamba_state: int = 16
    mamba_conv: int = 4
    mamba_expand: int = 2
    mamba_scan_expand: int = 1
    mamba_headdim: int = 64
    mamba_chunk: int = 64

    # Protein encoder.
    aa_dim: int = 128
    sequence_dim: int = 256
    esm_branch_dim: int = 384
    structure_branch_dim: int = 384
    protein_dim: int = 512
    max_protein_length: int = 1024

    dropout: float = 0.2
    structure_residual_weight: float = 0.05

    def validate(self) -> None:
        if self.gat_hidden * self.gat_heads <= 0:
            raise ValueError("GAT dimensions must be positive")
        if (self.mamba_dim * self.mamba_expand) % self.mamba_headdim:
            raise ValueError("mamba_dim * mamba_expand must be divisible by mamba_headdim")
