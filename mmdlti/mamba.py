"""Native bidirectional Mamba-2 block used by MMEL-DTI."""
import torch
from torch import nn
from mamba_ssm import Mamba2


class BidirectionalMamba(nn.Module):
    """Two independent native Mamba-2 scans in forward and reverse order."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 headdim=64, chunk_size=64):
        super().__init__()
        inner = d_model * expand
        if inner % headdim:
            raise ValueError("d_model * expand must be divisible by headdim")
        self.input_proj = nn.Linear(d_model, inner, bias=False)
        args = dict(d_model=inner, d_state=d_state, d_conv=d_conv,
                    expand=1, headdim=headdim, chunk_size=chunk_size)
        self.forward_scan = Mamba2(**args)
        self.reverse_scan = Mamba2(**args)
        self.output_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, length, channels], got {tuple(x.shape)}")
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        z = self.input_proj(x)
        forward = self.forward_scan(z)
        reverse = self.reverse_scan(z.flip(1)).flip(1)
        y = self.output_proj(forward + reverse)
        return y if mask is None else y * mask.unsqueeze(-1).to(y.dtype)
