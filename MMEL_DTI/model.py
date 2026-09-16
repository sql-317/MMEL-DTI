"""MMEL-DTI main model: GAT, bidirectional Mamba-2, protein branches and ISF."""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from rdkit import Chem

from .config import ModelConfig
from .graph import chemical_rule_features
from .mamba import BidirectionalMamba


class DrugEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gat = GATConv(cfg.atom_feature_dim, cfg.gat_hidden,
                           heads=cfg.gat_heads, concat=True, edge_dim=cfg.bond_feature_dim,
                           dropout=cfg.dropout, add_self_loops=True, fill_value=0.0)
        for name, parameter in self.gat.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter, gain=0.5)
            elif "bias" in name and parameter is not None:
                nn.init.zeros_(parameter)
        gat_dim = cfg.gat_hidden * cfg.gat_heads
        self.gat_norm = nn.BatchNorm1d(gat_dim)
        self.reduce = nn.Sequential(nn.Linear(gat_dim, cfg.drug_dim), nn.LayerNorm(cfg.drug_dim),
                                    nn.GELU(), nn.Dropout(cfg.dropout))
        self.to_mamba = nn.Sequential(nn.Linear(cfg.drug_dim, cfg.mamba_dim),
                                      nn.LayerNorm(cfg.mamba_dim), nn.Dropout(cfg.dropout))
        self.rank = nn.Sequential(nn.Linear(cfg.drug_dim, cfg.drug_dim // 2), nn.GELU(),
                                  nn.Linear(cfg.drug_dim // 2, 1))
        self.rule_coefficients = nn.Parameter(torch.tensor([1., 2., 1.5, .5, 1., .8]))
        self.rule_scale = nn.Parameter(torch.tensor(0.0))
        self.mamba = BidirectionalMamba(cfg.mamba_dim, cfg.mamba_state, cfg.mamba_conv,
                                        cfg.mamba_expand, cfg.mamba_scan_expand,
                                        cfg.mamba_headdim, cfg.mamba_chunk, residual_scale=0.1)
        self.output_proj = nn.Sequential(nn.Linear(cfg.mamba_dim, cfg.drug_dim),
                                         nn.LayerNorm(cfg.drug_dim), nn.Dropout(cfg.dropout))
        self.output_norm = nn.BatchNorm1d(cfg.drug_dim)
        self._molecule_cache = {}
        self._rule_cache = {}

    def _rules(self, smiles, node_count, device):
        smiles = str(smiles)
        if smiles not in self._molecule_cache:
            self._molecule_cache[smiles] = Chem.MolFromSmiles(smiles)
        mol = self._molecule_cache[smiles]
        if mol is None or mol.GetNumAtoms() != node_count:
            return None
        if smiles not in self._rule_cache:
            self._rule_cache[smiles] = chemical_rule_features(mol)
        return self._rule_cache[smiles].to(device)

    def forward(self, batch_graph):
        x = F.elu(self.gat(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr))
        x = F.dropout(x, self.cfg.dropout, self.training)
        if x.shape[0] > 1:
            x = self.gat_norm(x)
        x = self.reduce(x)
        ptr = batch_graph.ptr
        smiles = batch_graph.smiles if isinstance(batch_graph.smiles, list) else [batch_graph.smiles]
        lengths = [int(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
        max_len = max(lengths)
        device = x.device
        padded = x.new_zeros(len(lengths), max_len, x.shape[-1])
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
        orders = []
        for i, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            local = x[start:end]
            contextual = self.rank(local).squeeze(-1)
            rules = self._rules(smiles[i], len(local), device) if i < len(smiles) else None
            if rules is not None:
                chemical = rules @ self.rule_coefficients
            else:
                chemical = torch.zeros_like(contextual)
            scores = contextual + F.softplus(self.rule_scale) * chemical
            local = local * (1.0 + scores.sigmoid().unsqueeze(-1))
            order = scores.argsort(descending=True)
            padded[i, :len(local)] = local[order]
            mask[i, :len(local)] = True
            orders.append(order)
        z = self.to_mamba(padded)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        z = self.mamba(z, mask)
        # The released main configuration uses the learned chemical-rule
        # pooling path after Mamba.
        pooled_logits = z.new_zeros(len(lengths), max_len)
        for i, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            rules = self._rules(smiles[i], lengths[i], device) if i < len(smiles) else None
            if rules is not None:
                values = rules @ self.rule_coefficients
                pooled_logits[i, :lengths[i]] = values[orders[i]]
        pooled_logits = pooled_logits.masked_fill(~mask, torch.finfo(z.dtype).min)
        weights = pooled_logits.softmax(1)
        self.last_atom_weights = weights.detach()
        self.last_node_orders = orders
        output = self.output_proj((z * weights.unsqueeze(-1)).sum(1))
        return self.output_norm(output) if output.shape[0] > 1 else output


class ProteinEncoder(nn.Module):
    AA = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWYUOBZX")}

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.esm = nn.Sequential(nn.Linear(cfg.esm_dim, cfg.esm_branch_dim), nn.LayerNorm(cfg.esm_branch_dim),
                                 nn.GELU(), nn.Dropout(cfg.dropout))
        self.structure = nn.Sequential(nn.Linear(cfg.structure_dim, cfg.structure_branch_dim),
                                       nn.LayerNorm(cfg.structure_branch_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.aa = nn.Embedding(26, cfg.aa_dim, padding_idx=0)
        self.seq_mamba = BidirectionalMamba(cfg.aa_dim, cfg.mamba_state, cfg.mamba_conv,
                                            cfg.mamba_expand, 2,
                                            cfg.mamba_headdim, cfg.mamba_chunk, residual_scale=0.0)
        splits = [cfg.sequence_dim // 3 + cfg.sequence_dim % 3] + [cfg.sequence_dim // 3] * 2
        self.convs = nn.ModuleList([nn.Conv1d(cfg.aa_dim, c, k, padding=k // 2)
                                    for c, k in zip(splits, (3, 5, 7))])
        self.norms = nn.ModuleList([nn.BatchNorm1d(c) for c in splits])
        self.seq_attention = nn.Linear(cfg.sequence_dim, 1)
        self.seq_post = nn.Sequential(nn.Linear(cfg.sequence_dim, cfg.sequence_dim),
                                      nn.LayerNorm(cfg.sequence_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.seq_project = nn.Sequential(nn.Linear(cfg.sequence_dim, cfg.esm_branch_dim),
                                          nn.LayerNorm(cfg.esm_branch_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.gate = nn.Sequential(nn.Linear(cfg.esm_branch_dim * 3, 128), nn.LayerNorm(128), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(128, 2))
        self.gate_scale = nn.Parameter(torch.tensor(-2.1972))
        self.output = nn.Sequential(nn.Linear(cfg.esm_branch_dim, cfg.protein_dim),
                                    nn.LayerNorm(cfg.protein_dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.output_norm = nn.BatchNorm1d(cfg.protein_dim)

    def _indices(self, sequences, device):
        length = max(map(len, sequences))
        ids = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor([self.AA.get(a.upper(), 0) for a in seq], device=device)
        mask = torch.arange(length, device=device)[None] < torch.tensor([len(s) for s in sequences], device=device)[:, None]
        return ids, mask

    def forward(self, data):
        esm = self.esm(data["esm_embedding"])
        structure = self.structure(data["structure_embedding"])
        available = data.get("structure_available")
        if available is not None:
            structure = structure * available.to(structure.device).bool().unsqueeze(-1).to(structure.dtype)
        ids, mask = self._indices(data["sequence"], esm.device)
        x = self.aa(ids)
        x = self.seq_mamba(x, mask).transpose(1, 2)
        conv = [F.gelu(norm(layer(x))) * mask.unsqueeze(1) for layer, norm in zip(self.convs, self.norms)]
        x = torch.cat(conv, 1).transpose(1, 2)
        a = self.seq_attention(x).squeeze(-1).masked_fill(~mask, torch.finfo(x.dtype).min).softmax(1)
        self.last_residue_attention = a.detach().unsqueeze(-1)
        sequence = self.seq_post((x * a.unsqueeze(-1)).sum(1))
        sequence = self.seq_project(sequence)
        gates = self.gate(torch.cat([esm, structure, sequence], -1)).sigmoid()
        shared = torch.sigmoid(self.gate_scale)
        fused = esm + self.cfg.structure_residual_weight * gates[:, :1] * structure + shared * gates[:, 1:2] * sequence
        return self.output_norm(self.output(fused))


class ISFHead(nn.Module):
    """Pair-dependent ISF head without a raw-drug concatenation shortcut."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.drug_projection = nn.Linear(cfg.drug_dim, cfg.protein_dim)
        self.attention = nn.Sequential(nn.Linear(cfg.protein_dim, cfg.protein_dim // 4), nn.ReLU(),
                                       nn.Linear(cfg.protein_dim // 4, cfg.protein_dim), nn.Sigmoid())
        dims = [cfg.protein_dim, 512, 256, 128, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.LayerNorm(a), nn.Linear(a, b)]
            if b != 1:
                layers += [nn.GELU(), nn.Dropout(cfg.dropout)]
        self.predictor = nn.Sequential(*layers)

    def forward(self, drug, protein):
        d = self.drug_projection(drug)
        gate = self.attention(protein + d)
        fused = protein * gate + d * (1 - gate)
        return self.predictor(fused).squeeze(-1).clamp(-10, 10)


class MMELDTI(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.cfg.validate()
        self.drug_encoder = DrugEncoder(self.cfg)
        self.protein_encoder = ProteinEncoder(self.cfg)
        self.head = ISFHead(self.cfg)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, graph, protein_data):
        return self.head(self.drug_encoder(graph), self.protein_encoder(protein_data))
