"""RDKit molecular graph construction and chemical atom features."""
import torch
from torch_geometric.data import Data
from rdkit import Chem

ATOM_SYMBOLS = [
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B',
    'V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu',
    'Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
]


def atom_features(mol):
    rows = []
    for atom in mol.GetAtoms():
        row = [float(atom.GetSymbol() == x) for x in ATOM_SYMBOLS]
        row += [float(atom.GetDegree() == i) for i in range(11)]
        row += [float(atom.GetTotalNumHs() == i) for i in range(11)]
        try:
            valence = atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)
            row += [float(valence == i) for i in range(11)]
        except Exception:
            row += [0.0] * 11
        row += [float(atom.GetIsAromatic())]
        charge = atom.GetFormalCharge()
        row += [float(charge > 0), float(charge < 0)]
        hyb = atom.GetHybridization()
        row += [float(hyb == Chem.rdchem.HybridizationType.SP),
                float(hyb == Chem.rdchem.HybridizationType.SP2),
                float(hyb == Chem.rdchem.HybridizationType.SP3),
                float(hyb == Chem.rdchem.HybridizationType.SP3D),
                float(hyb not in (Chem.rdchem.HybridizationType.SP,
                                  Chem.rdchem.HybridizationType.SP2,
                                  Chem.rdchem.HybridizationType.SP3,
                                  Chem.rdchem.HybridizationType.SP3D))]
        row += [float(atom.IsInRing())]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def bond_features(mol):
    edges, values = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        kind = bond.GetBondType()
        one = [float(kind == k) for k in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                           Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)]
        one += [float(bond.GetIsAromatic()), float(bond.GetIsConjugated()), float(bond.IsInRing())]
        edges.extend(((a, b), (b, a))); values.extend((one, one))
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 7)
    return torch.tensor(edges, dtype=torch.long).t(), torch.tensor(values, dtype=torch.float32)


def structural_encoding(edge_index, n, pe_dim=8, walk_dim=8):
    adj = torch.zeros(n, n)
    if edge_index.numel():
        adj[edge_index[0], edge_index[1]] = 1.0
    adj = ((adj + adj.t()) > 0).float()
    degree = adj.sum(1)
    lap = torch.diag(degree) - adj + 1e-6 * torch.eye(n)
    _, vec = torch.linalg.eigh(lap)
    pe = torch.zeros(n, pe_dim)
    take = min(pe_dim, max(n - 1, 0))
    if take:
        pe[:, :take] = vec[:, 1:take + 1]
    transition = adj / degree.clamp_min(1).unsqueeze(1)
    current = torch.eye(n)
    walks = []
    for _ in range(walk_dim):
        current = current @ transition
        walks.append(torch.diag(current))
    se = torch.stack(walks, 1) if walks else torch.zeros(n, 0)
    return torch.cat([pe, se], 1)


def molecule_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles}")
    edge_index, edge_attr = bond_features(mol)
    x = torch.cat([atom_features(mol), structural_encoding(edge_index, mol.GetNumAtoms())], 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=mol.GetNumAtoms(), smiles=Chem.MolToSmiles(mol))


def chemical_rule_features(mol):
    rows = []
    for atom in mol.GetAtoms():
        rows.append([float(atom.GetAtomicNum() == 6),
                     float(atom.GetAtomicNum() in (7, 8)),
                     float(atom.GetAtomicNum() in (15, 16)),
                     float(atom.GetDegree()), float(atom.IsInRing()),
                     float(atom.GetIsAromatic())])
    return torch.tensor(rows, dtype=torch.float32)
