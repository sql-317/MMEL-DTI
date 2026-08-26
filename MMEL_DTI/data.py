"""BioSNAP loader with reusable molecular graphs and precomputed protein features."""
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from .graph import molecule_to_graph


def load_embedding_map(path):
    values = torch.load(path, map_location="cpu", weights_only=False)
    return {str(k): torch.as_tensor(v, dtype=torch.float32) for k, v in values.items()}


class DTIDataset(Dataset):
    def __init__(self, csv_path, esm_embeddings, structure_embeddings, max_protein_length=1024):
        self.frame = pd.read_csv(csv_path)
        self.frame = self.frame.rename(columns={"DrugBank ID": "drug_idx", "Gene": "protein_idx"})
        required = {"drug_idx", "protein_idx", "SMILES", "Target Sequence", "Label"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        self.esm = esm_embeddings
        self.structure = structure_embeddings
        self.max_protein_length = int(max_protein_length)
        self.graphs = {}

    def precompute_graphs(self):
        smiles = self.frame["SMILES"].astype(str).drop_duplicates().tolist()
        for number, value in enumerate(smiles, 1):
            self.graphs[value] = molecule_to_graph(value)
            if number % 500 == 0 or number == len(smiles):
                print(f"Prepared molecular graphs: {number}/{len(smiles)}", flush=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        protein_id = str(row["protein_idx"])
        if protein_id not in self.esm:
            raise KeyError(f"Missing ESM embedding: {protein_id}")
        structure = self.structure.get(protein_id)
        structure_available = structure is not None
        if structure is None:
            structure = torch.zeros_like(next(iter(self.structure.values())))
        smiles = str(row["SMILES"])
        if smiles not in self.graphs:
            self.graphs[smiles] = molecule_to_graph(smiles)
        return {
            "graph": self.graphs[smiles],
            "sequence": str(row["Target Sequence"])[:self.max_protein_length],
            "esm_embedding": self.esm[protein_id],
            "structure_embedding": structure,
            "structure_available": torch.tensor(structure_available, dtype=torch.bool),
            "label": torch.tensor(float(row.Label), dtype=torch.float32),
        }


def collate(items):
    return (
        Batch.from_data_list([x["graph"] for x in items]),
        {key: ([x[key] for x in items] if key == "sequence" else torch.stack([x[key] for x in items]))
         for key in ("sequence", "esm_embedding", "structure_embedding", "structure_available")},
        torch.stack([x["label"] for x in items]),
    )
