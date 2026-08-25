"""Minimal training entry point for one complete CSV dataset."""
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from .config import ModelConfig
from .data import DTIDataset, collate, load_embedding_map
from .model import MMELDTI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--esm-embeddings", required=True)
    p.add_argument("--structure-embeddings", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="mmel_dti.pt")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The native Mamba-2 kernels require CUDA.")
    device = torch.device("cuda")
    dataset = DTIDataset(args.csv, load_embedding_map(args.esm_embeddings),
                         load_embedding_map(args.structure_embeddings))
    n_val = max(1, int(len(dataset) * 0.1))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, collate_fn=collate,
                            pin_memory=True, num_workers=0)
    model = MMELDTI(ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for graph, protein, label in train_loader:
            graph = graph.to(device)
            protein = {k: (v if isinstance(v, list) else v.to(device)) for k, v in protein.items()}
            loss = loss_fn(model(graph, protein), label.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(label)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for graph, protein, label in val_loader:
                graph = graph.to(device)
                protein = {k: (v if isinstance(v, list) else v.to(device)) for k, v in protein.items()}
                val_loss += loss_fn(model(graph, protein), label.to(device)).item() * len(label)
        val_loss /= len(val_set)
        print(f"epoch={epoch:03d} train_loss={total / len(train_set):.5f} val_loss={val_loss:.5f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
