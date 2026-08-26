"""Train one fixed BioSNAP fold with the paper's main MMEL-DTI protocol."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Subset

from .config import ModelConfig
from .data import DTIDataset, collate, load_embedding_map
from .model import MMELDTI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--esm-embeddings", required=True)
    parser.add_argument("--structure-embeddings", required=True)
    parser.add_argument("--output-dir", default="runs/biosnap_fold1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-model", default=None,
                        help="Resume model weights and history from a previous run.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch, device):
    graph, protein, label = batch
    graph = graph.to(device)
    protein = {key: value if isinstance(value, list) else value.to(device)
               for key, value in protein.items()}
    return graph, protein, label.to(device)


def choose_threshold(labels, scores):
    thresholds = roc_curve(labels, scores)[2]
    thresholds = thresholds[np.isfinite(thresholds)]
    if not len(thresholds):
        return 0.5
    values = [f1_score(labels, scores >= threshold, zero_division=0)
              for threshold in thresholds]
    return float(thresholds[int(np.argmax(values))])


def evaluate(model, loader, device, threshold=None):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses, labels, scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            graph, protein, label = move_batch(batch, device)
            logits = model(graph, protein)
            losses.append(criterion(logits, label).item())
            labels.extend(label.cpu().numpy())
            scores.extend(logits.sigmoid().cpu().numpy())
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    threshold = choose_threshold(labels, scores) if threshold is None else float(threshold)
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    metrics = {
        "loss": float(np.mean(losses)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "accuracy": float((prediction == labels).mean()),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "sensitivity": float(recall_score(labels, prediction, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "threshold": threshold,
    }
    return metrics, labels, scores, prediction


def make_loader(dataset, indices, batch_size, shuffle, seed):
    return DataLoader(
        Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The native bidirectional Mamba-2 kernels require CUDA.")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    dataset = DTIDataset(
        args.csv, load_embedding_map(args.esm_embeddings),
        load_embedding_map(args.structure_embeddings), cfg.max_protein_length,
    )
    split = np.load(args.splits)
    train_indices = split[f"fold{args.fold}_train"]
    val_indices = split[f"fold{args.fold}_val"]
    test_indices = split[f"fold{args.fold}_test"]
    if max(map(np.max, (train_indices, val_indices, test_indices))) >= len(dataset):
        raise ValueError("Split indices do not match the input CSV row order.")
    print(f"BioSNAP fold {args.fold}: train={len(train_indices)}, "
          f"validation={len(val_indices)}, test={len(test_indices)}", flush=True)
    dataset.precompute_graphs()

    train_loader = make_loader(dataset, train_indices, args.batch_size, True,
                               args.seed + args.fold - 1)
    val_loader = make_loader(dataset, val_indices, args.batch_size, False, args.seed)
    test_loader = make_loader(dataset, test_indices, args.batch_size, False, args.seed)

    model = MMELDTI(cfg).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, patience_count = 0.0, 0, 0
    best_state, best_validation, history = None, None, []
    start_epoch = 1
    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device,
                                         weights_only=True))
        history_file = output_dir / "history.csv"
        if history_file.exists():
            history = pd.read_csv(history_file).to_dict("records")
            best_row = max(history, key=lambda row: row["val_auroc"])
            best_auc = float(best_row["val_auroc"])
            best_epoch = int(best_row["epoch"])
            patience_count = len(history) - best_epoch
            metric_names = ("loss", "auroc", "auprc", "f1", "accuracy",
                            "precision", "recall", "sensitivity", "specificity",
                            "mcc", "threshold")
            best_validation = {
                name: float(best_row[f"val_{name}"]) for name in metric_names
            }
            best_state = copy.deepcopy(model.state_dict())
            start_epoch = int(history[-1]["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} with best validation AUC {best_auc:.5f}.",
              flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        for batch in train_loader:
            graph, protein, label = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(graph, protein), label)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        validation, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step(validation["auroc"])
        record = {
            "epoch": epoch, "train_loss": float(np.mean(train_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.5f} "
            f"val_loss={validation['loss']:.5f} val_auc={validation['auroc']:.5f} "
            f"val_auprc={validation['auprc']:.5f} time={record['epoch_seconds']:.1f}s",
            flush=True,
        )
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        if validation["auroc"] > best_auc:
            best_auc, best_epoch, patience_count = validation["auroc"], epoch, 0
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    model.load_state_dict(best_state)
    test, labels, scores, predictions = evaluate(
        model, test_loader, device, threshold=best_validation["threshold"])
    rows = dataset.frame.iloc[test_indices][["drug_idx", "protein_idx", "Label"]].copy()
    rows["score"] = scores
    rows["prediction"] = predictions
    rows.to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "dataset": "BioSNAP", "fold": args.fold, "seed": args.seed,
        "train_size": len(train_indices), "validation_size": len(val_indices),
        "test_size": len(test_indices), "batch_size": args.batch_size,
        "parameters": trainable, "best_epoch": best_epoch,
        "epochs_completed": len(history), "training_seconds": time.perf_counter() - started,
        "validation": best_validation, "test": test,
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
