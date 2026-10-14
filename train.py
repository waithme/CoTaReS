"""Train CoTaReS on corrected MoleculeACE v3.0.0 CSV files.

This public runner performs the fixed training/validation protocol. It does not
read test labels or ship datasets, baseline implementations, or ablation code.
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import torch

from model import CoTaReS

RDLogger.DisableLog("rdApp.warning")


def load_training_data(path, features=8192, neighbors=8):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"smiles", "split"}.issubset(rows[0]):
        raise ValueError("Expected MoleculeACE rows with smiles and split columns")
    pool = np.array([i for i, row in enumerate(rows) if row["split"] == "train"])
    if not len(pool) or any(row["split"] not in {"train", "test"} for row in rows):
        raise ValueError("Missing or invalid official train/test partition")
    train, validation = train_test_split(
        pool, test_size=0.1, random_state=0, shuffle=True
    )
    if "y [pEC50/pKi]" in rows[0]:
        labels = np.array([float(rows[i]["y [pEC50/pKi]"]) for i in pool], np.float32)
    elif "exp_mean [nM]" in rows[0]:
        labels = np.array([9.0 - np.log10(float(rows[i]["exp_mean [nM]"])) for i in pool], np.float32)
    else:
        raise ValueError("Expected a pEC50/pKi column or exp_mean [nM]")
    if not np.isfinite(labels).all():
        raise ValueError("Nonfinite training activity")

    x = np.zeros((len(rows), features), np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
    canonical = {}
    sparse = {}
    for i in pool:
        molecule = Chem.MolFromSmiles(rows[i]["smiles"])
        if molecule is None:
            raise ValueError(f"Invalid training SMILES at row {i}")
        canonical[i] = Chem.MolToSmiles(molecule, isomericSmiles=True)
        counts = generator.GetSparseCountFingerprint(molecule).GetNonzeroElements()
        sparse[i] = counts
        for token, count in counts.items():
            x[i, int(token) % features] += count
    if {canonical[i] for i in train} & {canonical[i] for i in validation}:
        raise ValueError("Train/validation chemical-identity overlap")

    vocabulary = {token: col for col, token in enumerate(sorted({t for m in sparse.values() for t in m}))}
    row_index = {int(i): pos for pos, i in enumerate(pool)}
    sparse_rows, sparse_cols = [], []
    for i in pool:
        for token in sparse[i]:
            sparse_rows.append(row_index[int(i)])
            sparse_cols.append(vocabulary[token])
    presence = csr_matrix(
        (np.ones(len(sparse_rows), np.float32), (sparse_rows, sparse_cols)),
        shape=(len(pool), len(vocabulary)),
    )
    train_positions = np.array([row_index[int(i)] for i in train])
    intersection = (presence @ presence[train_positions].T).toarray()
    sizes = np.asarray(presence.sum(1)).reshape(-1)
    similarity = intersection / np.maximum(
        sizes[:, None] + sizes[train_positions][None, :] - intersection, 1
    )
    for i in train:
        same_identity = [j for j, idx in enumerate(train) if canonical[int(i)] == canonical[int(idx)]]
        similarity[row_index[int(i)], same_identity] = -np.inf
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :neighbors]
    scores = np.take_along_axis(similarity, order, axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Insufficient distinct training references")
    references = np.full((len(rows), neighbors), -1, np.int64)
    references[pool] = train[order]
    y = np.zeros(len(rows), np.float32)
    y[pool] = labels
    return x, y, train, validation, references, {
        "data_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "reference_indices": {str(int(i)): references[i].tolist() for i in pool},
    }


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def train_one(data_dir, output_dir, dataset, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    source = Path(data_dir) / f"{dataset}.csv"
    x, y, train, validation, references, manifest = load_training_data(source)
    mean, std = float(y[train].mean()), max(float(y[train].std()), 1e-6)
    tx = torch.tensor(x, device=device)
    ty = torch.tensor((y - mean) / std, device=device)
    ref_indices = torch.tensor(references, device=device)
    model = CoTaReS().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    target = Path(output_dir) / dataset / f"seed-{seed}"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output exists; choose a new directory: {target}")
    target.mkdir(parents=True)
    (target / "split_and_references.json").write_text(json.dumps(manifest, indent=2))
    (target / "config.json").write_text(json.dumps({
        "dataset": dataset, "seed": seed, "data_sha256": manifest["data_sha256"],
        "label_mean": mean, "label_std": std, "batch_size": 128,
        "learning_rate": 1e-4, "weight_decay": 1e-5, "max_epochs": 100,
        "patience": 15, "selection": "validation overall RMSE",
        "model_source_sha256": hashlib.sha256((Path(__file__).parent / "model.py").read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "device": str(device),
    }, indent=2))

    def forward(indices):
        ids = torch.as_tensor(indices, device=device)
        r = ref_indices[ids]
        return model(tx[ids], tx[r], ty[r])

    @torch.no_grad()
    def validation_rmse():
        model.eval()
        predictions = []
        for start in range(0, len(validation), 128):
            value, _ = forward(validation[start:start + 128])
            predictions.extend((value.cpu().numpy() * std + mean).tolist())
        return rmse(y[validation], predictions)

    best, stale, history = float("inf"), 0, []
    started = time.time()
    for epoch in range(1, 101):
        model.train()
        order = np.random.default_rng(seed * 100000 + epoch).permutation(train)
        total = 0.0
        for start in range(0, len(order), 128):
            ids = order[start:start + 128]
            predicted, details = forward(ids)
            index = torch.as_tensor(ids, device=device)
            main_loss = torch.mean((predicted - ty[index]) ** 2)
            loss = main_loss + 0.1 * torch.mean((details["base"] - ty[index]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        score = validation_rmse()
        record = {"epoch": epoch, "train_loss": total / len(train),
                  "validation_overall_rmse": score, "elapsed_seconds": time.time() - started}
        history.append(record)
        print(json.dumps({"dataset": dataset, "seed": seed, **record}), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation_rmse": score}, target / "best.pt")
        else:
            stale += 1
        (target / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= 15:
            break
    print(json.dumps({"completed": dataset, "seed": seed, "best_validation_rmse": best}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Directory of MoleculeACE v3.0.0 CSVs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if any(seed not in range(5) for seed in args.seeds):
        parser.error("Seeds must be 0–4")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    for dataset in args.datasets:
        for seed in args.seeds:
            train_one(args.data, args.output, dataset, seed, device)


if __name__ == "__main__":
    main()
