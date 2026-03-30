from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vcp.components.temporal import TemporalGRUNet, build_temporal_channel_groups


@dataclass(slots=True)
class TrainArgs:
    """Temporal GRU 학습 인자."""

    dataset_dir: Path
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    cnn_channels: int
    gru_layers: int
    head_hidden_dim: int
    dropout: float
    device: str
    num_workers: int
    project: Path
    name: str
    seed: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="processed dataset 기반 Temporal GRU 학습")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/dataset_vehicle_validation_yolo6cls_20260318_150443",
        help="train.jsonl/val.jsonl/label_map.json이 있는 데이터셋 경로",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cnn-channels", type=int, default=16)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--head-hidden-dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/exp_005_temporal_gru_training/runs",
    )
    parser.add_argument("--name", type=str, default="gru_e8_vehicle")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    return TrainArgs(
        dataset_dir=Path(args.dataset_dir),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=max(1e-6, float(args.lr)),
        hidden_dim=max(4, int(args.hidden_dim)),
        cnn_channels=max(4, int(args.cnn_channels)),
        gru_layers=max(1, int(args.gru_layers)),
        head_hidden_dim=max(0, int(args.head_hidden_dim)),
        dropout=max(0.0, min(0.5, float(args.dropout))),
        device=str(args.device),
        num_workers=max(0, int(args.num_workers)),
        project=Path(args.project),
        name=str(args.name),
        seed=int(args.seed),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


class SequenceDataset(Dataset):
    """JSONL 시퀀스 샘플을 Tensor로 변환하는 Dataset."""

    def __init__(self, rows: list[dict[str, Any]], feature_dim: int, window_size: int) -> None:
        self._rows = rows
        self.feature_dim = max(1, int(feature_dim))
        self.window_size = max(2, int(window_size))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._rows[index]
        features = row.get("features", {})
        seq = features.get("vision_feature_seq", [])

        normalized_seq = [self._normalize_vector(item) for item in seq if isinstance(item, list)]
        if len(normalized_seq) < self.window_size:
            pad = [[0.0] * self.feature_dim for _ in range(self.window_size - len(normalized_seq))]
            normalized_seq = pad + normalized_seq
        if len(normalized_seq) > self.window_size:
            normalized_seq = normalized_seq[-self.window_size :]

        target = row.get("target", {})
        context_index = int(target.get("context_index", 0))
        boundary_label = float(target.get("boundary_label", 0.0))
        importance_target = float(target.get("importance_target", 0.0))
        uncertainty_target = max(0.0, min(1.0, 1.0 - importance_target))

        return {
            "x": torch.tensor(normalized_seq, dtype=torch.float32),
            "context": torch.tensor(context_index, dtype=torch.long),
            "boundary": torch.tensor(boundary_label, dtype=torch.float32),
            "uncertainty": torch.tensor(uncertainty_target, dtype=torch.float32),
        }

    def _normalize_vector(self, vector: list[Any]) -> list[float]:
        output: list[float] = []
        for item in vector[: self.feature_dim]:
            try:
                output.append(float(item))
            except (TypeError, ValueError):
                output.append(0.0)
        if len(output) < self.feature_dim:
            output.extend([0.0] * (self.feature_dim - len(output)))
        return output


@torch.no_grad()
def evaluate(
    model: TemporalGRUNet,
    loader: DataLoader,
    context_loss_fn: nn.Module,
    boundary_loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_count = 0
    correct_context = 0

    tp = 0
    fp = 0
    fn = 0

    total_uncertainty_abs_error = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y_context = batch["context"].to(device)
        y_boundary = batch["boundary"].to(device)
        y_uncertainty = batch["uncertainty"].to(device)

        output = model(x)
        context_logits = output["context_logits"]
        boundary_logit = output["boundary_logit"]
        uncertainty_logit = output["uncertainty_logit"]

        loss_context = context_loss_fn(context_logits, y_context)
        loss_boundary = boundary_loss_fn(boundary_logit, y_boundary)
        pred_uncertainty = torch.sigmoid(uncertainty_logit)
        loss_uncertainty = torch.mean(torch.abs(pred_uncertainty - y_uncertainty))

        loss = loss_context + (0.7 * loss_boundary) + (0.3 * loss_uncertainty)

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        pred_context = torch.argmax(context_logits, dim=-1)
        correct_context += int((pred_context == y_context).sum().item())

        pred_boundary = (torch.sigmoid(boundary_logit) >= 0.5).to(torch.int)
        target_boundary = y_boundary.to(torch.int)

        tp += int(((pred_boundary == 1) & (target_boundary == 1)).sum().item())
        fp += int(((pred_boundary == 1) & (target_boundary == 0)).sum().item())
        fn += int(((pred_boundary == 0) & (target_boundary == 1)).sum().item())

        total_uncertainty_abs_error += float(torch.abs(pred_uncertainty - y_uncertainty).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)

    return {
        "loss": total_loss / max(1, total_count),
        "context_acc": correct_context / max(1, total_count),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "uncertainty_mae": total_uncertainty_abs_error / max(1, total_count),
    }


def infer_feature_dim(rows: list[dict[str, Any]]) -> int:
    dim = 0
    for row in rows:
        seq = row.get("features", {}).get("vision_feature_seq", [])
        for vector in seq:
            if isinstance(vector, list):
                dim = max(dim, len(vector))
    return max(1, dim)


def infer_window_size(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 8
    candidate = rows[0].get("window_size", 8)
    try:
        return max(2, int(candidate))
    except (TypeError, ValueError):
        return 8


def infer_feature_keys(rows: list[dict[str, Any]]) -> list[str] | None:
    for row in rows:
        candidate = row.get("features", {}).get("feature_keys")
        if isinstance(candidate, list) and candidate:
            return [str(item) for item in candidate]
    return None


def ensure_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def save_epoch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = args.dataset_dir / "train.jsonl"
    val_path = args.dataset_dir / "val.jsonl"
    label_map_path = args.dataset_dir / "label_map.json"

    if not train_path.exists() or not val_path.exists() or not label_map_path.exists():
        raise FileNotFoundError(
            f"필수 파일이 없습니다: {train_path}, {val_path}, {label_map_path}"
        )

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    if not train_rows or not val_rows:
        raise ValueError("학습/검증 샘플이 비어 있습니다.")

    feature_dim = infer_feature_dim(train_rows + val_rows)
    window_size = infer_window_size(train_rows)
    feature_keys = infer_feature_keys(train_rows + val_rows)
    channel_groups = build_temporal_channel_groups(feature_dim, feature_keys=feature_keys)
    num_contexts = max(2, len(label_map))

    train_dataset = SequenceDataset(train_rows, feature_dim=feature_dim, window_size=window_size)
    val_dataset = SequenceDataset(val_rows, feature_dim=feature_dim, window_size=window_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = ensure_device(args.device)

    model = TemporalGRUNet(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_contexts=num_contexts,
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
        channel_groups=channel_groups,
        gru_layers=args.gru_layers,
        head_hidden_dim=args.head_hidden_dim,
    ).to(device)

    context_loss_fn = nn.CrossEntropyLoss()

    positives = sum(int(row.get("target", {}).get("boundary_label", 0)) for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight_value = float(negatives / max(1, positives))
    boundary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_dir = args.project / args.name
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    epoch_metrics: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            y_context = batch["context"].to(device)
            y_boundary = batch["boundary"].to(device)
            y_uncertainty = batch["uncertainty"].to(device)

            output = model(x)
            context_logits = output["context_logits"]
            boundary_logit = output["boundary_logit"]
            uncertainty_logit = output["uncertainty_logit"]

            loss_context = context_loss_fn(context_logits, y_context)
            loss_boundary = boundary_loss_fn(boundary_logit, y_boundary)
            pred_uncertainty = torch.sigmoid(uncertainty_logit)
            loss_uncertainty = torch.mean(torch.abs(pred_uncertainty - y_uncertainty))

            loss = loss_context + (0.7 * loss_boundary) + (0.3 * loss_uncertainty)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(1, train_count)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            context_loss_fn=context_loss_fn,
            boundary_loss_fn=boundary_loss_fn,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "val_context_acc": round(float(val_metrics["context_acc"]), 6),
            "val_boundary_precision": round(float(val_metrics["boundary_precision"]), 6),
            "val_boundary_recall": round(float(val_metrics["boundary_recall"]), 6),
            "val_boundary_f1": round(float(val_metrics["boundary_f1"]), 6),
            "val_uncertainty_mae": round(float(val_metrics["uncertainty_mae"]), 6),
        }
        epoch_metrics.append(row)

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "num_contexts": num_contexts,
                "dropout": args.dropout,
                "window_size": window_size,
                "cnn_channels": args.cnn_channels,
                "gru_layers": args.gru_layers,
                "head_hidden_dim": args.head_hidden_dim,
                "channel_groups": channel_groups,
            },
            "train_config": {
                "dataset_dir": str(args.dataset_dir),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": str(device),
                "seed": args.seed,
            },
            "labels": label_map,
            "epoch": epoch,
            "metrics": row,
        }

        torch.save(checkpoint, checkpoints_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(checkpoint, checkpoints_dir / "best.pt")

    metrics_path = run_dir / "metrics.csv"
    save_epoch_metrics(metrics_path, epoch_metrics)

    best_row = min(epoch_metrics, key=lambda item: float(item["val_loss"]))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "feature_dim": feature_dim,
        "window_size": window_size,
        "num_contexts": num_contexts,
        "cnn_channels": args.cnn_channels,
        "gru_layers": args.gru_layers,
        "head_hidden_dim": args.head_hidden_dim,
        "channel_groups": channel_groups,
        "device": str(device),
        "best_epoch": int(best_row["epoch"]),
        "best_metrics": best_row,
        "paths": {
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_path),
            "best_ckpt": str(checkpoints_dir / "best.pt"),
            "last_ckpt": str(checkpoints_dir / "last.pt"),
        },
    }

    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (run_dir / "result_summary.md").write_text(
        "\n".join(
            [
                "# Temporal GRU 학습 요약",
                "",
                f"- dataset_dir: `{args.dataset_dir}`",
                f"- device: `{device}`",
                f"- feature_dim: `{feature_dim}`",
                f"- window_size: `{window_size}`",
                f"- cnn_channels: `{args.cnn_channels}`",
                f"- gru_layers: `{args.gru_layers}`",
                f"- head_hidden_dim: `{args.head_hidden_dim}`",
                f"- channel_groups: `{channel_groups}`",
                f"- best_epoch: `{best_row['epoch']}`",
                f"- val_loss: `{best_row['val_loss']}`",
                f"- val_context_acc: `{best_row['val_context_acc']}`",
                f"- val_boundary_f1: `{best_row['val_boundary_f1']}`",
                f"- val_uncertainty_mae: `{best_row['val_uncertainty_mae']}`",
                "",
                "## 체크포인트",
                f"- best: `{checkpoints_dir / 'best.pt'}`",
                f"- last: `{checkpoints_dir / 'last.pt'}`",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
