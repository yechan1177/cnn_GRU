from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vcp.components.temporal import TemporalGRUNet, build_temporal_channel_groups


@dataclass(slots=True)
class BenchmarkArgs:
    """맥락 비교 실험 인자."""

    dataset_dir: Path
    device: str
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    cnn_channels: int
    dropout: float
    seed: int
    project: Path
    name: str
    proposed_ckpt: Path | None


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="기본/문헌형/제안 맥락 모델 비교 실험")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/processed/dataset_manual_context_braking_expanded_feat16_t045_20260324_115253",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--cnn-channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/exp_016_context_model_benchmark/runs",
    )
    parser.add_argument("--name", type=str, default="context_model_benchmark")
    parser.add_argument(
        "--proposed-ckpt",
        type=str,
        default="experiments/exp_015_temporal_braking_expanded_feat16/runs/mcnn_gru_braking_expanded_feat16_t045_e40_h96_c24/checkpoints/best.pt",
    )
    args = parser.parse_args()
    ckpt = Path(args.proposed_ckpt) if args.proposed_ckpt else None
    return BenchmarkArgs(
        dataset_dir=Path(args.dataset_dir),
        device=str(args.device),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=max(1e-6, float(args.lr)),
        hidden_dim=max(4, int(args.hidden_dim)),
        cnn_channels=max(4, int(args.cnn_channels)),
        dropout=max(0.0, min(0.5, float(args.dropout))),
        seed=int(args.seed),
        project=Path(args.project),
        name=str(args.name),
        proposed_ckpt=ckpt,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


class SequenceDataset(Dataset):
    """맥락 benchmark용 sequence dataset."""

    def __init__(self, rows: list[dict[str, Any]], feature_dim: int, window_size: int) -> None:
        self.rows = rows
        self.feature_dim = feature_dim
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        seq = row.get("features", {}).get("vision_feature_seq", [])
        normalized = [self._normalize_vector(vec) for vec in seq if isinstance(vec, list)]
        if len(normalized) < self.window_size:
            pad = [[0.0] * self.feature_dim for _ in range(self.window_size - len(normalized))]
            normalized = pad + normalized
        if len(normalized) > self.window_size:
            normalized = normalized[-self.window_size :]

        target = row.get("target", {})
        feature_keys = row.get("features", {}).get("feature_keys", [])
        return {
            "x": torch.tensor(normalized, dtype=torch.float32),
            "context": torch.tensor(int(target.get("context_index", 0)), dtype=torch.long),
            "boundary": torch.tensor(float(target.get("boundary_label", 0.0)), dtype=torch.float32),
            "importance": torch.tensor(float(target.get("importance_target", 0.0)), dtype=torch.float32),
            "sample_id": row.get("sample_id", f"sample_{index}"),
            "run_id": row.get("run_id", ""),
            "frame_id": int(row.get("frame_ids", [0])[-1]),
            "feature_keys": feature_keys,
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


# 문헌형 단일채널 CNN-GRU는 공용 모듈로 이동했다(하위 호환용 재노출).
from vcp.components.temporal import LiteratureCNNGRUNet  # noqa: E402


def infer_feature_dim(rows: list[dict[str, Any]]) -> int:
    dim = 0
    for row in rows:
        for vector in row.get("features", {}).get("vision_feature_seq", []):
            if isinstance(vector, list):
                dim = max(dim, len(vector))
    return max(1, dim)


def infer_window_size(rows: list[dict[str, Any]]) -> int:
    for row in rows:
        candidate = row.get("window_size")
        if candidate is not None:
            try:
                return max(2, int(candidate))
            except (TypeError, ValueError):
                pass
    return 8


def macro_f1_score(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    f1_values: list[float] = []
    for cls in range(num_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == cls and yp != cls)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
        f1_values.append(f1)
    return sum(f1_values) / max(1, len(f1_values))


def binary_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 1 and yp == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1


def brake_critical_recall(y_true: list[int], y_pred: list[int], brake_indices: set[int]) -> float:
    target_positions = [idx for idx, cls in enumerate(y_true) if cls in brake_indices]
    if not target_positions:
        return 0.0
    hit = sum(1 for idx in target_positions if y_pred[idx] in brake_indices)
    return hit / len(target_positions)


def evaluate_rule_baseline(
    rows: list[dict[str, Any]],
    label_map: dict[str, int],
) -> dict[str, Any]:
    inverse_label_map = {value: key for key, value in label_map.items()}
    y_true: list[int] = []
    y_pred: list[int] = []
    boundary_true: list[int] = []
    boundary_pred: list[int] = []

    def label_index(name: str) -> int:
        return int(label_map.get(name, label_map.get("normal_drive", 0)))

    for row in rows:
        seq = row.get("features", {}).get("vision_feature_seq", [])
        last = seq[-1] if seq else [0.0] * 16
        vec = [float(v) for v in last] + [0.0] * max(0, 16 - len(last))
        det_norm, _, _, mean_area, _, vehicle_ratio, _, _, _, _, _, roi_risk, motion_delta, center, looming, occlusion = vec[:16]

        pred_label = "normal_drive"
        if det_norm >= 0.28 and vehicle_ratio >= 0.20 and mean_area < 0.01:
            pred_label = "dense_traffic"
        if vehicle_ratio >= 0.05 and roi_risk >= 0.34 and center >= 0.76 and looming >= 0.02 and (occlusion >= 0.60 or motion_delta <= 0.01):
            pred_label = "hard_brake_risk"
        elif vehicle_ratio >= 0.05 and roi_risk >= 0.34 and center >= 0.74 and looming >= 0.008:
            pred_label = "brake_warning"
        elif vehicle_ratio >= 0.05 and roi_risk >= 0.28 and center >= 0.70:
            pred_label = "front_vehicle_follow"
        elif occlusion >= 0.60 and looming < 0.006 and roi_risk >= 0.28:
            pred_label = "post_brake_recovery"

        pred_boundary = 1 if pred_label in {"brake_warning", "hard_brake_risk"} and looming >= 0.008 else 0

        target = row.get("target", {})
        y_true.append(int(target.get("context_index", 0)))
        y_pred.append(label_index(pred_label))
        boundary_true.append(int(target.get("boundary_label", 0)))
        boundary_pred.append(pred_boundary)

    brake_indices = {
        idx for idx, name in inverse_label_map.items() if name in {"brake_warning", "hard_brake_risk"}
    }
    precision, recall, f1 = binary_f1(boundary_true, boundary_pred)
    context_acc = sum(int(yt == yp) for yt, yp in zip(y_true, y_pred, strict=True)) / max(1, len(y_true))
    return {
        "model_name": "basic_yolo_rule",
        "context_acc": context_acc,
        "context_macro_f1": macro_f1_score(y_true, y_pred, len(label_map)),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "brake_critical_recall": brake_critical_recall(y_true, y_pred, brake_indices),
    }


@torch.no_grad()
def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_map: dict[str, int],
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    boundary_true: list[int] = []
    boundary_pred: list[int] = []

    for batch in loader:
        x = batch["x"].to(device)
        output = model(x)
        context_logits = output["context_logits"]
        pred_context = torch.argmax(context_logits, dim=-1).cpu().tolist()
        pred_boundary = (torch.sigmoid(output["boundary_logit"]) >= 0.5).to(torch.int).cpu().tolist()

        y_true.extend(batch["context"].cpu().tolist())
        y_pred.extend(pred_context)
        boundary_true.extend(batch["boundary"].to(torch.int).cpu().tolist())
        boundary_pred.extend(pred_boundary)

    inverse_label_map = {value: key for key, value in label_map.items()}
    brake_indices = {
        idx for idx, name in inverse_label_map.items() if name in {"brake_warning", "hard_brake_risk"}
    }
    precision, recall, f1 = binary_f1(boundary_true, boundary_pred)
    context_acc = sum(int(yt == yp) for yt, yp in zip(y_true, y_pred, strict=True)) / max(1, len(y_true))
    return {
        "context_acc": context_acc,
        "context_macro_f1": macro_f1_score(y_true, y_pred, len(label_map)),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "brake_critical_recall": brake_critical_recall(y_true, y_pred, brake_indices),
    }


@torch.no_grad()
def evaluate_proposed_hybrid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_map: dict[str, int],
    *,
    roi_thr: float,
    center_thr: float,
    looming_warn_thr: float,
    looming_hard_thr: float,
    occlusion_thr: float,
    motion_stop_thr: float,
    boundary_thr: float,
    warn_boundary_thr: float,
    hard_boundary_thr: float,
    warn_prob_thr: float,
    hard_prob_thr: float,
    follow_prob_thr: float,
    warn_boost: float,
    hard_boost: float,
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    boundary_true: list[int] = []
    boundary_pred: list[int] = []

    brake_warning_idx = int(label_map["brake_warning"])
    hard_brake_idx = int(label_map["hard_brake_risk"])
    follow_idx = int(label_map["front_vehicle_follow"])
    inverse_label_map = {value: key for key, value in label_map.items()}
    brake_indices = {
        idx for idx, name in inverse_label_map.items() if name in {"brake_warning", "hard_brake_risk"}
    }

    for batch in loader:
        x = batch["x"].to(device)
        output = model(x)
        probs = torch.softmax(output["context_logits"], dim=-1)
        boundary_probs = torch.sigmoid(output["boundary_logit"])
        boosted = probs.clone()

        last = x[:, -1, :]
        vehicle_ratio = last[:, 5]
        roi_risk = last[:, 11]
        motion_delta = last[:, 12]
        center = last[:, 13]
        looming = last[:, 14]
        occlusion = last[:, 15]

        warn_rule_feat = (
            (vehicle_ratio >= 0.05)
            & (roi_risk >= roi_thr)
            & (center >= center_thr)
            & (looming >= looming_warn_thr)
        )
        hard_rule_feat = (
            warn_rule_feat
            & (looming >= looming_hard_thr)
            & ((occlusion >= occlusion_thr) | (motion_delta <= motion_stop_thr))
        )

        warn_rule_signal = (
            (boundary_probs >= warn_boundary_thr)
            & (
                (probs[:, follow_idx] >= follow_prob_thr)
                | (probs[:, brake_warning_idx] >= warn_prob_thr)
            )
        )
        hard_rule_signal = (
            (boundary_probs >= hard_boundary_thr)
            & (
                (probs[:, follow_idx] >= follow_prob_thr)
                | (probs[:, brake_warning_idx] >= warn_prob_thr)
                | (probs[:, hard_brake_idx] >= hard_prob_thr)
            )
        )

        warn_rule = warn_rule_feat | warn_rule_signal
        hard_rule = hard_rule_feat | hard_rule_signal

        boosted[:, brake_warning_idx] = boosted[:, brake_warning_idx] + (warn_rule.float() * warn_boost)
        boosted[:, hard_brake_idx] = boosted[:, hard_brake_idx] + (hard_rule.float() * hard_boost)

        pred_context = torch.argmax(boosted, dim=-1)
        pred_context = torch.where(
            hard_rule & ((probs[:, hard_brake_idx] >= hard_prob_thr) | (boundary_probs >= boundary_thr)),
            torch.full_like(pred_context, hard_brake_idx),
            pred_context,
        )
        pred_context = torch.where(
            (~hard_rule)
            & warn_rule
            & ((probs[:, brake_warning_idx] >= warn_prob_thr) | (boundary_probs >= boundary_thr)),
            torch.full_like(pred_context, brake_warning_idx),
            pred_context,
        )

        pred_boundary = (
            (boundary_probs >= boundary_thr) | warn_rule | hard_rule
        ).to(torch.int)

        y_true.extend(batch["context"].cpu().tolist())
        y_pred.extend(pred_context.cpu().tolist())
        boundary_true.extend(batch["boundary"].to(torch.int).cpu().tolist())
        boundary_pred.extend(pred_boundary.cpu().tolist())

    precision, recall, f1 = binary_f1(boundary_true, boundary_pred)
    context_acc = sum(int(yt == yp) for yt, yp in zip(y_true, y_pred, strict=True)) / max(1, len(y_true))
    return {
        "context_acc": context_acc,
        "context_macro_f1": macro_f1_score(y_true, y_pred, len(label_map)),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "brake_critical_recall": brake_critical_recall(y_true, y_pred, brake_indices),
    }


def tune_hybrid_rule_gate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_map: dict[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    best_metrics: dict[str, float] | None = None
    best_params: dict[str, float] | None = None
    best_score = -1.0

    roi_candidates = [0.30, 0.34]
    center_candidates = [0.72, 0.74]
    warn_looming_candidates = [0.006, 0.008]
    hard_looming_candidates = [0.014, 0.018]
    occlusion_candidates = [0.58, 0.62]
    motion_stop_candidates = [0.010, 0.015]
    boundary_candidates = [0.45, 0.50]
    warn_boundary_candidates = [0.82, 0.86, 0.90]
    hard_boundary_candidates = [0.90, 0.93, 0.95]

    for roi_thr in roi_candidates:
        for center_thr in center_candidates:
            for looming_warn_thr in warn_looming_candidates:
                for looming_hard_thr in hard_looming_candidates:
                    for occlusion_thr in occlusion_candidates:
                        for motion_stop_thr in motion_stop_candidates:
                            for boundary_thr in boundary_candidates:
                                for warn_boundary_thr in warn_boundary_candidates:
                                    for hard_boundary_thr in hard_boundary_candidates:
                                        params = {
                                            "roi_thr": roi_thr,
                                            "center_thr": center_thr,
                                            "looming_warn_thr": looming_warn_thr,
                                            "looming_hard_thr": looming_hard_thr,
                                            "occlusion_thr": occlusion_thr,
                                            "motion_stop_thr": motion_stop_thr,
                                            "boundary_thr": boundary_thr,
                                            "warn_boundary_thr": warn_boundary_thr,
                                            "hard_boundary_thr": hard_boundary_thr,
                                            "warn_prob_thr": 0.05,
                                            "hard_prob_thr": 0.03,
                                            "follow_prob_thr": 0.20,
                                            "warn_boost": 0.22,
                                            "hard_boost": 0.30,
                                        }
                                        metrics = evaluate_proposed_hybrid(
                                            model,
                                            loader,
                                            device,
                                            label_map,
                                            **params,
                                        )
                                        composite = (
                                            0.25 * metrics["context_acc"]
                                            + 0.20 * metrics["context_macro_f1"]
                                            + 0.20 * metrics["boundary_f1"]
                                            + 0.35 * metrics["brake_critical_recall"]
                                        )
                                        if composite > best_score:
                                            best_score = composite
                                            best_metrics = metrics
                                            best_params = params

    if best_metrics is None or best_params is None:
        raise RuntimeError("hybrid rule gate tuning 결과가 없습니다.")
    best_metrics = dict(best_metrics)
    best_metrics["composite"] = best_score
    return best_metrics, best_params


def train_literature_model(
    model: LiteratureCNNGRUNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[LiteratureCNNGRUNet, dict[str, float], list[dict[str, float]]]:
    context_loss_fn = nn.CrossEntropyLoss()

    train_rows_count = len(train_loader.dataset)  # type: ignore[arg-type]
    positives = sum(int(batch["boundary"].sum().item()) for batch in train_loader)
    negatives = train_rows_count - positives
    pos_weight = torch.tensor([float(negatives / max(1, positives))], dtype=torch.float32, device=device)
    boundary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state: dict[str, Any] | None = None
    best_score = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y_context = batch["context"].to(device)
            y_boundary = batch["boundary"].to(device)

            output = model(x)
            loss_context = context_loss_fn(output["context_logits"], y_context)
            loss_boundary = boundary_loss_fn(output["boundary_logit"], y_boundary)
            loss = loss_context + (0.7 * loss_boundary)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        val_metrics = evaluate_torch_model(model, val_loader, device, label_map={})
        composite = (0.6 * val_metrics["context_macro_f1"]) + (0.4 * val_metrics["brake_critical_recall"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_sum / max(1, train_count),
                "context_acc": val_metrics["context_acc"],
                "context_macro_f1": val_metrics["context_macro_f1"],
                "boundary_f1": val_metrics["boundary_f1"],
                "brake_critical_recall": val_metrics["brake_critical_recall"],
                "composite": composite,
            }
        )
        if composite > best_score:
            best_score = composite
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("문헌형 baseline 학습 중 best state를 저장하지 못했습니다.")

    model.load_state_dict(best_state)
    return model, history[-1], history


def save_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.dataset_dir / "train.jsonl")
    val_rows = read_jsonl(args.dataset_dir / "val.jsonl")
    label_map = json.loads((args.dataset_dir / "label_map.json").read_text(encoding="utf-8"))
    inverse_label_map = {value: key for key, value in label_map.items()}

    feature_dim = infer_feature_dim(train_rows + val_rows)
    window_size = infer_window_size(train_rows + val_rows)
    feature_keys = train_rows[0].get("features", {}).get("feature_keys", [])

    train_dataset = SequenceDataset(train_rows, feature_dim=feature_dim, window_size=window_size)
    val_dataset = SequenceDataset(val_rows, feature_dim=feature_dim, window_size=window_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = ensure_device(args.device)

    # 비교군 1: 기본 YOLO 규칙형 baseline
    rule_metrics = evaluate_rule_baseline(val_rows, label_map)

    # 비교군 2: 문헌형 단일채널 CNN-GRU baseline
    literature_model = LiteratureCNNGRUNet(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_contexts=len(label_map),
        dropout=args.dropout,
        cnn_channels=args.cnn_channels,
    ).to(device)

    # evaluate_torch_model 내부 label_map 필요해서 임시 wrapper를 둔다.
    original_evaluate = evaluate_torch_model

    def literature_eval(model_obj: nn.Module, loader_obj: DataLoader) -> dict[str, float]:
        return original_evaluate(model_obj, loader_obj, device, label_map)

    # monkey patch 없이 수동 루프
    context_loss_fn = nn.CrossEntropyLoss()
    positives = sum(int(row.get("target", {}).get("boundary_label", 0)) for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight = torch.tensor([float(negatives / max(1, positives))], dtype=torch.float32, device=device)
    boundary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(literature_model.parameters(), lr=args.lr)
    best_state: dict[str, Any] | None = None
    best_composite = -1.0
    literature_history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        literature_model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y_context = batch["context"].to(device)
            y_boundary = batch["boundary"].to(device)

            output = literature_model(x)
            loss_context = context_loss_fn(output["context_logits"], y_context)
            loss_boundary = boundary_loss_fn(output["boundary_logit"], y_boundary)
            loss = loss_context + (0.7 * loss_boundary)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        val_metrics = literature_eval(literature_model, val_loader)
        composite = (0.6 * val_metrics["context_macro_f1"]) + (0.4 * val_metrics["brake_critical_recall"])
        literature_history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss_sum / max(1, train_count), 6),
                "context_acc": round(val_metrics["context_acc"], 6),
                "context_macro_f1": round(val_metrics["context_macro_f1"], 6),
                "boundary_f1": round(val_metrics["boundary_f1"], 6),
                "brake_critical_recall": round(val_metrics["brake_critical_recall"], 6),
                "composite": round(composite, 6),
            }
        )
        if composite > best_composite:
            best_composite = composite
            best_state = {key: value.detach().cpu() for key, value in literature_model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("문헌형 baseline 학습 중 best state를 찾지 못했습니다.")

    literature_model.load_state_dict(best_state)
    literature_metrics = literature_eval(literature_model, val_loader)
    torch.save(
        {
            "state_dict": literature_model.state_dict(),
            "model_config": {
                "input_dim": feature_dim,
                "hidden_dim": args.hidden_dim,
                "num_contexts": len(label_map),
                "dropout": args.dropout,
                "window_size": window_size,
                "cnn_channels": args.cnn_channels,
                "architecture": "single_channel_cnn_gru",
            },
            "labels": label_map,
            "feature_keys": feature_keys,
            "metrics": literature_metrics,
        },
        run_dir / "literature_single_channel_cnn_gru_best.pt",
    )

    # 비교군 3: 제안 멀티채널 CNN-GRU
    if args.proposed_ckpt is None or not args.proposed_ckpt.exists():
        raise FileNotFoundError(f"제안 모델 체크포인트가 없습니다: {args.proposed_ckpt}")

    proposed_checkpoint = torch.load(args.proposed_ckpt, map_location=device)
    proposed_config = proposed_checkpoint.get("model_config", {})
    proposed_model = TemporalGRUNet(
        input_dim=int(proposed_config.get("input_dim", feature_dim)),
        hidden_dim=int(proposed_config.get("hidden_dim", args.hidden_dim)),
        num_contexts=int(proposed_config.get("num_contexts", len(label_map))),
        dropout=float(proposed_config.get("dropout", args.dropout)),
        cnn_channels=int(proposed_config.get("cnn_channels", args.cnn_channels)),
        channel_groups=proposed_config.get(
            "channel_groups",
            build_temporal_channel_groups(feature_dim, feature_keys=feature_keys),
        ),
    ).to(device)
    proposed_model.load_state_dict(proposed_checkpoint["state_dict"])
    proposed_metrics = evaluate_torch_model(proposed_model, val_loader, device, label_map)
    hybrid_metrics, hybrid_params = tune_hybrid_rule_gate(proposed_model, val_loader, device, label_map)

    comparison_rows = [
        {
            "model": "basic_yolo_rule",
            "type": "basic_baseline",
            "context_acc": round(rule_metrics["context_acc"], 6),
            "context_macro_f1": round(rule_metrics["context_macro_f1"], 6),
            "boundary_f1": round(rule_metrics["boundary_f1"], 6),
            "brake_critical_recall": round(rule_metrics["brake_critical_recall"], 6),
        },
        {
            "model": "single_channel_cnn_gru",
            "type": "literature_style",
            "context_acc": round(literature_metrics["context_acc"], 6),
            "context_macro_f1": round(literature_metrics["context_macro_f1"], 6),
            "boundary_f1": round(literature_metrics["boundary_f1"], 6),
            "brake_critical_recall": round(literature_metrics["brake_critical_recall"], 6),
        },
        {
            "model": "proposed_multichannel_cnn_gru",
            "type": "proposed",
            "context_acc": round(proposed_metrics["context_acc"], 6),
            "context_macro_f1": round(proposed_metrics["context_macro_f1"], 6),
            "boundary_f1": round(proposed_metrics["boundary_f1"], 6),
            "brake_critical_recall": round(proposed_metrics["brake_critical_recall"], 6),
        },
        {
            "model": "proposed_hybrid_rule_gate",
            "type": "proposed_plus_rule",
            "context_acc": round(hybrid_metrics["context_acc"], 6),
            "context_macro_f1": round(hybrid_metrics["context_macro_f1"], 6),
            "boundary_f1": round(hybrid_metrics["boundary_f1"], 6),
            "brake_critical_recall": round(hybrid_metrics["brake_critical_recall"], 6),
        },
    ]

    csv_path = run_dir / "context_model_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    save_markdown_table(run_dir / "context_model_benchmark.md", comparison_rows)
    (run_dir / "literature_history.json").write_text(
        json.dumps(literature_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 루트 산출물 저장
    root_csv = Path("context_model_benchmark_actual_table.csv")
    root_md = Path("context_model_benchmark_actual_table.md")
    root_png = Path("context_model_benchmark_metrics.png")

    root_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    root_md.write_text((run_dir / "context_model_benchmark.md").read_text(encoding="utf-8"), encoding="utf-8")

    metric_names = ["context_acc", "context_macro_f1", "boundary_f1", "brake_critical_recall"]
    labels = [row["model"] for row in comparison_rows]
    x = range(len(labels))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for offset, metric_name in enumerate(metric_names):
        values = [row[metric_name] for row in comparison_rows]
        plt.bar([idx + ((offset - 1.5) * width) for idx in x], values, width=width, label=metric_name)
    plt.xticks(list(x), labels, rotation=10)
    plt.ylim(0.0, 1.0)
    plt.ylabel("score")
    plt.title("Context Model Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_png, dpi=160)
    plt.close()

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "proposed_ckpt": str(args.proposed_ckpt),
        "run_dir": str(run_dir),
        "metrics": metric_names,
        "models": comparison_rows,
        "root_outputs": {
            "csv": str(root_csv.resolve()),
            "markdown": str(root_md.resolve()),
            "figure": str(root_png.resolve()),
        },
        "notes": {
            "basic_baseline": "YOLO 출력 기반 규칙형 baseline",
            "literature_style": "문헌형 단일채널 CNN-GRU",
            "proposed": "멀티채널 CNN-GRU 제안모델",
            "proposed_plus_rule": "제안모델 뒤에 ROI/looming/occlusion 규칙 게이트 추가",
            "tuning_warning": "hybrid rule gate는 동일 validation split에서 exploratory tuning한 결과이므로 낙관적으로 보일 수 있음",
        },
        "hybrid_rule_params": hybrid_params,
    }
    (run_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path("context_model_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
