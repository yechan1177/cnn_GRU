from __future__ import annotations

"""시계열 맥락 모델 학습/추론(학습 서버 GPU와 클라우드 CPU 공용)."""

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ..components.temporal import build_temporal_channel_groups, build_temporal_model
from .data import gather_windows
from .metrics import macro_f1

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSpec:
    """실험 모델 정의."""

    name: str
    architecture: str  # gru | single_channel_cnn_gru | multichannel_cnn_gru | mlp_last
    feature_version: str = "v2"
    grouping: str = "semantic"  # semantic | balanced | single (multichannel 전용)
    loss: str = "ce"  # ce | cb_focal
    window: int = 8
    hidden_dim: int = 96
    cnn_channels: int = 24
    dropout: float = 0.1
    epochs: int = 14
    lr: float = 1e-3
    batch_size: int = 512
    patience: int = 4
    boundary_weight: float = 0.5
    boundary_tol: int = 2
    action_weight: float = 0.0  # >0이면 행동(가감속) 예측 보조 head 학습(VLA 연계)
    hybrid_gate: bool = False
    ema_smoothing: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionHeadWrapper(nn.Module):
    """기존 temporal 모델 위에 행동(미래 가감속) 회귀 head를 붙인다.

    맥락 분류 head와 같은 GRU 표현을 공유하므로, 같은 경량 모델이
    '맥락(언어로 서술 가능한 상태) + 행동'을 함께 내는 VLA용 교사/필터로 쓰일 수 있다.
    """

    def __init__(self, base: nn.Module, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.base = base
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self._feat: torch.Tensor | None = None
        base.context_head.register_forward_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._feat = inputs[0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = dict(self.base(x))
        assert self._feat is not None
        out["action"] = self.action_head(self._feat)
        return out


def build_model(spec: ModelSpec, input_dim: int, n_classes: int, action_dim: int = 0) -> nn.Module:
    groups = None
    arch = spec.architecture
    if arch == "multichannel_cnn_gru":
        groups = build_temporal_channel_groups(input_dim, grouping=spec.grouping, feature_version=spec.feature_version)
    model = build_temporal_model(
        architecture=arch,
        input_dim=input_dim,
        hidden_dim=spec.hidden_dim,
        num_contexts=n_classes,
        dropout=spec.dropout,
        cnn_channels=spec.cnn_channels,
        channel_groups=groups,
    )
    if action_dim > 0:
        model = ActionHeadWrapper(model, spec.hidden_dim, action_dim)
    return model


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _dilate_boundary(boundary: np.ndarray, group: np.ndarray, tol: int) -> np.ndarray:
    """경계 프레임을 ±tol 프레임으로 확장한 학습 타깃(같은 그룹 안에서만)."""

    out = boundary.astype(np.float32).copy()
    for shift in range(1, tol + 1):
        fwd = np.zeros_like(out)
        fwd[shift:] = boundary[:-shift] * (group[shift:] == group[:-shift])
        bwd = np.zeros_like(out)
        bwd[:-shift] = boundary[shift:] * (group[:-shift] == group[shift:])
        out = np.maximum(out, np.maximum(fwd, bwd))
    return out


def _class_weights(y: np.ndarray, n_classes: int, beta: float = 0.999) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = np.where(counts > 0, 1.0 / np.maximum(eff, 1e-9), 0.0)
    return (w / w[w > 0].mean()).astype(np.float32)


def _focal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    p_t = logp_t.exp()
    w = weight[target]
    return (-(w * (1.0 - p_t) ** gamma * logp_t)).sum() / w.sum().clamp_min(1e-9)


@dataclass(slots=True)
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int
    train_seconds: float
    params: int


def train_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    boundary: np.ndarray,
    group: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    n_classes: int,
    seed: int,
    device: str = "cpu",
    action: np.ndarray | None = None,
) -> TrainResult:
    """창 인덱스 행렬(train/val)로 모델을 학습하고 검증 macro-F1 최고 시점을 반환한다."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    action_dim = int(action.shape[1]) if (action is not None and spec.action_weight > 0) else 0
    model = build_model(spec, X.shape[1], n_classes, action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    soft_boundary = _dilate_boundary(boundary, group, spec.boundary_tol)

    y_train = y[train_index[:, -1]].astype(np.int64)
    pos = float(soft_boundary[train_index[:, -1]].sum())
    pos_weight = torch.tensor([min(20.0, (len(train_index) - pos) / max(1.0, pos))], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weights = torch.tensor(_class_weights(y_train, n_classes), device=device)
    ce = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_score, best_epoch, bad = -1.0, -1, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(spec.epochs):
        model.train()
        perm = rng.permutation(len(train_index))
        total = 0.0
        for b in range(0, len(perm), spec.batch_size):
            sel = train_index[perm[b : b + spec.batch_size]]
            xb = torch.from_numpy(gather_windows(X, sel)).to(device)
            last = sel[:, -1]
            yb = torch.from_numpy(y[last].astype(np.int64)).to(device)
            bb = torch.from_numpy(soft_boundary[last]).to(device)
            out = model(xb)
            if spec.loss == "cb_focal":
                loss = _focal_loss(out["context_logits"], yb, weights)
            else:
                loss = ce(out["context_logits"], yb)
            loss = loss + spec.boundary_weight * bce(out["boundary_logit"], bb)
            if action_dim and action is not None:
                ab = torch.from_numpy(action[last].astype(np.float32)).to(device)
                loss = loss + spec.action_weight * nn.functional.smooth_l1_loss(out["action"], ab)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(sel)
        logits, _, _ = predict(model, X, val_index, device=device)
        val_f1 = macro_f1(y[val_index[:, -1]], logits.argmax(1), n_classes)
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_macro_f1": val_f1})
        if val_f1 > best_score:
            best_score, best_epoch, bad = val_f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= spec.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, history, best_epoch, time.perf_counter() - started, count_parameters(model))


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    index: np.ndarray,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(context logits, boundary logits, action 예측 또는 None)."""

    model.eval()
    logits, bnd, acts = [], [], []
    for b in range(0, len(index), batch_size):
        xb = torch.from_numpy(gather_windows(X, index[b : b + batch_size])).to(device)
        out = model(xb)
        logits.append(out["context_logits"].cpu().numpy())
        bnd.append(out["boundary_logit"].cpu().numpy())
        if "action" in out:
            acts.append(out["action"].cpu().numpy())
    return (
        np.concatenate(logits) if logits else np.zeros((0, 1)),
        np.concatenate(bnd) if bnd else np.zeros(0),
        np.concatenate(acts) if acts else None,
    )


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(1e-6, temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """검증셋 NLL 최소 온도(temperature scaling, 1차원 탐색)."""

    best_t, best_nll = 1.0, math.inf
    for t in np.exp(np.linspace(math.log(0.3), math.log(5.0), 60)):
        p = softmax(logits, float(t))
        nll = -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def ema_smooth(probs: np.ndarray, group: np.ndarray, alpha: float) -> np.ndarray:
    """인과(causal) 지수이동평균 확률 평활화. 그룹 경계에서 초기화."""

    out = probs.copy()
    for i in range(1, len(probs)):
        if group[i] == group[i - 1]:
            out[i] = alpha * probs[i] + (1.0 - alpha) * out[i - 1]
    return out
