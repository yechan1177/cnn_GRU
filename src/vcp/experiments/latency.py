from __future__ import annotations

"""지연시간/모델 크기 측정(측정한 장치를 결과에 함께 기록).

- temporal 모델: PyTorch CPU(1/4 스레드), ONNX Runtime CPU, 배치 1, 창 1개 기준
- YOLO: 640/320 입력, CPU
- Jetson Orin Nano 수치는 이 스크립트를 실기기에서 실행해야 얻을 수 있다(현재 미측정).
"""

import json
import logging
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .trainer import ModelSpec, build_model, count_parameters

logger = logging.getLogger(__name__)


def _bench(fn: Any, warmup: int = 20, iters: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


def temporal_latency(spec: ModelSpec, input_dim: int = 16, n_classes: int = 6) -> dict[str, Any]:
    model = build_model(spec, input_dim, n_classes).eval()
    x = torch.randn(1, spec.window, input_dim)
    out: dict[str, Any] = {"model": spec.name, "params": count_parameters(model), "window": spec.window}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        with torch.no_grad():
            out[f"torch_cpu_t{threads}"] = _bench(lambda: model(x))
    try:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")

            class _Wrap(torch.nn.Module):
                def __init__(self, m: torch.nn.Module) -> None:
                    super().__init__()
                    self.m = m

                def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    o = self.m(inp)
                    return o["context_logits"], o["boundary_logit"]

            torch.onnx.export(_Wrap(model), (x,), path, input_names=["x"], output_names=["logits", "boundary"], opset_version=17, dynamo=False)
            out["onnx_bytes"] = os.path.getsize(path)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            xn = x.numpy()
            ref = model(x)["context_logits"].detach().numpy()
            got = sess.run(None, {"x": xn})[0]
            out["onnx_max_abs_diff"] = float(np.abs(ref - got).max())
            out["onnx_cpu_t1"] = _bench(lambda: sess.run(None, {"x": xn}))
    except Exception as exc:  # pragma: no cover - 환경 의존
        logger.warning("ONNX 측정 실패(%s): %s", spec.name, exc)
        out["onnx_error"] = str(exc)
    torch.set_num_threads(4)
    return out


def yolo_latency(weights: Path, video: Path | None) -> list[dict[str, Any]]:
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError:  # pragma: no cover
        return []
    model = YOLO(str(weights))
    frame = None
    if video is not None and video.exists():
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
        ok, frame = cap.read()
        cap.release()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rows = []
    for imgsz in (640, 320):
        for threads in (1, 4):
            torch.set_num_threads(threads)
            stats = _bench(lambda: model.predict(source=frame, imgsz=imgsz, device="cpu", verbose=False, conf=0.45), warmup=5, iters=40)
            rows.append({"imgsz": imgsz, "threads": threads, **stats})
    torch.set_num_threads(4)
    return rows


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def run_latency(out_dir: Path, comma_dir: Path | None = None) -> Path:
    specs = [
        ModelSpec("mlp_last", "mlp_last"),
        ModelSpec("gru", "gru"),
        ModelSpec("cnn_gru_single", "single_channel_cnn_gru"),
        ModelSpec("mc_cnn_gru_balanced", "multichannel_cnn_gru", grouping="balanced"),
        ModelSpec("mc_cnn_gru_semantic_v2", "multichannel_cnn_gru", "v2", grouping="semantic"),
        ModelSpec("mc_cnn_gru_semantic_v2_w16", "multichannel_cnn_gru", "v2", grouping="semantic", window=16),
    ]
    temporal = [temporal_latency(s) for s in specs]
    video = Path("data/raw/external/comma_speedchallenge/train.mp4")
    yolo = yolo_latency(Path("models/checkpoints/yolo3cls_best.pt"), video)
    out = {"device": device_info(), "temporal": temporal, "yolo": yolo, "jetson_orin_nano": "미측정(실기기 필요)"}
    path = out_dir / "latency_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("저장: %s", path)
    return path
