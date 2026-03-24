from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass(slots=True)
class TrainArgs:
    data: str
    model: str
    epochs: int
    final_epochs: int
    imgsz: int
    batch: int
    fraction: float
    device: str
    project: str
    name: str
    workers: int
    seed: int
    optimizer: str
    lr0: float
    lrf: float
    patience: int
    weight_decay: float
    cache: bool
    amp: bool
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    fliplr: float
    flipud: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    mosaic: float
    mixup: float
    copy_paste: float
    erasing: float
    plots: bool


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="YOLO 3클래스 학습 실행 도구")
    parser.add_argument("--data", type=str, default="configs/datasets/yolo3cls_merged.yaml")
    parser.add_argument("--model", type=str, default="models/pretrained/yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--final-epochs", type=int, default=20, help="마지막 증강 비활성 미세조정 epoch 수")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="experiments/exp_011_yolo3cls_training/runs")
    parser.add_argument("--name", type=str, default="yolov8n_3cls_from_pretrained")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.set_defaults(cache=False)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.08)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.2)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.set_defaults(plots=True)
    args = parser.parse_args()

    return TrainArgs(
        data=str(args.data),
        model=str(args.model),
        epochs=max(1, int(args.epochs)),
        final_epochs=max(0, int(args.final_epochs)),
        imgsz=max(64, int(args.imgsz)),
        batch=max(1, int(args.batch)),
        fraction=max(0.001, min(1.0, float(args.fraction))),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
        workers=max(0, int(args.workers)),
        seed=int(args.seed),
        optimizer=str(args.optimizer),
        lr0=max(1e-6, float(args.lr0)),
        lrf=max(1e-6, float(args.lrf)),
        patience=max(1, int(args.patience)),
        weight_decay=max(0.0, float(args.weight_decay)),
        cache=bool(args.cache),
        amp=bool(args.amp),
        degrees=float(args.degrees),
        translate=float(args.translate),
        scale=float(args.scale),
        shear=float(args.shear),
        perspective=float(args.perspective),
        fliplr=float(args.fliplr),
        flipud=float(args.flipud),
        hsv_h=float(args.hsv_h),
        hsv_s=float(args.hsv_s),
        hsv_v=float(args.hsv_v),
        mosaic=float(args.mosaic),
        mixup=float(args.mixup),
        copy_paste=float(args.copy_paste),
        erasing=float(args.erasing),
        plots=bool(args.plots),
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return str(value)


def model_profile(model_name: str) -> dict[str, Any]:
    lower = model_name.lower()
    if "yolov8n" in lower:
        return {
            "model_variant": "yolov8n",
            "params_m": 3.1572,
            "gflops": 8.9,
            "jetson_fit": "preferred",
            "reason": "Jetson Orin Nano 8GB 기준으로 가장 현실적인 메모리/지연 균형",
        }
    if "yolov8s" in lower:
        return {
            "model_variant": "yolov8s",
            "params_m": 11.1666,
            "gflops": 28.8,
            "jetson_fit": "risky",
            "reason": "정확도 여지는 있으나 Jetson Orin Nano 8GB에서 지연/메모리 부담이 큼",
        }
    return {
        "model_variant": Path(model_name).name,
        "params_m": None,
        "gflops": None,
        "jetson_fit": "unknown",
        "reason": "사전 정의되지 않은 모델 경로",
    }


def train_stage(
    model: YOLO,
    *,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    fraction: float,
    device: str,
    project: str,
    name: str,
    workers: int,
    seed: int,
    optimizer: str,
    lr0: float,
    lrf: float,
    patience: int,
    weight_decay: float,
    cache: bool,
    amp: bool,
    plots: bool,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
    perspective: float,
    fliplr: float,
    flipud: float,
    hsv_h: float,
    hsv_s: float,
    hsv_v: float,
    mosaic: float,
    mixup: float,
    copy_paste: float,
    erasing: float,
) -> Any:
    return model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,
        device=device,
        project=str(Path(project).resolve()),
        name=name,
        exist_ok=True,
        workers=workers,
        verbose=True,
        pretrained=True,
        seed=seed,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        weight_decay=weight_decay,
        cache=cache,
        amp=amp,
        plots=plots,
        cos_lr=True,
        rect=False,
        multi_scale=False,
        close_mosaic=0,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        fliplr=fliplr,
        flipud=flipud,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        erasing=erasing,
    )


def main() -> None:
    args = parse_args()
    profile = model_profile(args.model)
    total_epochs = args.epochs
    final_epochs = min(args.final_epochs, max(0, total_epochs - 1))
    stage1_epochs = max(1, total_epochs - final_epochs)

    stage1_name = f"{args.name}_stage1_aug"
    model = YOLO(args.model)
    stage1_results = train_stage(
        model,
        data=args.data,
        epochs=stage1_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=args.device,
        project=args.project,
        name=stage1_name,
        workers=args.workers,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        weight_decay=args.weight_decay,
        cache=args.cache,
        amp=args.amp,
        plots=args.plots,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
    )

    stage1_dir = Path(stage1_results.save_dir)
    stage1_best = stage1_dir / "weights" / "best.pt"

    stage2_summary: dict[str, Any] | None = None
    final_best_ckpt = stage1_best
    final_last_ckpt = stage1_dir / "weights" / "last.pt"

    if final_epochs > 0 and stage1_best.exists():
        stage2_name = f"{args.name}_stage2_noaug"
        stage2_model = YOLO(str(stage1_best))
        stage2_results = train_stage(
            stage2_model,
            data=args.data,
            epochs=final_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            device=args.device,
            project=args.project,
            name=stage2_name,
            workers=args.workers,
            seed=args.seed,
            optimizer=args.optimizer,
            lr0=max(args.lr0 * 0.3, 1e-5),
            lrf=max(args.lrf * 0.5, 1e-5),
            patience=max(5, min(args.patience, 10)),
            weight_decay=args.weight_decay,
            cache=args.cache,
            amp=args.amp,
            plots=args.plots,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
        )
        stage2_dir = Path(stage2_results.save_dir)
        final_best_ckpt = stage2_dir / "weights" / "best.pt"
        final_last_ckpt = stage2_dir / "weights" / "last.pt"
        stage2_summary = {
            "save_dir": str(stage2_dir),
            "best_ckpt": str(final_best_ckpt),
            "last_ckpt": str(final_last_ckpt),
            "metrics": to_serializable(getattr(stage2_results, "results_dict", {})),
            "epochs": final_epochs,
            "augmentation": "disabled",
        }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": args.data,
        "model": args.model,
        "model_profile": profile,
        "activation_note": "YOLOv8 기본 Conv 블록은 SiLU 활성함수를 사용한다.",
        "training_plan": {
            "total_epochs": total_epochs,
            "stage1_epochs": stage1_epochs,
            "stage2_final_no_aug_epochs": final_epochs,
            "optimizer": args.optimizer,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "early_stopping_patience": args.patience,
        },
        "augmentation_stage1": {
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "shear": args.shear,
            "perspective": args.perspective,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "erasing": args.erasing,
        },
        "stage1": {
            "save_dir": str(stage1_dir),
            "best_ckpt": str(stage1_best),
            "last_ckpt": str(stage1_dir / "weights" / "last.pt"),
            "metrics": to_serializable(getattr(stage1_results, "results_dict", {})),
            "epochs": stage1_epochs,
            "augmentation": "enabled",
        },
        "stage2": stage2_summary,
        "final_best_ckpt": str(final_best_ckpt),
        "final_last_ckpt": str(final_last_ckpt),
    }

    summary_dir = Path(args.project) / args.name
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
