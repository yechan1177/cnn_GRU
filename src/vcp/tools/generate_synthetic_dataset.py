from __future__ import annotations

"""물리 기반 합성 주행 데이터셋 생성 CLI.

예시::

    python -m vcp.tools.generate_synthetic_dataset --out data/processed/synth/main_15fps_mid \\
        --n-episodes 700 --fps 15 --noise mid --seed-base 0 --export-jsonl 5
"""

import argparse
import logging
from pathlib import Path

from vcp.sim.camera import NOISE_LEVELS
from vcp.sim.dataset import SynthConfig, generate_dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="물리 기반 합성 주행 데이터셋 생성")
    parser.add_argument("--out", required=True, help="출력 경로(확장자 제외)")
    parser.add_argument("--n-episodes", type=int, default=700)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--noise", default="mid", help="low | mid | high 또는 숫자")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", default="driving", help="driving | robot")
    parser.add_argument("--export-jsonl", type=int, default=0, help="검출 JSONL로 내보낼 에피소드 수")
    args = parser.parse_args()

    level = NOISE_LEVELS.get(args.noise)
    if level is None:
        level = float(args.noise)
    cfg = SynthConfig(
        n_episodes=args.n_episodes,
        fps=args.fps,
        duration_s=args.duration,
        noise_level=level,
        seed_base=args.seed_base,
        conf_threshold=args.conf,
        workers=args.workers,
        domain=args.domain,
    )
    generate_dataset(Path(args.out), cfg, export_jsonl_episodes=args.export_jsonl)


if __name__ == "__main__":
    main()
