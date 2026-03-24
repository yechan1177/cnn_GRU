from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .pipeline import VisionContextPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="비전인식-맥락 연계형 파이프라인 baseline 실행"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="실행할 YAML 설정 경로",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config_path = Path(args.config)
    pipeline = VisionContextPipeline(config_path=config_path)
    summary = pipeline.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
