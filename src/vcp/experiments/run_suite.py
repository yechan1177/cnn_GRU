from __future__ import annotations

"""자동 실험 진입점.

예시::

    python -m vcp.experiments.run_suite synthetic --out experiments/exp_100_paper_suite/summary
    python -m vcp.experiments.run_suite comma --out experiments/exp_100_paper_suite/summary
    python -m vcp.experiments.run_suite latency --out experiments/exp_100_paper_suite/summary
    python -m vcp.experiments.run_suite all --out experiments/exp_100_paper_suite/summary
"""

import argparse
import logging
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="논문용 자동 실험 스위트")
    parser.add_argument("suite", choices=["synthetic", "comma", "latency", "vla", "report", "all"])
    parser.add_argument("--out", default="experiments/exp_100_paper_suite/summary")
    parser.add_argument("--data-dir", default="data/processed/synth")
    parser.add_argument("--comma-dir", default="data/processed/comma_speedchallenge")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="스모크 테스트용 축소 실행")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = (0,) if args.quick else (0, 1, 2)

    if args.suite in {"synthetic", "all"}:
        from .synthetic_suite import SuiteConfig, run_synthetic_suite

        run_synthetic_suite(
            SuiteConfig(out_dir=out, data_dir=Path(args.data_dir), seeds=seeds, workers=args.workers, quick=args.quick)
        )
    if args.suite in {"comma", "all"}:
        from .comma_suite import CommaSuiteConfig, run_comma_suite

        run_comma_suite(CommaSuiteConfig(out_dir=out, data_dir=Path(args.comma_dir), seeds=seeds, workers=args.workers, quick=args.quick))
    if args.suite in {"latency", "all"}:
        from .latency import run_latency

        run_latency(out, comma_dir=Path(args.comma_dir))
    if args.suite in {"vla", "all"}:
        from .vla_suite import run_vla_suite

        run_vla_suite(out, data_dir=Path(args.data_dir), comma_dir=Path(args.comma_dir), quick=args.quick)
    if args.suite in {"report", "all"}:
        from .report import build_report

        build_report(out)


if __name__ == "__main__":
    main()
