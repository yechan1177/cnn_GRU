#!/usr/bin/env bash
# 논문 실험 전체 자동 실행: 데이터 수집/생성 → 실험 → 리포트 → 원고
# 사용: bash scripts/run_paper_suite.sh [--quick]
#  - GPU가 있으면 자동 사용(--device auto), 없으면 CPU
#  - 결과: experiments/exp_100_paper_suite/summary/, paper/manuscript_ko.md
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
QUICK=${1:-}
OUT=experiments/exp_100_paper_suite/summary
LOG=experiments/exp_100_paper_suite/logs
mkdir -p "$OUT" "$LOG"
export PYTHONPATH=src

# 1) 공개 실주행 데이터 수집 + YOLO 검출 + 학습 테이블
bash scripts/download_comma_speedchallenge.sh
DET=data/processed/comma_speedchallenge/detections_yolo3cls.jsonl
if [ ! -f "$DET" ] || [ "$(wc -l < "$DET")" -lt 20400 ]; then
  $PY -m vcp.tools.extract_detections --video data/raw/external/comma_speedchallenge/train.mp4 \
      --output "$DET" --device auto --conf 0.25 --resume 2>&1 | tee "$LOG/extract.log"
fi
$PY -m vcp.tools.build_comma_dataset

# 2) 실험(합성 데이터는 synthetic 단계에서 시드 고정 생성)
$PY -m vcp.experiments.run_suite synthetic --out "$OUT" $QUICK 2>&1 | tee "$LOG/synthetic.log"
$PY -m vcp.experiments.run_suite comma --out "$OUT" $QUICK 2>&1 | tee "$LOG/comma.log"
$PY -m vcp.experiments.run_suite vla --out "$OUT" $QUICK 2>&1 | tee "$LOG/vla.log"
$PY -m vcp.experiments.run_suite latency --out "$OUT" 2>&1 | tee "$LOG/latency.log"

# 3) 표/그림 리포트 + 논문 원고
$PY -m vcp.experiments.run_suite report --out "$OUT"
$PY scripts/build_paper.py
echo "완료: $OUT/RESULTS.md, paper/manuscript_ko.md"
