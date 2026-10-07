#!/usr/bin/env bash
# v3 핵심 지표 재실험 일괄 실행(docs/33·34). 단계별 캐시가 있어 중단 후 재실행하면 이어서 진행한다.
# 사용: bash scripts/run_vla_v3.sh [단계...]  (기본: prepare pilot tune main lang robot comma report)
set -euo pipefail
PY=${PY:-python}
ROOT=${ROOT:-experiments/exp_120_vla_v3}
WORKERS=${WORKERS:-4}
STAGES=${*:-prepare pilot tune main lang robot comma report}
export PYTHONPATH=${PYTHONPATH:-src}
mkdir -p "$ROOT/logs"
for s in $STAGES; do
  echo "=== $(date '+%F %T') 단계 $s 시작" | tee -a "$ROOT/logs/stages.log"
  if [ "$s" = report ]; then
    $PY -m vcp.experiments.vla_v3_report --root "$ROOT" 2>&1 | tee -a "$ROOT/logs/report.log"
  else
    $PY -m vcp.experiments.vla_v3_suite "$s" --root "$ROOT" --workers "$WORKERS" 2>&1 | grep --line-buffered -v "policy: step" >> "$ROOT/logs/$s.log"
  fi
  echo "=== $(date '+%F %T') 단계 $s 완료" | tee -a "$ROOT/logs/stages.log"
done
