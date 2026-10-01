#!/usr/bin/env bash
# VLA 연계 큐레이션 실험 일괄 실행(단계별 캐시, 중단 후 재실행 시 이어서 진행).
# 사용: bash scripts/run_vla_curation.sh [단계...]  (기본: prepare pilot tune main lang robot comma summary)
set -euo pipefail
PY=${PY:-python}
ROOT=${ROOT:-experiments/exp_110_vla_curation}
STAGES=${*:-prepare pilot tune main lang robot comma summary}
mkdir -p "$ROOT/logs"
for s in $STAGES; do
  case $s in
    robot)
      $PY -m vcp.experiments.vla_curation_suite prepare --domain robot --root "$ROOT" 2>&1 | grep -v "policy: step" | tee -a "$ROOT/logs/robot.log"
      $PY -m vcp.experiments.vla_curation_suite robot --domain robot --root "$ROOT" 2>&1 | grep -v "policy: step" | tee -a "$ROOT/logs/robot.log" ;;
    *)
      $PY -m vcp.experiments.vla_curation_suite "$s" --root "$ROOT" 2>&1 | grep -v "policy: step" | tee -a "$ROOT/logs/$s.log" ;;
  esac
done
