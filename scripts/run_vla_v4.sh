#!/usr/bin/env bash
# v4 개선 실험 일괄 실행(docs/36·37). 단계별 캐시가 있어 중단 후 재실행하면 이어서 진행한다.
# 사용: bash scripts/run_vla_v4.sh [단계...]
#   개발(사전 등록 전): prepare dev robotdev
#   확증(사전 등록 docs/37 커밋 후): main lang robot comma report
#   세 번째 테스트(사전 등록 docs/39 커밋 후): third robot3 commap5 thirdreport
#   네 번째 평가 세트·새 학습 시드(사전 등록 docs/40 커밋 후): fourth fourthreport p5dev
# WAIT_PID가 있으면 그 프로세스(예: v3 실행)가 끝날 때까지 기다린 뒤 시작한다(CPU 4개 공유 방지).
set -euo pipefail
PY=${PY:-python}
ROOT=${ROOT:-experiments/exp_130_vla_v4}
WORKERS=${WORKERS:-4}
STAGES=${*:-prepare dev robotdev}
export PYTHONPATH=${PYTHONPATH:-src}
mkdir -p "$ROOT/logs"
if [ -n "${WAIT_PID:-}" ]; then
  echo "=== $(date '+%F %T') PID $WAIT_PID 종료 대기" | tee -a "$ROOT/logs/stages.log"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
for s in $STAGES; do
  echo "=== $(date '+%F %T') 단계 $s 시작" | tee -a "$ROOT/logs/stages.log"
  if [ "$s" = report ]; then
    $PY -m vcp.experiments.vla_v4_report --root "$ROOT" 2>&1 | tee -a "$ROOT/logs/report.log"
  elif [ "$s" = fourthreport ]; then
    $PY -m vcp.experiments.vla_third_report --root "$ROOT" --test-ev test4 --cf-ev cf4 --tag fourth 2>&1 | tee -a "$ROOT/logs/report.log"
  elif [ "$s" = thirdreport ]; then
    $PY -m vcp.experiments.vla_third_report --root "$ROOT" 2>&1 | tee -a "$ROOT/logs/report.log"
  else
    # grep은 출력할 줄이 없으면 1을 돌려주므로 감싼다(파이썬 실패는 pipefail로 그대로 잡힌다)
    $PY -m vcp.experiments.vla_v4_suite "$s" --root "$ROOT" --workers "$WORKERS" 2>&1 | { grep --line-buffered -v "policy: step" || true; } >> "$ROOT/logs/$s.log"
  fi
  echo "=== $(date '+%F %T') 단계 $s 완료" | tee -a "$ROOT/logs/stages.log"
done
