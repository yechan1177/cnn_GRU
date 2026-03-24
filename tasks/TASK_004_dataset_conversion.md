# TASK_004: 데이터셋 분석 및 학습 포맷 변환

## 목표
- 최신 `outputs` run 데이터를 분석해 분포/품질을 확인한다.
- 학습 가능한 시퀀스 샘플 포맷으로 변환한다.
- 변환 결과를 `data/processed/`에 재현 가능한 구조로 저장한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 분석 대상 run 확정
- [x] frame/event 기본 통계 산출
- [x] 학습 샘플 스키마 정의
- [x] 변환 스크립트 구현
- [x] train/val split 생성
- [x] 변환 결과 manifest 작성
- [x] README/문서 갱신
- [x] 완료 상태 반영

## 산출물
- `src/vcp/tools/convert_dataset.py`
- `data/processed/<dataset_id>/...`
- `experiments/exp_001_baseline/dataset_analysis.md`
- 관련 문서 갱신

## 진행 메모
- 2026-03-18: 작업 시작
- 2026-03-18: `debug_after_adaptive_20260318_145038` 기준 변환 완료 (`data/processed/dataset_debug_after_adaptive_20260318_145038`)
