# TASK_051 특징벡터 규칙 기반 pseudo-label GRU 재구성

## 목적
- [x] 영상 구간 라벨 중심 학습에서 벡터 상태 규칙 중심 학습으로 전환한다.
- [x] 16차원 feature 시퀀스에서 `normal/follow/brake_warning/hard_brake/post_brake/dense_traffic`를 자동 부여한다.
- [x] 새 pseudo-label dataset으로 GRU를 재학습한다.
- [x] 실행/문서/태스크를 함께 갱신한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 규칙 기반 dataset builder 구현
- [x] pseudo-label dataset 생성
- [x] GRU 재학습
- [x] 스모크 검증
- [x] 문서 반영

## 산출물
- dataset: `data/processed/dataset_rule_context_feat16_v2_20260327_144832`
- checkpoint: `experiments/exp_018_rule_context_temporal/runs/mcnn_gru_rule_context_feat16_v2_e16/checkpoints/best.pt`
- backup checkpoint: `models/checkpoints/temporal_final_best_manual_segments_backup.pt`
- active checkpoint: `models/checkpoints/temporal_final_best.pt`
