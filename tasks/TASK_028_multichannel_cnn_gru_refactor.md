# TASK_028: 멀티채널 CNN-GRU 구조 반영 및 프로젝트 정리

## 목표
- 기존 단일 입력 GRU temporal 구조를 멀티채널 CNN-GRU 구조로 교체한다.
- 입력 벡터를 의미/구간별 채널로 분리하고 채널별 1D CNN 특징 추출 후 GRU로 결합한다.
- GAP 기반 분류 헤드를 반영한다.
- 실제 학습 스크립트와 런타임 체크포인트를 새 구조에 맞게 갱신한다.
- 불필요한 산출물/폴더 중 명확히 재생성 가능한 항목은 정리한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 멀티채널 temporal 구조 구현
- [x] 학습/추론 코드 갱신
- [x] 실제 체크포인트 재학습
- [x] 문서/설정 갱신
- [x] 불필요 항목 정리
- [x] 테스트 및 TODO 반영

## 상태
- 완료

## 진행 메모
- 2026-03-23: 작업 시작
- 2026-03-23: `TemporalGRUNet`을 멀티채널 1D CNN + GRU + GAP 구조로 교체하고 채널 그룹 자동 분리 로직 추가
- 2026-03-23: `train_temporal_gru.py`가 channel_groups/cnn_channels를 checkpoint에 저장하도록 갱신
- 2026-03-23: `exp_008_multichannel_temporal_training`에서 실제 vehicle processed dataset 기준 체크포인트 재학습 완료
- 2026-03-23: runtime config를 새 체크포인트로 갱신하고 race/stopcar 스모크 실행 확인
- 2026-03-23: `__pycache__` 폴더 정리 및 pytest 10건 통과 확인
