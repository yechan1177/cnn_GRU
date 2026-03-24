# 수동 맥락 annotation

이 폴더는 `YOLO 객체검출 결과`와 별개로 `시계열 맥락(context)` 학습을 위한 구간 단위 annotation을 저장한다.

## 목적
- 객체 존재 여부가 아니라 `상황 변화`를 학습시키기 위한 라벨 소스
- `stopcar`, `race` 같은 실제 영상에 대해 사람 검수 기반 구간 라벨 축적
- 향후 `human-in-the-loop` 검수와 재학습의 기준 데이터 제공

## 파일 구성
- `context_label_schema.json`: 라벨 정의
- `stopcar_context_segments.jsonl`: `stopcar.mp4` 구간 라벨
- `race_context_segments.jsonl`: `race.mp4` 구간 라벨

## 주요 라벨
- `normal_drive`: 일반 주행
- `front_vehicle_follow`: 전방 차량 추종
- `brake_warning`: 중앙 ROI에서 전방 박스가 짧은 프레임 사이 빠르게 커지는 경고 구간
- `hard_brake_risk`: 박스가 더 급격히 커지거나 움직임이 둔화/정지하는 강한 제동 위험 구간
- `post_brake_recovery`: 제동 후 회복 구간
- `dense_traffic`: 혼잡 주행 구간

## 주의
- 현재 annotation은 모두 `draft`다.
- detector와 temporal 재학습에 연결하는 기준 데이터이므로, run_id를 새 run으로 갱신하며 관리한다.
- 현재 feature는 detector 통계 + ROI/looming/stop-motion 기반이며, 추적기 기반 TTC는 아직 포함되지 않는다.
