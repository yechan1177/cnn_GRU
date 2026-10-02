# 피지컬 AI · 로봇 · VLA 연계 설계

## 1. 위치 설정: 경량 맥락 모델 = 온디바이스 System-1
최근 VLA(Vision-Language-Action) 구조는 느린 추론 계통(VLM 기반, System-2)과 빠른 반응 계통(System-1)을 나누는 방향으로 가고 있다(예: GR00T N1). 본 프로젝트의 경량 맥락 모델(YOLOv8n + 16차원 의미 특징 + 멀티채널 CNN-GRU)은 VLA 자체를 대체하지 않는다. 대신 다음 세 역할을 맡는 엣지 계층으로 정의한다.

| 역할 | 내용 | 구현 위치 |
|---|---|---|
| 안전 반사 계층 | 역 TTC 기반 감속/정지 맥락을 수 ms 안에 판단 | `vcp.features.semantic_v2`, `vcp.components.temporal` |
| 데이터 필터 | 이벤트 점수 + 불확실성 상위 클립만 저장/전송 | `vcp.vla.curation` |
| 주석기 | 프레임 단위 맥락·언어 서술·행동 타깃을 붙여 VLA 학습 에피소드 생성 | `vcp.vla.language`, `vcp.vla.export` |

## 2. (관측, 언어, 행동) 정렬
| 축 | 주행(실데이터 comma) | 주행(합성) | 실내 이동로봇(합성) |
|---|---|---|---|
| 관측 | 영상 프레임 참조 + 박스 + 16차원 특징 + 속도/가속도 | 박스 + 특징 + 자차 상태 | 박스 + 특징 + 로봇 상태 |
| 언어 | 과업 지시문 + 맥락 서술(한/영) | 동일 | 동일(로봇 어휘) |
| 행동 | 0.5초/1.0초 뒤 가감속(속도 센서) | 시뮬레이터 가감속 | 시뮬레이터 가감속 |

- 언어 서술은 템플릿 기반(`narrate`)이며 속도·TTC·모델 확신도를 포함한다. 프레임 단위 추론 서술(embodied chain-of-thought) 형식으로 쓸 수 있다.
- 행동 head(`ActionHeadWrapper`)는 맥락 head와 GRU 표현을 공유한다. 같은 경량 모델이 맥락과 행동 사전(prior)을 함께 낸다.

## 3. 내보내기 형식 (LeRobot v2 구조)
```
data/processed/vla_export/<name>/
  meta/info.json      # fps, feature 스키마, 에피소드/프레임 수
  meta/tasks.jsonl    # 과업 지시문(영/한)
  meta/episodes.jsonl # 에피소드 길이, 출처, 영상 경로
  data/chunk-000/episode_000000.parquet
```
- 주요 열: `observation.state`, `observation.semantic_features`, `observation.context_probs`, `observation.boxes`, `observation.video_frame`, `action`, `annotation.context`, `annotation.narration_ko/en`, `curation.event_score`, `curation.uncertainty`
- 영상은 복사하지 않고 원본 경로 + 프레임 번호로 참조한다.
- 공식 LeRobot 로더 검증은 아직 하지 않았다(호환 지향 형식).
- 저장소에는 메타데이터 샘플만 `experiments/exp_100_paper_suite/summary/vla_sample/`에 둔다.

## 4. 로봇 도메인 전이
- 같은 3클래스 검출 체계를 실내 물류 환경에서 재해석한다: person=작업자, vehicle=지게차/AGV, bike=대차.
- 6맥락 대응: normal_drive→normal_move, front_vehicle_follow→follow_agent, brake_warning→slow_down, hard_brake_risk→safety_stop, post_brake_recovery→resume, dense_traffic→crowded.
- 이 체계는 협동 로봇의 속도·분리 감시(ISO/TS 15066 SSM) 개념(접근 시 감속, 임계 거리에서 정지)과 대응된다.
- 실험: 로봇 합성 데이터 단독 학습, 주행 모델 zero-shot, 로봇 데이터 10% 미세조정(주행 사전학습 vs 처음부터).

## 5. 실행
```bash
PYTHONPATH=src python -m vcp.experiments.run_suite vla --out experiments/exp_100_paper_suite/summary
```

## 6. 다음 단계 (미구현)
- [ ] 내보낸 데이터셋을 LeRobot 공식 로더로 검증하고 소형 VLA(OpenVLA/π0 계열) 미세조정 실험
- [ ] 실제 로봇(AMR 또는 매니퓰레이터) 카메라 + 오도메트리/관절 상태 수집 경로 추가(`vcp.components.camera` 확장)
- [ ] 템플릿 서술을 VLM 기반 서술로 확장하고, 경량 맥락 모델을 서술 검증기로 사용
- [ ] Jetson Orin Nano에서 System-1 지연·전력 실측
