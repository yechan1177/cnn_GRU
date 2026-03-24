# 08. YOLO 대비 파이프라인 매핑

## 전제
- 사용자는 YOLO 중심 경험을 보유하고 있다.
- 본 프로젝트는 YOLO를 대체하는 것이 아니라, YOLO 신호를 포함해 시간 맥락/이벤트/자동 정제/정렬 export까지 확장하는 구조다.

## 단계별 매핑
1. `spatial encoder`
- YOLO에서 대응되는 부분: 백본 + 넥 + 헤드의 per-frame 인식 결과
- 본 구조에서 역할: 객체/장면 신호를 고정 길이 벡터로 요약

2. `feature packing`
- YOLO에서 대응되는 부분: detection 결과 후처리(NMS 이후 박스/클래스/점수 정리)
- 본 구조에서 역할: 시계열 인코더가 소비할 공통 feature 포맷 생성

3. `temporal context encoder`
- YOLO 단독에서는 약한 부분(프레임 독립 추론)
- 본 구조에서 역할: 연속 프레임의 문맥/상태 전이 추정

4. `context classification / event boundary / uncertainty`
- YOLO에서 대응되는 부분: confidence score(부분 대응)
- 본 구조에서 역할: 경계/불확실성/상태전이를 명시적으로 분리 산출

5. `auto-curation`
- YOLO 단독 파이프라인에서는 보통 별도 구현
- 본 구조에서 역할: 중요 이벤트 중심 저장(clip/keyframe/discard)

6. `aligned dataset export`
- YOLO 학습셋 포맷은 주로 이미지+bbox 라벨
- 본 구조에서 역할: frame/event/aligned 멀티모달 학습 포맷으로 누적

## YOLO 단독 대비 장점
- 시간 문맥 처리: 프레임 단위 오검출을 시계열 정보로 완화 가능
- 이벤트 중심 저장: 모든 프레임 저장 대신 중요 구간 위주로 데이터 품질/용량 동시 관리
- 확장성: EEG/EMG/IMU/robot_state 등 멀티모달 정렬을 스키마에 자연스럽게 결합
- 재현성: run -> processed dataset -> experiment report 흐름이 구조화되어 논문/발표 재사용이 쉬움

## 현실적인 트레이드오프
- 구현 복잡도와 유지보수 비용이 YOLO 단독보다 높다.
- 초기에는 시간모듈/정제정책 튜닝이 필요하다.
- 따라서 초기 검증은 “정확도 최고점”보다 “구조가 실제 데이터에서 안정 동작하는지”를 우선한다.
