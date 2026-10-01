# 경량 의미 특징 기반 시계열 맥락 인식의 재평가와 피지컬 AI 데이터 자동 구축으로의 확장: 주행 및 실내 이동로봇 적용

<!-- 이 파일은 템플릿이다. `python scripts/build_paper.py`가 실험 결과 JSON으로 이중 중괄호 자리표시자를 채워
     paper/manuscript_ko.md 를 생성한다. 수치를 직접 고치지 말고 실험을 다시 실행할 것. -->

**초록** — 실시간 카메라 영상의 객체 검출 결과를 의미 특징으로 요약하고 경량 시계열 모델로 주행 맥락(추종, 제동 경고, 강한 제동 위험 등)을 인식하는 온디바이스 파이프라인을 다룬다. 2026년 3월 제출본(YOLOv8n + 16차원 특징 + 멀티채널 CNN-GRU + 하이브리드 룰 게이트)을 재검토하여 같은 영상 안 구간 분할과 검증셋 재사용에 의한 평가 누수, 특징 피크 기반 라벨을 같은 특징의 룰로 평가하는 순환 구조, 프레임 차분 특징의 FPS 의존성을 확인하였다. 이를 바로잡기 위해 (i) 추적·시간 정규화 기반 역 TTC 특징(v2), (ii) 라벨이 특징과 독립인 물리 기반 합성 벤치마크(주행 {{synth_episodes}}개, 실내 이동로봇 {{robot_episodes}}개 에피소드), (iii) 속도 센서 라벨의 공개 실주행 영상 검증, (iv) 에피소드/시간 블록 분할·검증셋 전용 튜닝·다중 시드의 자동 실험 프로토콜을 제시한다. 재평가 결과 성능을 좌우한 것은 특징 설계였다. 같은 구조에서 v1을 v2로 바꾸면 macro-F1이 합성 주행에서 {{syn:mc_cnn_gru_balanced_v1:main:macro_f1}}에서 {{syn:mc_cnn_gru_balanced_v2:main:macro_f1}}로, 실주행 영상(10-fold)에서 {{comma:mc_cnn_gru_balanced_v1:test_20fps:macro_f1}}에서 {{comma:mc_cnn_gru_balanced_v2:test_20fps:macro_f1}}로 올랐다. 반면 의미 기반 채널 그룹과 하이브리드 룰 게이트는 macro-F1 개선을 보이지 않았고, 정확도는 다수 클래스에 끌려 macro-F1과 반대 순위를 냈다. 합성 데이터로만 학습한 모델은 실영상 제동 구간을 AUROC {{s2r_v2}}로 판별하였다. 나아가 같은 경량 모델을 피지컬 AI 데이터 엔진으로 확장하여 미래 가감속 예측 head, 맥락 기반 한/영 서술, 예산 기반 큐레이션, LeRobot 구조 내보내기를 구현하였다. 저장 예산 20%에서 모델 점수 기반 선별은 제동 프레임의 {{cur20_model}}를 회수하였고(무작위 {{cur20_random}}), 주행 사전학습은 로봇 데이터 10% 미세조정에서 macro-F1을 {{robot_scratch}}에서 {{robot_ft}}로 높였다. temporal 모델은 클라우드 CPU 단일 스레드에서 창당 {{lat_prop_ms}} ms였으며, Jetson Orin Nano 실측은 향후 과제로 남긴다.

**주제어**: 맥락 인식, CNN-GRU, 역 TTC, 평가 누수, 합성 데이터, 피지컬 AI, VLA, 데이터 큐레이션, 이동로봇

---

## 1. 서론

로봇과 자율주행을 포괄하는 피지컬 AI는 대규모 (관측, 언어, 행동) 데이터로 학습한 VLA(Vision-Language-Action) 모델로 빠르게 옮겨 가고 있다[1-4]. 그러나 이런 데이터는 수집 비용이 크고, 실제 운용 중 녹화한 영상의 대부분은 학습 가치가 낮은 평범한 구간이다. 따라서 엣지 장치에서 "지금이 의미 있는 상황인가"를 싸게 판단하고, 그런 구간만 정렬된 형태로 남기는 경량 맥락 인식 계층이 필요하다. 최근 VLA 구조가 느린 추론 계통(System-2)과 빠른 반응 계통(System-1)을 나누는 흐름[4, 5]도 같은 요구를 보여 준다.

본 연구의 출발점은 2026년 3월에 제출한 모델이다. 이 모델은 YOLOv8n[6, 7]으로 사람/차량/이륜차를 검출하고, 박스를 16차원 의미 특징으로 바꾼 뒤, 특징 그룹별 1D-CNN과 GRU[8]를 결합한 멀티채널 CNN-GRU와 하이브리드 룰 게이트로 맥락을 판단하였다. 당시 보고에서 하이브리드 구성은 브레이크 관련 recall을 크게 높였다. 그러나 재검토 결과 다음 문제를 확인하였다.

1. **평가 누수**: 영상 6개 안에서 구간 단위로 학습/검증을 나눠 인접한 8프레임 창이 양쪽에 걸쳤고, 룰 임계값을 고른 검증셋에서 성능을 다시 보고했으며, 학습에 쓴 영상을 실측 비교에 다시 사용했다[9].
2. **라벨 순환**: 제동 구간 라벨을 ROI/looming 특징 피크로 부트스트랩했는데, 같은 특징으로 만든 룰 게이트가 이를 평가받았다.
3. **표본 부족과 지표 편향**: 검증셋의 브레이크 표본이 한 자릿수였고, 다수 클래스 비율에 가까운 정확도가 주 지표였다.
4. **구현-문서 불일치**: 배포 체크포인트의 채널 그룹이 문서와 달리 의미 그룹이 아닌 인덱스 균등 분할이었고, v1 특징의 클래스 비율 값 배치가 key 이름과 달랐다(값은 person, vehicle 순, 이름은 vehicle, person 순).
5. **FPS 의존 특징**: looming/motion이 프레임 차분이어서 배포 장치의 FPS가 바뀌면 같은 상황도 다른 값이 된다.

본 논문의 기여는 다음과 같다.

- 위 문제를 고친 평가 프로토콜(에피소드/시간 블록 단위 분할, 검증셋 전용 튜닝, 다중 시드, macro-F1·이벤트 단위 지표·bootstrap 신뢰구간)과 자동 실험 스위트를 제시한다.
- 박스 크기 변화율로 역 TTC를 추정하고 Δt로 정규화한 특징 v2를 제안하고, 세 도메인에서 v1 대비 효과와 FPS·노이즈 변화에 대한 강건성을 정량 비교한다.
- 특징과 독립인 물리량(TTC, 요구 감속도, 자차 감속도)으로 라벨을 만드는 합성 벤치마크를 주행과 실내 이동로봇(AMR) 두 도메인에 대해 구축한다.
- 속도 센서로 라벨을 만든 공개 실주행 영상에서 같은 파이프라인을 검증한다.
- 경량 맥락 모델을 VLA 데이터 구축과 연결한다: 행동 예측 보조 head, 맥락 기반 언어 서술, 예산 기반 큐레이션, LeRobot 구조 내보내기.

## 2. 관련 연구

**객체 검출과 시계열 맥락.** 단일 단계 검출기[6]와 그 경량 구현[7]은 엣지 장치에서 널리 쓰인다. 검출 결과나 프레임 특징을 RNN/GRU[8]나 시간 합성곱[10]으로 묶어 사고 예측[11]이나 운전 행동 인식[12]에 쓰는 연구가 있다. 본 연구는 원 영상 특징 대신 해석 가능한 저차원 의미 특징을 써서 모델을 수만 개 파라미터 규모로 유지한다.

**충돌 시간(TTC).** 단안 영상에서 물체 상의 크기 변화율로 TTC를 추정하는 원리는 지각 심리학에서 τ 이론으로 알려져 있다[13]. 박스 폭 w가 거리 Z에 반비례하면 d ln w / dt = −Ż/Z = 1/TTC 이므로, 거리 추정 없이 역 TTC를 얻을 수 있다.

**피지컬 AI와 VLA.** RT-2[1], OpenVLA[2], π0[3] 등은 웹 규모 사전학습과 로봇 시연 데이터를 결합한다. Open X-Embodiment[14]와 LeRobot[15]은 이종 로봇 데이터를 공통 형식으로 모으는 기반을 제공한다. 프레임 단위 추론 서술(embodied chain-of-thought)[16]이 정책 성능을 높인다는 보고와, 주행 분야에서 VLM을 계획에 쓰는 연구[17]도 있다. GR00T N1[4]은 느린 VLM 계통과 빠른 행동 계통을 분리한다. 본 연구는 VLA 자체가 아니라, 그 학습 데이터를 엣지에서 고르고 주석하는 경량 System-1 계층을 다룬다.

**평가 누수.** 시계열 데이터의 무작위/인접 분할은 성능을 과대평가하며[9, 18], 블록 단위 교차검증이 권장된다.

**불균형과 보정.** 희소 이벤트 클래스에는 focal loss[19]와 class-balanced 가중치[20]를, 확률 보정에는 temperature scaling[21]을 사용한다. 협동 로봇의 속도·분리 감시(SSM)[22]는 사람과의 거리·접근 속도에 따라 감속/정지하도록 규정하며, 본 연구의 AMR 라벨 체계(감속, 안전 정지)와 대응된다.

## 3. 방법

### 3.1 파이프라인
그림 1과 같이 카메라 프레임마다 YOLOv8n(3클래스)이 박스를 내고, 특징 추출기가 16차원 의미 벡터를 만든다. 최근 W(=8) 프레임의 벡터를 멀티채널 CNN-GRU에 넣어 맥락 확률, 경계 점수, (선택적으로) 행동을 출력한다. 출력은 경고/이벤트 트리거와 자동 큐레이션에 쓰이고, 선택된 구간은 (관측, 언어, 행동) 에피소드로 저장된다.

![그림 1. 시스템 구조](figures/fig_architecture.png)

*그림 1. 시스템 구조. 온디바이스 System-1(맥락 인식·큐레이션)과 서버 측 데이터셋 구축·VLA 학습의 분리.*

### 3.2 의미 특징
**v1(2026-03).** 검출 수, 평균/최대 신뢰도, 평균 면적과 분산, 클래스 비율 3개, 클래스별 평균 신뢰도 3개, 중앙 ROI 위험도, motion, 중심 근접도, looming, occlusion으로 구성된다. looming은 ROI 박스 평균 면적의 프레임 간 증가량에 20을 곱한 값으로, FPS와 신규 객체 진입에 민감하다.

**v2(제안).** 전역 5개(검출 밀도, 평균 신뢰도, 평균 면적, 면적 분산, 검출 밀도 변화율), 선행 차량 6개(차량 비율, 존재, 크기, 하단 위치, 크기 변화율, 역 TTC), 취약 도로 이용자(VRU) 5개(사람/이륜차 비율, 존재, 크기, 접근 지표)로 구성된다. 선행 차량은 화면 하단 중앙 통로의 차량 중 하단 y가 가장 큰 박스이며, 직전 프레임 박스와 IoU ≥ 0.3이면 같은 객체로 연결한다. 크기 변화율은

  r_t = r_{t−1} + α_t ( (ln w_t − ln w_{t−1}) / Δt_t − r_{t−1} ),  α_t = 1 − exp(−Δt_t / τ)

로 계산하며(τ = 0.25 s), 역 TTC 특징은 max(0, r_t)를 [0, 1]로 자른 값이다. 연결이 끊긴 신규 객체는 r = 0에서 시작한다. Δt로 정규화하므로 같은 물리 상황이면 FPS와 무관하게 비슷한 값을 낸다(단위 테스트로 10/30fps에서 확인).

### 3.3 멀티채널 CNN-GRU
입력 X ∈ R^{W×D}를 의미 그룹 G_1..G_K로 나누고, 그룹마다 두 층 1D-CNN(커널 3, 채널 C)을 적용한 뒤 시간축으로 이어 붙여 GRU(은닉 H)에 넣는다. GRU 출력의 시간 평균에 맥락(softmax), 경계(sigmoid), 행동(회귀) head를 둔다. 의미 그룹은 v2에서 {global, vehicle, VRU}, v1에서 {global, vehicle, person, bike}이다. 2026-03 체크포인트는 실제로는 인덱스 균등 분할 [0–4], [5–8], [9–12], [13–15]를 사용했으며, 본 연구는 두 방식을 모두 비교한다. 학습 손실은 L = L_ctx + 0.5·L_bnd (+ λ·L_act)이며, 경계 타깃은 ±2프레임으로 확장하였다.

### 3.4 하이브리드 룰 게이트와 후처리
2026-03 하이브리드 게이트(ROI·중심·looming 조건과 경계/확률 조건으로 경고/강한 제동 클래스를 승격)를 그대로 벡터화해 재현하였다. 임계값은 검증셋에서만 고른다. 후처리로 인과 지수이동평균 평활화와 temperature scaling을 비교한다.

### 3.5 피지컬 AI / VLA 연계
- **행동 head**: 0.5초와 1.0초 뒤 자차 가감속을 smooth-L1로 회귀한다. 실데이터에서는 속도 센서, 합성 데이터에서는 시뮬레이터 값이 정답이다.
- **언어 서술**: 맥락 라벨과 물리량(속도, TTC, 확신도)으로 한국어/영어 서술을 템플릿 생성해 프레임 단위 추론 주석으로 저장한다. 에피소드 단위 과업 지시문은 도메인별로 둔다.
- **자동 큐레이션**: 이벤트 점수(제동 계열 확률 합) + 0.5 × 정규화 엔트로피로 2초 클립을 순위화해 저장 예산 k% 안에서 고른다.
- **내보내기**: LeRobot v2 디렉터리 구조(meta/info.json, tasks.jsonl, episodes.jsonl, data/chunk-000/*.parquet)로 관측(특징, 박스, 자차 상태, 영상 프레임 참조), 언어, 행동, 큐레이션 점수를 프레임 정렬해 저장한다. 공식 LeRobot 로더 검증은 향후 과제다.
- **로봇 전이**: 같은 3클래스 검출 체계를 실내 물류 환경에서 person=작업자, vehicle=지게차/AGV, bike=대차로 해석하고, 6맥락을 {이동, 추종, 감속, 안전 정지, 재개, 혼잡}으로 대응시킨다.

## 4. 데이터

### 4.1 물리 기반 합성 주행 데이터
3차선 도로에서 자차는 IDM[23]과 반응 지연(0.3–1.0 s), 1차 구동 지연을 따른다. 시나리오는 자유 주행, 추종, 선행차 급제동(3–8 m/s²), 끼어들기, 보행자/이륜차 횡단, 정체, 정지-출발 7종이다. 객체는 핀홀 카메라(640×480, f = 520 px, 높이 1.3 m)로 투영하고, 크기 의존 미검출, 박스 흔들림, 신뢰도 잡음, 사람↔이륜차 혼동, 오검출, 가림, 자차 가감속에 따른 피치 변화를 적용한다. 라벨은 GT 상태에서만 계산한다(표 1).

| 라벨 | 규칙(우선순위 높은 순) |
|---|---|
| hard_brake_risk | 진행 경로 객체 TTC < 1.8 s 또는 자차 감속 > 4 m/s² |
| brake_warning | TTC < 3.5 s 또는 자차 감속 > 2 m/s² |
| post_brake_recovery | 강한 제동 이벤트 종료 후 3 s 이내, 가속도 ≥ −0.5, 속도 < 이벤트 전의 85% |
| dense_traffic | 전방 35 m 내 차량 ≥ 4대이고 자차 속도 < 10 m/s |
| front_vehicle_follow | 진행 경로 선행차의 시간 간격 < 3 s 또는 거리 < 25 m |
| normal_drive | 그 외 |

표 1. 합성 데이터 라벨 규칙(TTC는 접근 속도 ≥ 1 m/s일 때만 계산).

에피소드는 30초, 15fps이며, 에피소드 단위로 train/val/test를 나눈다. 강건성 평가를 위해 테스트 에피소드와 같은 시드(같은 물리 상황)를 10/30fps, 노이즈 low/high로 다시 생성하였다. {{synth_data_line}}

### 4.2 실내 이동로봇(AMR) 합성 데이터
폭 2 m 통로, 로봇 속도 0.8–1.6 m/s, 카메라 높이 0.8 m, 최대 제동 2.5 m/s²로 바꾸고, 통로 주행, AGV 추종, AGV 급정지, 작업자 횡단, 마주 오는 작업자, 혼잡 6종 시나리오를 만든다. 감속/안전 정지 임계값은 TTC 3.0/1.5 s, 감속도 0.6/1.2 m/s²이다. {{robot_data_line}}

### 4.3 공개 실주행 영상
comma.ai speedchallenge[24]의 학습 영상(20fps, 20,400프레임, 약 17분)과 프레임별 속도를 사용한다. 3클래스 YOLOv8n으로 모든 프레임을 검출해 저장한 뒤(CPU 평균 {{extract_ms}} ms/프레임), 특징을 계산한다. 라벨은 속도를 0.5초 이동평균하고 미분한 가속도로 정한다: 속도 < 1.0이면 정지, 가속도 < −0.6이면 제동, > 0.6이면 가속, 그 외 정속(원 데이터셋은 속도 단위를 명시하지 않는다). {{comma_data_line}} 영상을 10개 연속 블록으로 나눠 fold마다 test 1, val 1, train 8 블록을 쓰고, 블록 경계 ±1초는 평가에서 제외한다. 10fps 강건성 평가는 짝수 프레임만으로 특징을 다시 계산해 수행한다.

## 5. 실험 설정
- **비교군**: 다수 클래스, 룰(v1 특징: 2026-03 YOLO+rule 형태 / v2 특징: 역 TTC), MLP(마지막 프레임), GRU, 단일채널 CNN-GRU(문헌형), 멀티채널 CNN-GRU(균등 분할 / 의미 그룹), 2026-03 구성(+하이브리드 게이트), 제안 구성의 변형(CB-focal 손실, 창 16, EMA 평활화).
- **학습**: Adam(lr 10⁻³), 배치 512, 최대 14 에폭, 검증 macro-F1 기준 조기 종료(인내 4), 은닉 96, CNN 채널 24, 학습 창 stride 2. 시드 3개(합성), fold 10개(실데이터).
- **튜닝 원칙**: 룰 임계값, 하이브리드 게이트, temperature, EMA 계수, 경계 임계값은 모두 검증셋에서만 고르고 테스트셋에는 고정 적용한다.
- **지표**: 정확도, macro-F1, 클래스별 F1, 제동 이벤트 recall(GT 이벤트 시작 0.5초 전부터 종료까지 한 번이라도 검출), 검출 지연, 분당 오경보(GT 이벤트 ±1초와 겹치지 않는 2프레임 이상 예측 구간), 분당 라벨 전환, ECE, 에피소드 bootstrap 95% CI, 엔트로피의 오류 탐지 AUROC.
- **환경**: 학습·평가는 GPU 없는 클라우드 컨테이너({{cpu_model}}, {{cpu_count}} 논리 코어)에서 수행했다. 동일 코드는 `--device auto`로 RTX 3080 Ti에서도 실행된다.

## 6. 결과

{{section:results_ko}}

## 7. 논의

{{section:discussion_ko}}

## 8. 결론
{{section:conclusion_ko}}

## 재현 방법
```bash
python -m pip install -e . && python -m pip install onnx onnxruntime pyarrow matplotlib koreanize-matplotlib
bash scripts/run_paper_suite.sh          # 데이터 수집/생성 → 실험 → 리포트 → 원고
```

## 참고문헌
(제출 전 서지 정보를 원문과 대조할 것)

[1] A. Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," CoRL, 2023.
[2] M. J. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
[3] K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024.
[4] NVIDIA, "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots," arXiv:2503.14734, 2025.
[5] D. Kahneman, *Thinking, Fast and Slow*, 2011. (System-1/2 비유의 출처)
[6] J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016.
[7] G. Jocher et al., Ultralytics YOLOv8, https://github.com/ultralytics/ultralytics, 2023.
[8] K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," EMNLP, 2014.
[9] S. Kapoor and A. Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science," Patterns, 2023.
[10] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018.
[11] F.-H. Chan et al., "Anticipating Accidents in Dashcam Videos," ACCV, 2016.
[12] V. Ramanishka et al., "Toward Driving Scene Understanding: A Dataset for Learning Driver Behavior and Causal Reasoning," CVPR, 2018.
[13] D. N. Lee, "A Theory of Visual Control of Braking Based on Information about Time-to-Collision," Perception, 1976.
[14] Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," ICRA, 2024.
[15] R. Cadene et al., LeRobot, https://github.com/huggingface/lerobot, 2024.
[16] M. Zawalski et al., "Robotic Control via Embodied Chain-of-Thought Reasoning," CoRL, 2024.
[17] X. Tian et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models," CoRL, 2024.
[18] C. Bergmeir and J. M. Benítez, "On the Use of Cross-Validation for Time Series Predictor Evaluation," Information Sciences, 2012.
[19] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.
[20] Y. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR, 2019.
[21] C. Guo et al., "On Calibration of Modern Neural Networks," ICML, 2017.
[22] ISO/TS 15066:2016, Robots and robotic devices — Collaborative robots.
[23] M. Treiber, A. Hennecke, and D. Helbing, "Congested Traffic States in Empirical Observations and Microscopic Simulations," Physical Review E, 2000.
[24] comma.ai, speedchallenge, https://github.com/commaai/speedchallenge, 2017.
