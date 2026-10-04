# TASK_064 (A14) v4 정책 블록: 시간 특징 인코더(P2)·위험 맥락 보조 헤드(P3)·지시문 드롭아웃(T1)

계획: `docs/36_v4_알고리즘_모델_개선계획.md` 2절(P2·P3·T1), 6절(인터페이스 계약).
소유 범위: `src/vcp/vla/policy.py`, `tests/test_vla_policy_v4.py`, `docs/31_VLA_lite_정책.md` 11절,
`docs/28a_VLA_모듈_인터페이스_계약.md` "v4 추가 계약(A14)". A15 소유(`obs.py`, `closed_loop.py`, `curation.py`)는 수정하지 않는다.

## 체크리스트
- [x] 작업 전 v3 기준값(초기 가중치 해시·고정 시드 학습 결과·예측 해시) 기록
- [x] P2: `PolicyConfig.feature_history/feature_stride/feature_encoder`, `encode_features` [B,H,D] 일반화
- [x] P2: `"gru"` 인코더(nan→0, clamp(−3,3), 시간 순서 뒤집기, Linear+ReLU → GRU 1층 → 마지막 은닉)
- [x] P2: `PolicyData.observation(idx, history, stride)`(에피소드 시작에서 잘라 냄), `train_policy`·`predict_open_loop`·`make_policy_fn` 연결, 정책 함수 속성 `feature_history`·`feature_stride`
- [x] P3: `PolicyConfig.aux_weight/aux_classes`, `PolicyData.aux_targets`, 보조 헤드(마지막 생성), soft CE 손실, `forward(..., return_aux=True)`, info `final_aux_loss`
- [x] T1: `PolicyConfig.lang_dropout`, 전용 Generator(seed+4), 빈 지시 = PAD만 있는 시퀀스
- [x] 추가 계약(본 세션 지시): `PolicyData.feature_group`(특징 프리롤), 테스트 1건, docs/31 11.3
- [x] 기본값에서 v3와 비트 단위 동일(작업 전 기준값과 비교 + 테스트)
- [x] `tests/test_vla_policy_v4.py`(14개)
- [x] 파라미터 수·1스레드 처리량 측정(v3 설정 대비 v4 기본 조합)
- [x] 문서: docs/31 11절, docs/28a v4 추가 계약(A14)
- [x] 전체 테스트 통과(124개)
- [x] 커밋

## 결과 메모
- 비트 동일성: 변경 전 코드로 만든 해시(초기 가중치 4구성, 학습 4종의 final_loss·손실 곡선·가중치·예측·정책 함수 출력, 관측 배열)와
  변경 후 값이 모두 같다(`experiments/exp_130_vla_v4/summary/a14_policy_blocks/`). 테스트는 v3 커밋 policy.py를 git에서 꺼내 비교한다.
- 파라미터(어휘 89): v3 설정 544,040, v4 기본 조합(H=8 s=2 gru, aux 0.2) 564,526(+20,486).
- 1스레드 처리량(배치 128, 200단계, 무작위 입력, CPU 공유, 2회): v3 1,657/1,579, v4 1,466/1,433 samples/s(약 10% 느림).
- 합성 추세 문제(행동 = 4·(f[t] − f[t−14]), 처음 보는 에피소드): MAE/기준선 gru(H=8) 약 0.43, mlp(H=2) 약 0.89~0.91.
- 빈 지시: PAD만 있는 시퀀스. 마스크 평균 분모가 clamp_min(1)이라 0으로 나누기가 없어 null 임베딩을 두지 않았다.
- `final_loss`·`loss_curve`는 행동 손실만 기록한다(보조 손실은 `final_aux_loss`·`aux_loss_curve`로 분리).
