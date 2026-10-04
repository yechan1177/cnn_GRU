# TASK_065 (A15) v4 특징 이력 일반화·점수 몫 에피소드 상한(S1)

계획: `docs/36_v4_알고리즘_모델_개선계획.md`(2절 P2·S1, 6절 인터페이스 계약).
계약 메모: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "v4 추가 계약(A15)".
소유 범위: `src/vcp/vla/obs.py`, `src/vcp/vla/closed_loop.py`, `src/vcp/vla/curation.py`의 `select_shared`. `policy.py`(A14)는 수정하지 않는다.

## 체크리스트
- [x] obs.py: `FeatureHistory(batch, history=2, stride=2)` → [B,H,32], 링 버퍼 (H−1)·s+1 슬롯, 에피소드 시작에서 잘라 냄
- [x] obs.py: `stack_feature_history(feats, t, history=2, stride=2)`, `feature_history_offsets`, 기본값 상수
- [x] closed_loop.py: `run_closed_loop(feature_history=None, feature_stride=None)`, policy_fn 속성 → 2/2 순서로 결정, `config` 기록
- [x] curation.py: `select_shared(per_group_cap=None, info=None)`. 점수 몫 group당 상한, 부족 시 상한 무시 채움(로그·info), 무작위 기준선 무관
- [x] group = 에피소드 id 확인(`vla_v3_suite.select_v3`가 `pool["ep"]`를 넘김), docstring 명시
- [x] `tests/test_vla_v4_obs_curation.py`(12건): H=8 형태·잘라 냄·학습 쪽 규칙 동등성, 폐루프 전달, 기본값 v3 동일성, S1 상한·선택량·저장소·채움
- [x] 기존 테스트 전부 통과(전체 122건)
- [x] v3 코드(857a4fe)와 직접 비교: 기본값 폐루프(주행 7에피소드, 특징 의존 정책)·`select_shared` 12건 결과 동일
- [x] 측정: H=8 vs H=2 특징 갱신 시간(주행 21에피소드, 1회), v3 주행 풀 2%에서 c=1 vs None(CARE·트리거 혼합, 시드 0·1·2, 보조 10%)
- [x] 문서: docs/29 10절, docs/30 7절, docs/28a v4(A15) 메모
- [ ] 개발·새 테스트 시드 세트(docs/36 5절 A15 항목): 이번 지시 범위 밖. 본 세션 v4 스위트에서 정한다

## 결과 메모
- 특징 갱신 시간(배치 21, 프레임당): H=2 0.80 ms, H=8 0.84 ms. 쌓기 연산만은 3.6 → 4.9 µs. 차이는 1회 측정 편차와 구분되지 않는다.
- S1(2%, ρ=0.9, 시드 0): CARE는 v3 점수 몫 36클립이 이미 36개 에피소드에서 나와 c=1이 선택을 바꾸지 않았다(시드 1·2도 같음).
  트리거 혼합은 에피소드 31→36, 점수 몫 위험 비중 0.636→0.581.
- 학습·폐루프 이력 경계 차이: v3 스위트는 `PolicyData(group=클립 id)`라 학습 이력이 클립 시작에서 잘린다(docs/29 10.3절). v4 스위트에서 결정이 필요하다.
- 결과 파일: `experiments/exp_121_v4_a15_obs_curation/summary/a15_measure.json`, 스크립트 `scripts/measure_v4_a15.py`
