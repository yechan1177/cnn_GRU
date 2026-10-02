# TASK_058 A3 큐레이션 방법 10종·comma 64×64 프레임

- 상위 태스크: `tasks/TASK_060_vla_curation_paper.md`
- 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`(A3 소유 부분)
- 방법 정의 문서: `docs/30_큐레이션_방법_정의.md`

## 체크리스트
- [x] 계약·기존 코드(`curation.py`, `vla_suite.py`, `build_comma_dataset.py`, `semantic_v2.py`) 확인
- [x] `clip_starts`(그룹별 비중첩 클립 격자) 구현
- [x] `select`: random, uniform, action_trigger, rule_ittc, uncertainty, event, coreset, offline_loss, oracle, ours
- [x] 모든 방법의 선택 프레임 수 동일 보장(k = max(1, round(β·N/L)))
- [x] `selection_stats`(클래스 비율, 라벨 엔트로피, 위험 프레임·구간 회수율, 그룹 커버리지)
- [x] `event_coverage` 벡터화(이전 구현과 결과 동일, 무작위 300건 대조)
- [x] coreset 대규모 실행 시간 측정(문서 30 §4)
- [x] `tools/extract_small_frames.py` 작성, crop 근거 확인(프레임 0/5000/10000/17000)
- [x] `frames_64.npz` 생성(20400×64×64×3) 및 `comma_table.npz` frame_id 정렬 확인(항등)
- [x] 샘플 몽타주 `docs/assets/comma_frames_64_sample.png`
- [x] 테스트 `tests/test_curation_methods.py`(15건) 및 전체 테스트 통과
- [x] 방법 정의 문서 `docs/30_큐레이션_방법_정의.md`
