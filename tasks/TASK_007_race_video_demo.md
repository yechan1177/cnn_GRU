# TASK_007: race 영상 실시간 모델 테스트 및 시각화

## 목표
- `data/raw/videos/race.mp4`를 입력으로 YOLO Spatial + Temporal 모델을 실행한다.
- 실시간 화면에서 context/score/policy를 확인할 수 있는 오버레이 뷰어를 제공한다.
- 실행 결과를 재현 가능한 파일(`outputs`, `experiments`)로 저장한다.

## 체크리스트
- [x] 태스크 문서 생성
- [x] 비디오 입력 데모 스크립트 구현
- [x] YOLO spatial encoder의 비디오 프레임 입력 지원
- [x] 실시간 표시(on/off) + 결과 영상 저장 구현
- [x] YOLO 박스/클래스/confidence + top-context 확률 시각화 개선
- [x] 실행 요약/메타데이터 저장
- [x] README/문서 업데이트
- [x] 단일 실행 파일(`python launchers/run_race_demo.py`) 제공
- [x] race 영상 실행 검증
- [x] 테스트/검증 완료 반영

## 산출물
- `src/vcp/tools/run_video_demo.py`
- `launchers/run_race_demo.py`
- `outputs/runs/race_video_demo_20260318_194039/`
- `outputs/runs/race_video_demo_preview_20260318_194039.mp4`
- `artifacts/screenshots/race_demo_20260318_194039.jpg`
- `experiments/exp_006_race_video_demo/`

## 상태
- 완료

## 진행 메모
- 2026-03-18: 작업 시작 (`race` 입력 영상 확인 완료)
- 2026-03-18: `--no-display --max-frames 300` 실행 검증 완료 (avg_fps 37.09)
