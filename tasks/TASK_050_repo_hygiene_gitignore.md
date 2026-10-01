# TASK_050 저장소 정리 및 .gitignore 재설계

## 목적
- 잘못 커밋된 가상환경 제거, 생성물/요약 결과 추적 규칙 정리

## 체크리스트
- [x] `.venv_readme_check/`(1052개 파일) 추적 해제 및 삭제
- [x] `.venv*/`, `build/`, `dist/`, `*.onnx`, `*.engine`, `*.npz`, `*.log` 무시
- [x] `experiments/exp_1*/summary/**`만 추적(기존 exp_0xx 로컬 폴더는 계속 무시)
- [x] `paper/*.md`, `paper/sections/`, `paper/figures/*.png|svg`만 추적
- [x] `git check-ignore`로 규칙 동작 확인
