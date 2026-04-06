# TASK_068 spatial 간단 재현 코드 추가

## 목적
- spatial.py의 핵심 동작을 초보자/중급자가 쉽게 따라해볼 수 있는 루트 실행 스크립트를 추가한다.
- YOLO 검출 -> 16차원 feature 생성 -> 시각화 저장까지 최소 흐름만 제공한다.

## 체크리스트
- [x] 스크립트 범위 확정
- [x] 루트 실행 스크립트 작성
- [x] 사용 가이드 문서 작성
- [x] README/TODO 반영
- [x] 문법/스모크 검증
- [x] GitHub 반영

## 검증 메모
- `python -m py_compile practice_spatial_encoder.py` 통과
- `.\.venv\Scripts\python.exe practice_spatial_encoder.py` 실행 성공
- 루트 출력 이미지 `practice_spatial_encoder_preview.png` 생성 확인
