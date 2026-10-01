# TASK_054 공개 실주행 데이터 수집 및 속도 기반 라벨

## 체크리스트
- [x] comma.ai speedchallenge train.mp4/train.txt 다운로드(LFS 해시 확인) 스크립트
- [x] 20,400프레임 YOLO 3클래스 검출 JSONL 추출(CPU 약 62.6 ms/프레임)
- [x] 속도 → 가속도 → 4상태 라벨, 미래 가감속 행동 타깃
- [x] 10fps 재계산 테이블(FPS 강건성 실데이터 검증용)
- [x] 보닛 오검출 여부 점검(선행차 후보 박스 하단 ≤ y 339)
