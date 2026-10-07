"""VLA(Vision-Language-Action) / 피지컬 AI 연계 모듈.

경량 맥락 모델(온디바이스 System-1)이 만든 맥락·불확실성·행동 신호를
VLA 학습용 (관측, 언어, 행동) 에피소드로 변환한다.

- `language`: 맥락 라벨 + 물리량 → 한국어/영어 서술(프레임 단위 추론 주석), 과업 지시문
- `export`: LeRobot v2 구조를 따르는 에피소드 내보내기(parquet 또는 jsonl)
- `curation`: 이벤트/불확실성 점수 기반 클립 선별(저장 예산 대비 이벤트 회수율)
- `instructions`: 주행 스타일 3종·지시문 패러프레이즈·VOCAB·토큰화
- `obs`: VLA-lite 관측 규격(64×64 프레임 2장, proprio, 정규화 상수)
- `pool`: 스타일 지시문이 붙은 큐레이션 풀 생성·로드
- `closed_loop`: lockstep 배치 폐루프 평가와 지표
"""

from .curation import select_clips
from .export import VLAEpisode, export_lerobot_like
from .language import action_phrase, narrate, task_instruction

__all__ = ["select_clips", "VLAEpisode", "export_lerobot_like", "action_phrase", "narrate", "task_instruction"]
