from __future__ import annotations

"""실행 장치(device) 선택 유틸리티.

학습 서버(RTX 3080 Ti), 클라우드 CPU 컨테이너, Jetson Orin Nano에서
같은 코드를 쓰기 위해 `auto` 지정 시 사용 가능한 장치를 자동으로 고른다.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_device(requested: str | None = "auto") -> str:
    """요청 문자열을 실제 사용 가능한 torch device 문자열로 변환한다.

    - ``auto`` / 빈 값: CUDA가 있으면 ``cuda:0``, 없으면 ``cpu``
    - ``cuda*``를 요청했지만 CUDA가 없으면 경고 후 ``cpu``로 대체
    - 그 외 값은 그대로 반환
    """

    text = (requested or "auto").strip().lower()
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except ModuleNotFoundError:  # pragma: no cover - torch 미설치 환경
        cuda_ok = False

    if text in {"", "auto"}:
        return "cuda:0" if cuda_ok else "cpu"
    if text.startswith("cuda") and not cuda_ok:
        logger.warning("CUDA를 사용할 수 없어 '%s' 대신 cpu로 실행합니다.", requested)
        return "cpu"
    return text
