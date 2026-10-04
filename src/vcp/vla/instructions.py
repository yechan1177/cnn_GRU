from __future__ import annotations

"""에피소드 단위 주행 스타일 지시문(언어 조건)과 토큰화.

VLA-lite 정책의 언어 입력은 "목표 속도 + 주행 스타일(간격·태도)"을 담은 짧은 영어 지시문이다.
- 스타일 3종(cautious / normal / brisk)은 전문가 IDM 파라미터(시간 headway, 최대 가속,
  쾌적 감속)와 목표 속도 배율로 정의한다. 도메인(주행/로봇)마다 값이 다르다.
- 패러프레이즈 4개를 두어 같은 의미를 다른 표현으로 쓴다(언어 일반화·절제 실험용).
- 한국어 지시문은 meta 기록용이며, 모델 입력은 영어 지시문을 `encode_instruction`으로 만든 토큰이다.
- `VOCAB`은 영어 템플릿의 모든 단어와 가능한 숫자 토큰을 정렬해 결정적으로 만든다(0=PAD, 1=UNK).
"""

import logging
import random
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

STYLE_NAMES: tuple[str, ...] = ("cautious", "normal", "brisk")
N_PARAPHRASES: int = 4
PAD_ID: int = 0
UNK_ID: int = 1


@dataclass(frozen=True)
class DrivingStyle:
    """주행 스타일. IDM 파라미터와 목표 속도 배율.

    - t_head: 원하는 시간 headway(초)
    - a_max: 최대 가속도(m/s^2)
    - b_comf: 쾌적 감속도(m/s^2)
    - v_factor: 시나리오 기본 속도(v0)에 곱하는 목표 속도 배율
    """

    name: str
    t_head: float
    a_max: float
    b_comf: float
    v_factor: float


_STYLES: dict[str, dict[str, DrivingStyle]] = {
    "driving": {
        "cautious": DrivingStyle("cautious", t_head=2.0, a_max=1.2, b_comf=2.0, v_factor=0.85),
        "normal": DrivingStyle("normal", t_head=1.4, a_max=1.8, b_comf=2.5, v_factor=1.0),
        "brisk": DrivingStyle("brisk", t_head=1.0, a_max=2.5, b_comf=3.0, v_factor=1.0),
    },
    # 실내 AMR: world.ROBOT의 IDM 범위(t_head 0.8~1.5, a_max 0.5~1.0, b_comf 0.8~1.2)에 맞춰 축소
    "robot": {
        "cautious": DrivingStyle("cautious", t_head=1.8, a_max=0.5, b_comf=0.8, v_factor=0.85),
        "normal": DrivingStyle("normal", t_head=1.2, a_max=0.75, b_comf=1.0, v_factor=1.0),
        "brisk": DrivingStyle("brisk", t_head=0.8, a_max=1.0, b_comf=1.2, v_factor=1.0),
    },
}

# 목표 속도 반올림 단위와 범위(숫자 토큰 VOCAB 생성에도 쓴다)
_DRIVING_KMH_STEP = 10
_DRIVING_KMH_RANGE = (10, 150)
_ROBOT_MPS_STEP = 0.1
_ROBOT_MPS_RANGE = (0.1, 3.0)

# 스타일별 표현(영어/한국어): 간격(gap), 부사(adv), 형용사(adj)
_STYLE_WORDS_EN: dict[str, dict[str, str]] = {
    "cautious": {"gap": "long", "adv": "cautiously", "adj": "cautious"},
    "normal": {"gap": "moderate", "adv": "smoothly", "adj": "normal"},
    "brisk": {"gap": "short", "adv": "briskly", "adj": "sporty"},
}
_STYLE_WORDS_KO: dict[str, dict[str, str]] = {
    "cautious": {"gap": "넉넉한", "adv": "조심스럽게", "adj": "신중한"},
    "normal": {"gap": "보통", "adv": "부드럽게", "adj": "보통의"},
    "brisk": {"gap": "짧은", "adv": "민첩하게", "adj": "활발한"},
}

# 패러프레이즈 템플릿. {speed}에는 "60 km/h" 또는 "0.8 m/s"가 들어간다.
_TEMPLATES_EN: dict[str, tuple[str, ...]] = {
    "driving": (
        "Drive at {speed} and keep a {gap} gap to the vehicle ahead, {adv}.",
        "Cruise around {speed} in a {adj} style with a {gap} following distance.",
        "Target speed {speed}. Follow the lead car with a {gap} gap and drive {adv}.",
        "Keep {speed} on the road, drive {adv} and leave a {gap} distance ahead.",
    ),
    "robot": (
        "Move along the aisle at {speed} and keep a {gap} gap to anyone ahead, {adv}.",
        "Travel around {speed} in a {adj} style with a {gap} following distance.",
        "Target speed {speed}. Follow the agent ahead with a {gap} gap and move {adv}.",
        "Keep {speed} in the aisle, move {adv} and leave a {gap} distance ahead.",
    ),
}
_TEMPLATES_KO: dict[str, tuple[str, ...]] = {
    "driving": (
        "{speed}로 주행하고 앞차와 {gap} 간격을 유지하며 {adv} 운전하라.",
        "{adj} 스타일로 {speed} 안팎을 유지하고 {gap} 차간 거리를 두어라.",
        "목표 속도는 {speed}이다. {gap} 간격으로 선행차를 따라 {adv} 주행하라.",
        "도로에서 {speed}를 유지하고 {adv} 운전하며 앞쪽에 {gap} 거리를 남겨라.",
    ),
    "robot": (
        "통로를 따라 {speed}로 이동하고 앞사람과 {gap} 간격을 유지하며 {adv} 움직여라.",
        "{adj} 스타일로 {speed} 안팎을 유지하고 {gap} 추종 거리를 두어라.",
        "목표 속도는 {speed}이다. {gap} 간격으로 앞선 운반체를 따라 {adv} 이동하라.",
        "통로에서 {speed}를 유지하고 {adv} 이동하며 앞쪽에 {gap} 거리를 남겨라.",
    ),
}

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:/[a-z]+)?")


def _check_domain(domain: str) -> str:
    key = str(domain).strip().lower()
    if key not in _STYLES:
        raise KeyError(f"지원하지 않는 도메인: {domain} (지원: {sorted(_STYLES)})")
    return key


def styles_for_domain(domain: str) -> dict[str, DrivingStyle]:
    """도메인별 스타일 표("cautious" | "normal" | "brisk")를 반환한다."""

    return dict(_STYLES[_check_domain(domain)])


def sample_style(rng: random.Random, domain: str) -> DrivingStyle:
    """주어진 난수 생성기로 스타일 하나를 균등하게 고른다(결정적)."""

    styles = _STYLES[_check_domain(domain)]
    return styles[rng.choice(STYLE_NAMES)]


def target_speed(v0: float, style: DrivingStyle, domain: str) -> float:
    """시나리오 기본 속도 v0에 스타일 배율을 곱하고 지시문 단위로 반올림한 목표 속도(m/s).

    - 주행: 10 km/h 단위로 반올림, [10, 150] km/h로 제한
    - 로봇: 0.1 m/s 단위로 반올림, [0.1, 3.0] m/s로 제한
    """

    key = _check_domain(domain)
    raw = float(v0) * style.v_factor
    if key == "driving":
        kmh = round(raw * 3.6 / _DRIVING_KMH_STEP) * _DRIVING_KMH_STEP
        kmh = min(max(kmh, _DRIVING_KMH_RANGE[0]), _DRIVING_KMH_RANGE[1])
        return kmh / 3.6
    mps = round(raw / _ROBOT_MPS_STEP) * _ROBOT_MPS_STEP
    mps = min(max(mps, _ROBOT_MPS_RANGE[0]), _ROBOT_MPS_RANGE[1])
    return round(mps, 1)


def speed_text(v_target: float, domain: str) -> str:
    """목표 속도 표기 문자열("60 km/h" 또는 "0.8 m/s")."""

    if _check_domain(domain) == "driving":
        return f"{int(round(v_target * 3.6))} km/h"
    return f"{v_target:.1f} m/s"


def instruction_text(style_name: str, v_target: float, domain: str, paraphrase: int, lang: str = "en") -> str:
    """스타일·목표 속도·패러프레이즈 번호로 지시문을 만든다(lang: "en" | "ko")."""

    key = _check_domain(domain)
    if style_name not in STYLE_NAMES:
        raise KeyError(f"알 수 없는 스타일: {style_name}")
    idx = int(paraphrase) % N_PARAPHRASES
    if lang == "en":
        template, words = _TEMPLATES_EN[key][idx], _STYLE_WORDS_EN[style_name]
    elif lang == "ko":
        template, words = _TEMPLATES_KO[key][idx], _STYLE_WORDS_KO[style_name]
    else:
        raise KeyError(f"지원하지 않는 언어: {lang}")
    return template.format(speed=speed_text(v_target, key), **words)


def tokenize(text: str) -> list[str]:
    """소문자화 후 단어·숫자·단위(km/h, m/s) 토큰으로 나눈다. 구두점은 버린다."""

    return _TOKEN_RE.findall(text.lower())


def _build_vocab() -> dict[str, int]:
    words: set[str] = set()
    for templates in _TEMPLATES_EN.values():
        for template in templates:
            for style in STYLE_NAMES:
                words.update(tokenize(template.format(speed="0 km/h 0 m/s", **_STYLE_WORDS_EN[style])))
    for kmh in range(_DRIVING_KMH_RANGE[0], _DRIVING_KMH_RANGE[1] + 1, _DRIVING_KMH_STEP):
        words.add(str(kmh))
    n_robot = int(round((_ROBOT_MPS_RANGE[1] - _ROBOT_MPS_RANGE[0]) / _ROBOT_MPS_STEP)) + 1
    for i in range(n_robot):
        words.add(f"{_ROBOT_MPS_RANGE[0] + i * _ROBOT_MPS_STEP:.1f}")
    words.discard("0")
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for word in sorted(words):
        vocab[word] = len(vocab)
    return vocab


VOCAB: dict[str, int] = _build_vocab()


def goal_value_table(speed_scales: dict[str, float] | None = None) -> np.ndarray:
    """어휘 id → 정규화 목표 속도 값 float32 [len(VOCAB)](숫자 토큰이 아니면 NaN). v4 P6 수치 목표 인코딩.

    - 주행 지시문의 정수 토큰("60")은 km/h → m/s(÷3.6) → ÷주행 속도 스케일(기본 30 m/s)
    - 로봇 지시문의 소수 토큰("0.8")은 m/s → ÷로봇 속도 스케일(기본 3 m/s)
    두 도메인의 숫자 표기가 겹치지 않으므로(정수 10~150 / 소수 0.1~3.0) 도메인 정보 없이 값이 정해진다.
    정규화는 `obs.proprio`의 속도 정규화와 같아 현재 속도와 바로 비교할 수 있다.
    """

    sc = {"driving": 30.0, "robot": 3.0} if speed_scales is None else speed_scales
    table = np.full(len(VOCAB), np.nan, dtype=np.float32)
    for word, idx in VOCAB.items():
        if word.isdigit():
            table[idx] = float(word) / 3.6 / sc["driving"]
        elif _is_decimal(word):
            table[idx] = float(word) / sc["robot"]
    return table


def _is_decimal(word: str) -> bool:
    head, dot, tail = word.partition(".")
    return bool(dot) and head.isdigit() and tail.isdigit()


def encode_instruction(text: str, max_len: int = 24) -> np.ndarray:
    """지시문을 int64 [max_len] 토큰 id로 바꾼다(뒤를 PAD로 채우고 넘치면 자른다)."""

    ids = [VOCAB.get(tok, UNK_ID) for tok in tokenize(text)]
    if len(ids) > max_len:
        logger.debug("지시문 토큰 %d개를 %d개로 자름: %s", len(ids), max_len, text)
        ids = ids[:max_len]
    out = np.full(max_len, PAD_ID, dtype=np.int64)
    out[: len(ids)] = ids
    return out
