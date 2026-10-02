#!/usr/bin/env bash
# comma.ai speedchallenge 학습 영상/속도 라벨 다운로드 (공개 저장소, Git LFS 미디어 URL)
# 출처: https://github.com/commaai/speedchallenge
set -euo pipefail
OUT=${1:-data/raw/external/comma_speedchallenge}
mkdir -p "$OUT"
BASE=https://media.githubusercontent.com/media/commaai/speedchallenge/master/data
[ -f "$OUT/train.mp4" ] || curl -fL -o "$OUT/train.mp4" "$BASE/train.mp4"
[ -f "$OUT/train.txt" ] || curl -fL -o "$OUT/train.txt" https://raw.githubusercontent.com/commaai/speedchallenge/master/data/train.txt
# LFS 객체 해시 확인(2026-10 기준)
echo "a9f16228de735cbe65946093287bc32e317bea52d0738b81340bcd28f72ae06f  $OUT/train.mp4" | sha256sum -c -
