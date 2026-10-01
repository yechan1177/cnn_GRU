"""SVG 그림을 PNG로 렌더링한다(Playwright Chromium, 한글 폰트는 시스템/사용자 폰트 사용).

사용: python scripts/render_svg.py paper/figures/fig_care_overview.svg [--scale 2]
"""

import argparse
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def render(svg_path: Path, scale: float = 2.0) -> Path:
    svg = svg_path.read_text(encoding="utf-8")
    w = int(float(re.search(r'width="([\d.]+)"', svg).group(1)))
    h = int(float(re.search(r'height="([\d.]+)"', svg).group(1)))
    out = svg_path.with_suffix(".png")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None)
        page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
        page.set_content(f"<html><body style='margin:0;background:#fff'>{svg}</body></html>")
        page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": w, "height": h})
        browser.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="+")
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    for s in a.svg:
        print(render(Path(s), a.scale))
