from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path


def _as_bool(value: str, default: bool) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _maybe_reexec_with_venv(root: Path) -> None:
    if os.getenv("VCP_NO_REEXEC", "0") == "1":
        return

    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if current == target:
        return

    os.environ["VCP_NO_REEXEC"] = "1"
    os.execv(
        str(target),
        [str(target), str(root / "launchers" / "run_race_demo.py"), *sys.argv[1:]],
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _maybe_reexec_with_venv(root)

    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vcp.tools.run_video_demo import run_demo

    args = Namespace(
        video=os.getenv("VCP_VIDEO", "data/raw/videos/race.mp4"),
        config=os.getenv("VCP_CONFIG", "configs/runtime_rtx3080ti_yolo_gru.yaml"),
        display=_as_bool(os.getenv("VCP_DISPLAY", "1"), True),
        save_video=_as_bool(os.getenv("VCP_SAVE_VIDEO", "1"), True),
        max_frames=int(os.getenv("VCP_MAX_FRAMES", "0")),
        run_name=os.getenv("VCP_RUN_NAME", "race_video_demo"),
    )
    summary = run_demo(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
