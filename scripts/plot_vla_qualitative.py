"""정성 그림: 한 에피소드의 맥락 점수 시계열, GT 위험 구간, 방법별 선택 클립, 렌더링 프레임.

사용: python scripts/plot_vla_qualitative.py [--root experiments/exp_110_vla_curation] [--episode 3]
출력: paper/figures/fig_vla_episode.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import koreanize_matplotlib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

from vcp.experiments.vla_curation_suite import CurationSuiteConfig, HAZARD_IDS, pool_path
from vcp.sim.render import render_frame
from vcp.vla.curation import select
from vcp.vla.pool import frame_boxes, load_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_110_vla_curation")
    ap.add_argument("--episode", type=int, default=-1, help="-1이면 CARE가 위험 클립을 고른 급제동 에피소드를 자동 선택")
    ap.add_argument("--budget", type=float, default=0.02)
    args = ap.parse_args()
    root = Path(args.root)
    cfg = CurationSuiteConfig(root=root)
    pool, meta = load_pool(pool_path(cfg))
    sc = dict(np.load(root / "cache" / f"scores_{cfg.domain}.npz"))
    tune = json.loads((root / "cache" / "tune_driving.json").read_text(encoding="utf-8"))
    common = dict(event_score=sc["event_score"], entropy=sc["entropy"], features=pool["X_v1v2"], action=sc["action"], ittc=sc["ittc"], oracle=sc["oracle"])
    masks = {}
    # 본 실험과 같은 선택(시드 0의 선택 난수 1000)을 재현한다
    for name, m, kw in (
        ("무작위", "random", {}),
        ("감속 트리거", "action_trigger", {}),
        (f"CARE(ρ={tune['reservoir']})", "ours", {"lam": tune["lam"], "reservoir": tune["reservoir"]}),
    ):
        masks[name] = select(m, args.budget, pool["ep"], cfg.clip_len, np.random.default_rng(1000), **common, **kw)
    ep_ids = pool["ep"]
    if args.episode < 0:
        # 사례 선정 기준(심사 m8): 급제동 시나리오 에피소드 가운데 고정 시드(2026)로 무작위 추출
        cands = sorted(e["index"] for e in meta["episodes"] if e["scenario"] == "lead_brake")
        ep = int(np.random.default_rng(2026).choice(cands))
    else:
        ep = args.episode
    idx = np.where(ep_ids == ep)[0]
    t = pool["t"][idx]
    info = meta["episodes"][ep]

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.25, 1.0, 1.0, 0.55], hspace=0.55)
    # 렌더링 프레임 4장
    haz = sc["oracle"][idx]
    first_h = int(np.argmax(haz)) if haz.any() else len(idx) // 2
    picks = sorted({max(0, first_h - 45), max(0, first_h - 10), min(len(idx) - 1, first_h + 10), min(len(idx) - 1, first_h + 60)})
    extra = iter(np.linspace(0, len(idx) - 1, 8).astype(int).tolist())
    while len(picks) < 4:  # 에피소드 초반 사건이면 중복이 생기므로 등간격 프레임으로 채운다
        picks = sorted(set(picks) | {next(extra)})
    for k, j in enumerate(picks[:4]):
        ax = fig.add_subplot(gs[0, k])
        i = idx[j]
        ax.imshow(render_frame(frame_boxes(pool, i), float(pool["horizon_y"][i]), "driving"), interpolation="nearest")
        ax.set_title(f"t={t[j]:.1f}s, v={pool['ego_v'][i]:.1f} m/s", fontsize=9)
        ax.axis("off")
    ax1 = fig.add_subplot(gs[1, :])
    ax1.fill_between(t, 0, 1, where=haz, color="#ffcdd2", step="mid", label="GT 위험 구간")
    ax1.plot(t, sc["event_score"][idx], color="#1565c0", lw=1.6, label="위험 확률 $e_t$")
    ax1.plot(t, sc["entropy"][idx], color="#7e57c2", lw=1.0, ls="--", label="불확실성 $u_t$")
    ax1.plot(t, np.clip(sc["ittc"][idx], 0, 1), color="#8d6e63", lw=0.9, alpha=0.8, label="역 TTC 특징")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("점수")
    ax1.legend(fontsize=8, ncol=4, loc="upper left")
    ax1.set_title(f"에피소드 {ep} (선행차 급제동, 스타일 {info['style']}) — 지시문: \"{info['instruction_en']}\"", fontsize=9.5)
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
    ax2.plot(t, pool["ego_a"][idx], color="#ff9800", lw=1.2, label="자차 가속도(감속 트리거 입력)")
    ax2.plot(t, pool["expert_cmd"][idx], color="#424242", lw=0.9, alpha=0.7, label="전문가 명령(행동 라벨)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("m/s²")
    ax2.legend(fontsize=8, loc="lower left")
    ax3 = fig.add_subplot(gs[3, :], sharex=ax1)
    for r, (name, mk) in enumerate(masks.items()):
        sel = mk[idx]
        ax3.fill_between(t, r, r + 0.8, where=sel, step="mid", color=["#9e9e9e", "#ff9800", "#1565c0"][r])
    ax3.set_yticks([0.4, 1.4, 2.4])
    ax3.set_yticklabels(list(masks), fontsize=8)
    ax3.set_xlabel("시간(s)")
    ax3.set_title(f"선택된 클립(예산 {int(args.budget * 100)}%, 시드 0 선택을 재현; 풀 전체 선택 중 이 에피소드 부분)", fontsize=9)
    out = Path("paper/figures/fig_vla_episode.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(out, "episode", ep)


if __name__ == "__main__":
    main()
