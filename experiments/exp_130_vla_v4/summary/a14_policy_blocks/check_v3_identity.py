"""v3 기준값(초기 가중치·학습 결과 해시) 기록/비교 스크립트(A14, docs/31 11.5).

사용: PYTHONPATH=src OMP_NUM_THREADS=1 python check_v3_identity.py out.json [v3_reference_hashes.json]
v3_reference_hashes.json은 v4 변경 전(커밋 857a4fe) 코드로 만든 기준값이다. 두 번째 인자를 주면 IDENTICAL_TO_REF를 출력한다.
"""
import hashlib
import json
import sys

import numpy as np
import torch

from vcp.vla.policy import (ArrayImageSource, PolicyConfig, PolicyData, VLALitePolicy, count_parameters,
                            make_policy_fn, predict_open_loop, train_policy)


def sd_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def data(n_groups=6, glen=20, feats=True):
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(n_groups), glen)
    n = len(group)
    frames = rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)
    return PolicyData(images=ArrayImageSource(frames), group=group, ego_v=rng.uniform(0, 30, n).astype(np.float32),
                      action=rng.normal(0, 2, n).astype(np.float32),
                      tokens=rng.integers(0, 20, size=(n, 12)).astype(np.int64), domain="driving",
                      features=rng.normal(0, 1, (n, 32)).astype(np.float32) if feats else None)


out = {}
for name, kw in {"default": {}, "feat": {"use_features": True}, "nolang": {"use_language": False},
                 "feat_nolang": {"use_features": True, "use_language": False}}.items():
    torch.manual_seed(0)
    m = VLALitePolicy(vocab_size=89, **kw)
    out[f"init_{name}"] = [count_parameters(m), sd_hash(m)]
d = data()
idx = np.arange(len(d))
for name, kw in {"train_default": {}, "train_feat": {"use_features": True},
                 "train_feat_noaug": {"use_features": True, "augment": False},
                 "train_nolang_feat": {"use_features": True, "use_language": False}}.items():
    cfg = PolicyConfig(steps=30, batch=16, threads=1, seed=3, **kw)
    m, info = train_policy(d, idx, cfg)
    pred = predict_open_loop(m, d, idx)
    fn = make_policy_fn(m, "driving")
    pf = fn(d.observation(idx[:40]))
    out[name] = {"final_loss": info["final_loss"], "curve": info["loss_curve"], "state": sd_hash(m),
                 "pred": hashlib.sha256(pred.tobytes()).hexdigest(), "fn": hashlib.sha256(pf.tobytes()).hexdigest(),
                 "n_params": info["n_params"]}
obs = d.observation(idx)
out["obs"] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() + str(v.shape) + str(v.dtype)
              for k, v in obs.items()}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items():
    print(k, v if k.startswith("init") else (v.get("final_loss"), v.get("state", "")[:12]) if "final_loss" in v else v)
if len(sys.argv) > 2:
    ref = json.load(open(sys.argv[2]))
    same = ref == json.loads(json.dumps(out))
    print("IDENTICAL_TO_REF:", same)
    if not same:
        for k in ref:
            if ref[k] != json.loads(json.dumps(out)).get(k):
                print("DIFF", k)
