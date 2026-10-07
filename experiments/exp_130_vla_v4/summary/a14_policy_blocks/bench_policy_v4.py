"""v3 대 v4 정책 파라미터 수·1스레드 학습 처리량(배치 128, 200단계, 무작위 입력)."""
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from vcp.vla.policy import ArrayImageSource, PolicyConfig, PolicyData, VLALitePolicy, count_parameters, train_policy

torch.set_num_threads(1)
VOCAB = 89
n_groups, glen = 40, 100
n = n_groups * glen
rng = np.random.default_rng(0)
group = np.repeat(np.arange(n_groups), glen)
frames = rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)
feats = rng.uniform(-1, 1, size=(n, 32)).astype(np.float32)
q = rng.dirichlet(np.ones(6), size=n).astype(np.float32)
tokens = rng.integers(2, 60, size=(n, 12)).astype(np.int64)
data = PolicyData(images=ArrayImageSource(frames), group=group, ego_v=rng.uniform(0, 30, n).astype(np.float32),
                  action=rng.normal(0, 2, n).astype(np.float32), tokens=tokens, domain="driving", features=feats,
                  aux_targets=q)
configs = {
    "v3 (use_features, H=2 mlp)": dict(use_features=True),
    "v4 (H=8 s=2 gru, aux 0.2, lang_dropout 0.15)": dict(use_features=True, feature_history=8, feature_stride=2,
                                                       feature_encoder="gru", aux_weight=0.2, lang_dropout=0.15),
}
order = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "1", "0"]
names = list(configs)
res = {k: {"samples_per_s": []} for k in names}
for name in names:
    torch.manual_seed(0)
    res[name]["n_params"] = count_parameters(VLALitePolicy.from_config(PolicyConfig(**configs[name]), VOCAB))
for o in order:
    name = names[int(o)]
    cfg = PolicyConfig(steps=200, batch=128, threads=1, seed=0, augment=True, **configs[name])
    load0 = os.getloadavg()[0]
    _, info = train_policy(data, np.arange(n), cfg)
    res[name]["samples_per_s"].append(round(info["samples_per_s"], 1))
    res[name].setdefault("load1_before", []).append(round(load0, 2))
res["env"] = {"torch": torch.__version__, "python": platform.python_version(), "cpu_count": os.cpu_count(),
              "processor": platform.processor(), "threads": 1}
print(json.dumps(res, indent=1, ensure_ascii=False))
