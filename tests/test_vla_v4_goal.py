"""v4 P6 지시문 수치 목표 인코딩(docs/36) 테스트."""

from __future__ import annotations

import numpy as np
import torch

from vcp.vla.instructions import VOCAB, encode_instruction, goal_value_table, instruction_text
from vcp.vla.policy import PolicyConfig, VLALitePolicy


def test_goal_table_values() -> None:
    t = goal_value_table()
    assert len(t) == len(VOCAB)
    assert np.isclose(t[VOCAB["60"]], 60 / 3.6 / 30)
    assert np.isclose(t[VOCAB["0.8"]], 0.8 / 3.0)
    assert np.isnan(t[VOCAB["<pad>"]]) and np.isnan(t[VOCAB["drive"]])


def test_encode_goal_from_instruction_and_empty() -> None:
    m = VLALitePolicy(vocab_size=len(VOCAB), goal_encoding=True)
    tok_d = torch.from_numpy(encode_instruction(instruction_text("normal", 60 / 3.6, "driving", 0)))[None]
    tok_r = torch.from_numpy(encode_instruction(instruction_text("normal", 0.8, "robot", 1)))[None]
    pad = torch.zeros(1, 24, dtype=torch.int64)
    v = torch.tensor([[0.3]])
    gd, gr, gp = m.encode_goal(tok_d, v), m.encode_goal(tok_r, v), m.encode_goal(pad, v)
    assert torch.allclose(gd, torch.tensor([[60 / 108, 1.0, 60 / 108 - 0.3]]), atol=1e-6)
    assert torch.allclose(gr[0, :2], torch.tensor([0.8 / 3, 1.0]), atol=1e-6)
    assert torch.equal(gp, torch.zeros(1, 3))


def test_goal_disabled_without_language_and_by_default() -> None:
    assert not VLALitePolicy(vocab_size=len(VOCAB)).goal_encoding
    m = VLALitePolicy.from_config(PolicyConfig(use_language=False, goal_encoding=True), vocab_size=len(VOCAB))
    assert not m.goal_encoding and not hasattr(m, "goal_mlp")
    m2 = VLALitePolicy.from_config(PolicyConfig(goal_encoding=True), vocab_size=len(VOCAB))
    out = m2(torch.rand(2, 6, 64, 64), torch.from_numpy(np.stack([encode_instruction("drive at 60 km/h")] * 2)), torch.zeros(2, 1))
    assert out.shape == (2, m2.chunk)
