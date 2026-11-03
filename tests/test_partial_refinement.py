"""Phase 0 consistency tests for models/partial_refinement.py.

GPU tests need the LM1B-small checkpoint under artifacts/lm1b (see AGENTS.md) and the local
LM1B test split; they are skipped otherwise. Override paths with NCM_TEST_CONFIG,
NCM_TEST_CKPT and NCM_TEST_DATA.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import partial_refinement as pr

CONFIG = os.environ.get(
    "NCM_TEST_CONFIG", os.path.join(ROOT, "config/eval/lm1b-small.yaml")
)
CKPT = os.environ.get("NCM_TEST_CKPT", os.path.join(ROOT, "artifacts/lm1b/small"))
DATA = os.environ.get(
    "NCM_TEST_DATA",
    "/scratch/prj0000000245/sakurai/data/lm1b/lm1b_test_bs128_wrapped.dat",
)


@pytest.mark.parametrize("length", [1, 3, 8, 16, 128])
@pytest.mark.parametrize("passes", [1, 2, 4, 8])
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_commit_counts(length, passes, schedule):
    c = pr.commit_counts(length, passes, schedule)
    assert c[-1] == length
    assert len(c) == min(passes, length)
    assert all(b > a for a, b in zip([0] + c[:-1], c))


gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not os.path.exists(os.path.join(CKPT, "generator.pt")),
    reason="needs CUDA and the LM1B checkpoint",
)


@pytest.fixture(scope="module")
def model():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, tk, net = pr.load_ncm(CONFIG, CKPT)
    return cfg, tk, net


@pytest.fixture(scope="module")
def gt(model):
    import datasets

    _cfg, tk, net = model
    ds = datasets.load_from_disk(DATA)
    ids = torch.tensor(ds[:8]["input_ids"], device="cuda")
    idx = tk.vqvae.encode(ids)
    eos = tk.pre_tokenizer.eos_token_id
    prefix = torch.full((ids.shape[0], net.prefix_len), eos, device="cuda")
    return ids, idx, prefix


def _split(vq, idx):
    out, off = [], 0
    for l in vq.levels:
        out.append(idx[:, off : off + l])
        off += l
    return out


@gpu
def test_contribution_boundaries(model, gt):
    _, tk, _net = model
    vq = tk.vqvae.vq
    codes = _split(vq, gt[1])
    for t in range(1, len(vq.levels)):
        B, l = codes[t].shape
        empty = torch.zeros(B, l, dtype=torch.bool, device="cuda")
        full = torch.ones_like(empty)
        assert torch.equal(
            pr.partial_contribution(vq, t, codes[t], empty),
            torch.zeros(B, vq.levels[-1], vq.embed_dim, device="cuda"),
        )
        assert torch.equal(
            pr.partial_contribution(vq, t, codes[t], full),
            pr.level_operator(vq, t, vq.embeddings[t](codes[t])),
        )
    # quant_resi has a bias, so the zero baseline is not trivially zero
    f0 = pr.level_operator(
        vq, 1, torch.zeros(1, vq.levels[1], vq.embed_dim, device="cuda")
    )
    print("||F_1(0)|| per position:", f0.norm(dim=-1).mean().item())


@gpu
def test_cumulative_matches_quantizer(model, gt):
    _, tk, _ = model
    vq = tk.vqvae.vq
    Hs = pr.cumulative_reconstructions(vq, gt[1])
    assert torch.allclose(Hs[-1], vq.idx_to_z_q(gt[1]), atol=1e-5)


@gpu
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_one_pass_matches_generate(model, gt, temperature):
    _, tk, net = model
    vq = tk.vqvae.vq
    prefix = gt[2]
    l0 = _split(vq, gt[1])[0]
    torch.manual_seed(123)
    ref = net.generate(l0, prefix, temperature=temperature, top_k=0, top_p=1.0)
    torch.manual_seed(123)
    out = pr.generate_refined(
        net, l0, prefix, passes=1, temperature=temperature, top_k=0, top_p=1.0
    )
    assert torch.equal(ref, out)


@gpu
def test_level_logits_match_generate_path(model, gt):
    """S = {}: the helper's level inputs/logits equal those inside generate() bit for bit."""
    _, tk, net = model
    vq = tk.vqvae.vq
    ps = pr.prefix_state(net, gt[2])
    Hs = pr.cumulative_reconstructions(vq, gt[1])
    # Feed ground-truth codes level by level through generate()'s own update code.
    z_hat = torch.zeros_like(Hs[0])
    codes = _split(vq, gt[1])
    for t in range(1, len(vq.levels)):
        z_hat, x = vq.get_next_autoregressive_input(
            t - 1, z_hat, vq.embeddings[t - 1](codes[t - 1])
        )
        if net.use_post_upsample_conv:
            x = net._apply_post_upsample_conv_single(x, t - net._lvl_first)
        off = sum(net.levels[net._lvl_first : t])
        x = net.transformer.wte(x) + ps["lpe"][:, off : off + net.levels[t], :]
        ref = net._decode_level_logits(
            x, ps["prefix_emb"], t - 1, ps["pbias"], hidx=t - net._lvl_first
        )
        assert torch.equal(ref, pr.level_logits(net, t, Hs[t - 1], ps)), t


@gpu
def test_level_logits_match_teacher_forcing(model, gt):
    """S = {}: per-level decoding agrees with the full teacher-forced forward pass (training path)."""
    _, tk, net = model
    vq = tk.vqvae.vq
    ids, idx, prefix = gt
    ps = pr.prefix_state(net, prefix)
    z_prefix = tk.vqvae.encoder.wte(prefix)
    parts = [z_prefix]
    if net.query_level0:
        parts.append(net.build_l0_query_input(ids.shape[0], z_prefix.dtype))
    parts.append(vq.idx_to_ncm_input(idx))
    logits_tf = net(torch.cat(parts, dim=1))
    Hs = pr.cumulative_reconstructions(vq, idx)
    off = 0 if net.query_level0 else -vq.levels[0]
    for t, l in enumerate(vq.levels):
        if t >= 1:
            a = logits_tf[:, off : off + l, : net.vocab_sizes[t - net._lvl_first]]
            b = pr.level_logits(net, t, Hs[t - 1], ps)
            assert torch.allclose(a, b, atol=2e-3, rtol=1e-3), (
                t,
                (a - b).abs().max().item(),
            )
        off += l


@gpu
@pytest.mark.parametrize("policy", ["confidence", "random", "l2r"])
def test_multi_pass_no_double_counting(model, gt, policy):
    _, tk, net = model
    vq = tk.vqvae.vq
    ps = pr.prefix_state(net, gt[2])
    Hs = pr.cumulative_reconstructions(vq, gt[1])
    t = 5
    H_prev = Hs[t - 1].clone()
    trace = []
    codes, H_t = pr.refine_level(
        net, t, H_prev, ps, 4, policy=policy, temperature=1.0, trace=trace
    )
    assert torch.equal(H_prev, Hs[t - 1])  # previous state untouched
    assert [r["n_committed"] for r in trace] == pr.commit_counts(vq.levels[t], 4)
    assert torch.equal(H_t, H_prev + pr.level_operator(vq, t, vq.embeddings[t](codes)))


@gpu
def test_oracle_empty_and_full(model, gt):
    _, tk, net = model
    vq = tk.vqvae.vq
    ps = pr.prefix_state(net, gt[2])
    Hs = pr.cumulative_reconstructions(vq, gt[1])
    codes = _split(vq, gt[1])
    t = 3
    empty = torch.zeros_like(codes[t], dtype=torch.bool)
    _ce0, lg0, _ = pr.oracle_level_ce(net, t, codes[t], Hs[t - 1], ps, empty)
    assert torch.equal(lg0, pr.level_logits(net, t, Hs[t - 1], ps))
    # Committing everything feeds H_t (the next level's state) into level t's slot.
    _, lg1, _ = pr.oracle_level_ce(net, t, codes[t], Hs[t - 1], ps, ~empty)
    assert torch.equal(lg1, pr.level_logits(net, t, Hs[t], ps))
