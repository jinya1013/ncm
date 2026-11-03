"""Within-level partial commitment for NCM (see plans.md).

A level k is committed in several passes. After each pass, the contribution of the
committed codes is added to the cumulative reconstruction H_{k-1}, and the same
pretrained generator re-predicts the remaining positions through its normal level-k
input path. The commit rule (sample every open position, keep the most confident ones
on a cosine/linear schedule) follows MaskGIT (Chang et al., 2022).

Level indices here are absolute VQ levels: target level ``t`` is predicted from the
reconstruction of levels ``< t``. Level 0 is never refined.
"""

import math
import os

import torch
import torch.nn.functional as F


def load_ncm(config_path, checkpoint_dir, device="cuda"):
    """Load tokenizer and generator exactly as scripts/eval_shard.py does."""
    from omegaconf import OmegaConf

    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    cfg = OmegaConf.load(config_path)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(
        checkpoint_dir, cfg.checkpoints.tokenizer
    )
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(
        os.path.join(checkpoint_dir, cfg.checkpoints.generator),
        map_location="cpu",
        weights_only=False,
    )
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    return cfg, tk, net


# ---------------------------------------------------------------------------
# Reconstruction-side helpers
# ---------------------------------------------------------------------------


def level_operator(vq, t, q_BLC):
    """F_t: the existing per-level update (upsample to full length, then quant_resi)."""
    if t != len(vq.levels) - 1:
        q_BLC = vq._up_to_full(q_BLC, t, vq.levels[-1])
    return vq._apply_quant_resi(q_BLC, t)


def partial_contribution(vq, t, codes, committed, zero_baseline=True):
    """C_t(S) = F_t(m_S * e_t(r)) - (1 - |S|/l_t) F_t(0).

    Gives exactly 0 for S = {} and exactly F_t(e_t(r)) for the full set.
    ``zero_baseline=False`` is the plain F_t(m_S * e_t(r)) ablation.
    """
    m = committed.to(vq.embeddings[t].weight.dtype)[..., None]
    out = level_operator(vq, t, vq.embeddings[t](codes) * m)
    if zero_baseline:
        frac = committed.float().mean(dim=1)[:, None, None].to(out.dtype)
        out = out - (1.0 - frac) * level_operator(
            vq, t, torch.zeros_like(m.expand(-1, -1, vq.embed_dim))
        )
    return out


def cumulative_reconstructions(vq, idx):
    """[H_0, H_1, ...] where H_k is the sum of level contributions 0..k (fresh tensors)."""
    B = idx.shape[0]
    H = torch.zeros(B, vq.levels[-1], vq.embed_dim, device=idx.device)
    out, off = [], 0
    for t, l in enumerate(vq.levels):
        H = H + level_operator(vq, t, vq.embeddings[t](idx[:, off : off + l]))
        out.append(H)
        off += l
    return out


# ---------------------------------------------------------------------------
# Generator-side helpers (mirror NextConceptTransformer.generate, eager path, no CFG)
# ---------------------------------------------------------------------------


@torch.no_grad()
def prefix_state(net, prefix_tokens):
    """prefix_emb, prefix pad bias and level/position embeddings, as in generate()."""
    dev = prefix_tokens.device
    P = net.prefix_len
    assert not net.use_ctx_encoder, "ctx_encoder checkpoints are not supported yet"
    if bool(getattr(net.config, "raw_token_prefix", False)):
        enc = net.vqvae_local[0].encoder
        if (
            bool(getattr(net.config, "contextual_prefix", False))
            and len(getattr(enc, "blocks", [])) > 0
        ):
            z_prefix = enc.forward_nopos(prefix_tokens)[:, :P, :]
        else:
            z_prefix = enc.wte(prefix_tokens)[:, :P, :]
    else:
        z_prefix = net.vqvae_local[0].encode_z(prefix_tokens)[:, :P, :]
    prefix_emb = net.transformer.wte(z_prefix)
    if not net.use_rope:
        prefix_emb = prefix_emb + net.wpe(net.wp_prefix)

    if net.config.use_level_encoding:
        lpe = net.transformer.wle(net.lvl_1L)
    else:
        lpe = torch.zeros(1, net.L - net.levels[0], net.config.n_embd, device=dev)
    if not net.use_rope:
        lpe = lpe + net.wpe(net.wp_1L)

    pbias = None
    if net.prefix_pad_id is not None:
        pm = prefix_tokens[:, :P] == net.prefix_pad_id
        pbias = torch.zeros(pm.shape[0], 1, 1, P, device=dev)
        pbias[:, :, :, :P] = torch.where(
            pm[:, None, None, :],
            torch.tensor(float("-inf"), device=dev),
            torch.tensor(0.0, device=dev),
        )
    return {"prefix_emb": prefix_emb, "pbias": pbias, "lpe": lpe}


@torch.no_grad()
def level_input(net, t, H):
    """Area-downsample H to l_t, then post_upsample_conv (the normal level-t input)."""
    vq = net.vqvae_local[0].vq
    x = F.interpolate(H.transpose(-1, -2), size=(vq.levels[t],), mode="area").transpose(
        -1, -2
    )
    if net.use_post_upsample_conv:
        x = net._apply_post_upsample_conv_single(x, t - net._lvl_first)
    return x


@torch.no_grad()
def level_logits(net, t, H, ps):
    """Logits for target level t (>= 1) given the cumulative reconstruction H of levels < t."""
    assert t >= 1
    hidx = t - net._lvl_first
    off = sum(net.levels[net._lvl_first : t])
    x = (
        net.transformer.wte(level_input(net, t, H))
        + ps["lpe"][:, off : off + net.levels[t], :]
    )
    return net._decode_level_logits(x, ps["prefix_emb"], t - 1, ps["pbias"], hidx=hidx)


# ---------------------------------------------------------------------------
# Commit schedule and selection (MaskGIT-style)
# ---------------------------------------------------------------------------


def commit_counts(length, num_passes, schedule="linear"):
    """Cumulative number of committed positions after each pass; strictly increasing, last == length."""
    n = max(1, min(int(num_passes), length))
    if schedule == "linear":
        frac = [(p + 1) / n for p in range(n)]
    elif schedule == "cosine":  # MaskGIT: masked ratio cos(pi/2 * (p+1)/n)
        frac = [1.0 - math.cos(math.pi / 2 * (p + 1) / n) for p in range(n)]
    else:
        raise ValueError(schedule)
    counts, prev = [], 0
    for p, f in enumerate(frac):
        c = length if p == n - 1 else round(f * length)
        c = max(c, prev + 1)
        c = min(c, length - (n - 1 - p))  # leave at least one per remaining pass
        counts.append(c)
        prev = c
    return counts


def select_new(score, committed, n_new):
    """Mark the n_new highest-score open positions in each row."""
    s = score.masked_fill(committed, float("-inf"))
    top = s.topk(n_new, dim=1).indices
    return torch.zeros_like(committed).scatter_(1, top, True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def refine_level(
    net,
    t,
    H_prev,
    ps,
    num_passes,
    policy="confidence",
    schedule="linear",
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    zero_baseline=True,
    rng=None,
    trace=None,
):
    """Commit level t in ``num_passes`` passes. Returns (codes, H_t). H_prev is not modified."""
    vq = net.vqvae_local[0].vq
    B, length = H_prev.shape[0], vq.levels[t]
    committed = torch.zeros(B, length, dtype=torch.bool, device=H_prev.device)
    codes = torch.zeros(B, length, dtype=torch.long, device=H_prev.device)
    counts = commit_counts(length, num_passes, schedule)
    prev = 0
    for p, c in enumerate(counts):
        H = (
            H_prev
            if p == 0
            else H_prev + partial_contribution(vq, t, codes, committed, zero_baseline)
        )
        logits = level_logits(net, t, H, ps)
        cand = net._sample_level(logits, temperature, top_k, top_p)
        if c == length and prev == 0:
            sel = torch.ones_like(committed)
        elif policy == "confidence":
            lp = (
                F.log_softmax(logits.float(), dim=-1)
                .gather(-1, cand[..., None])
                .squeeze(-1)
            )
            sel = select_new(lp, committed, c - prev)
        elif policy == "random":
            sel = select_new(
                torch.rand(B, length, device=H.device, generator=rng),
                committed,
                c - prev,
            )
        elif policy == "l2r":
            sel = torch.zeros_like(committed)
            sel[:, prev:c] = True
        else:
            raise ValueError(policy)
        sel &= ~committed
        codes = torch.where(sel, cand, codes)
        committed |= sel
        if trace is not None:
            trace.append({"level": t, "pass": p, "n_committed": int(c)})
        prev = c
    assert bool(committed.all())
    return codes, H_prev + level_operator(vq, t, vq.embeddings[t](codes))


@torch.no_grad()
def generate_refined(
    net,
    l0_idx,
    prefix_tokens,
    passes,
    policy="confidence",
    schedule="linear",
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    zero_baseline=True,
    rng=None,
    trace=None,
):
    """NCM generation with per-level pass counts.

    ``passes`` maps absolute level -> number of passes (missing levels use 1), or is an int
    applied to every level >= 1. Level-0 codes must be given (as in the unconditional eval
    protocol, which samples them from a unigram). With all passes == 1 this reproduces
    ``net.generate(l0_idx, prefix_tokens, ...)`` exactly, including the RNG stream.
    """
    assert l0_idx is not None, (
        "pass level-0 codes (the eval protocol samples them from a unigram)"
    )
    vq = net.vqvae_local[0].vq
    ps = prefix_state(net, prefix_tokens)
    B = prefix_tokens.shape[0]
    H = torch.zeros(B, vq.levels[-1], vq.embed_dim, device=prefix_tokens.device)
    H = H + level_operator(vq, 0, vq.embeddings[0](l0_idx))
    out = [l0_idx]
    for t in range(1, len(vq.levels)):
        n = passes if isinstance(passes, int) else passes.get(t, 1)
        codes, H = refine_level(
            net,
            t,
            H,
            ps,
            n,
            policy,
            schedule,
            temperature,
            top_k,
            top_p,
            zero_baseline,
            rng,
            trace,
        )
        out.append(codes)
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# Oracle conditioning (Phase 1)
# ---------------------------------------------------------------------------


@torch.no_grad()
def oracle_level_ce(
    net, t, gt_codes, H_prev, ps, committed, fill="gt", zero_baseline=True, rng=None
):
    """Per-position CE of the true level-t codes given H_prev + C_t(S, fill codes).

    fill: "gt" (true codes), "random" (uniform codes), "shuffled" (true codes permuted
    across the batch). Returns (ce[B, l_t], logits, H) where H is the conditioning state.
    """
    vq = net.vqvae_local[0].vq
    if fill == "gt":
        codes = gt_codes
    elif fill == "random":
        codes = torch.randint(
            0,
            vq.vqvae_vocab_sizes[t],
            gt_codes.shape,
            device=gt_codes.device,
            generator=rng,
        )
    elif fill == "shuffled":
        perm = torch.randperm(gt_codes.shape[0], device=gt_codes.device, generator=rng)
        codes = gt_codes[perm]
    else:
        raise ValueError(fill)
    H = (
        H_prev
        if not committed.any()
        else H_prev + partial_contribution(vq, t, codes, committed, zero_baseline)
    )
    logits = level_logits(net, t, H, ps)
    ce = F.cross_entropy(logits.float().transpose(1, 2), gt_codes, reduction="none")
    return ce, logits, H
