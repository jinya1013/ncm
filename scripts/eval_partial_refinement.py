"""Phase 1: oracle partial-commitment CE on held-out text (plans.md section 5).

For each level t >= 1, commit a fraction of positions S with ground-truth codes (or random /
batch-shuffled control codes) and measure the CE of the true codes on the remaining
positions U = S^c, against the S = {} baseline on the same U.

Example:
  python scripts/eval_partial_refinement.py --config config/eval/lm1b-small.yaml \
      --checkpoint-dir artifacts/lm1b/small \
      --data /scratch/prj0000000245/sakurai/data/lm1b/lm1b_test_bs128_wrapped.dat \
      --n 2048 --out outputs/partial_refinement/oracle_lm1b_small.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from models import partial_refinement as pr

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument(
    "--data", required=True, help="HF save_to_disk dir with input_ids (flm format)"
)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--fractions", default="0.25,0.5,0.75,0.9")
ap.add_argument("--orders", default="l2r,random,confidence")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import datasets

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg, tk, net = pr.load_ncm(a.config, a.checkpoint_dir)
vq = tk.vqvae.vq
L = vq.levels
EOS = tk.pre_tokenizer.eos_token_id
ctx = int(tk.vqvae.context_length)
ds = datasets.load_from_disk(a.data)
n = min(a.n, len(ds))
fractions = [float(f) for f in a.fractions.split(",")]
orders = a.orders.split(",")
rng = torch.Generator(device="cuda").manual_seed(a.seed)

# Conditions: (fill, zero_baseline). Controls use the default zero-baseline form.
CONDS = [("gt", True), ("gt", False), ("random", True), ("shuffled", True)]

acc = {}  # key -> running sums


def add(key, **vals):
    d = acc.setdefault(key, {})
    for k, v in vals.items():
        d[k] = d.get(k, 0.0) + float(v)


def commit_mask(order, n_commit, conf0, length, B):
    if order == "l2r":
        m = torch.zeros(B, length, dtype=torch.bool, device="cuda")
        m[:, :n_commit] = True
        return m
    if order == "random":
        score = torch.rand(B, length, device="cuda", generator=rng)
    elif order == "confidence":
        score = conf0
    else:
        raise ValueError(order)
    return pr.select_new(
        score, torch.zeros(B, length, dtype=torch.bool, device="cuda"), n_commit
    )


t0 = time.time()
with torch.no_grad():
    for s in range(0, n, a.bs):
        ids = torch.tensor(ds[s : min(n, s + a.bs)]["input_ids"], device="cuda")
        assert ids.shape[1] == ctx, (ids.shape, ctx)
        B = ids.shape[0]
        idx = tk.vqvae.encode(ids)
        prefix = torch.full((B, net.prefix_len), EOS, device="cuda")
        ps = pr.prefix_state(net, prefix)
        Hs = pr.cumulative_reconstructions(vq, idx)
        off = L[0]
        for t in range(1, len(L)):
            gt = idx[:, off : off + L[t]]
            off += L[t]
            length = L[t]
            x0 = pr.level_input(net, t, Hs[t - 1])
            lg0 = pr.level_logits(net, t, Hs[t - 1], ps).float()
            lp0 = F.log_softmax(lg0, -1)
            ce0 = -lp0.gather(-1, gt[..., None]).squeeze(-1)
            acc0 = (lg0.argmax(-1) == gt).float()
            conf0 = lp0.max(-1).values
            add(("base", t), ce=ce0.sum(), acc=acc0.sum(), n=ce0.numel())
            for order in orders:
                for f in fractions:
                    k = min(max(round(f * length), 1), length - 1) if length > 1 else 0
                    if k == 0:
                        continue
                    S = commit_mask(order, k, conf0, length, B)
                    U = (~S).float()
                    nU = U.sum()
                    for fill, zb in CONDS:
                        ce, lg, H = pr.oracle_level_ce(
                            net,
                            t,
                            gt,
                            Hs[t - 1],
                            ps,
                            S,
                            fill=fill,
                            zero_baseline=zb,
                            rng=rng,
                        )
                        lg = lg.float()
                        dx = (pr.level_input(net, t, H) - x0).norm(dim=-1).mean()
                        add(
                            (fill, zb, order, f, t),
                            ce_base=(ce0 * U).sum(),
                            ce=(ce * U).sum(),
                            acc_base=(acc0 * U).sum(),
                            acc=((lg.argmax(-1) == gt).float() * U).sum(),
                            dlogit=((lg - lg0).norm(dim=-1) * U).sum(),
                            dx_sum=dx * B,
                            nb=B,
                            n=nU,
                        )
        print(f"{min(n, s + a.bs)}/{n} seqs, {time.time() - t0:.0f}s", flush=True)

rows = []
for key, d in acc.items():
    if key[0] == "base":
        rows.append(
            {
                "kind": "base",
                "level": key[1],
                "len": L[key[1]],
                "ce": d["ce"] / d["n"],
                "acc": d["acc"] / d["n"],
            }
        )
        continue
    fill, zb, order, f, t = key
    nb = d["nb"]
    rows.append(
        {
            "kind": "oracle",
            "fill": fill,
            "zero_baseline": zb,
            "order": order,
            "fraction": f,
            "fraction_real": round(min(max(round(f * L[t]), 1), L[t] - 1) / L[t], 4),
            "level": t,
            "len": L[t],
            "ce_base_U": d["ce_base"] / d["n"],
            "ce_U": d["ce"] / d["n"],
            "delta": (d["ce_base"] - d["ce"]) / d["n"],
            "acc_base_U": d["acc_base"] / d["n"],
            "acc_U": d["acc"] / d["n"],
            "dlogit_U": d["dlogit"] / d["n"],
            "dinput": d["dx_sum"] / nb,
        }
    )
meta = {
    "config": a.config,
    "checkpoint": a.checkpoint_dir,
    "data": a.data,
    "n": n,
    "levels": list(L),
    "seconds": time.time() - t0,
}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as fh:
    json.dump({"meta": meta, "rows": rows}, fh, indent=1)
print("wrote", a.out)
