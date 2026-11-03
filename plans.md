# NCM Within-Level Partial Commitment and Residual Updates: Implementation Plan

**Date:** November 2, 2026  
**Target:** Next-Concept Modeling (NCM)  
**Strategy:** Keep the pretrained tokenizer and NCM Transformer fixed. First evaluate oracle conditioning and generation without additional training. Consider fine-tuning on partially committed states only after analyzing these results.

## 1. Objective and Hypotheses

NCM generates hierarchical levels of discrete codes from coarse to fine. Codes within each level are predicted in parallel. After a level is committed, its contribution is incorporated into the cumulative reconstruction used to condition the next level.

**Proposal:** Apply the same update mechanism *within* a level, not only *between* levels. After committing a subset of codes at level $k$, add their contribution to the cumulative reconstruction and use the **same pretrained NCM Transformer** to predict the remaining codes again.

We will test three hypotheses:

- **H1 (information utilization):** Conditioning on a subset of ground-truth codes reduces the prediction cross-entropy (CE) of the remaining codes.
- **H2 (generation quality):** Iterative generation that commits model-predicted codes in stages can outperform standard one-pass-per-level generation under some compute budgets.
- **H3 (conditional dependence):** The resulting improvements provide evidence about conditional dependencies remaining within a level. However, CE improvements from an unchanged pretrained model must **not** be reported as direct estimates of conditional total correlation (TC).

**Priority:** H1 → H2 → H3.

## 2. Mathematical Formulation

Let the codes at level $k$ be $r_k=(r_k^1,\ldots,r_k^{\ell_k})$, and let the cumulative reconstruction up to the preceding level be

$$
H_{k-1}=\hat z^{(k-1)}\in\mathbb R^{L\times d}.
$$

Let the code embeddings be $e_k(r_k)\in\mathbb R^{\ell_k\times d}$. Represent the existing level-update operator by

$$
F_k(u)=\operatorname{QuantResi}_k\!\left(\operatorname{Upsample}_k(u)\right).
$$

In the implementation, this corresponds to `vq._up_to_full` and `vq._apply_quant_resi`. Depending on the configuration, these operators can include learned convolutions.

The standard complete-level update is

$$
\hat z^{(k)}=H_{k-1}+F_k(e_k(r_k)).
$$

### 2.1 Partially Committed States

Let $S\subseteq\{1,\ldots,\ell_k\}$ denote the committed positions and let $m_S\in\{0,1\}^{\ell_k}$ be their binary mask. Define the following **partial update**:

$$
\begin{aligned}
\rho_S&=|S|/\ell_k,\\
C_k(S,r_S)&=F_k(m_S\odot e_k(r_k))-(1-\rho_S)F_k(0),\\
H_{k,S}&=H_{k-1}+C_k(S,r_S).
\end{aligned}
$$

Embeddings at uncommitted positions are set to zero. The zero-baseline correction guarantees the boundary conditions

$$
C_k(\varnothing,\cdot)=0,\qquad
C_k(\{1,\ldots,\ell_k\},r_k)=F_k(e_k(r_k))
$$

**exactly**, including configurations in which biases in the upsampling or quantization-residual operators make $F_k(0)\neq0$. The intermediate interpolation remains a new design choice and may be outside the original NCM training distribution. As an initial ablation, also test the simpler form $F_k(m_Se_k(r_k))$.

Construct the current-level Transformer input as

$$
X_{k,S}=\operatorname{Downsample}_{\ell_k}(H_{k,S}).
$$

Apply **the same input pipeline as the existing implementation**, including `area` resampling, `post_upsample_conv`, position and level embeddings, prefix handling, and level-specific prediction heads. Do not pass $H_{k,S}$ to the Transformer through an ad hoc path: reuse the normal input path for level $k$.

**Important:** At inference time, the true latent representation $z=E_\phi(x)$ is unknown. Hence the true residual $z-H_{k,S}$ cannot be updated. The updated object is the **cumulative reconstruction, which provides the generator's conditioning input**.

### 2.2 Generation Procedure

1. Start level $k$ with $S=\varnothing$ and perform the standard parallel prediction.
2. Select a subset $A\subseteq S^c$ of uncommitted positions based on their predicted probabilities.
3. Commit the predicted codes at positions $A$ and set $S\leftarrow S\cup A$.
4. **Rebuild** $H_{k,S}$ from the fixed previous-level state $H_{k-1}$, then predict the remaining positions again.
5. Once all positions are committed, verify that the final cumulative reconstruction matches the standard complete-level update, then proceed to level $k+1$.

**Prevent double counting:** Do not repeatedly execute `z_hat += full_level_contribution` after each pass. Doing so would add previously committed contributions multiple times. Always reconstruct the state as $H_{k-1}+C_k(S,r_S)$.

## 3. Mapping to the Existing Codebase

Official repository: <https://github.com/HazyResearch/ncm>

| Location | Existing responsibilities | Proposed change |
|---|---|---|
| `models/quant.py` | `MultiscaleResidualVectorQuantizer`, code embeddings, `_up_to_full`, `_apply_quant_resi`, `get_next_autoregressive_input` | Add a pure helper function, `get_partial_level_contribution`. Leave the existing complete-level update unchanged. |
| `models/next_concept.py` | `NextConceptTransformer.generate`, `_decode_level_logits`, sampling, level heads, prefix/position/level encodings | Retain standard one-pass generation; add a level-wise repeated-prediction helper and `generate_refined`. |
| `config/eval/` | Configurations for pretrained checkpoints | Initially load existing configurations and weights without modification. |
| `evaluate.py` and `scripts/` | Published evaluation pipeline | Add an independent evaluation script with outputs comparable to the official evaluation. |

`vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)` adds the contribution of an **entire level** and returns the input for the **next level**. Therefore, it should not be reused directly for within-level prediction. Instead, reuse the underlying upsampling and quantization-residual operations.

**Level-0 exception:** Some checkpoints use `query_level0`, predicting the first level from learned queries rather than the standard residual-conditioning input. For the minimum viable prototype (MVP), generate level 0 as usual and apply partial refinement only for $k\ge1$. The published implementation also has a separate `l0_autoregressive` mechanism; do not conflate it with this experiment.

**Proposed new files:**

```text
models/partial_refinement.py
scripts/eval_partial_refinement.py
config/partial_refinement.yaml
tests/test_partial_refinement.py
```

## 4. Minimal Implementation API

The following is **illustrative pseudocode**, not an exact transcription of the public model's method signatures.

```python
@torch.no_grad()
def partial_contribution(vq, level_idx, code_ids, committed_mask):
    code_vectors = vq.embeddings[level_idx](code_ids)
    masked_vectors = code_vectors * committed_mask[..., None]
    zero_vectors = torch.zeros_like(masked_vectors)

    def reconstruct(vectors):
        full = vq._up_to_full(vectors, level_idx, vq.levels[-1])
        return vq._apply_quant_resi(full, level_idx)

    fraction = committed_mask.float().mean(dim=1)[:, None, None]
    return reconstruct(masked_vectors) - (1.0 - fraction) * reconstruct(zero_vectors)


@torch.no_grad()
def refine_level(
    model,
    vq,
    level_idx,
    previous_reconstruction,
    prefix_state,
    num_passes,
    policy,
    temperature,
):
    batch_size = previous_reconstruction.shape[0]
    level_size = vq.levels[level_idx]
    committed = torch.zeros(
        batch_size, level_size, dtype=torch.bool, device=previous_reconstruction.device
    )
    codes = torch.zeros(
        batch_size, level_size, dtype=torch.long, device=previous_reconstruction.device
    )

    for step in range(num_passes):
        contribution = partial_contribution(vq, level_idx, codes, committed)
        current_reconstruction = previous_reconstruction + contribution
        level_input = build_existing_level_input(current_reconstruction, level_idx)
        logits = predict_with_existing_head(model, prefix_state, level_input, level_idx)
        candidate_codes = sample_uncommitted(logits, committed, temperature)
        selection = select_new_positions(logits, committed, step, num_passes, policy)
        codes[selection] = candidate_codes[selection]
        committed |= selection

    assert committed.all()
    final_contribution = partial_contribution(vq, level_idx, codes, committed)
    return codes, previous_reconstruction + final_contribution
```

Implementation notes:

- `build_existing_level_input` must preserve the current preprocessing steps, including `F.interpolate(..., mode="area")` and `post_upsample_conv`.
- `predict_with_existing_head` must reuse the existing level-specific logits computation, prefix handling, and attention configuration. A direct generic Transformer call may not reproduce the same outputs.
- The committed set grows monotonically. Do not resample committed codes. Treat schemes that allow revisions as separate experiments.
- Commit at least one position per pass and all remaining positions in the final pass. Ensure that `num_passes=1` reproduces fully parallel generation.
- Start with greedy evaluation (`temperature=0`), then evaluate the original sampling configuration.

## 5. Experimental Plan

### Phase 0: Reproduction and Consistency Tests

**Goal:** Verify that the repeated-prediction implementation matches the original NCM whenever no additional refinement is performed.

- [ ] Load a published checkpoint and dataset; reproduce standard NCM generation and evaluation.
- [ ] Begin with a published TinyStories or LM1B checkpoint.
- [ ] Verify that `num_passes=1` produces the same logits and samples as standard generation under identical random-number conditions.
- [ ] For $S=\varnothing$, verify that the partial contribution is zero and that both level inputs and logits match the original implementation.
- [ ] For $S=\{1,\ldots,\ell_k\}$, verify that the result matches the standard level update in `get_next_autoregressive_input`.
- [ ] Verify that expanding the committed set does not modify the fixed previous-level state or double-count earlier contributions.
- [ ] Check masks, codebook indices, tensor shapes after upsampling, device placement, and dtypes.
- [ ] Document unsupported paths for `query_level0`, level heads, RoPE, CFG, and `post_upsample_conv` configurations.

**Exit criteria:** Exact one-pass equivalence, correct complete-level updates, and passing unit tests.

### Phase 1: Oracle Partial Refinement — Highest Priority

**Goal:** Test whether conditioning on some correct codes at a level improves predictions of the remaining codes.

For each evaluation example, obtain the ground-truth codes $r_k^*$ from the tokenizer and construct the **ground-truth cumulative reconstruction through the preceding level**, $H_{k-1}^*$.

- **Baseline:** Set $S=\varnothing$ and record CE against each true code.
- **Oracle:** Add some **ground-truth codes** to $S$ and record CE only on uncommitted positions.
- **Control:** Insert random or shuffled codes instead, to help distinguish useful conditioning information from effects caused merely by altering the input distribution.

For a fixed committed set $S$ and the same uncommitted positions $U=S^c$, measure

$$
\Delta_{k,S}^{\mathrm{oracle}}=
\frac{1}{|U|}\sum_{i\in U}
\left[
-\log q_\theta(r_k^{*,i}\mid H_{k-1}^*)
+\log q_\theta(r_k^{*,i}\mid H_{k,S}^*)
\right].
$$

A positive value indicates a CE improvement due to oracle conditioning.

**Commitment orders:** Left-to-right, random, and descending initial-prediction confidence. Since confidence-based selection introduces selection bias, give fixed-order results priority when interpreting conditional dependence.

**Committed fractions:** 0%, 25%, 50%, 75%, and 90%, rounded to accommodate each level's length.

**Outputs:** CE improvement by level, CE versus committed fraction, correct-prediction rate, change in logits after updating, and norm differences between updated and standard inputs. Record the number of codes at each level.

**Interpretation constraint:** If the model is not calibrated or accurate on partially committed states, the CE improvement is not a conditional TC estimate. A lack of oracle CE improvement does not imply that the true codes have weak conditional dependence.

### Phase 2: Generation Evaluation Without Additional Training

Compare the standard NCM sampler against the following samplers, all using the **same model and checkpoint**:

| Configuration | Within-level procedure | Purpose |
|---|---|---|
| P1 | Commit all positions in one pass | Standard NCM baseline |
| P2 | Commit the top 50% by confidence, then the remainder | Detect potential gains with two passes |
| P4 | Commit additional uncommitted positions progressively over four passes | Quality–NFE comparison |
| P8 | Finer-grained commitment over eight passes | Assess the limit of repeated refinement |
| R4 | Commit positions in random order over four passes | Control for confidence-based selection |

**Priority:** Levels with the largest oracle CE improvements in Phase 1. Also compare against allocating extra passes uniformly across all levels.

**Metrics:**

- LM1B / TinyStories / OWT: GenPPL under the published evaluation protocol, unigram entropy, and, where feasible, distinct-n and qualitative samples.
- Sudoku: solved rate, cell accuracy, and valid-grid rate. Sudoku need not be part of the initial MVP.
- Across all tasks: NFE, tokens/second, per-sample latency, GPU memory use, and error rate among committed positions.
- Fix checkpoints, prompts, random seeds, evaluators, and sample counts. Report **NFE-matched** comparisons separately from **wall-clock-matched** comparisons.
- Standard NCM makes one forward pass per level. Repeated updates increase the forward-pass count, and different levels can have different computational costs; therefore NFE alone does not characterize runtime.

**Primary plot:** Generation quality (GenPPL) versus measured latency or total forward computation. **Secondary plot:** Oracle CE improvement by level and committed fraction.

### Phase 3: Fine-Tuning on Partially Committed States — Only If Needed

If Phase 1 suggests that the pretrained model cannot adequately process partially committed states, introduce fine-tuning:

- Initially keep the tokenizer fixed.
- Sample a random committed subset $S$ for each training example and construct $H_{k,S}$ with **ground-truth codes**.
- Fine-tune the same NCM Transformer, computing CE only on uncommitted positions $i\notin S$.
- Include $S=\varnothing$ for a nonzero fraction of training examples to preserve parallel-generation performance.
- Sample committed fractions broadly and optionally mix in confidence-selected subsets.
- Keep training and validation disjoint and verify that baseline GenPPL does not substantially deteriorate.

$$
\mathcal L_{\mathrm{partial}}=
\mathbb E_{x,k,S}\left[
\frac{1}{|S^c|}\sum_{i\notin S}
-\log q_\theta(r_k^{*,i}\mid H_{k,S}^*,k)
\right].
$$

If necessary, retain the original parallel NCM objective:

$$
\mathcal L=\mathcal L_{\mathrm{NCM}}+\lambda\mathcal L_{\mathrm{partial}}.
$$

Because the committed set $S$ may not be uniquely identifiable from the reconstructed features alone, also compare a variant that **explicitly supplies the committed-position mask** during fine-tuning.

## 6. Relationship to Conditional Total Correlation

The conditional total correlation of codes within level $k$ is

$$
\mathrm{TC}(R_k\mid H_{k-1})=
\sum_i H(R_k^i\mid H_{k-1})-
H(R_k\mid H_{k-1}).
$$

For a fixed ordering $1,\ldots,\ell_k$, if the predictor represents the true conditional distributions and the partial reconstruction preserves the information in the code prefix $R_k^{<i}$, then

$$
\mathrm{TC}(R_k\mid H_{k-1})=
\sum_i\left[
H(R_k^i\mid H_{k-1})-
H(R_k^i\mid H_{k-1},R_k^{<i})
\right].
$$

Thus, the ideal improvement from AR-like within-level conditioning is directly related to conditional TC.

However, the proposed input $H_{k,S}$ is **not** the code prefix itself. It is a transformed representation obtained through upsampling and convolutions, and it may not be a sufficient statistic for the committed codes. Even an ideal predictor may therefore have access to less information. Furthermore, the original NCM was not trained on partial states. **Do not report measurements from Phase 1 or 2 as direct TC estimates.** Even after Phase 3, approximation quality and information preservation require separate validation.

To regularize TC directly in a later study, consider bilevel optimization based on the difference between parallel CE and fixed-order teacher-forced CE. This would require explicit treatment of approximation errors in both modes, gradient estimation through the tokenizer's discrete assignments, and the reconstruction-quality trade-off.

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Partial updates produce out-of-distribution Transformer inputs. | Compare oracle and random controls; monitor input norms and changes in logits. Proceed to Phase 3 if necessary. |
| Upsampling or `quant_resi` includes biases or convolutions. | Measure $F_k(0)$ and unit-test the empty/full-set boundary conditions. |
| Partial reconstructions do not preserve sufficient information about committed positions. | Test explicit mask conditioning and diagnostics for information retention. |
| Incorrect high-confidence predictions are committed too early. | Compare oracle and sampled performance, and test random, confidence-based, and fixed-order policies. |
| Repeated inference is expensive. | Reuse the zero correction and prefixes where valid; allocate passes by level. A KV cache cannot automatically be reused within a level because the entire conditioning input changes. |
| Refinement improvements are mistaken for reductions in TC. | Treat TC estimation and generation-quality improvements as distinct claims. |
| Improvements limited to the final level are mistaken for global gains. | Report both per-level and end-to-end evaluations. |

## 8. Milestones and Exit Criteria

- [ ] **M0:** Reproduce baseline NCM quality and generation speed.
- [ ] **M1:** Implement `get_partial_level_contribution`; pass empty-set, full-set, and no-double-counting tests.
- [ ] **M2:** Implement level-wise repeated prediction with exactly matching one-pass logits.
- [ ] **M3:** Measure oracle CE improvements by level, committed fraction, and commitment order.
- [ ] **M4:** Compare P1/P2/P4/P8/R4 generation and produce quality–compute curves.
- [ ] **M5 (conditional):** Fine-tune on partial states and assess whether gains can be obtained without sacrificing P1 performance.

**Initial go/no-go decision:** At M3, determine whether oracle conditioning yields consistent CE improvements relative to the random-code control. If not, do **not** immediately conclude that TC is small; first inspect the input construction and out-of-distribution effects.

## 9. References

- Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, 2026. <https://arxiv.org/abs/2610.14162>
- Official implementation: <https://github.com/HazyResearch/ncm>
- Generator implementation: <https://github.com/HazyResearch/ncm/blob/main/models/next_concept.py>
- Multiscale VQ implementation: <https://github.com/HazyResearch/ncm/blob/main/models/quant.py>
- Checkpoints and evaluation datasets: <https://huggingface.co/collections/hazyresearch/ncm>

---

**First implementation task:** Using a pretrained TinyStories or LM1B model, implement the exact-equivalence test for `num_passes=1` and the oracle CE improvement curves. **Defer tokenizer retraining and explicit TC regularization until those results have been established.**

