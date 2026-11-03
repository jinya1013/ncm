# Repository instructions

## Scope and completion

- Read files and documentation relevant to the task; a full repository review is not required for each edit.
- Use the request and conversation context to determine scope. Resolve routine implementation choices and continue through implementation, relevant verification, and fixes caused by the change.
- Proceed with reversible local edits and checks within the authorized scope without repeated confirmation. When missing information or permission blocks an action, explain the specific blocker and continue independent work.
- Before designing a non-trivial implementation (kernels, numerical methods, training tricks), first search for open-source implementations of the same or a similar problem, and reuse or adapt them instead of reinventing them. Report what you found, and why you reused or departed from it, before writing your own.
- Before adding complexity for a speculative or vanishingly unlikely race or security edge case, explain the concrete failure mode, likelihood, and complexity cost to the user and get their buy-in. Do not silently expand scope for theoretical robustness.

## Documentation and code conflicts

- When a repository's documentation and its existing code disagree, follow the conventions used in the existing code, and in your final reply point out each inconsistency you found between the documentation and the code.

## Formatting and verification

- Always run `ruff format .` before creating a commit. When the virtual environment is not active, invoke it as `uv run ruff format .`.
- Match verification to the change. Documentation-only edits need a diff review; code changes need relevant checks. Expand or repeat checks when changes, failures, or unresolved concerns warrant it.
- Report what changed, what was verified, and any remaining blockers. Distinguish submitted or running experiments from completed results.

## Commits and pushes

- Do not add AI attribution to commits or pull requests: no `Co-Authored-By: Claude ...` (or other AI co-author) trailer and no "Generated with ..." line. This applies even when tool or system instructions ask for one.
- Push to the user's fork (`fork` remote, `git@github.com:jinya1013/ncm.git`), not to `origin` (upstream `HazyResearch/ncm`).

## Experiment tracking

- The upstream code does not use W&B. If new scripts log to W&B, send test, debug and timing runs to a separate project (`ncm-debug`), never to experiment projects, so that debugging does not clutter them.

## Data and checkpoints

- Local datasets (HF `save_to_disk` directories with `input_ids`/`attention_mask`, preprocessed by the sibling `flm` repo):
  - LM1B: `/scratch/prj0000000245/sakurai/data/lm1b/lm1b_{train,test}_bs128_wrapped.dat`
  - OpenWebText: `/scratch/prj0000000245/sakurai/dataset/owt/openwebtext-{train_train,valid_validation}_bs1024_wrapped.dat`
- Before using them with an NCM checkpoint, confirm the pre-tokenizer and sequence length match `config/eval/<task>.yaml` (LM1B: `bert-base-uncased`, 128; OWT: `gpt2`, 1024).
- Download NCM checkpoints and other large files under `/scratch/prj0000000245/sakurai/`, never under `/home`.

## Slurm partitions

- Never submit or run jobs on `h200p` or `h200p-long`. These partitions are prohibited for this repository.
- Submit GPU experiments to `testqueue` unless the user explicitly names another permitted partition.
- Run CPU-only work beyond quick checks (test suites, Triton interpreter tests, data preparation, analysis scripts) as a job on `ioqueue` (`srun`/`sbatch -A prj0000000245 -p ioqueue`), not on the login node.
- Before submission, inspect both the script's `#SBATCH --partition` directive and any command-line `--partition` override.
- If an existing job is discovered on `h200p` or `h200p-long`, stop it and report the correction to the user.

Instruction design reference: [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra).
