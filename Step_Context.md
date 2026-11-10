# Step Context — E0 through G4.2 Research Journey

Current chronological research record, beginning with the Phase E entry point
and now extending through the G4.2 hierarchy campaign. Earlier entries are
preserved as written; the consolidated 2026-11-10 entry below records the
subsequent E-F0 and G-series work in experimental order. Context.md retains
the wider historical background; this file is the active experiment and
decision handoff.

---

## 2026-09-16 — Phase E+F roadmap established; E0 ready to begin

### What this stage establishes

The repo has completed Milestone D on all seven Definition-of-Done criteria.
The substrate is validated on synthetic tasks. The next two phases are now
defined:

- **Phase E (Real-Data Bridge):** Apply the validated substrate to Facebook's
  bAbI dataset — a published, well-benchmarked reasoning task suite with
  ~150-token vocabulary. Goal: prove D2C works on real data, make the
  interpretability story tangible, and establish the slow-channel finding
  on a recognised benchmark.

- **Phase F (Autonomy Stack):** Build the self-learning machinery (runtime
  spine, HJB critics, predictive coding, composite objective, birth/death,
  gamma adaptation, agent sandbox) on top of the real-data-proven substrate.

### Key decisions made

1. **Dataset: bAbI** over CBT, SimpleQuestions, WikiQA, or custom synthetic.
   Rationale: ~150 token vocabulary fits existing infrastructure; 20 tasks
   decompose by reasoning type; well-studied baselines (Memory Networks,
   LSTMs).

2. **State dimensions: both 64 and 150.** Run at both sizes, report the
   comparison. Tests whether D2C can compress.

3. **Task ordering: QA1 → QA2 → QA3 → QA5 → QA11 → QA14.** Start with
   purest memory retrieval (QA1–QA3), then entity tracking (QA5, QA11),
   then temporal reasoning (QA14). Tasks 17–20 as negative controls.

4. **Phase E before Phase F.** Building self-learning machinery without
   knowing what it needs to learn is designing in a vacuum. E0–E3 provide
   the empirical grounding for F0–F6 design.

### What is ready to build

**E0: bAbI Integration and Supervised QA** — three new files:

| File | Purpose |
|---|---|
| `code/python/d2c/digital/babi.py` | Task loader, story parser, episode builder |
| `code/python/d2c/experiments/babi_qa.py` | Experiment entrypoint |
| `code/python/tests/test_babi.py` | Tests |

No existing files need modification. The bAbI integration composes:
- `Vocabulary` (tokens.py) — construct from bAbI token set
- `TokenForcingBridge` (bridge.py) — deterministic dense codes at 64/150-dim
- `DigitalDirector` / `DigitalEpisode` (director.py) — episode runner
- `QueryConditionedMLPReadout` (readout.py) — answer classification
- `DigitalMemoryConfig` (bridge.py) — validated D1/D2 kernel

### Gate for E0

Full kernel achieves >80% accuracy on QA1 (single supporting fact) at both
64-dim and 150-dim. Ablations (full / no_slow / collapsed_gamma) show the
expected gradient: full >= collapsed_gamma > no_slow.

### Gate for E1–E3

- E1: D_eff(QA3) > D_eff(QA2) > D_eff(QA1). Channel specialisation visible.
- E2: Full beats no_slow with paired CI (p < 0.05) on QA1–QA3.
- E3: Persistent-state reading matches per-story accuracy. Consolidation fires.

### Roadmap file

`d2c_milestone_f.md` has been restructured as the unified E+F roadmap
(previously Milestone F only). Backup at `d2c_milestone_f.md.bak`.

---

## 2026-09-17 — bAbI dataset acquired and verified; machine suitability confirmed

### Dataset acquired

**Source:** https://huggingface.co/datasets/Muennighoff/babi
(en-valid variant, all 20 tasks, JSONL format — adapted from Stanford HELM's
bAbI scenario, originally from `tasks_1-20_v1-2`)

**Location:** `code/python/d2c/progress/babi_data/`

| Split | File | Stories | Disk |
|---|---|---|---|
| train | babi_train.jsonl | 18,013 (900/task) | 7.0 MB |
| test | babi_test.jsonl | 20,000 (1000/task) | 7.8 MB |
| valid | babi_valid.jsonl | 1,987 | 0.8 MB |
| **total** | | **40,000** | **15.2 MB** |

**Format:** one JSON record per line:
```json
{"passage": "Mary moved to the bathroom.\nJohn went to the hallway.\n",
 "question": "Where is Mary?", "answer": "bathroom", "task": 1}
```

**Vocabulary:** 270 unique tokens across all 20 tasks; **57 tokens** for the
QA1+QA2+QA3 target subset. Well within the existing `Vocabulary` /
`TokenForcingBridge` infrastructure.

**CLI download (verified, no UI or `datasets` library needed):**
```bash
curl -sL -o babi_train.jsonl "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl"
curl -sL -o babi_test.jsonl  "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_test.jsonl"
curl -sL -o babi_valid.jsonl "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_valid.jsonl"
```

### Sourcing notes (why this mirror)

- Original tarball (`thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz`):
  **dead (404)** — confirmed.
- `facebookarchive/bAbI-tasks` (GitHub): the Lua task *generator*; README
  states generated output does not match the public dataset exactly.
- HuggingFace `facebook/babi_qa` parquet conversion: covers **qa1 only**.
- `Muennighoff/babi`: only verified full-20-task mirror, CLI-fetchable,
  15.2 MB total. **Selected.**
- Backup mirrors (unverified): `RawthiL/babi_tasks/data/`;
  `RMT-team/babilong` (longer-context bAbI redesign, relevant at Phase F scale).

### Machine suitability (this workstation)

2 cores x86_64, 3.7 GB RAM (~800 MB available), 1.8 GB disk free, Python 3.10,
venv has numpy + matplotlib + pytest. **Adequate for E0–E2.** The D2C SOE
stepper is O(state_dim x channels) per step; at 64-dim/3 channels a full
epoch of QA1 (~900 stories x ~10 steps = ~9k memory steps) is seconds.
No GPU required. Disk headroom after dataset: ~1.8 GB — sufficient for
progress bundles at these experiment sizes.

### Yes — the same dataset supports the full Phase E progression

The single `Muennighoff/babi` download supports every E stage and most F gates:

| Stage | Uses dataset how |
|---|---|
| E0 (supervised QA) | QA1 first, then QA2/QA3; train/valid/test splits as-is |
| E1 (D_eff analysis) | Same runs, diagnostics extracted per task |
| E2 (ablation suite) | QA1, QA2, QA3, QA5, QA11, QA14 from the same files |
| E3 (continuous reading) | Concatenated multi-story sessions from train split |
| F0 (runtime smoke) | QA1–QA3 streaming for 10k steps |
| F1–F6 gates | bAbI-specific acceptance checks (roadmap §6) |

No further dataset acquisition is needed for the E+F roadmap. When Phase F
reaches larger-scale experiments (SQuAD-style reading comprehension,
interactive Q&A), additional datasets will be sourced then.

### Published bAbI benchmarks for later comparison

Recorded reference points from the original papers (Weston et al. 2015,
arXiv:1502.05698; Sukhbaatar et al. 2015, arXiv:1503.08895) — all trained on
10k story versions, error rates reported as % of stories wrong:

| Method | Task 1 | Task 2 | Task 3 | Tasks solved (<5% error) |
|---|---|---|---|---|
| LSTM (random baseline 50%) | 0% | 63% | 93% | 8/20 |
| MemNN (original, no ench.) | 6% | 97% | 98% | 0/20 initially |
| MemNN w/ enhancements | 0% | 0% | 0% | 19/20 |
| End-to-End MemNN (10k) | 0% | 0% | 1% | 20/20 |

Interpretation for D2C positioning:

- Our E0 target (>80% accuracy on QA1) is well below MemNN's 100% — that is
  expected and acceptable. The claim is **not** leaderboard parity; it is the
  interpretability story: D_eff / channel specialisation maps onto task
  structure, which no MemNN/Transformer report includes.
- Tasks where D2C fails (expected: QA17–QA20) mirror exactly where the weak
  baselines fail — the failure *pattern* is the comparable, publishable
  finding.
- Caveat: benchmark numbers are for the 1k/10k task versions; our mirror is
  the en-valid variant with 900–1000 stories/task. Numbers remain comparable
  because the task distribution is the same; we must state the variant in
  any report.

### Next action

Build `code/python/d2c/digital/babi.py` — the bAbI task loader, pointing at
`code/python/d2c/progress/babi_data/` as the default data path. Followed by
tests, then the experiment entrypoint.

---

## 2026-09-18 — E0 bAbI loader + experiment implementation complete; smoke test results

### Files implemented

| File | Lines | Purpose |
|---|---|---|
| `code/python/d2c/digital/babi.py` | ~190 | BabiDataset loader, BabiStory, vocab/bridge/config helpers, episode builder |
| `code/python/d2c/experiments/babi_qa.py` | ~195 | BabiQAConfig, run/save/CLI; supervised MLP readout over SOE memory |
| `code/python/tests/test_babi.py` | ~155 | 15 tests covering dataset loading, helpers, episode construction |

**Tests:** 15/15 passed (dataset load, filter, vocab, bridge dense/one-hot,
memory configs, episode structure/silence/query/dim).

### Architecture

Each bAbI story is converted to a DigitalEpisode: every word becomes a forcing
step through TokenForcingBridge (deterministic dense codes). A bag-of-words
query encoding is built from question words. The SOE memory is stepped through
all story words + question words + one zero-forcing readout step. Then
QueryConditionedMLPReadout classifies the answer from
`query_conditioned_features(state, memory_cfg, query)`.

### E0 smoke test results

All on QA1 (single supporting fact), 1000 test stories:

| dim | variant | train stories | seeds | mean accuracy |
|---|---|---|---|---|
| 16 | full | 100 | 3 | 0.178 ± 0.013 |
| 16 | full | 500 | 3 | 0.163 ± 0.006 |
| 16 | no_slow | 500 | 3 | 0.222 ± 0.021 |
| 16 | collapsed_gamma | 500 | 3 | 0.178 ± 0.006 |
| 32 | full | 300 | 3 | 0.168 ± 0.008 |

**Interpretation:** Above chance (1/~10 unique answers ≈ 10%) but well below
the E0 gate (>80%). The ablation pattern (no_slow > full) is inverted from the
expected — this is because:
1. The MLP readout is a 32-hidden-dim network classifying from 288-dim
   features (dim=16) — the feature space dominates the hidden layer.
2. The slow channel (γ=0.1) at dim=16 doesn't yet show its interference
   filtering advantage — bAbI QA1 is a simple retrieval task with no
   distractors.
3. The MLP needs more epochs (currently 5) and possibly a larger hidden dim.

These are honest smoke results establishing the pipeline, not claims.

### Performance constraint identified

`query_conditioned_features` returns `dim + dim + dim*dim` features
(gated interaction). At dim=64 that is 4224 features — the pure-Python
MLP training is O(features × hidden × output) per story, which makes
full-scale runs (>500 stories × 5 epochs × 64-dim) take >300s on this
2-core machine. The SOE stepper is not the bottleneck — the MLP readout is.

**Resolution options (for later, not blocking E0):**
- numpy-ize the MLP forward/backward (order-of-magnitude speedup)
- Use a linear readout instead of MLP for initial experiments
- Run E0 claim runs on a faster machine or with reduced dim (32)
- Restrict to 200 stories × 10 epochs for dim=64 claim (still >80%)

The current machine is adequate for code development, testing, and
dim-16/dim-32 exploration. The E0 gate (>80% on QA1) will require either
the numpy-ized MLP or a faster CPU.

---

## 2026-09-21 — Machine constraint analysis: detailed benchmarks

### Hardware

```
CPU:    Intel Celeron 3855U @ 1.60GHz, 2 cores, 2 MB cache
RAM:    3.7 GB total, ~870 MB available, 1.6 GB swap (77 MB used)
Disk:   29 GB eMMC, 2.7 GB free
Arch:   x86_64, Linux 5.15, Python 3.10.12
```

This is a low-end Chromebook-class processor (2016 Skylake, no turbo boost,
no hyperthreading). It is the weakest link — RAM and disk are not the
constraint.

### Benchmark results (pure Python MLP, the actual bottleneck)

| dim | features | train_step (fwd+bwd) | predict (fwd only) |
|---|---|---|---|
| 16 | 288 | 12.7 ms | 1.68 ms |
| 32 | 1088 | 29.9 ms | 5.26 ms |
| 64 | 4224 | 114.8 ms | 18.19 ms |

**SOE stepper (not the bottleneck):**

| dim | step time | 10k steps |
|---|---|---|
| 16 | 0.050 ms | 0.5 s |
| 32 | 0.098 ms | 1.0 s |
| 64 | 0.197 ms | 2.0 s |

**NumPy forward (baseline for speedup estimate):**

| dim | numpy fwd | pure-Python train_step | speedup |
|---|---|---|---|
| 16 | 0.046 ms | 12.7 ms | **279x** |
| 32 | 0.117 ms | 29.9 ms | **255x** |
| 64 | 0.401 ms | 114.8 ms | **286x** |

### Time estimates per experiment cell (900 stories × 5 epochs)

| dim | seeds | pure-Python | numpy-ized (est.) |
|---|---|---|---|
| 16 | 3 | **2.9 min** | ~6s |
| 16 | 20 | **19 min** | ~4s |
| 32 | 3 | **6.7 min** | ~16s |
| 32 | 20 | **45 min** | ~11s |
| 64 | 3 | **26 min** | ~5s |
| 64 | 20 | **172 min** (2.9 hrs) | ~36s |

### What this means for each E stage

| Stage | Requirement | dim | Pure-Python time | Feasible? |
|---|---|---|---|---|
| E0 smoke | 3 seeds, quick | 16 | ~1 min | **Yes** |
| E0 claim (3 variants × 20 seeds) | 60 cells × 900 stories | 16 | **~114 min** (1.9 hrs) | Marginal |
| E0 claim | 60 cells | 64 | **~172 hrs** | **No** (pure Python) |
| E1 D_eff analysis | 3 tasks × 3 seeds | 16 | ~9 min | **Yes** |
| E2 full ablation | 6 tasks × 3 variants × 20 seeds = 360 cells | 16 | **~684 min** (11.4 hrs) | Marginal |
| E2 full ablation | 360 cells | 64 | **~1032 hrs** | **No** |

### The honest constraint

**The SOE memory stepper is not the bottleneck.** At any dimension, the SOE
stepper completes 10k steps in under 2 seconds. The bottleneck is the
**pure-Python MLP readout** — specifically the nested list-comprehension
matrix operations in `QueryConditionedMLPReadout._forward()` and
`train_step_with_input_gradient()`.

**NumPy would remove the constraint entirely.** A numpy-ized MLP is
~270x faster, which would make the full E0 claim (60 cells × 64-dim)
complete in ~2 minutes instead of ~172 hours.

### Three resolution paths

**Path A: NumPy-ize the MLP readout (recommended)**

Replace the pure-Python matrix operations in `QueryConditionedMLPReadout`
with numpy arrays and `np.dot` / `np.maximum`. This is a localised change
to one file (`readout.py`) — the class interface stays identical. NumPy is
already in the venv (2.2.6). No new dependencies.

Estimated effort: ~2-3 hours of work. Expected speedup: ~270x.
After this, dim=64 with 20 seeds per cell completes in seconds.

**Path B: Linear readout (faster, simpler, interpretable)**

Replace the MLP with a single linear layer: `features @ W → logits →
softmax → argmax`. This is even faster than numpy-MLP, directly
interpretable (each weight maps to a feature), and sufficient for the
E0 claim. The interpretability story actually improves: "every feature
contributes linearly to the answer" is a cleaner claim than "the MLP
hides its reasoning in a hidden layer."

Estimated effort: ~1 hour. Expected speedup: ~500x.
After this, dim=150 with 20 seeds completes in seconds.

**Path C: Run on a faster machine**

Move claim runs to a cloud VM (2-core modern Xeon/EPYC) or a desktop.
The code is identical; only the CPU wall-clock changes. A modern 2-core
desktop (Zen 4, Alder Lake) would be ~5-10x faster even in pure Python.

### Recommendation

**Implement Path B (linear readout) first** — it is the fastest to
implement, gives the biggest speedup, and strengthens the interpretability
story. Then implement Path A (numpy-MLP) as a second readout option for
the E0 gate comparison (linear vs. MLP, same as the 64-vs-150 dim
comparison we already committed to).

Both paths are localised to `readout.py` and `babi_qa.py`. No
architectural changes, no new dependencies.

### Revised E0 gate timeline

With a numpy or linear readout on this machine:

- E0 claim (QA1, 64-dim + 150-dim, 3 variants × 20 seeds): **~5-10 minutes**
- E1 D_eff analysis (QA1-3, 2 dims, 3 seeds): **~30 seconds**
- E2 full ablation (6 tasks × 3 variants × 20 seeds): **~30-60 minutes**

All within this machine's capability after the readout speedup.

### Next action

Implement a `LinearReadout` class in `readout.py` (single-layer, softmax,
cross-entropy, numpy operations). Add `--readout linear|mlp` flag to
`babi_qa.py`. Run E0 smoke at dim=64 with linear readout to validate
the speedup, then proceed to E0 claim runs.

### Honest distance to the Q&A vision

What exists vs. what the full Q&A system needs:

| What exists | What's needed | How to close |
|---|---|---|
| Token forcing (30-40 tokens) | Learned embedding (5k-10k vocab) | Standard NLP layer |
| 14-dim state, 3 channels | 256-512 dim, birth/death channels | Parameter change + F4 |
| Episode-based operation | Continuous operation | F0 (runtime spine) |
| Supervised learning | Self-supervised learning | F1 + F3 |
| Fixed channels | Self-organising channels | F4 |
| Classification readout | Autoregressive generation | Generation head |

The bAbI experiment is the first bridge from "toy experiments" to "real data"
and it strengthens every downstream claim.

---

## 2026-09-19 — E0 bAbI convergence sweep + ablation results (dim=16, QA1)

### Infrastructure

- Installed MSYS2 + MinGW-w64 (gcc 16.2.0, cmake, ninja) for Windows builds
- C++ core: all 13 tests pass (static build); shared library builds for wrappers
- Python venv with numpy, scipy, matplotlib, pytest, ruff
- Added `NumpyMLPReadout` to `readout.py` — numpy-ized forward/backward pass
  for ~10x speedup over pure-Python `QueryConditionedMLPReadout`
- bAbI dataset downloaded to `code/python/d2c/progress/babi_data/`

### Hyperparameter sweep (dim=16, QA1, 5 seeds each, 500 train stories)

| Config | Accuracy | Notes |
|---|---|---|
| hd=128, ep=20, lr=**0.05** | 0.199 ± 0.012 | Original Step_Context config — overfits |
| hd=128, ep=30, lr=0.05 | 0.195 ± 0.005 | More epochs doesn't help at this lr |
| hd=256, ep=20, lr=0.05 | 0.196 ± 0.012 | Bigger hidden dim doesn't help at this lr |
| hd=128, ep=10, lr=0.05, 900 stories | 0.209 ± 0.006 | More data marginally helps |
| **hd=128, ep=20, lr=0.02** | **0.334 ± 0.005** | **Dominated: lr is the key knob** |

**Key finding:** The learning rate was the dominant factor, not hidden dim or
epoch count. lr=0.02 gives 0.334 vs lr=0.05 at 0.199 — a +68% relative
improvement. The higher learning rate was causing gradient explosion on the
288-dim feature space.

### Ablation results (best config: hd=128, ep=20, lr=0.02, 500 train, 5 seeds)

| Variant | Accuracy | Channels |
|---|---|---|
| full (γ=[2.0, 0.5, 0.1]) | 0.334 ± 0.005 | 3 |
| **no_slow** (γ=[2.0, 0.5]) | **0.402 ± 0.008** | 2 |
| collapsed_gamma (γ=[1.0, 1.0, 1.0]) | 0.368 ± 0.006 | 3 |

**Honest interpretation:** The ablation pattern is inverted from the D2C
hypothesis. `no_slow` (removing the slow channel) gives the *best* accuracy
on QA1. This is consistent with the original E0 smoke findings (Step_Context,
2026-09-18).

**Why this happens at dim=16 on QA1:**
1. QA1 is a simple single-fact retrieval with no distractors — fast memory
   suffices. The slow channel's interference-filtering advantage (proven in
   Milestone D on synthetic tasks with distractors) is not needed here.
2. At dim=16, the state space is too small for the slow channel to encode
   useful long-range structure without creating noise.
3. The slow channel (γ=0.1) retains old forcing patterns that conflict with
   the correct answer signal at readout time.

**This does NOT invalidate the Milestone D claims.** D3-R/D4A proved slow-
channel advantage on synthetic tasks with explicit distractors and interference.
QA1 has neither. The claim needs to be scoped: slow-channel advantage appears
under interference load, not on clean single-fact retrieval.

### What this tells us about E0–E2

- The E0 gate (>80% on QA1) is **not achievable at dim=16** with a frozen SOE
  kernel and a learned readout. The readout is doing all the work; the memory
  features are not discriminative enough.
- To reach the E0 gate we likely need:
  - dim=64 or 150 (more representational capacity)
  - OR a learned embedding layer (replace deterministic dense codes)
  - OR a different readout architecture (attention over memory channels)
- The slow-channel story will come alive on QA2/QA3 (multi-fact chains) and
  QA5/QA11 (entity tracking under load), not on QA1.

### Next action

1. Run the same ablation sweep at **dim=32** to test whether representational
   capacity changes the ablation ordering.
2. Run at **dim=16 on QA2 and QA3** (multi-fact chains) where the slow channel
   should start helping.
3. Consider adding a **linear readout baseline** (softmax regression) to
   separate feature-quality issues from MLP-capacity issues.
4. If dim=32/QA2/QA3 still shows no_slow >= full, the E0 claim needs to be
   reframed: "D2C slow-channel advantage is task-dependent and requires
   interference load to manifest."

---

## 2026-09-19 (cont.) — dim=32 ablation on QA1

### Configuration

Same best config from dim=16 sweep: hd=128, ep=20, lr=0.02, 500 train stories,
5 seeds. Only change: dim=32 (feat_dim=1088 vs 288).

### Results

| Variant | dim=16 | dim=32 | Δ |
|---|---|---|---|
| full | 0.334 ± 0.005 | 0.349 ± 0.007 | +0.015 |
| no_slow | **0.402 ± 0.008** | **0.414 ± 0.003** | +0.012 |
| collapsed_gamma | 0.368 ± 0.006 | 0.404 ± 0.004 | +0.036 |

### Interpretation

1. **The ablation ordering is unchanged.** no_slow > collapsed_gamma > full
   at both dim=16 and dim=32. Doubling the state dimension does not flip
   the ordering.

2. **Dim=32 gives a uniform ~1-4 point lift** across all variants. The extra
   capacity helps all kernels equally — it does not selectively help the
   slow channel.

3. **collapsed_gamma closed the gap to no_slow** at dim=32 (0.404 vs 0.414,
   gap=0.010) compared to dim=16 (0.368 vs 0.402, gap=0.034). This is
   consistent with more capacity allowing the collapsed kernel to better
   utilise its uniform fast timescales.

4. **The slow channel is still net-negative on QA1** at both dimensions.
   The hypothesis that "dim=16 was too small for the slow channel" is
   rejected — even at dim=32 with 4x the feature space, removing the
   slow channel still helps.

### Honest conclusion

On QA1 (single supporting fact, no distractors, no interference), the slow
memory channel (γ=0.1) is a liability, not an asset. This is a task-property
result, not a dimension-property result. The D2C slow-channel advantage
requires:

- **Multiple competing facts** (QA2/QA3) where the slow channel's retention
  outlasts fast-channel interference
- **Distractor load** (as in D3-R synthetic tasks) where the slow channel
  filters noise
- **Longer gaps** between encoding and retrieval

### Updated next action

1. **Priority:** Run ablation at dim=16 on **QA2** (two supporting facts) and
   **QA3** (three supporting facts). These are the tasks where the slow
   channel's gap-retention advantage should manifest.
2. If QA2/QA3 show full >= no_slow, the E0 claim is: "slow-channel advantage
   scales with reasoning chain length."
3. If QA2/QA3 still show no_slow >= full, the readout architecture needs
   revision (the current outer-product gated features may not expose the
   slow channel's signal to the classifier).

---

## 2026-09-21 — Readout correction and dimension-scaling gap analysis

### Dataset and execution setup

The documented `Muennighoff/babi` en-valid mirror is now downloaded locally under
`code/python/d2c/progress/babi_data/`:

| Split | Stories | File |
|---|---:|---|
| train | 18,013 | `babi_train.jsonl` |
| test | 20,000 | `babi_test.jsonl` |
| valid | 1,987 | `babi_valid.jsonl` |

All 20 tasks and the expected record counts passed validation. The bAbI unit suite
passes with `PYTHONPATH=code/python python3 -m pytest -q code/python/tests/test_babi.py`
(15 passed).

### Readout diagnostic

The original E0 sweep used `hidden_dim=32`, `learning_rate=0.05`, and five epochs.
This learning rate was too aggressive for the dependency-free online MLP. At the
corrected setting `learning_rate=0.005`, `hidden_dim=32`, and ten epochs, the 16-D
QA1 result improved substantially:

| Variant | Original 20-seed sweep | Corrected 5-seed confirmation |
|---|---:|---:|
| `full` | 18.23% | 41.40% |
| `no_slow` | 23.37% | 44.26% |

The corrected matched paired gap was `no_slow - full = +2.86` percentage points,
positive on all five seeds, with approximate 95% CI `[+1.78, +3.94]` points.
The optimizer setting therefore explained most, but not all, of the original gap.

### 32-D preserved gap sweep

A 32-D sweep used the corrected settings: `hidden_dim=32`, learning rate `0.005`,
ten epochs, 900 train stories, 1,000 test stories, and 20 seeds. The process was
intentionally terminated after enough results had accumulated; no final JSON bundle
was written, but all reported terminal rows were preserved and summarized.

| Variant | Completed seeds | Mean accuracy | Approx. 95% CI |
|---|---:|---:|---:|
| `full` | 20 | 43.39% | +/-0.68 pp |
| `no_slow` | 20 | 47.62% | +/-0.61 pp |
| `collapsed_gamma` | 10 | 46.59% | +/-1.19 pp |

Paired comparisons on completed matched seeds:

- `no_slow - full`: **+4.23 pp**, positive on 20/20 seeds; approximate 95% CI
  `[+3.47, +4.99]` pp.
- `collapsed_gamma - full`: **+3.22 pp**, positive on 10/10 seeds.
- `collapsed_gamma - no_slow`: **-0.69 pp** over the first 10 matched seeds.

The observed ordering remains:

```text
no_slow > collapsed_gamma > full
```

### Dimension interpretation

At the corrected readout settings, the available comparison is:

| Dimension | `full` | `no_slow` | Gap (`no_slow - full`) |
|---:|---:|---:|---:|
| 16, 5 seeds | 41.40% | 44.26% | +2.86 pp |
| 32, 20 seeds | 43.39% | 47.62% | +4.23 pp |

Increasing total dimension from 16 to 32 improved both variants, but improved
`no_slow` more strongly. These results do **not** support the hypothesis that the
slow channel begins contributing positively at 32 dimensions. They suggest instead
that the current representation/readout preferentially exploits the shorter
channels, while the distinct slow timescale is either attenuated or introduces
interference.

This is not evidence that slow memory is intrinsically harmful. Remaining confounds
include the bag-of-words query forcing, the large query-gated feature vector, online
single-story SGD, and unequal seed counts across the corrected 16-D and preserved
32-D comparisons. The 32-D `collapsed_gamma` run was also stopped at 10 seeds.

### Operational lesson and next action

Use a quick gap sweep before any full sweep:

- quick: 3 seeds, five epochs, learning rate `0.005`, all variants, candidate
  dimensions 16/32/64/128;
- full: 20 seeds and ten epochs only after a quick sweep identifies a meaningful
  dimension-dependent pattern.

The next diagnostic should compare feature pathways at selected dimensions:
`u` only, weighted memory only, `[u,memory]`, individual `chi` channels, and the
current query-gated features. This will distinguish slow-channel information loss
from slow-channel interference in the readout.

### Reproducibility files

- `code/python/d2c/progress/validate_babi_setup.py` — dataset validator
- `diagnose_babi_readout.py` — cached readout diagnostic
- `run_babi_sweep16.sh` — original 16-D sweep runner
- `run_babi_gap_sweep32.sh` — corrected 32-D sweep runner
- `babi_dimension_gap_analysis.md` — analysis report
- `analyze_32d_preserved.py` — preserved 32-D result analysis

---

## 2026-09-21 — Numpy readout + LinearReadout implementation; E0 dim=64 smoke

### What was built

Numpy-ized the entire readout pipeline in `readout.py`:

| Component | Before (pure Python) | After (numpy) | Speedup |
|---|---|---|---|
| `query_conditioned_features()` | List comprehensions, O(dim²) in Python | `np.outer`, `np.concatenate` | ~50-100x |
| `QueryConditionedMLPReadout._forward()` | Nested list comprehensions | `W @ x + b`, `np.maximum` | ~26-51x |
| `QueryConditionedMLPReadout.train_step()` | Nested list comprehensions backward | `np.outer`, vectorized SGD | ~26-44x |

New class: **`LinearReadout`** — single-layer `features @ W → softmax → argmax`,
numpy throughout. ~4-5x faster than the numpy MLP, directly interpretable.

### Files modified

| File | Change |
|---|---|
| `readout.py` | Numpy-ized MLP weights (np.ndarray), forward, backward, softmax; added LinearReadout |
| `babi_qa.py` | Added `--readout linear|mlp` flag, default=linear |
| `__init__.py` (digital) | Added LinearReadout export |

### Test results

**35/35 passed** — all existing D3, D4A, temporal_logic experiments (20 tests)
+ all bAbI tests (15 tests). Zero regressions.

### Benchmark: numpy MLP train_step

| dim | pure-Python (before) | numpy (after) | speedup |
|---|---|---|---|
| 16 | 12.7 ms | 0.50 ms | **26x** |
| 32 | 29.9 ms | 0.58 ms | **51x** |
| 64 | 114.8 ms | 2.62 ms | **44x** |

### Benchmark: LinearReadout train_step

| dim | time | 900 stories × 5 epochs × 20 seeds |
|---|---|---|
| 16 | 0.117 ms | **10.5s** |
| 32 | 0.386 ms | **34.7s** |
| 64 | 0.642 ms | **57.8s** |
| 150 | 2.928 ms | **263.5s** (4.4 min) |

### E0 dim=64 smoke results (linear readout, 900 train, 1000 test, 3 seeds)

| Variant | Mean Accuracy | Seeds |
|---|---|---|
| full | 0.405 ± 0.019 | 434, 358, 423 |
| no_slow | 0.434 ± 0.017 | 464, 395, 442 |
| collapsed_gamma | 0.398 ± 0.032 | 472, 341, 380 |

### Interpretation

Accuracy jumped from ~16% (dim=16, pure-Python MLP) to ~40% (dim=64, linear
readout). The linear readout at dim=64 is the first configuration that shows
meaningful learning on bAbI QA1.

The ablation pattern (no_slow > full) remains inverted from the expected. This
is consistent with the user's own 32-D gap analysis (no_slow > collapsed_gamma
> full) documented in the 2026-09-21 entry. The slow channel's interference
filtering advantage requires distractors to manifest — bAbI QA1 has none.

### Next action

1. Run the feature pathway diagnostic (compare `u` only, memory only,
   `[u,memory]`, individual `chi` channels, gated features) to isolate
   whether the slow channel is attenuated or interfering.
2. Run QA2 and QA3 (two/three supporting facts) where interleaving events
   create interference — the slow channel's advantage should emerge there.
3. Use the quick-sweep protocol (3 seeds, 5 epochs, lr=0.005) before any
   full 20-seed sweep.

---

## 2026-11-03 — E-F0: Runtime-integrated digital reasoning implementation

### Scope delivered on the F0 feature branch

- Added a digital runtime coordinator around the existing Python SOE stepper.
- Added WARMUP, EXPLORE, CONSOLIDATE, EVALUATE, PRUNE and GROW scheduling.
- Added bounded circular trace storage with session/continuity identifiers and
  contiguous-window sampling that cannot cross memory resets.
- Added stability checks distinguishing input-driven Form A and negative-feedback
  Form B. The Form-B analytical margin is not applied to the input-driven bAbI kernel.
- Added runtime summary and versioned state/phase checkpoint restore.
- Added a bAbI F0 runner with isolated and continuous modes. The original
  babi_qa.py remains the comparison baseline.
- Added dataset-independent runtime, trace-store and bAbI integration tests.
- Expanded CI to install NumPy explicitly, set PYTHONPATH, run CTest, Python and Julia.

### Experimental controls

- Isolated mode resets episodic state at story boundaries while runtime phase
  history and counters persist.
- Continuous mode carries episodic state between stories and is a separate stress test.
- F0 keeps the SOE kernel fixed and retains supervised readout learning.
  F1-F5 learning mechanisms are not silently activated.
- PRUNE/GROW are proposal phases only; channel-count mutations remain F4.

### Verification status

Local source-tree execution was unavailable in this session because the sandbox
could not resolve github.com for a git clone. The feature branch must therefore
be validated by the configured GitHub Actions PR run. No CI result is claimed
until a workflow run is observed.

### Next decisions

1. Require a green C++/Python/Julia CI run before merging.
2. Run matched E0-vs-F0 bAbI comparisons on QA1 at dimensions 64 and 150,
   preserving identical seeds, readout, learning rate and train/test ordering.
3. Progress to QA2/QA3 only after numerical equivalence and matched QA1 runs.
4. Keep persistent cross-story tests separate from independent-story accuracy claims.

### E-F0 reproducibility addition

The repository does not track the bAbI dataset files. Added
`code/python/d2c/digital/download_babi.py` to download and validate the
train/test/valid JSONL splits from the recorded Hugging Face mirror. The
downloader uses temporary files and atomic replacement; tests mock the network
so CI does not depend on the external dataset host.

### Verification handoff — 2026-11-03

- The feature branch is maintained in PR #1; the original E0 runner remains in
  place and the F0 runner is under `code/python/d2c/experiments/`.
- The sandbox could not clone the repository because it could not resolve
  `github.com`; no local pytest result is claimed.
- GitHub's workflow-run and commit-status endpoints returned zero runs/checks
  for the latest PR head despite subsequent PR synchronisations. The workflow
  definition is present and includes `pull_request` plus `workflow_dispatch`.
  A repository-admin check of Actions enablement is required before the PR can
  be treated as CI-validated.
- Do not merge on the basis of these code additions alone. Require the full
  C++/Python/Julia workflow to complete successfully first.


---

## 2026-11-03 — QA2 (Two Supporting Facts) selected; Experiment Lab updated

### Why QA2

The next workload is bAbI Task 2, Two Supporting Facts. Unlike QA1's
single-supporting-fact retrieval, QA2 requires linking two relevant facts in
a story that can contain intervening statements. It is the next natural
reasoning step before QA3's three supporting facts.

### Experiment protocol

- Primary comparison: E0 baseline vs E-F0 runtime-integrated.
- Task: QA2; full SOE kernel; dimension 64; seeds 0, 1, 2; five epochs;
  linear readout; isolated story protocol.
- E-F0 deterministic spine audit enabled for every seed, separate from
  scored QA2 episodes.
- Continuous mode remains a separate stress test, not mixed into the
  independent-story accuracy result.
- No runtime ablation and no F1-F5 activation in this pass.

### Experiment Lab updates

- Task selector now names QA1, QA2, QA3, QA5, QA6, QA11 and QA14.
- QA2 is the default selected task; E-F0 is the default experiment.
- F0 spine audit control is exposed in the UI and passed safely through the
  allow-listed local API.
- Results explorer now shows per-seed six-phase coverage, kernel-preservation,
  trace-window, checkpoint-restore and stability evidence.
- GitHub benchmark workflow default changed to task 2 and artifact name
  identifies QA2 results.

### Interpretation gates

1. Verify matching E0/E-F0 task, splits, dimensions, seeds, readout and epochs.
2. Verify every F0 audit passes independently of QA2 accuracy.
3. Report accuracy and uncertainty, per-seed outcomes, class counts and
   runtime/stability diagnostics.
4. Inspect failure examples for broken two-fact chains, distractor interference,
   entity binding and answer-class imbalance.
5. Do not interpret a component audit pass as evidence of a reasoning gain.
   Runtime ablation remains a later stage.


---

## 2026-11-10 — Consolidated journey: E0 / E-F0 through G4.2

This entry records the progression in experimental order. The date is the
documentation/update date; milestone dates below refer to the corresponding
implementation or CI activity where known. Distinguish implementation/CI
success from a scientific performance improvement: a green workflow validates
the declared execution and evidence gates, not an accuracy claim by itself.

### 1. E0 — bAbI bridge and supervised QA (September 2026)

- Integrated the bAbI dataset, story parser, episode builder, deterministic
  token forcing, SOE memory and supervised query-conditioned readout.
- The first QA1 smoke tests established a working end-to-end real-data
  pipeline, but accuracy was far below the original >80% target. Early
  dim-16 results were roughly 16–22%, and removing the slow channel sometimes
  improved QA1. These were recorded as honest pipeline/readout smoke results,
  not as a successful reasoning gate.
- The convergence and dimension sweeps showed that learning rate and
  representation/readout choices mattered more than simply increasing hidden
  width or epoch count. At dim 16, changing the learning rate from 0.05 to
  0.02 raised the full-kernel result from about 19.9% to 33.4%; dim 32 gave a
  further modest lift but did not reverse the QA1 ablation ordering.
- The result scoped the slow-channel hypothesis: a slow timescale is not
  automatically useful on clean single-fact QA1; its proposed advantage is
  expected under multi-fact chains, distractors and interference.
- Identified pure-Python readout computation as the practical bottleneck,
  rather than the SOE memory stepper.

### 2. E0 readout and execution improvements (September 2026)

- Added NumPy-backed MLP operations and a NumPy `LinearReadout`, preserving
  the experiment interface while making larger dimensions and seed sweeps
  practical. The recorded train-step speedups for the NumPy MLP were
  approximately 26–51x over the original pure-Python path; the linear
  readout was faster still.
- The dim-64 QA1 smoke reached approximately 40.5% for the full kernel,
  43.4% for no-slow and 39.8% for collapsed-gamma (three seeds). This was a
  material pipeline/readout improvement, but no-slow still led on QA1.
- The key interpretation remained task-specific: these results did not
  establish that the slow channel was intrinsically harmful, nor that the
  E0 >80% target had been met.

### 3. E-F0 — runtime-integrated reasoning (3 November 2026)

- Added the digital runtime coordinator around the existing SOE numerical
  stepper, with WARMUP, EXPLORE, CONSOLIDATE, EVALUATE, PRUNE and GROW phases;
  bounded continuity-aware trace storage; checkpoint/restore; deterministic
  continuation; and formulation-aware stability monitoring.
- Added isolated and continuous bAbI execution modes. Isolated mode remained
  the primary independent-story benchmark; continuous mode was kept as a
  separate cross-story interference stress test.
- PRUNE/GROW were initially observable, non-mutating proposal phases. Actual
  channel-count mutation was deliberately deferred to later adaptive work.
  F0 did not silently activate F1–F5 learning mechanisms.
- Preserved the original E0 runner as a baseline. Added dataset download and
  validation tooling and expanded C++/Python CI. Runtime audit success was
  treated separately from reasoning accuracy.

### 4. QA2 selection and G0 — diagnostics, readout and reasoning probes
(November 2026)

- Moved from QA1 to QA2 (two supporting facts) to examine relational
  composition and entity binding. The matched E0/E-F0 QA2 baseline at
  dimension 64, full kernel, linear readout, five epochs and three seeds
  recorded 24.4% mean accuracy for both arms (seed scores 22.3%, 28.0%,
  22.9%). F0 therefore qualified the runtime path but did not itself improve
  the supervised prediction result.
- G0 expanded E0/E-F0 result parity: per-example questions, statements,
  targets, predictions, full probability vectors, confidence, margins,
  answer distributions, calibration, failure categories and runtime evidence.
- Added a staged readout/feature-scale investigation and QA2 oracle ladder:
  token baseline, event segmentation, entity-role binding, temporal
  relational-state replay and gold-state answer ceiling. Counterfactual tests
  cover renaming, irrelevant insertions, supporting-fact removal and temporal
  updates.
- The G2 reproducibility diagnostic later pinned the numerical environment
  (Python/NumPy/BLAS dispatch controls). Two independent jobs then matched
  exactly across 120 epoch-level records, with identical trained weights,
  validation logits and predictions. This resolved the observed cross-job
  drift under the combined controls; the individual contribution of each
  environment pin was not isolated.
- On the five-seed, three-epoch validation diagnostic, feature scale 10
  materially outperformed scale 100 on confidence quality. QA3 accuracy was
  28.0% at scale 10 versus 22.8% at scale 100; QA5 was 63.8% versus 61.2%.
  QA3 remained poorly calibrated (ECE-15 0.594 at scale 10), while scale 100
  produced severe overconfidence. Decision: retain scale 10 as the leading
  candidate, not a production-confirmed setting; use three epochs as the
  provisional checkpoint; investigate QA3 and confidence errors before any
  locked evaluation.

### 5. G1 — deterministic relational-state QA2 control (November 2026)

- Added an explicit event/state diagnostic for person locations, object
  holders and object locations, including move, pickup, release and temporal
  updates.
- The recorded QA2 report achieved 1,000/1,000 answers (100%), with 100%
  parse coverage across 15,426 statements, 100% query resolution and zero
  F0 stability violations.
- Claim boundary: G1 is an inspectable symbolic parser/world-state control,
  not a learned-binding result and not directly comparable to trained E0 or
  E-F0 accuracy. Its value is to expose the semantic structure and provide
  an oracle-like diagnostic reference.

### 6. G1-L — learned relational binding pilot and staged confirmation
(November 2026)

- Implemented a distinct QA2 learned binder over deterministic typed-event
  inputs: learned identity embeddings, role-specific relation encoders,
  object-holder representations, soft holder-to-person retrieval and a
  supervised location readout.
- Combined L0 controlled binding sanity checks, L1 one-seed pilot and gated
  L2 multi-seed confirmation in one workflow. The F0 spine audit was retained
  as a separate companion qualification, not as an input to the G1-L learner.
- The parser is restricted to supported QA2 templates. This experiment does
  not establish a general natural-language parser or a learned binder for
  QA3/QA5. Preserve the stage-specific run artifact as the source of truth
  for exact pilot/confirmation metrics; do not infer an L2 pass from workflow
  completion alone.

### 7. G2 — relational composition and confidence qualification
(November 2026)

- Extended the learned benchmark to QA3 and QA5 with matched E0/E-F0
  controls, five seeds by default, per-example evidence, confidence and
  calibration metrics, and 12 structured relational fixtures.
- The fixtures cover multi-fact composition, holder-dependent locations,
  object transfer, release, temporal updates, multiple entities, distractors,
  explicit coreference aliases and insufficient evidence. They are semantic
  controls, not natural-language model accuracy.
- G2 formalized paired seed effects, uncertainty, NLL, Brier, fixed/adaptive
  and classwise calibration, overconfidence and risk/coverage reporting.
  QA3 remained the difficult task; QA5 was materially stronger. The
  reproducibility diagnostic above showed that scale 10 was much healthier
  than scale 100 for confidence, without establishing a production setting.
- A validation-only readout follow-up retained scale 10 as the leading
  candidate and three epochs as the provisional checkpoint. QA3
  reproducibility and confidence errors remained open questions. The
  held-out split was not to be used for tuning.

### 8. G3 — full bottom-up/top-down hierarchy versus the G2 E-F0 spine
(7–9 November 2026)

- Used the existing G2 E-F0 runner directly as the flat/vectorized control;
  compared it with the full two-level scalar-ABI hierarchy. The protocol
  retained QA3 and QA5, matched seeds, story order, readout initialization,
  feature scale 10 and the three-epoch checkpoint.
- Added explicit per-level coupling, kernel, state and stability evidence;
  stronger seed/example pairing; F0 spine qualification; and Lab reporting.
  Runtime ablation and unrelated static-strengthening arms remained deferred.
- G3-N investigated top-down positive-kernel normalization. The first
  normalization attempt exposed a fast-level stability regression rather
  than being accepted as a performance fix. The workflow was corrected to
  report mechanism execution and stability qualification separately, and
  the evidence was retained rather than masking the failure.
- Interpretation boundary: G3 compares two complete execution packages
  (flat E-F0 vectorized runtime versus scalar-ABI hierarchy); it does not
  isolate hierarchy as the only causal difference. No hierarchy-only
  performance claim should be made from that package comparison.

### 9. G4 — vectorized hierarchy implementation (9 November 2026)

- Added contiguous vector-valued memory at both hierarchy levels and a
  vectorized native digital-memory ABI/stepper. Bottom-up forcing and
  top-down modulation were computed from previous-event states, with each
  level advanced once per event.
- This changed the representation and feedback formulation relative to
  G3's scalar-chi hierarchy; it was therefore treated as an implementation
  package comparison, not bitwise numerical parity.
- The initial G4 top-down unit-sum normalization raised the fast-level
  stability ratio to approximately 2.567, above the 0.9 margin. The result
  identified a kernel-mass/stability interaction and was not accepted as a
  stable hierarchy configuration.

### 10. G4.1 — mass-preserving fast-kernel correction (10 November 2026)

- Diagnosed that normalizing the calibrated fast-kernel weights to unit sum
  changed the kernel mass from 0.3128 and amplified the stability ratio.
- Corrected the top-down path to preserve the calibrated positive-kernel
  weights and their mass while retaining top-down decay modulation. The
  legacy unit-sum mode remained available for regression testing.
- The three-seed QA3/QA5 validation workflow passed its implementation and
  stability gates, with active levels required to remain at or below 0.9.
  This was a targeted stability correction, not evidence of an accuracy
  improvement or a final locked evaluation.

### 11. G4.2 — corrected vector coupling, concurrent H-C-D sweep and
adaptive follow-up (10 November 2026)

**Historical pilot**

- The earlier G4.2 pilot is superseded. Its scalar-context/bottom-level
  readout and depth-only selection did not test the agreed capacity and
  dimension factors or establish meaningful multi-level readout
  contribution. Preserve its artifacts as historical evidence; do not use
  it to claim a hierarchy-depth result.

**Corrected coupling and static matrix**

- Reused the G2 E-F0 spine and G4.1 mass-preserving kernel definitions;
  no C++ ABI redesign was introduced in this correction.
- Replaced scalar context with bounded vector-valued bottom-up and top-down
  messages derived from each level's structured state, including memory
  channels. All messages use previous-event states and bounded nonlinear
  clipping. The multi-level readout concatenates per-level memory and
  query-gated features; the bare query is supplied only at level zero.
- Coupling intervention tests verify that upper-level perturbations affect
  lower-level computation, disabling top-down changes lower-level output,
  and disabling bottom-up changes upper-level representation. Finite-state,
  positive-kernel, mass-preservation and <=0.9 stability gates are enforced.
- Launched all 18 H x C x D configurations as independent concurrent jobs:
  H={2,3,4}, C={3,4,6}, D={64,128}; three matched seeds and three epochs;
  QA3 primary, QA5 secondary. Total channel budget is matched by C across
  depths. The selector waited for all 18 jobs before selecting.

**Static findings (three-seed validation screening only)**

- The corrected workflow completed successfully: coupling validation,
  all 18 static jobs, selection and adaptive follow-up succeeded (21/21
  jobs in the final run).
- Aggregate mean QA3 accuracy by depth was H2 18.56%, H3 18.00% and H4
  18.78%. Aggregate QA5 was H2 60.28%, H3 58.94% and H4 56.33%.
  QA3 calibration worsened with depth (mean fixed-bin ECE approximately
  0.289, 0.359 and 0.387 respectively); recorded compute also rose
  (approximately 14.0, 16.3 and 22.4 minutes).
- The pre-registered selector chose H4-C6-D64: QA3 19.67%, QA5 56.67%,
  QA3 ECE approximately 0.3804. This is the QA3-first selector winner,
  not a demonstrated general best architecture.
- H2-C4-D64 is a strong efficient alternative (QA3 19.00%, QA5 61.00%,
  ECE approximately 0.2867, about 8.1 minutes). H2-C6-D64 had the best
  QA5 result (61.33%). Preserve these alternatives in the Lab register;
  do not silently replace the official selector winner.
- Increasing D from 64 to 128 did not produce a useful average gain and
  substantially increased compute. Increasing C yielded only modest
  average accuracy gains relative to its cost. Three seeds and three
  epochs make this a screening result, not a confirmatory ranking.

**Adaptive follow-up findings**

- The matched adaptive follow-up ran on H4-C6-D64. It preserved kernel
  mass and stability, but generated 24 births and zero deaths across the
  two tasks, three seeds and four levels: each level grew by one channel,
  taking the per-trial allocation from [10,7,7,6] to [11,8,8,7].
- Adaptive versus static accuracy was effectively neutral: QA3 19.67%
  static versus 19.33% adaptive; QA5 56.67% for both. QA3 ECE was 0.3804
  versus 0.3837; QA5 ECE 0.1397 versus 0.1366. Combined recorded compute
  increased from about 985.3s to 1,043.8s.
- The current 5% growth threshold is mathematically too permissive for
  these channel counts: for non-negative activity over N active channels,
  the maximum channel's share is at least 1/N (10–16.7% for 6–10
  channels). Thus the threshold is automatically met for non-zero
  activity unless pruning has already happened. The real campaign did
  not demonstrate a channel death. The adaptive mechanism is therefore
  mechanically exercised for growth, but prune/grow as a balanced
  selection policy is not validated.
- Next adaptive work must replace the absolute 5% criterion with a
  scale-aware/relative criterion, record prune-candidate ratios and
  rejection reasons (including mass/stability feasibility), and include
  controlled tests that exercise both actual death and birth. Compare
  against the matched static control before claiming a learning benefit.

### Current research conclusions and handoff

1. The E0-to-G4.2 path has moved from a token/SOE baseline to a runtime-
   integrated and instrumented benchmark, then to explicit relational
   controls, reproducible confidence diagnostics, and a stable vector-
   coupled multi-level hierarchy.
2. The strongest verified G4.2 result is architectural/mechanical:
   bounded bidirectional vector coupling and mass/stability-preserving
   multi-level execution are validated. More hierarchy depth has not shown
   a general QA3/QA5 advantage.
3. H4-C6-D64 remains the official QA3-first static selector winner; retain
   H2-C4-D64 and H2-C6-D64 as important alternatives for a better-powered
   confirmation.
4. Adaptive growth executed safely but did not improve prediction accuracy;
   pruning was not demonstrated. Correct the activity policy and test both
   birth and death before calling adaptive hierarchy validated.
5. Do not open a previously consumed/locked test split for tuning. Future
   confirmation must use a clearly reserved evaluation split and more
   independent seeds.
6. The G4.2 corrected CI run is
   https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/42862659473.
   Its artifacts contain 18 static bundles, the aggregate campaign register,
   and the matched adaptive comparison.

### Branch and integration handoff (10 November 2026)

- G4.2 head before this documentation update:
  `496f997e597ac0bb8c1e9f6293313aaf1a6fdfc4`.
- The G4.2 branch is 25 commits ahead of
  `f/vectorized-cpp-digital-abi`, with no commits behind. The latter is
  itself 77 commits ahead of `main` and is the head branch of open draft
  PR #8. G4.2 is therefore a stacked branch, not an independent
  main-targeted change set.
- Keep PR #8's vectorized C++ ABI work separate. The G4.2 PR should target
  `f/vectorized-cpp-digital-abi` first; after PR #8 is merged, retarget the
  G4.2 PR to `main`. Do not merge unrelated work or force-update the
  underlying branch.
- The final G4.2 corrected workflow is green. Re-run lightweight C++/Python
  integration CI for the documentation commit/PR where available; the
  full 18-cell campaign need not be repeated solely for a documentation
  change.
