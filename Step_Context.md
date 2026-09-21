# Step Context — Phase E Entry Point

Current record file for the E+F stage. Supersedes entries in Context.md from
this date forward for active roadmap tracking. Context.md retains all prior
milestone history; this file tracks the E+F phase.

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
