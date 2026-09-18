# D2C Milestones E+F: From Real-Data Bridge to Self-Learning Agent

Status: active roadmap; Milestone D substrate validated (2026-09-15)
Scope: Python director first, C++ core changes only where the existing ABI cannot support a mechanism
Primary aim: (Phase E) prove the validated memory substrate works on real structured-reasoning data, then (Phase F) build the autonomy stack specified in D2C.md SS5.5-SS9 on top of that proven foundation.

---

## 1. Strategic Context

Milestone D proved the substrate on synthetic tasks:

- **Memory block**: stable SOE kernels, proven dynamics, all three formulations
- **Stability**: margin contract enforced everywhere; no violations under adaptation (E1-R)
- **Three-factor rule**: signed Hebbian + error-modulated decay passes the mechanism gate
- **Hierarchy**: live bidirectional stepping via the D4B ABI; top-down modulation functional
- **Diagnostics**: D_eff, M_cap, H_mem, spectral units at every step -- the strongest component
- **Formation capacity**: self-formed bindings work without interference (0.55 above chance)
- **Retrieval capacity**: slow-channel retention is load-bearing under interference (D4A: full beats no_slow +0.047, t=3.81, 20 seeds)

What D did **not** prove is that the system works on real data or that it can *learn by itself*.

**Phase E** bridges from synthetic to real: apply the validated substrate to a published reasoning benchmark (bAbI).

**Phase F** builds the autonomy stack: from validated substrate to self-learning memory system.

---

## 2. Non-Negotiable Principles

1. **Every stage must be validated before the next begins.** Pre-declared gates, bounded claims.
2. **The stability contract is never relaxed.** Margin <= 0.9 at every step.
3. **Self-supervision complements task supervision.** Transition measured by performance without labels.
4. **The memory channels remain the identity layer.** Spectral diagnostics remain valid.
5. **New modules follow the same testing policy as D.** Unit, deterministic, smoke, quick-mode tests.
6. **Real data before autonomy.** Phase E completes before Phase F begins.

---

## 3. Current Repo Capabilities

| Component | Status | E+F build on |
|---|---|---|
| SOE kernel + stability contract | Complete | Foundation for all stages |
| Spectral diagnostics | Complete | Health monitoring |
| DigitalDirector + DigitalEpisode | Complete | Episode runner (E0), runtime spine (F0) |
| Token vocabulary + deterministic codes | Complete | bAbI vocabulary (E0) |
| TokenForcingBridge | Complete | Token-to-forcing for bAbI (E0) |
| Hierarchy stepping (D4B ABI) | Complete | Hierarchy predictive coding (F2) |
| Three-factor rule (E1-R) | Mechanism-validated | HJB credit assignment (F1) |
| Consolidation helpers | Exists | CONSOLIDATE phase (F0) |
| Per-channel TD errors | Exists | HJB per-channel critics (F1) |
| QueryConditionedMLPReadout | Complete | bAbI answer classification (E0) |
| TraceStore (DigitalTrace) | Exists | Contiguous-window sampling (F0) |

---

## 4. Dataset Selection and Justification

### Why bAbI

Facebook's bAbI dataset (Weston et al., 2015) is the right first real-data benchmark:

1. **Small vocabulary (~150 tokens).** Fits existing Vocabulary and TokenForcingBridge.
2. **Tasks decompose by reasoning type.** 20 tasks let us make targeted claims about what D2C can and cannot do.
3. **Well-studied baselines.** Memory Networks solved 19/20 at 10k; LSTMs solve ~12/20.

### Datasets considered and rejected

| Dataset | Why not |
|---|---|
| Children's Book Test | Vocab ~50k; needs NLP pipeline |
| SimpleQuestions | Needs Freebase KB; vocab ~50k |
| WikiQA | Needs passage retrieval; vocab ~10k+ |
| Custom synthetic | Weakens "works on real data" claim |

### bAbI Task Tiers for D2C

**Tier 1 -- Core memory retrieval (expect strong):**
- QA1: Single Supporting Fact
- QA2: Two Supporting Facts
- QA3: Three Supporting Facts
- QA5: Three Argument Relations
- QA6: Yes/No Questions
- QA11: Basic Coreference

**Tier 2 -- Structured memory (expect partial):**
- QA4: Two Argument Relations
- QA7: Counting
- QA8: Lists/Sets
- QA14: Time Reasoning
- QA16: Basic Induction

**Tier 3 -- Compositional reasoning (expect failure):**
- QA9-QA13: Negation, indefinite knowledge, conjunction, compound coreference, deduction
- QA17-QA20: Positional, size, path, motivation reasoning

### Target task ordering

1. **QA1** -- simplest memory retrieval; proves the bridge works
2. **QA2** -- chain two facts with interleaving
3. **QA3** -- longer chains; multi-gap retention
4. **QA5** -- entity tracking under load
5. **QA11** -- pronoun resolution
6. **QA14** -- temporal ordering

Tasks 17-20 serve as negative controls.

---

## 5. Phase E: Real-Data Bridge

### E0: bAbI Integration and Supervised QA

**What to build:**
- code/python/d2c/digital/babi.py: task loader, story parser, episode builder
- code/python/d2c/experiments/babi_qa.py: experiment entrypoint
- code/python/tests/test_babi.py: tests

**Architecture:** Each bAbI story is statements followed by a question. Statements become forcing steps through TokenForcingBridge. The question becomes the query for QueryConditionedMLPReadout. Answer vocabulary is the set of unique answers for the task.

**Configurations:**
- State dimensions: 64 and 150 (run both, report comparison)
- Kernel: validated D1/D2 kernel (gamma=[2.0, 0.5, 0.1], w=[0.04, 0.08, 0.12])
- Variants: full / no_slow / collapsed_gamma
- Seeds: 20 per cell for claims, 3 for quick tests
- Training: supervised cross-entropy, SOE kernel frozen, readout trained

**Gate:** Full kernel achieves >80% on QA1 at both 64-dim and 150-dim. Ablations show expected gradient.

---

### E1: D_eff and Channel Specialisation Analysis

**Protocol:** Run QA1, QA2, QA3 at both dimensions. Record per-step D_eff, per-channel weight evolution, gamma distribution.

**Gate:**
- D_eff(QA3) > D_eff(QA2) > D_eff(QA1)
- Channel specialisation visible across tasks
- Slow channel role consistent with D1-D4A findings

---

### E2: Ablation Suite

**Protocol:** Full ablation across QA1, QA2, QA3, QA5, QA11, QA14. Variants: full / no_slow / collapsed_gamma. 20 seeds per cell. Both dimensions.

**Gate:** Full beats no_slow with paired CI (p < 0.05) on at least QA1-QA3.

---

### E3: Continuous Reading

**What to build:** A ContinuousDirector maintaining memory state across stories.

**Protocol:** Run QA1-QA3 with persistent state across 50 stories.

**Gate:** Accuracy matches per-story accuracy. At least one consolidation event. Stability ratio < 0.9 throughout.

---

## 6. Phase F: Autonomy Stack

### F0: Runtime Spine

**Reference:** D2C.md SS9.2-SS9.4

**What to build:** State machine: WARMUP -> EXPLORE -> CONSOLIDATE -> EVALUATE -> PRUNE -> GROW.

**Acceptance (bAbI):** Runs QA1-QA3 continuously for 10,000 steps, zero stability violations.

---

### F1: Per-Channel Value Learning (HJB Critics)

**Reference:** D2C.md SS6.2-SS6.4

**What to build:** L per-channel critic networks V_l. Replace activity-proxy values with learned critics.

**Acceptance (bAbI):** HJB critics improve held-out bAbI accuracy above frozen-kernel baseline.

---

### F2: Predictive Coding Heads

**Reference:** D2C.md SS5.5

**What to build:** Per-level prediction heads. Prediction error drives learning.

**Acceptance (bAbI):** Prediction error decreases on bAbI passages.

---

### F3: Composite Objective

**Reference:** D2C.md SS9.5

**What to build:** Composite loss with task, prediction, value, stability, and memory terms.

**Acceptance (bAbI):** Improves on bAbI AND prediction error decreases simultaneously.

---

### F4: Channel Birth and Death

**Reference:** D2C.md SS8.4

**What to build:** Prune negligible channels; grow when D_eff saturates at uncovered timescales.

**Acceptance (bAbI):** Channels grow for tasks requiring new timescales.

---

### F5: Timescale Adaptation

**Reference:** D2C.md SS8.2 (Eq 34)

**What to build:** dgamma/dt = -alpha*gamma*sign(chi*epsilon). alpha << eta. gamma_min=0.01, gamma_max=5.0.

**Acceptance (bAbI):** gamma values cluster near bAbI task timescales.

---

### F6: D5 Agent Sandbox (Capstone)

**Reference:** Roadmap SS7

**What to build:** Streaming environment, delayed reward, rule shifts, no episode boundaries.

**Acceptance (bAbI):** bAbI tasks solved in continuous-operation mode. D_eff and channel landscape evolve.

---

## 7. Proposed Module Layout

```
code/python/d2c/
  digital/
    babi.py                    # E0: bAbI task loader
  runtime/
    state_machine.py           # F0: state machine
    trace_store.py             # F0: trajectory memory
    scheduler.py               # F0: phase scheduling
    stability_monitor.py       # F0: StabilityMonitor
    continuous_director.py     # E3: persistent-state runner
  learning/
    critics.py                 # F1: value critics
    predictive.py              # F2: prediction heads
    composite_objective.py     # F3: composite loss
    birth_death.py             # F4: channel lifecycle
    gamma_adaptation.py        # F5: timescale adaptation
  experiments/
    babi_qa.py                 # E0: bAbI experiment
    f0_runtime_smoke.py        # F0: smoke test
    f1_hjb_probe.py            # F1: HJB probe
    f2_predictive_hierarchy.py # F2: predictive coding
    f3_composite_integration.py# F3: composite integration
    f5_agent_sandbox.py        # F6: agent sandbox
```

---

## 8. Implementation Order and Dependencies

```
Phase E:
  E0 -> E1 -> E2 -> E3
                    |
Phase F:            |
  F0 -------------------
  |
  +-- F1 --+
  |        +-- F3 -- F5
  +-- F2 --+
  |
  +-- F4 --+
  |
  +-- F6 <-- all F stages
```

---

## 9. What This Establishes If Successful

**Phase E:**
1. D2C works on real data (published benchmark)
2. Interpretability story is tangible (D_eff, channel specialisation)
3. Slow-channel finding generalises to real reasoning
4. Performance pattern is predictable (memory retrieval succeeds, compositional reasoning fails)

**Phase F:**
5. Self-supervised learning from prediction error and reward
6. Provable stability during continuous adaptation
7. Self-organising memory (birth/death, timescale adaptation)
8. The always-on agent
9. The complete D2C architecture

---

## 10. What We Are Not Doing

- Training a large language model
- Claiming transformer-scale superiority
- Building a full symbolic theorem prover
- Adding heavy ML dependencies
- Replacing the C++ core with Python dynamics
- Competing on bAbI leaderboards (our contribution is the mechanism story)

---

## 11. Testing Policy

For every new module: unit tests, deterministic tests, smoke tests, quick-mode tests.
For every experiment: quick-mode (N=1-3), CLI entrypoint test, report test, reproducibility test.
Full statistical runs: N=20 seeds, report CIs.

---

## 12. Context and Progress Ledger

**Active record file: `Step_Context.md`** — tracks all E+F stage progress,
decisions, verification results, and next actions. Created 2026-09-16.

`Context.md` retains all prior milestone history (D0 through D4A, through
2026-09-15). New entries for the E+F phase go in Step_Context.md.

Update Step_Context.md when:
- a new E or F module lands
- a new experiment runs end-to-end
- a hypothesis is validated, weakened, or rejected
- a major design decision changes the roadmap

Each entry: date, milestone title, changes, verification, why it matters, next steps.
