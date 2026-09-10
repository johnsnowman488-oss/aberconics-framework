# D2C Milestone D: Digital Autonomy Technical Plan

Status: active roadmap; D0, D1/D1.2, and D2 are complete at single-level scope
Scope: Python D2C layer first, C ABI extensions only when the experiment path requires them
Primary aim: move D2C from physical-system demonstrations toward autonomous digital-space operation while preserving the architecture's core strengths: explicit multiscale memory, stability monitoring, interpretable diagnostics, and online credit assignment.

## 1. Strategic Pivot

The previous implementation path focused on completing the physical-modelling interpretation of D2C: Lorenz63, OU-like memory, hierarchy diagnostics, and stability-bound learning. That remains valuable, but it should no longer be the only path.

Milestone D reframes the next phase:

```text
Physical modelling substrate:
  continuous state + SOE memory + stability + hierarchy

Digital autonomy substrate:
  token/event streams + continuous memory + prediction + action + delayed feedback
```

The key claim is not that the current D2C implementation already reasons. It does not. The claim is that the current implementation contains the right low-level machinery for reasoning-like digital behavior:

- non-Markovian memory via SOE channels
- multiscale working memory via gamma-separated channels
- explicit memory load diagnostics via D_eff and related spectral units
- delayed credit assignment via per-channel traces and TD-style errors
- top-down context modulation in the hierarchy
- online consolidation from fast adaptation into slower memory

Milestone D must turn that architectural argument into experiments.

## 2. Non-Negotiable Principles

1. Do not claim digital reasoning from physical benchmarks.

   Lorenz63 and OU-style tasks validate memory/stability machinery, not symbolic reasoning.

2. Every demonstration must have a baseline and an ablation.

   At minimum, compare against a simple token-synchronous recurrent baseline and ablate slow channels, free-evolution intervals, or top-down modulation as appropriate.

3. Digital experiments must expose memory diagnostics.

   Accuracy alone is not enough. Every run should report D_eff, M_cap or equivalent memory-capacity indicators, channel usage by timescale, and stability ratios.

4. The first digital experiments should be synthetic and small.

   The goal is to isolate mechanism before scale. We are not trying to train a language model in this milestone.

5. Keep the ABI thin.

   Implement digital task orchestration in Python first. Add C ABI only when Python cannot access required state or step behavior.

6. Hierarchy must be used deliberately.

   The hierarchy modules are mature enough for run/report diagnostics, but not yet for streaming digital control from Python. Hierarchical digital experiments must wait for either a stateful hierarchy-step ABI or a carefully bounded batch-style experiment.

## 3. Current Repo Capabilities

The repo already supports:

- C++ ABERSOE single-level runtime with stateful step support.
- C ABI and Python wrapper for `gfe_c_abersoe_step`.
- Python `AberconicsDirector` for Lorenz63 online step loops.
- Python learning primitives:
  - next-state prediction helpers
  - per-channel discounts
  - TD-error helpers
  - three-factor update proposals
  - fast/slow consolidation
- Python trace windows for Lorenz memory traces.
- C++ hierarchy runtime:
  - `HierarchicalMINModel`
  - `HierarchicalState`
  - `step(...)`
  - `run(...)`
  - cross-level diagnostics
  - adjacent renormalisation analyses
- C ABI and Python wrappers for hierarchy run/report paths:
  - scenario runs
  - constrained chain-spec runs
  - cross-level reports
  - renorm reports
- Python typed hierarchy config objects and schemas:
  - `HierarchyExperimentConfig`
  - `HierarchyLevelConfig`
  - `HierarchyEdgeConfig`
  - `ChainLevelSpec`
  - `ChainEdgeSpec`
- Digital D2C substrate:
  - deterministic vocabulary, event stream, forcing, and trace schemas
  - Python-side stable SOE memory stepping with evolving and frozen silence
  - `DigitalDirector` for task-provided forcing episodes and event traces
  - explicit key/value outer-product bindings and a finite-window baseline
  - deterministic binding decoding and a trained query-conditioned MLP readout
  - symbolic retrieval and thinking-between-tokens experiment entrypoints

## 4. Current Gaps

These gaps define the implementation order.

### 4.1 Digital Stream Gap — closed for D0

`digital/tokens.py`, `streams.py`, `traces.py`, and `tasks.py` provide the
generic substrate. The symbolic retrieval experiment also has deterministic
held-out assignment splits. New tasks should reuse these types.

### 4.2 Token Bridge and Readout Gap — closed for D0/D1.2

`TokenForcingBridge`, `KeyValueBindingBridge`, forcing schedules, silence
controls, deterministic decoders, and `QueryConditionedMLPReadout` now cover
the initial scope. The MLP trains only readout weights over frozen state; the
outer-product binding writer remains an explicit control, not learned grounding.

### 4.3 Digital Director Gap — stepping closed; learning integration open

`DigitalDirector` owns state initialization, SOE stepping, and generic traces
for arbitrary task-provided `DigitalEpisode` schedules. Reward hooks, phase
scheduling, online updates, and consolidation are the D3 integration work.

### 4.4 Hierarchy Step Gap

C++ hierarchy has `step(...)`, but the C ABI exposes only hierarchy run/report paths. Python cannot yet run an arbitrary hierarchy one digital event at a time.

Required before serious hierarchical digital autonomy:

- `gfe_c_hierarchical_step_chain_spec(...)` or equivalent stateful hierarchy-step ABI
- Python binding for hierarchy state roundtrip
- active-kernel export after top-down modulation
- per-level state and spectral diagnostics at each step

### 4.5 Baseline Gap — minimum D1 baseline closed

`window_limited_lookup_prediction` is the finite-context baseline for D1/D2.
A token-synchronous recurrent baseline remains optional, but is useful for
D3/D5 comparisons and should not block delayed-feedback integration.

## 5. Proposed Module Layout

Add a digital subpackage rather than folding this into Lorenz modules.

```text
code/python/d2c/digital/
  __init__.py
  tokens.py
  streams.py
  bridge.py
  readout.py
  tasks.py
  metrics.py
  traces.py
  director.py
  baselines.py
  reports.py

code/python/d2c/experiments/
  symbolic_induction.py
  token_bridge.py
  temporal_logic.py
  hierarchy_digital_probe.py
```

Responsibilities:

- `tokens.py`
  - `Vocabulary`
  - token ids and labels
  - deterministic token code generation

- `streams.py`
  - timestamped token/event streams
  - regular and irregular event spacing
  - no-input interval representation

- `bridge.py`
  - token-to-forcing mapping
  - forcing pulse shapes
  - free-evolution gaps
  - deterministic and learned bridge variants

- `readout.py`
  - linear readout from `u`
  - readout from `chi`
  - combined state readout
  - nearest-code decoder for early experiments

- `tasks.py`
  - symbolic induction task
  - delayed-copy/retrieve task
  - temporal logic task
  - distribution-shift task

- `metrics.py`
  - next-token accuracy
  - accuracy vs gap length
  - retrieval latency
  - memory saturation
  - D_eff correlation with task complexity
  - stability violation count

- `traces.py`
  - generic digital trace records
  - generic fixed-window extraction
  - per-event `(token, t, u, chi, prediction, target, reward)` records

- `director.py`
  - digital episode loop
  - step scheduling
  - learning/evaluation phases
  - consolidation triggers

- `baselines.py`
  - table baseline
  - window-limited baseline
  - token-synchronous recurrent baseline

- `reports.py`
  - human-readable experiment summaries
  - JSON-serializable result bundles
  - diagnostic tables

## 6. Director Structure

The Digital Director should follow the shape already established by the Python D2C plan and the Lorenz63 `AberconicsDirector`, but the Lorenz-specific assumptions must move behind task adapters.

The desired structure is:

```text
Director
  owns runtime library access
  owns scheduler
  owns trace store
  owns current state
  owns learning/evaluation phase state
  delegates task-specific observation, reward, target, and report logic
```

Digital Director responsibilities:

- load or receive the C runtime library
- seed an initial continuous state from a token/event task
- step ABERSOE state through token pulses and silent intervals
- collect generic digital traces
- compute task prediction errors
- compute reward or feedback signals when available
- run phase transitions such as WARMUP, EXPLORE, CONSOLIDATE, and EVALUATE
- apply or propose stable memory updates
- trigger consolidation based on memory load, error trends, or task feedback
- emit progress messages and final reports

Task adapter responsibilities:

- generate token/event streams
- provide the next forcing vector
- define prediction target and action space
- score predictions or actions
- provide delayed reward/error where relevant
- define task-specific metrics

Trace store responsibilities:

- store event-level state records independent of Lorenz63
- retain `(t, token, forcing, u, chi, prediction, target, reward, phase)` where available
- expose fixed windows for prediction, TD analysis, and report generation
- serialize cleanly to JSON

Scheduler responsibilities:

- provide deterministic episode order
- manage seed lists
- manage ablation variants
- support quick-mode and statistical-mode runs

Design rule:

The Digital Director should not become a symbolic task implementation. It should be an orchestration layer. Symbolic induction, temporal logic, and digital sandbox behavior belong in task adapters and experiment modules.

Refactor rule:

Do not immediately break the Lorenz63 Director. First build the Digital Director beside it. After the interface stabilizes, extract shared pieces into a base Director or runtime helpers.

## 7. Milestone Sequence

### D0: Digital Substrate — complete

Goal:

Build the minimal digital runtime substrate without claiming reasoning.

Deliverables:

- `d2c/digital/tokens.py`
- `d2c/digital/streams.py`
- `d2c/digital/bridge.py`
- `d2c/digital/readout.py`
- `d2c/digital/traces.py`
- focused tests for token streams, forcing schedules, readout, and trace serialization

Acceptance criteria:

- can create a vocabulary and timestamped token stream
- can map tokens to forcing vectors
- can evolve ABERSOE state through forced and silent intervals
- can decode a next-token prediction from state
- can serialize a digital trace
- no Lorenz-specific imports in the digital package

Initial implementation choice:

- Use a Python-side forced-memory stepper that mirrors stable ABERSOE forms until the C ABI exposes arbitrary per-step forcing.
- Use deterministic token codes before learned embeddings.
- Use nearest-code or ridge-free linear readout before training a neural decoder.

### D1: Symbolic Retrieval over Long Gaps — complete, bounded claim

Goal:

Test whether explicit slow memory channels support symbolic retrieval across gaps.

Task:

```text
Training stream:
  A B ... gap ... A -> predict B

General form:
  key value filler... key -> value
```

Hypothesis:

Slow SOE channels improve retrieval accuracy as gap length grows.

Baselines:

- window-limited table baseline
- token-synchronous recurrent baseline
- D2C without slow channels
- D2C with fixed or collapsed gamma values

Metrics:

- accuracy vs gap length
- accuracy vs vocabulary size
- D_eff during first key, gap, second key, and prediction
- per-channel contribution to readout
- stability ratio and rescale count

Acceptance criteria:

- D2C beats the window-limited baseline beyond the baseline window.
- Removing slow channels degrades long-gap accuracy.
- D_eff increases with symbolic load or gap complexity in a measurable way.
- Stability remains within configured bounds.

### D2: Thinking Between Tokens — complete, bounded claim

Goal:

Test whether continuous free evolution between tokens improves prediction or memory preparation.

Task:

```text
token pulse -> silent interval -> token pulse -> silent interval -> prediction
```

Hypothesis:

Allowing latent state to evolve during no-input intervals produces better state organization than token-synchronous updates.

Result:

Free evolution does not improve passive retrieval in the fixed linear
substrate; it lowers confidence. It does encode elapsed interval in the timing
probe, while the duration-matched frozen control remains at chance. Separated
slow modes preserve retrieval through silence better than the no-slow and
collapsed-timescale ablations. This establishes interval representation and
retention, not productive latent reasoning.

Baselines:

- same D2C model with zero-length silent intervals
- token-synchronous recurrent baseline
- bridge ablation where `chi` is reset or ignored

Metrics:

- next-token accuracy
- retention after no-input interval
- readout confidence over silent time
- D_eff trajectory during silence
- distance between state and target token code

Acceptance criteria:

- nonzero thinking intervals improve or stabilize prediction on at least one delayed task
- the improvement disappears or shrinks when memory channels are removed
- silent intervals do not create stability violations

### D3: Temporal Logic and Delayed Inference — active; randomized causal task next

Goal:

Move from retrieval to rule-conditioned prediction/action.

Current finding and evaluation correction:

The initial four-case temporal-rule control demonstrated a narrow
full/no-slow memory and persistent-credit attribution separation, but its
nominal held-out episodes repeat the same four canonical streams as training.
Its `0.25`, `0.50`, and `1.00` scores are consequently quantized case counts,
not generalization evidence. The task-gated `1.00` result is an upper-bound
control because it supplies the relevant premise event explicitly.

Next task: randomized indexed rule-conditioned retrieval

```text
MODE: IDENTITY or INVERT
...
SLOT_3 = BIT_0
SLOT_7 = BIT_1
... variable-order, matched distractor events ...
QUERY: SLOT_7
-> ACTION_1 for IDENTITY; ACTION_0 for INVERT
```

Generate train and test episodes independently. Randomize assignments, event
order, target position, distractor identities/count, delay, and silence
timing, while stratifying the rule/action distribution. The first version may
retain an explicit binding writer as a bounded representation control; the
learning question is whether D2C can select and use the causally relevant
earlier event from delayed feedback.

Legacy task form:

```text
IF X occurred earlier AND Y occurs now THEN predict/action Z
```

Examples:

- if `OPEN` appeared earlier and `KEY` appears now, predict `UNLOCK`
- if `A` appeared before `B`, then choose `C`; otherwise choose `D`
- if a contradiction token appears, assign negative credit to the earlier premise token

Hypothesis:

SOE memory traces provide a natural eligibility substrate for delayed rule composition.

Implementation prerequisite:

Wire the existing per-channel TD-error, three-factor-update, and fast/slow
consolidation helpers into `DigitalDirector` episode phases. Begin with a
fixed kernel and explicit binding writer to isolate delayed credit assignment.

Baselines:

- token-synchronous recurrent baseline
- D2C without slow channels
- D2C without TD/credit signal

Metrics:

- delayed-rule accuracy
- per-rule, per-delay, and confidence-interval accuracy on independently
  generated held-out streams
- error attribution to earlier target events versus matched distractors
- TD-error by channel
- rule horizon at which performance collapses
- noise/jitter tolerance
- intervention sensitivity: changing the queried stored value must change the
  action, while changing an unqueried distractor must not

Acceptance criteria:

- Full D2C exceeds finite-window/no-gate and no-slow controls over longer
  delays on genuinely unseen episodes; report at least 20 seeds.
- TD/error traces identify the earlier queried event more strongly than
  matched irrelevant events, and intervention tests confirm that this feature
  is causally used for action selection.
- The task-gated result remains an upper bound, not evidence of learned
  eligibility. Do not advance adaptive-kernel or hierarchy claims unless the
  ungated learned path clears the above tests.

### D4: Hierarchical Digital Probe

Goal:

Use hierarchy diagnostics in a digital task without overclaiming full hierarchical autonomy.

This milestone has two tracks.

#### D4A: Batch Hierarchy Probe

Use existing hierarchy chain-spec run/report ABI.

Task:

- construct synthetic two-level chain specs that represent fast local events and slower context
- run predefined chain scenarios
- compare bottom-up-only, top-down-only, and bidirectional variants

Metrics:

- level-wise D_eff
- cross-level relation deltas
- kernel shift summaries
- renorm consistency
- prediction or retrieval score if attached externally

Acceptance criteria:

- reports show interpretable cross-level differences between ablations
- top-down modulation changes lower-level memory diagnostics in the expected direction
- renorm reports pass consistency checks or log actionable failures

#### D4B: Stateful Hierarchy Step ABI — complete

Add the missing bridge for real streaming hierarchy experiments.

Required C ABI:

```text
gfe_c_hierarchical_step_chain_spec(...)
```

Minimum capabilities:

- pass current state for each level
- pass or reconstruct active kernels
- apply bottom-up and top-down couplings
- return next state for each level
- return active kernel per level
- return spectral diagnostics per level

Python deliverables:

- raw wrapper in `gfe_ctypes.py`
- convenience wrapper in `d2c/ffi.py`
- typed digital/hierarchy state schema
- smoke tests mirroring current ABERSOE step tests

Acceptance criteria:

- Python can step a two-level chain one event at a time
- per-level state and active kernel round-trip correctly
- existing hierarchy run/report tests remain green

Implementation note (2026-09-08): All deliverables landed. The C++ side adds
`step_with_external_forcing()` which wraps the existing `step()` with a
per-step forcing override at the target level. The C ABI function
`gfe_c_hierarchical_step_chain_spec` exposes this with in/out state buffers,
output kernel buffers, and output spectral-unit arrays. Python wrappers in
`gfe_ctypes.step_hierarchical_chain_spec` and `ffi.step_hierarchy_raw`. Smoke
test `test_ctypes_hierarchical_step_chain_spec` covers structural checks,
stateful multi-step progression, and silent steps. All 63 Python tests and
13 C++ tests pass.

### D5: Minimal Digital Agent Sandbox

Goal:

Demonstrate closed-loop digital autonomy on a tiny synthetic environment.

Loop:

```text
observe token/event
update continuous memory
predict next event or hidden condition
choose action token
receive delayed reward/error
assign credit through memory traces
consolidate useful changes
```

Candidate environments:

- key-door symbolic world
- delayed instruction following
- small file-operation simulator
- memory-card retrieval game
- streaming rule-change environment

Metrics:

- task success rate
- delayed reward assignment accuracy
- adaptation after rule shift
- forgetting after return to old rule
- D_eff saturation and consolidation events
- stability violation count

Acceptance criteria:

- model completes a multi-step digital task requiring memory of an earlier instruction
- delayed feedback changes future action selection
- consolidation preserves useful behavior across repeated episodes
- generated report explains which memory horizons were used

## 8. Hierarchy Usage Plan

The hierarchy modules are important, but they should not be forced into D0-D2.

Use hierarchy in this order:

1. Keep D0-D3 single-level unless the experiment specifically needs cross-level context.
2. Use existing hierarchy run/report APIs for D4A diagnostic probes.
3. Add hierarchy step ABI only when D4A proves the diagnostic pathway is useful.
4. Use stateful hierarchy in D5 or later for context-modulated digital autonomy.

Reason:

The existing hierarchy code is strong for diagnostics and renorm reporting. It is not yet exposed to Python as a live event-by-event substrate. Milestone D should not block early digital experiments on that ABI work.

## 9. Stability and Memory Diagnostics

Every D2C digital run must report:

- gamma values
- kernel weights
- D_eff
- M_cap
- M_scale where available
- stability ratio before and after update
- rescale count
- finite-state pass/fail
- memory-channel saturation or near-saturation
- consolidation events

For hierarchy runs, also report:

- per-level D_eff
- per-level M_cap and M_scale
- cross-level kernel shifts
- relation deltas
- actionable warnings
- renorm consistency checks

Rules:

- Stability violations are failures unless explicitly marked as exploratory.
- Auto-rescaling is allowed but must be counted.
- A model that succeeds only by saturating all memory channels is not considered a clean success.

## 10. Baseline Policy

Milestone D baselines should start simple.

Required early baselines:

- finite-window lookup baseline
- last-token or n-gram baseline
- no-slow-channel D2C ablation
- zero-interval D2C ablation

Optional later baselines:

- small RNN
- small LSTM
- small structured state-space baseline
- small transformer with constrained context

Do not let external baseline dependencies block D0-D2. The first goal is to prove internal mechanisms and ablations.

## 11. Experiment Reporting

Each experiment bundle should include:

- config JSON
- summary JSON
- human-readable report
- metrics table
- memory diagnostics table
- seed list
- ablation table
- failure modes
- next actions

Generated artifacts belong under:

```text
code/python/d2c/progress/<experiment_name>/<run_id>/
```

Commit policy:

- commit source, schemas, tests, and stable docs
- do not commit timestamped generated progress artifacts by default
- keep only curated reports when they are referenced by a paper, release, or milestone note

## 12. Context and Progress Ledger

`Context.md` remains the repo-wide implementation ledger.

Update `Context.md` when:

- a new Milestone D module lands
- a new experiment runs end-to-end
- an ABI surface is added or changed
- a hypothesis is validated, weakened, or rejected
- a major design decision changes the roadmap
- a meaningful failure mode is discovered

Each entry should include:

- date
- short milestone title
- changes made
- verification commands and results
- why the change matters
- next steps

Use `code/python/d2c/progress/` for experiment-local artifacts and detailed run outputs. Use `Context.md` for durable repo-level narrative.

## 13. Testing Policy

For every new module:

- unit tests for dataclass/schema roundtrips
- deterministic tests for token stream generation
- deterministic tests for token forcing
- smoke tests for digital stepping
- metric tests with known toy data

For every experiment:

- quick-mode test with very small vocab/gap/seed count
- CLI or function entrypoint test
- report generation test
- reproducibility test for fixed seed

Full statistical runs:

- target N = 20 seeds for claims
- quick tests may use N = 1-3
- report confidence intervals or min/median/max across seeds

## 14. Immediate Implementation Order

Implement in this order:

1. Create `code/python/d2c/digital/`.
2. Add token vocabulary and stream schemas.
3. Add deterministic token-to-forcing bridge.
4. Add digital trace store.
5. Add simple readout and metrics.
6. Add symbolic induction task generator.
7. Add `symbolic_induction.py` experiment entrypoint.
8. Add quick tests.
9. Add human-readable report.
10. Run D1 ablations:
    - full D2C
    - no slow channels
    - collapsed gamma
    - window-limited lookup
11. D1/D2 are complete. Implement D3 delayed-feedback integration before
    revisiting the hierarchy step ABI.

## 15. Definition of Done for Milestone D

Milestone D is done when:

- the repo contains a reusable digital event/token substrate
- at least two digital tasks run end-to-end, including one with delayed
  feedback that changes future behaviour
- every task has a baseline and an ablation
- reports include memory diagnostics and stability checks
- D_eff or channel usage is shown to vary with task difficulty in at least one controlled experiment
- hierarchy diagnostics are used in at least one digital probe
- the plan for stateful hierarchy stepping is either implemented or documented with exact ABI requirements

## 16. What We Are Not Doing Yet

Not in Milestone D:

- training a large language model
- claiming transformer-scale superiority
- building a full symbolic theorem prover
- adding heavy ML dependencies as a first step
- replacing the C++ core with Python dynamics
- making hierarchy mandatory for all digital tasks

## 17. Working Claim

The working claim for this milestone is:

```text
D2C can become an autonomous digital-space model by treating token and event
streams as forcing signals into a stable continuous memory substrate, then
using explicit multiscale memory traces for prediction, delayed credit
assignment, consolidation, and eventually action selection.
```

This claim is narrow enough to test and broad enough to guide the next phase.
