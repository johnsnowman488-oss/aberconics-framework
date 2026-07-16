# ABERSOE and Adaptation: Role in Non-Markovian Dynamics Research

## High-Level Picture

The Aberconics framework has three logical layers for non-Markovian research:

```
┌─────────────────────────────────────────────────────────┐
│ Research Question & Data (papers/experiments)           │
│ "Learn memory kernels for chaotic/turbulent systems"    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ GFE Core (spectral numerics substrate)                  │
│ - Kernel fitting (NNLS, Prony, regularized methods)    │
│ - Spectral units (capacity, entropy, effective dim)    │
│ - Parameter packing/unpacking for optimization         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ ABERSOE Runtime (model assembly + execution)            │
│ - Couples learned kernels → state dynamics              │
│ - Formulation choices (A/B/C) for coupling semantics    │
│ - Adaptation hooks (Hebbian memory-weight tuning)       │
│ - Diagnostics & verification (energy, renorm, drift)    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Hierarchical MIN (multi-scale + cross-level analysis)   │
│ - Multi-level composition for complex systems           │
│ - Cross-level diagnostics & renormalization             │
│ - Scaling & coupling strategies                         │
└─────────────────────────────────────────────────────────┘
```

---

## Role 1: ABERSOE as Model Materialization & Execution

**What it does:**
- Takes abstract kernel parameters `(gamma, w)` from `gfe_core` fitting and materializes them as **runtime dynamical systems**.
- Provides explicit coupling formulations:
  - **Formulation A** (Input-Driven): memory driven by external forcing
  - **Formulation B** (Negative-Feedback): memory driven by state, subtracted from dynamics (dissipative)
  - **Formulation C** (Resonant): second-order state layout with coupled q/v dynamics
- Executes time integration via augmented ODEs: `du/dt = L[u] + sum(w_l * chi_l)`, `dchi_l/dt = -gamma_l * chi_l + drive_l(u,t)`

**Why it matters for non-Markovian research:**
- Non-Markovian effects come from **explicit convolution of past state through memory kernels**.
- ABERSOE is the **executable bridge** between the mathematical abstract `sum(int_0^t w(t-s) u(s) ds)` and actual numerical simulations.
- Different formulations (A/B/C) embody different **physical hypotheses** about how memory couples to state. Researchers can test which formulation best explains data.

**Example:** 
- Fit OU-noise or chaotic dynamics using Prony regression → get `gamma, w` from `gfe_core`.
- Pass to `abersoe_cli --scenario lorenz63 --form B` with custom kernels.
- Get trajectory + diagnostics showing whether memory improved basin structure (Lyapunov shift).

---

## Role 2: Adaptation Mechanisms (Hebbian + Learning Hooks)

**What it does:**
ABERSOE provides **adaptive memory-weight tuning** during or after execution:
- **Hebbian learning** (`abersoe_cli --hebbian oja`): update `w_l` on-the-fly based on co-variance of memory channel `chi_l` and state `u`.
- **Error-modulated learning**: track objective (e.g., prediction error, energy drift) and adjust `w` to minimize it.
- **Gradient-based fitting**: pack `(gamma, w)` → `theta`, optimize via backprop or evolutionary methods.

**Why it matters for non-Markovian research:**
- Raw Prony/NNLS fits may be suboptimal for **specific tasks** (chaos suppression, energy preservation, etc.).
- Adaptation allows **closed-loop discovery**: fit a kernel, run dynamics, measure performance, adjust kernel weights, repeat.
- Hebbian rules are **biologically plausible** and **mechanistic** — they reveal which memory timescales are "important" for the task.
- Bridges **data-driven learning** (fit from observed trajectories) and **physics-driven adaptation** (tune weights to preserve energy or suppress chaos).

**Example:**
- Learn initial kernel from OU trajectory using `fit_soe_kernel`.
- Run through `abersoe` with Hebbian learning → `w` evolves to capture dominant structures.
- Record `w(t)` over steps to see which timescales the system prioritizes.
- Compare learned `w` to Prony baseline and to domain knowledge (e.g., expected vortex timescales in fluids).

---

## Role 3: Diagnostics & Verification (Closed-Loop Feedback)

**What it does:**
ABERSOE emits rich **runtime diagnostics**:
- **Spectral metrics**: Mcap (memory capacity), Deff (effective dimension), Hnorm (normalized entropy)
- **Stability**: norm growth, NaN/Inf detection, boundedness flags
- **Energy tracking**: residual dE/dt vs expected dissipation
- **Regression snapshots**: CSV baselines for trajectory/metric consistency checks
- **Renormalization diagnostics**: cross-level analysis, drift metrics (hierarchical MIN)

**Why it matters for non-Markovian research:**
- A learned memory kernel is only useful if it **demonstrates measurable improvement** or **interpretable change** to system behavior.
- Diagnostics allow researchers to:
  - Validate that memory is not degenerate (Deff > 1, Mcap > 0).
  - Check **regime change** (Lyapunov suppression, chaos → order transition).
  - Detect overfitting or model collapse (energy drift, norm explosion).
  - Quantify memory "quality" (spectral entropy, effective dimension) and compare across methods.
- Renormalization analysis reveals whether memory effects **scale across levels** in multiscale systems (hierarchical MIN use case).

**Example:**
- Fit kernel to Lorenz63 with target: suppress Lyapunov exponent by 25% (from paper examples).
- Run `abersoe_cli --scenario lorenz63 --form B --fit-backend prony --diagnostics-csv /tmp/diag.csv`.
- Check CSV: does max-norm(`u`) decrease? Is Hnorm stable? Does renorm report meaningful cross-level energy balance?
- If diagnostics show instability, adjust fit constraints (e.g., Sobolev regularization, basis pruning) and re-fit.

---

## Role 4: Connection to Literature (GLE/Mori–Zwanzig/ML Papers)

**How ABERSOE maps to research frameworks:**

| Research Concept | GFE/Core Tool | ABERSOE Realizes As |
|---|---|---|
| **Memory kernel identification** | `fit_soe_kernel`, Prony, NNLS | Kernel ingestion + diagnostics |
| **Non-Markovian closure** | `gfe_dynamics` + `CouplingForm` enums | Formulation A/B/C explicit choices |
| **Adaptive closure tuning** | Parameter packing + optimization API | Hebbian + error-modulated learning loops |
| **Coupling semantics** (which sign, where forcing enters) | `apply_L`, `apply_N`, `forcing` callbacks | Model spec + coupling index |
| **Verification** (does memory help?) | Spectral metrics, energy observer | Diagnostics CSV + regression snapshots |
| **Multi-scale effects** | Hierarchical MIN + GST adapter | Cross-level renorm report + coupling |

---

## Role 5: Practical Integration Path for GLE/MZ Research

### Workflow A: Kernel Learning → Runtime Verification
```
1. Get trajectory data (e.g., chaotic system, partial observation)
2. Use gfe_core:  fit_soe_kernel(...) → get (gamma, w)
3. Use abersoe:   assemble model, choose Formulation B/C
4. Run abersoe_cli with learned kernel
5. Check diagnostics CSV (energy, spectral units, drift)
6. If good → publish kernel + metrics; if poor → iterate fit strategy
```

### Workflow B: Data-Driven MZ Closure Discovery
```
1. Identify coarse-grained observables (e.g., slow modes via SVD)
2. Use gfe_core:  mori_zwanzig_projection(...) helper
   - compute regression kernel from restricted dynamics
   - get SOE approximation (gamma, w) to MZ memory term
3. Use abersoe:   couple as Formulation B (dissipative closure)
4. Run hierarchy or single-level with learned closure
5. Compare reduced model prediction vs full-system data
6. Refine basis / regularization if needed
```

### Workflow C: Adaptive Memory Discovery (Online Learning)
```
1. Start with weak prior on kernel (e.g., uniform w, sparse gamma)
2. Use abersoe with Hebbian learning enabled
3. Run ensemble or long trajectory, record w(t) evolution
4. Analyze which timescales (gamma values) grew important
5. Use final w as starting point for offline optimization
6. Plug optimized kernel into hierarchical MIN for multi-scale experiments
```

---

## Summary: Where ABERSOE Sits

- **GFE Core** = *What are the kernels?* (fitting, metrics, theory)
- **ABERSOE** = *How do we use kernels to alter dynamics?* (runtime, coupling, adaptation, verification)
- **Hierarchical MIN** = *How do memory effects propagate across scales?* (multi-level composition, renormalization)
- **C ABI / Python Wrappers** = *How do we integrate into ML/scientific workflows?* (end-to-end differentiable, adjoint, operator learning)

**ABERSOE is the executable heart** of the framework—the bridge from theory (GLE / Mori-Zwanzig) to practice (running, measuring, adapting, publishing).

---

## Next Experiments for GLE/MZ Integration

1. **Learn kernel from OU, inject into Lorenz**: verify memory effect on chaos → support GLE theory claim.
2. **Adaptive Hebbian tuning + Lyapunov tracking**: show online learning discovers timescales without supervision.
3. **Hierarchical MIN + Renorm diagnostics**: demonstrate cross-scale memory consistency (validate MZ reduction).
4. **Compare formulations A/B/C on same kernel**: show how coupling choice affects final dynamics (physics interpretability).
5. **End-to-end gradient-based fitting**: embed abersoe stepping in autodiff loop, learn kernels end-to-end (modern ML integration).

---

## Appendix: How ABERSOE Addresses Broken/Suboptimal Trends in Kernel Learning

Recent research has identified common pitfalls in data-driven kernel identification. Below is how six key papers diagnose these pitfalls and where **ABERSOE + gfe_core** fit as solutions:

### Pitfall 1: Ill-Posed Kernel Fitting Without Regularization
**Problem:** Naive NNLS fitting of memory kernels from noisy trajectories leads to:
- Overfitting to measurement noise
- Degenerate solutions (few large weights, many zeros)
- Unstable extrapolation beyond training range

**Literature Fix:** 
- **Lang & Lu (2024)** *Learning Memory Kernels in GLEs*: Use **Sobolev-norm regularization** + Prony method to balance fit quality vs. smoothness, and provide **error bounds** showing kernel approximation error decays with true system properties.

**ABERSOE Response:**
- `gfe_core` already has NNLS + Prony backends; extend with `SoeFitOptions::regularization = Sobolev` option.
- Add **regression-stability tests** in smoke tests: verify fitted kernel produces bounded trajectories for 100+ steps.
- Emit **kernel quality metrics** in CSV: condition number of design matrix, effective rank, residual growth rate.
- *Implementation path:* Add `FitBackend::PronyRegularized` variant and `SoeFitOptions.sobolev_lambda` parameter; document trade-off curves (lambda vs. fit error vs. stability).

### Pitfall 2: Non-Physical Kernels (Violating Fluctuation-Dissipation & Energy Constraints)
**Problem:** Learned kernels may:
- Violate the fluctuation-dissipation theorem (FDT), leading to perpetual motion or unphysical dissipation.
- Produce energy growth on timescales where systems should dissipate.
- Fail on multi-scale systems where energy must be conserved across levels.

**Literature Fix:**
- **Xie, Car, E (2024)** *Ab Initio Generalized Langevin Equations*: **FDT-consistent kernel learning** via Lagrange multipliers; ensures learned `(gamma, w)` respect equilibrium fluctuation-dissipation relations.

**ABERSOE Response:**
- `gfe_energy` module already has energy observers; add **FDT-check hooks**:
  - Compute kernel-derived Green-Kubo relation (e.g., integral of kernel ≈ viscosity in fluids).
  - Flag kernels that would violate energy dissipation direction.
- `gfe_assumptions` module: add FDT-evidence field to `AssumptionEvaluation` struct.
- `abersoe_diagnostics`: emit CSV column `energy_dissipation_valid` (boolean), `fdt_margin` (how close to boundary).
- *Implementation path:* Add `gfe_c_fdt_check_kernel(gamma, w, reference_viscosity)` function returning deviation metrics; integrate into `fit_soe_kernel` as optional post-fit validation; expose through C ABI.

### Pitfall 3: Black-Box Closures Without Interpretability
**Problem:** ML-learned closures (e.g., neural networks, opaque regressions) lack:
- Mechanistic interpretation (which timescales matter? what physical processes do they represent?)
- Auditability (why did learning choose this closure over alternatives?)
- Transferability (does closure learned for System A work on System B?)

**Literature Fix:**
- **de Wit et al. (PNAS 2026)** *Data-Driven Mori-Zwanzig of Lagrangian Dynamics*: Use **regression-based MZ projection** to extract interpretable memory kernels with explicit timescales and physical meaning.
- **Lin et al. (2022/2023)** *Regression-Based Projection for MZ Operators*: Systematic operator identification via least-squares regression of restricted dynamics.

**ABERSOE Response:**
- Add `gfe_core` helper: `mori_zwanzig_projection(full_state_traj, reduced_state_indices) -> (gamma, w)` that:
  - Projects full dynamics onto reduced subspace via SVD.
  - Regresses restricted system to extract memory term.
  - Returns **SOE approximation** of MZ memory kernel.
- Emit **projection quality metrics**: reconstruction error, effective rank of memory matrix, coverage of timescale spectrum.
- `abersoe` scenario: expose `"mz_closure"` scenario type that takes `reduced_indices` as input and automatically builds augmented model with regression-learned kernel.
- *Implementation path:* Implement `mori_zwanzig_projection` in `gfe_core.cpp` using ATA/ATb patterns from existing NNLS code; wire into `abersoe_cli` as `--scenario mz_lorenz63 --reduced-indices 0,2` (project Lorenz to x,z, let y be memory-modeled).

### Pitfall 4: Kernels Learned from Full State Fail Under Partial Observation
**Problem:** Most kernels are fitted assuming access to full system state; when only partial sensors are available:
- Learned kernel becomes invalid (memory was implicitly learning unobserved degrees of freedom).
- Reduced model with learned kernel diverges rapidly.
- No clear way to adapt kernels for different sensor configurations.

**Literature Fix:**
- **Monsel et al. (2026)** *Neural Delay Differential Equations*: Learn non-Markovian closures as **delay differential equations (DDEs)** or **neural DDEs** directly from partial observations, treating unobserved variables as implicit delays.

**ABERSOE Response:**
- Add `gfe_dynamics` option: `PartialObservationMode` enum with variants:
  - `FULL_STATE`: current behavior (all state observed).
  - `DELAY_EMBEDDED`: learn kernel from time-delay embeddings (treat past as memory).
  - `NEURAL_DDE`: parameterized DDE kernel (allow NN-parameterized `drive` function).
- `abersoe_runtime_config`: add optional `observation_indices` field to signal which state dims are observable.
- `abersoe_cli`: support `--scenario lorenz63 --partial-observation 0,1` to learn kernel from x,y only (z hidden).
- *Implementation path:* Add `GFEDynamicsAdapter::learnable_drive(...)` callback type; allow embedding NN layers as `drive` function (requires PyTorch/Julia binding). Wire into C ABI as opaque handle for scripting.

### Pitfall 5: Architectural Rigidity — Hard to Design Closures for Different System Topologies
**Problem:** One kernel design doesn't translate across different graph structures or PDE domains:
- Closure learned for grid graph doesn't apply to complex networks.
- 1D closure doesn't generalize to 2D/3D PDEs.
- No principled way to design closure architecture for new domain.

**Literature Fix:**
- **Yu, Harlim, et al. (2024)** *Learning Coarse-Grained Dynamics on Graph*: **MZ-informed GNN architecture** where graph convolution is designed to capture leading MZ memory term at each node; learn closure via graph structure.

**ABERSOE Response:**
- Add `gfe_gst` adapter: `GraphOperatorAdapter` that:
  - Takes graph Laplacian or adjacency as input.
  - Designs SOE basis (gamma) informed by spectral gaps and connectivity.
  - Fits weights (w) via regression on graph-restricted dynamics.
- `abersoe` model spec: allow `GST_ADAPTER` = `GraphOperator` with graph-construction lambda.
- Expose **graph scenario** in `hierarchical_min_cli`: compose graph-structured agents, each with learned MZ closure.
- *Implementation path:* Implement `gfe_gst::GraphOperatorAdapter` that builds `gamma` from graph spectrum + fits `w` via regression; expose in C ABI as `gfe_c_hierarchical_run_graph_scenario(...)`.

### Pitfall 6: Lack of End-to-End Differentiability for Modern ML Integration
**Problem:** Current kernel fitting workflows are separate from downstream learning:
- Kernel fit offline via NNLS/Prony.
- Kernel plugged into simulator (non-differentiable).
- Hard to backprop through memory effects for hybrid physics-ML models.
- Adjoint methods require hand-coding for each coupling form.

**Literature Fix:**
- Multiple papers (Lang & Lu, Monsel et al.) note need for **differentiable memory models** in modern autodiff ecosystems (PyTorch, JAX).

**ABERSOE Response:**
- Add **differentiable stepping path** in `gfe_dynamics`:
  - C++ core stays as is (numerical stability).
  - Provide **PyTorch/JAX wrapper** (via ctypes or direct binding) exposing `abersoe_step` as differentiable operation.
  - Implement **implicit adjoint** for augmented ODE (using backsolve through memory channels).
- `abersoe_runtime_config`: add `differentiable_mode = true` flag; when enabled, use forward-mode or reverse-mode autodiff tracing.
- Example workflow:
  ```python
  kernel = gfe.fit_soe_kernel(data)
  model = abersoe.DifferentiableModel(kernel, form='B')
  loss = sum( (model.trajectory(t_end) - target) ** 2 )
  loss.backward()  # gradient w.r.t. kernel weights
  ```
- *Implementation path:* Build PyTorch extension via pybind11; expose `AberSOEFunction` as `torch.autograd.Function` with custom backward. Wire C ABI gradient calls through ctypes for lighter-weight setup.

### Summary: Coverage of Pitfalls

| Pitfall | Key Paper | ABERSOE Solution | Status |
|---|---|---|---|
| Ill-posed fitting (noise, overfitting) | Lang & Lu 2024 | Sobolev regularization + stability tests in `gfe_core` | *To implement* |
| Non-physical kernels (FDT, energy) | Xie et al. 2024 | FDT-check hooks in `gfe_energy` / `gfe_assumptions` | *To implement* |
| Black-box closures (no interpretability) | de Wit et al., Lin et al. | `mori_zwanzig_projection` helper in `gfe_core` | *To implement* |
| Partial observation failures | Monsel et al. 2026 | Delay-embedding + neural-DDE mode in `gfe_dynamics` | *To implement* |
| Rigid architectures (graph/PDE) | Yu et al. 2024 | `GraphOperatorAdapter` in `gfe_gst` | *To implement* |
| Lack of end-to-end differentiability | Multiple papers | PyTorch/JAX wrapper with implicit adjoint | *To implement* |

**Critical Implementation Priority:**
1. **Sobolev regularization + FDT checks** (Pitfalls 1–2): These are **foundational** for validating any learned kernel. Should be in `gfe_core` before Workflows A/B.
2. **Mori-Zwanzig projection helper** (Pitfall 3): Enables interpretability and reproducibility. Direct mapping from literature → code.
3. **Differentiable wrapper** (Pitfall 6): Unlocks modern ML + operator learning. High impact but requires binding infrastructure.
