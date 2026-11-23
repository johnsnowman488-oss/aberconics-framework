# M4 Persistent Custom-Chain Hierarchy Runtime
## ABI Contract, Ownership Model, and Implementation Ledger

**Status:** Persistent execution and tested D4B parity passed on Linux/Windows; checkpoint/restore and performance benchmark remain open — PR not merge-ready  
**Branch:** `m4/persistent-hierarchy-abi-contract`  
**Base:** `main`  
**Scope:** Evolve the existing scalar custom-chain hierarchy ABI into a persistent native runtime for HLA/HLB. Preserve the current D4B numerical contract.

---

## 1. Purpose and architectural boundary

The current D4B entry point, `gfe_c_hierarchical_step_chain_spec(...)`, is stateful at the data level but not persistent at the model level. Each call converts the chain specification, constructs a `HierarchicalMINModel`, rebuilds a `HierarchicalState` from caller buffers, advances one step, and copies state, active kernels, and spectral diagnostics back.

The Python `step_hierarchical_chain_spec` wrapper also rebuilds the chain-spec ctypes buffers on each call.

M4 adds a persistent runtime handle. It is an extension of the existing custom-chain implementation, not a replacement hierarchy and not a migration to the vector-valued G4 digital-memory representation.

**Boundary rule:** C++ owns the SOE hierarchy execution model and evolving numerical state. Python owns experiment orchestration, Leptograph objects/graphs, relational decisions, scheduling, and experiment records. The ABI exchanges typed numerical inputs, state snapshots, diagnostics, and checkpoint bytes; it does not expose arbitrary C++ or Leptograph objects.

## 2. Evidence from the existing implementation

Inspected on `main`:

- `code/c++ core/include/gfe/gfe_c_api.h`
- `code/c++ core/src/gfe_c_api.cpp`
- `code/python/gfe_ctypes.py`
- `code/python/d2c/ffi.py`
- `code/python/README.md`
- `d2c_milestone_d_roadmap.md`

Existing relevant types include `gfe_c_hierarchical_chain_spec_view`, `gfe_c_state_view`, `gfe_c_state_mut_view`, `gfe_c_memory_kernel_mut_view`, and `gfe_c_spectral_units`.

Existing status values are OK=0, INVALID_ARGUMENT=1, RUNTIME_ERROR=2, BUFFER_TOO_SMALL=3. Any new values must be appended; existing numeric values must not change.

The current step function validates the chain, constructs the model inside the call, reconstructs current per-level state from caller-owned buffers, applies one external forcing vector to one `forcing_level`, and returns per-level state/kernel/spectral outputs. It validates output capacities before copying, avoiding partial output writes.

The Python wrapper builds specification, state, forcing, kernel, and spectral ctypes buffers for each call. Its current state contract uses scalar per-channel `chi` arrays. Do not conflate this with vector-valued memory state.

## 2.1 Confirmed stepping semantics from the C++ implementation

Inspection of `abersoe::step` in `hierarchical_min.cpp` settles the multi-feedback question:

- The stepper first normalizes a single pre-step hierarchy state and computes observables from that state.
- It then computes each level's next state from that same pre-step hierarchy state.
- Bottom-up coupling contributions are accumulated into the target level's forcing during that level's one step.
- Top-down modulation is computed from the source level's pre-step state and applied to the target's active kernel for that cycle.
- The old `step_with_external_forcing` shallow-copies the model and wraps one target level's base forcing before calling `step`.

Therefore multi-level external feedback can be applied in one synchronisation cycle without repeated calls to the stepper: attach each validated vector to its target level's local forcing closure during the existing per-level step loop, then evaluate the hierarchy exactly once. The new `step_with_external_forcings` injects each validated vector directly into the corresponding per-level forcing closure inside the existing step loop. It validates unique targets and exact target state dimensions, computes the full next state before returning, and does not copy or reconstruct the full hierarchy model.

**Important D4B parity detail:** the existing `gfe_c_hierarchical_step_chain_spec` rebuilds each level's active kernel from the immutable chain specification on every event before stepping. The persistent C API explicitly preserves this reset at each event. This prevents top-down-modulated kernels from accidentally becoming persistent across events when the D4B caller-owned state contract did not preserve them.

## 3. Design decisions (proposed contract)

### 3.1 Opaque handle and lifecycle

Introduce an opaque 64-bit handle token (not a raw pointer). The implementation uses a native registry of shared runtime owners, so stale or unknown tokens can be rejected without dereferencing caller-supplied addresses. Conceptually:

```c
typedef uint64_t gfe_c_hierarchy_handle;

int gfe_c_hierarchy_create(
    const gfe_c_hierarchical_chain_spec_view* spec,
    const gfe_c_hierarchy_runtime_options* options,
    gfe_c_hierarchy_handle* out_handle,
    char* error_msg, size_t error_msg_capacity);

int gfe_c_hierarchy_destroy(
    gfe_c_hierarchy_handle handle,
    char* error_msg, size_t error_msg_capacity);
```

The handle is created once from a validated, copied chain specification. A successful destroy removes its token from the registry and marks the shared runtime closed. Null, unknown, or already-destroyed tokens return the appended INVALID_HANDLE status. The Python owner makes close idempotent; direct C callers must not reuse a destroyed token.

The runtime owns:
- the copied immutable chain specification and topology;
- the constructed C++ hierarchy model;
- the current `HierarchicalState` for every level;
- the current step/event counter and runtime metadata;
- any runtime-owned synchronization primitive.

The runtime does not own Python memory, caller buffers, Leptograph nodes, Python callbacks, or experiment orchestration.

Creation copies all specification data. No pointer from the caller's specification may be retained after create returns.

### 3.2 State ownership and exchange

**Default: handle-owned state.** Initial state is taken from each chain level's `initial_state` during creation. A successful step commits the next state inside the handle. Python no longer needs to round-trip every level's state on every event.

Expose explicit inspection and controlled restoration/update operations:
- query level count and stable level metadata;
- get one level's state, active kernel, and spectral diagnostics;
- get all levels' snapshots using caller-owned output views;
- set/restore state only through a validated all-level operation, not by mutating internal pointers.

Get operations use the existing capacity/size convention: report required sizes, validate every output capacity, then copy. A short buffer must not partially modify caller data. The API must document whether size fields are updated on BUFFER_TOO_SMALL (recommended: yes, required sizes are returned).

State writes must validate all level counts, dimensions, finite values (when strict-finite is enabled), and time consistency before committing any level. The write is atomic across the hierarchy.

The active kernel is part of the runtime snapshot. In the initial implementation it is read-only through the public runtime API unless the existing numerical engine already changes it during a step. Do not add kernel adaptation implicitly.

### 3.3 Multi-level feedback and synchronisation

Replace the single `forcing_level` input for the persistent interface with an array of addressed feedback records. Conceptual v1 shape:

```c
typedef struct gfe_c_hierarchy_feedback_view {
    size_t target_level;
    const double* values;
    size_t values_size;
    /* reserved/versioned flags may be added */
} gfe_c_hierarchy_feedback_view;

int gfe_c_hierarchy_step(
    gfe_c_hierarchy_handle handle,
    const gfe_c_hierarchy_feedback_view* feedback,
    size_t feedback_count,
    char* error_msg, size_t error_msg_capacity);
```

Semantics:
- zero feedback records means a silent/free-evolution hierarchy step;
- each record addresses exactly one hierarchy level;
- v1 permits at most one external feedback vector per target level per synchronisation cycle; duplicate target levels are invalid rather than implicitly summed;
- feedback dimensions and target levels are validated before execution;
- all feedback records are applied within one hierarchy synchronisation/step operation, not by calling the step function repeatedly;
- the existing engine's forcing semantics and ordering are preserved for the one-input case;
- multi-input combination/order semantics must be specified by the C++ hierarchy implementation and tested. The ABI must not invent a summation rule that the engine does not support.

The first implementation supports multiple independently addressed level vectors in one call, without arbitrary callbacks. Duplicate targets are rejected. Empty feedback means free evolution. The C++ stepper receives all feedback in a single call, so it evaluates all levels against one pre-step state. More expressive typed feedback channels can be added later with an explicit ABI version.

### 3.4 Transaction and failure semantics

A step is a transaction:
1. validate handle and all input records;
2. construct/prepare temporary step inputs;
3. compute the next hierarchy state without mutating committed state;
4. validate resulting dimensions and finite-state policy;
5. commit the complete next state and increment the event counter;
6. return OK.

On any validation or runtime failure, committed state and event counter remain unchanged. Error text is diagnostic only; callers should branch on status codes.

This is the key guarantee that lets Python coordinate Leptograph feedback without observing a half-applied hierarchy cycle.

### 3.5 Threading and reentrancy

Initial contract: operations on one handle are serialized. The implementation should protect each handle's state with an internal mutex, so independent handles can run concurrently. Calls to destroy must not race with any other operation on that same handle; this remains a caller lifecycle responsibility. No callback from C++ into Python is permitted.

Document this explicitly in the header and Python wrapper. Do not promise parallel stepping of one handle.

### 3.6 ABI compatibility and versioning

- Preserve all existing exported functions and struct layouts.
- Add new declarations and append new status enum values only.
- New option/config structs should begin with `struct_size` and `abi_version` fields to permit forward-compatible extension.
- Use opaque handles; never expose C++ class layouts.
- Use fixed-width scalar types only where needed for serialized formats; keep the existing ABI's `size_t` conventions for in-process array lengths.
- Keep the existing scalar `gfe_c_state_view`/`gfe_c_state_mut_view` contract and numerical semantics unchanged.
- The old `gfe_c_hierarchical_step_chain_spec` remains available as a compatibility/reference path and as the parity oracle.

### 3.7 Status and error handling

Keep current status values stable. Append distinct statuses if needed for invalid handle, incompatible checkpoint, or busy/lifecycle errors; do not overload INVALID_ARGUMENT for every runtime-handle condition. All exported functions must catch C++ exceptions and translate them to status + bounded error text. No exception may cross the C boundary.

### 3.8 Checkpoint and restore

Checkpointing is a later implementation stage, but its contract is reserved now.

Provide a two-call buffer protocol:
- query required checkpoint byte count;
- serialize into a caller-owned buffer after capacity validation.

Checkpoint payload must be versioned and self-describing. It should include:
- checkpoint format version;
- a deterministic fingerprint of the immutable hierarchy specification/topology and numerical contract;
- event/step counter;
- per-level scalar state (`u`, `chi`, `t`);
- active-kernel data if it is mutable in the selected runtime;
- any additional state required for exact continuation.

Restore validates format, fingerprint, counts, dimensions, and finite-state policy before atomically replacing runtime state. A checkpoint for a different topology/specification must be rejected. Do not serialize raw pointers, compiler-dependent object memory, mutexes, or C++ object layouts.

### 3.9 Configuration and topology mutation

Topology is immutable for the lifetime of a v1 handle. No add/remove/reorder level or edge operation is included.

Runtime parameter updates are also excluded from the first ABI slice unless a concrete experiment requires a narrowly typed, validated update. State restoration is not topology mutation. A future reconfiguration API must define whether it creates a new runtime or performs an explicit migration with a reproducible mapping; never silently mutate topology mid-episode.

## 4. Proposed API surface (logical, not yet declarations)

| Operation | Responsibility | Initial priority |
|---|---|---:|
| `gfe_c_hierarchy_create` | Copy spec, validate, construct model/state once | P1 |
| `gfe_c_hierarchy_destroy` | Release native resources | P1 |
| `gfe_c_hierarchy_get_level_count` | Discover hierarchy dimensions | P1 |
| `gfe_c_hierarchy_step` | One atomic synchronisation with multi-level feedback | P2 |
| `gfe_c_hierarchy_get_state` | Inspect state/kernel/spectral snapshot | P2 |
| `gfe_c_hierarchy_set_state` | Atomic validated state restore/set | P3 |
| `gfe_c_hierarchy_checkpoint_size` / `checkpoint` | Versioned serialization | P4 |
| `gfe_c_hierarchy_restore` | Validated atomic restoration | P4 |

Names and exact C signatures remain provisional until header review. Do not implement a generic `update_config` or topology-edit function in v1.

## 5. Python integration contract

Add a small owning Python class (candidate: `PersistentHierarchyRuntime`) above `gfe_ctypes.py`/ `d2c/ffi.py`.

Responsibilities:
- create native handle once from ordinary Python level/edge definitions;
- retain the `ctypes.CDLL` object for at least as long as the handle;
- retain only buffers needed for each call, not the full hierarchy spec on every step;
- provide `step(feedback_by_level)`, `get_state()`, `set_state(snapshot)`, `checkpoint()`, `restore(blob)`, and context-manager/close semantics as stages land;
- close idempotently and release native handle deterministically;
- guard against use-after-close;
- avoid a Python destructor as the only cleanup mechanism.

The Python coordinator remains the authority for Leptograph execution and feedback construction. It must not reconstruct the C++ hierarchy on each event.

## 6. Numerical compatibility and D4B parity policy

The old D4B step function is the reference implementation. Persistent runtime is accepted only after deterministic parity tests show that, for identical specification, initial state, forcing sequence, and strict-finite policy:
- every level's `u`, `chi`, and `t` match within the established numerical tolerance;
- active kernel arrays match;
- all spectral diagnostics match within tolerance;
- silent steps and one-level forcing match;
- multi-step sequences preserve the same ordering and state progression.

Do not compare the persistent runtime against the vector-valued G4 digital-memory API. They represent different state models.

Parity tests must include at least two- and three-level chains, bottom-up and top-down edges, asymmetric dimensions where supported, zero/silent forcing, and invalid-input/no-mutation cases. Record compiler/build details and tolerances in the test report.

## 7. Implementation ledger

This is the active work ledger. Status values: **NOT STARTED**, **IN PROGRESS**, **BLOCKED**, **PASSED**, **FAILED**. Update it when code, tests, or decisions change; do not mark a stage complete on design agreement alone.

| ID | Work package | Deliverable / exit gate | Status |
|---|---|---|---|
| M4-ABI-0 | Contract and ownership design | C++ step semantics, handle token, status, and ownership contract reconciled with implementation | PASSED |
| M4-ABI-1 | Native runtime type | Shared registry-owned runtime containing persistent model/state and per-handle mutex; Linux/Windows builds and C++ tests pass | PASSED |
| M4-ABI-2 | Create/destroy/discovery ABI | 64-bit opaque token, create/destroy, level count, stale-token rejection test | PASSED |
| M4-ABI-3 | Persistent single-level-input step | Handle-owned step, zero or one feedback, D4B active-kernel reset semantics and per-event parity | PASSED |
| M4-ABI-4 | Multi-level feedback transaction | Multiple addressed vectors, exact dimensions, duplicate/out-of-range validation, compute-before-commit; no-mutation test passes | PASSED |
| M4-ABI-5 | State and diagnostics inspection | Per-level state/kernel/spectral snapshot; exercised by persistent-vs-D4B parity tests | PASSED |
| M4-ABI-6 | State restore/set | Native atomic all-level validation and commit; dimension/time/kernel/finite checks; step counter restored | PASSED |
| M4-ABI-7 | Python owning wrapper | Context-managed native owner; D2C director creates once per episode and sends feedback per event | PASSED |
| M4-ABI-8 | D4B parity campaign | Two-/three-level deterministic parity, silent/forced sequences, multi-level feedback and invalid-input invariance | PASSED (tested scope; Linux and Windows 198 passed/4 skipped) |
| M4-ABI-9 | Checkpoint/restore | Version-1 portable JSON; SHA-256 spec fingerprint and payload checksum; atomic file write; exact three-level continuation and rejection tests | PASSED |
| M4-ABI-10 | HLA/HLB integration seam | Python feedback adapter and inspection path; hierarchy topology held fixed | NOT STARTED |
| M4-ABI-11 | Confirmatory runtime campaign | Repeated create/step/checkpoint/restore/destroy cycles and stale-handle rejection | PASSED (24-cycle lifecycle test; native heap-leak profiling not instrumented) |
| M4-ABI-12 | Persistent-vs-D4B benchmark | 3-level / 4-edge workload; 400 events, 7 repeats, 2 warm-ups; alternating order; parity gate; raw JSON evidence | PASSED (1.865x measured episode speedup; native allocator count not instrumented) |

### Execution order and dependencies

```text
ABI-0 -> ABI-1 -> ABI-2 -> ABI-3 -> ABI-4 -> ABI-5 -> ABI-6
                                      |                       |
                                      +------ ABI-7 <---------+
                                                   |
                                                ABI-8
                                                   |
                                      ABI-9 (checkpoint)
                                                   |
                                                ABI-10
                                                   |
                                                ABI-11
```

Parallel work is appropriate only after the shared contract is frozen: Python wrapper scaffolding can proceed alongside native inspection/state APIs, and checkpoint format design can proceed after the state schema is stable. Do not parallelize competing ABI definitions.

### Research/engineering safeguards

1. Preserve the scalar per-channel `chi` representation and existing D4B semantics.
2. No Leptograph C++ object ABI and no arbitrary Python callback crossing.
3. No topology mutation in v1.
4. No HLA/HLB performance claims before parity and controlled integration.
5. Benchmark end-to-end latency, model-construction count, and Python allocation/peak-memory proxy against D4B. Native C++ allocator counts are not instrumented in this slice; do not infer native allocation reductions or speedup without measured evidence.
6. Keep numerical correctness, interface performance, and Leptograph research outcomes as separate evidence categories.

## 7.1 Verification record

### Final cross-platform verification

Workflow run [45173581148](https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/45173581148), tested code/workflow head `9f5273ff15cb78c44900bff328053ae16a309294`:
- Ubuntu and Windows C++ build/CTest: passed.
- Ubuntu Python 3.12.16: **198 passed, 4 skipped**.
- Windows Python 3.12.10: **198 passed, 4 skipped**.
- Linux module compilation and experiment-lab JavaScript validation: passed.
- The 24-cycle create/step/checkpoint/restore/destroy lifecycle test passed on both operating systems.
- The three-level D4B parity campaign remains green; checkpoint exact-continuation and native atomic-restore tests are included in the 198 passing tests.

The latest follow-up commit preserves benchmark evidence and updates this ledger; it does not change runtime code.

### Checkpoint contract now implemented

- Native `gfe_c_hierarchy_restore` validates every level's state and active kernel into a candidate snapshot, recomputes spectral diagnostics, and commits state plus step counter only after all levels pass.
- Native `gfe_c_hierarchy_get_step_count` exposes the committed synchronisation count.
- Python checkpoint format `gfe.persistent_hierarchy.checkpoint`, version 1: JSON-serializable per-level `u/chi/t/gamma/w`, step count, SHA-256 fingerprint of the normalized immutable chain specification, and SHA-256 payload checksum.
- Checkpoint file writes use a same-directory temporary file, flush/fsync, and atomic replace. Restore checks schema/version, checksum, spec identity, level count, dimensions, and finite/positive kernel constraints before calling native restore.
- The Python owner serializes its public operations with a reentrant lock. Tests verify exact continuation on a three-level chain, immediate restored-state identity, mismatched-spec rejection, payload corruption, malformed dimensions, non-finite values, and native all-level atomicity when a later level is invalid.
- A checkpoint is portable between processes provided the caller recreates the same immutable chain specification and uses a compatible checkpoint-format version; native handles themselves are process-local.

### Persistent-vs-D4B benchmark

Raw result preserved at [`docs/benchmarks/M4_PERSISTENT_VS_D4B_2026-11-23.json`](benchmarks/M4_PERSISTENT_VS_D4B_2026-11-23.json). CI artifact: [`m4-persistent-vs-d4b`](https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/45173581148/artifacts/14817039055), artifact ZIP SHA-256 `5fc7d3e0ed5b556febbade3bf4c6cba9b2ac43622fcb9d0b531fce64c2246ce6`.

Workload: three levels, four coupling edges, 400 events per episode, two warm-up episodes, seven measured repeats, alternating execution order; CPython 3.12.16 on the Linux CI runner. Timing includes Python ctypes calls and per-event state/diagnostic transfer on both paths.

| Metric | D4B | Persistent |
|---|---:|---:|
| Median episode time | 47.185 ms | 25.304 ms including creation |
| Median per-event time | 0.11796 ms | 0.06326 ms amortized |
| Chain model constructions / episode | 400 | 1 |
| Python tracemalloc peak | 1,773,929 bytes | 1,784,314 bytes |

Measured persistent episode speedup: **1.865x**. Maximum absolute numerical discrepancy over the full benchmark trace: **0.0**. The persistent path's Python peak was about 0.6% higher; this is not a memory reduction result. Native C++ allocation counts and native heap usage were not instrumented, so no native allocation or memory-saving claim is made. This is one controlled workload, not a universal speedup claim.


## 8. Remaining work and branch policy

1. M4 persistent execution, multi-level feedback, atomic restore, versioned checkpoint/restore, exact continuation, cross-platform tests, lifecycle behavior, and the measured D4B comparison are operational and passed for the tested scope.
2. Native allocator/heap instrumentation is a separate performance-diagnostics improvement; it is not claimed by the current benchmark.
3. HLA/HLB coupling and the broader confirmatory runtime campaign remain subsequent research work packages. Keep the topology and scalar custom-chain ABI fixed until those are explicitly opened.
4. This branch is being retained as an operational research branch. No merge into `main` is planned.

