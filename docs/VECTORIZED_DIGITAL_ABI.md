# Vectorized digital-memory C++ ABI

## Scope

This ABI adds vector-valued digital memory beside the existing scalar-channel
ABERSOE API. It does not change the scalar ABERSOE step contract.

For state dimension D and channel count C, state is u[D] and memory chi[C,D].
chi is one contiguous row-major buffer with index channel * D + dimension.
Kernel parameters are gamma[C] and weights[C].

## Numerical contract

For each channel and state coordinate:

    chi_next[c,d] = chi[c,d] + dt * (-gamma[c] * chi[c,d] + source[d])

The source is forcing for input-driven mode and the previous state for
negative-feedback mode. Feedback is aggregated from the updated memory:

    feedback[d] = sum_c weights[c] * chi_next[c,d]
    u_next[d] = u[d] + dt * (-leak_rate*u[d] + forcing[d] +/- feedback[d])

The positive sign is input-driven; the negative sign is negative-feedback.
No kernel normalization or implicit stability correction is performed.

## C ABI

- gfe_c_digital_step performs one vector-valued update.
- gfe_c_digital_run_sequence consumes row-major forcings [steps,D] and can emit
  final state plus optional trajectories. Trajectories include the initial
  state at row zero and then each post-step state.
- Output views use capacities and required-size pointers. A too-small buffer
  reports required sizes and returns GFE_C_STATUS_BUFFER_TOO_SMALL.

## Python integration

CppVectorizedMemoryStepper is selected with BabiF0Config(memory_backend=
"cpp_vectorized") or G2Config(memory_backend="cpp_vectorized"). The default
remains python_ref. The adapter changes only the numerical memory update;
phase control, the 0.9 stability margin, diagnostics, traces, readout,
checkpointing and episode resets remain in Python.

The whole-sequence call is available for parity checks and controlled
benchmarks. The current F0 runtime continues to request one update per event
so Python can preserve per-step phase and stability observations.

## Validation

C++ tests cover both forms, update order, sequence parity, ABI layouts and
buffer-too-small reporting. Python tests compare native steps and sequences
against the Python reference across several dimensions and channel counts.
