#pragma once
#include <cstddef>
#include <vector>
namespace gfe::digital {
enum class MemoryForm : int { InputDriven=0, NegativeFeedback=1 };
struct MemoryConfig { std::size_t state_dim=0, channel_count=0; double dt=0.05, leak_rate=1.0; MemoryForm form=MemoryForm::InputDriven; };
struct MemoryKernel { std::vector<double> gamma, weights; };
// chi is contiguous row-major [channel, state_dimension].
struct MemoryState { std::vector<double> u, chi; double t=0.0; };
struct MemoryDiagnostics { std::size_t steps_executed=0; bool all_finite=true; double max_abs_u=0.0, max_abs_chi=0.0; };
struct SequenceResult { MemoryState final_state; MemoryDiagnostics diagnostics; std::vector<double> times, u_trajectory, chi_trajectory; };
void validate(const MemoryConfig&, const MemoryKernel&, const MemoryState&);
MemoryState step(const MemoryConfig&, const MemoryKernel&, const MemoryState&, const std::vector<double>& forcing);
SequenceResult run_sequence(const MemoryConfig&, const MemoryKernel&, const MemoryState&, const std::vector<double>& forcings, std::size_t steps);
} // namespace gfe::digital
