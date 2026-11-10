#include "gfe/gfe_digital_memory.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace gfe::digital {
namespace {
bool finite(const std::vector<double>& v) { return std::all_of(v.begin(),v.end(),[](double x){return std::isfinite(x);}); }
void max_abs(const std::vector<double>& v,double& m) { for(double x:v)m=std::max(m,std::fabs(x)); }
std::size_t chi_size(const MemoryConfig& c) {
 if(c.state_dim && c.channel_count>std::numeric_limits<std::size_t>::max()/c.state_dim) throw std::invalid_argument("channel_count * state_dim overflows");
 return c.channel_count*c.state_dim;
}
}
void validate(const MemoryConfig& c,const MemoryKernel& k,const MemoryState& s) {
 if(!c.state_dim||!c.channel_count) throw std::invalid_argument("state_dim and channel_count must be positive");
 if(!std::isfinite(c.dt)||c.dt<=0) throw std::invalid_argument("dt must be finite and positive");
 if(!std::isfinite(c.leak_rate)||c.leak_rate<=0) throw std::invalid_argument("leak_rate must be finite and positive");
 if(c.form!=MemoryForm::InputDriven&&c.form!=MemoryForm::NegativeFeedback) throw std::invalid_argument("unknown digital memory form");
 if(k.gamma.size()!=c.channel_count||k.weights.size()!=c.channel_count) throw std::invalid_argument("kernel sizes must equal channel_count");
 if(s.u.size()!=c.state_dim||s.chi.size()!=chi_size(c)) throw std::invalid_argument("state dimensions do not match config");
 if(!std::isfinite(s.t)||!finite(k.gamma)||!finite(k.weights)||!finite(s.u)||!finite(s.chi)) throw std::invalid_argument("state and kernel values must be finite");
 if(std::any_of(k.gamma.begin(),k.gamma.end(),[](double x){return x<=0;})) throw std::invalid_argument("gamma values must be positive");
}
MemoryState step(const MemoryConfig& c,const MemoryKernel& k,const MemoryState& s,const std::vector<double>& x) {
 validate(c,k,s);
 if(x.size()!=c.state_dim||!finite(x)) throw std::invalid_argument("forcing must be finite and match state_dim");
 const auto D=c.state_dim,C=c.channel_count;
 const bool input=c.form==MemoryForm::InputDriven;
 const auto& source=input?x:s.u;
 MemoryState n; n.u.resize(D); n.chi.resize(C*D); n.t=s.t+c.dt;
 for(std::size_t ch=0;ch<C;++ch) for(std::size_t d=0;d<D;++d) {
  const auto i=ch*D+d; const double old=s.chi[i];
  n.chi[i]=old+c.dt*(-k.gamma[ch]*old+source[d]);
 }
 const double sign=input?1.0:-1.0;
 for(std::size_t d=0;d<D;++d) {
  double feedback=0; for(std::size_t ch=0;ch<C;++ch) feedback+=k.weights[ch]*n.chi[ch*D+d];
  n.u[d]=s.u[d]+c.dt*(-c.leak_rate*s.u[d]+x[d]+sign*feedback);
 }
 if(!finite(n.u)||!finite(n.chi)||!std::isfinite(n.t)) throw std::runtime_error("digital memory state became non-finite");
 return n;
}
SequenceResult run_sequence(const MemoryConfig& c,const MemoryKernel& k,const MemoryState& initial,const std::vector<double>& xs,std::size_t steps) {
 validate(c,k,initial);
 const auto max=std::numeric_limits<std::size_t>::max();
 if(steps>max/c.state_dim||xs.size()!=steps*c.state_dim) throw std::invalid_argument("forcing sequence must be contiguous [steps,state_dim]");
 if(!finite(xs)||steps==max) throw std::invalid_argument("forcing sequence is non-finite or too large");
 const auto rows=steps+1, per_chi=chi_size(c);
 if(rows>max/c.state_dim||rows>max/per_chi) throw std::invalid_argument("trajectory size overflows");
 SequenceResult r; r.final_state=initial;
 r.times.reserve(rows); r.u_trajectory.reserve(rows*c.state_dim); r.chi_trajectory.reserve(rows*per_chi);
 max_abs(initial.u,r.diagnostics.max_abs_u); max_abs(initial.chi,r.diagnostics.max_abs_chi);
 r.times.push_back(initial.t); r.u_trajectory.insert(r.u_trajectory.end(),initial.u.begin(),initial.u.end());
 r.chi_trajectory.insert(r.chi_trajectory.end(),initial.chi.begin(),initial.chi.end());
 std::vector<double> row(c.state_dim);
 for(std::size_t i=0;i<steps;++i) {
  std::copy_n(xs.begin()+static_cast<std::ptrdiff_t>(i*c.state_dim),c.state_dim,row.begin());
  r.final_state=step(c,k,r.final_state,row); ++r.diagnostics.steps_executed;
  max_abs(r.final_state.u,r.diagnostics.max_abs_u); max_abs(r.final_state.chi,r.diagnostics.max_abs_chi);
  r.times.push_back(r.final_state.t);
  r.u_trajectory.insert(r.u_trajectory.end(),r.final_state.u.begin(),r.final_state.u.end());
  r.chi_trajectory.insert(r.chi_trajectory.end(),r.final_state.chi.begin(),r.final_state.chi.end());
 }
 r.diagnostics.all_finite=true; return r;
}
} // namespace gfe::digital
