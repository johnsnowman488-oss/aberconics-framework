#include "gfe/gfe_c_api.h"
#include "gfe/gfe_digital_memory.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void err(char* p,std::size_t n,const char* s){if(!p||!n)return;auto k=std::min(n-1,std::strlen(s));std::memcpy(p,s,k);p[k]='\0';}
int fail(const std::exception& e,char* p,std::size_t n){err(p,n,e.what());return dynamic_cast<const std::invalid_argument*>(&e)?GFE_C_STATUS_INVALID_ARGUMENT:GFE_C_STATUS_RUNTIME_ERROR;}
gfe::digital::MemoryConfig cfg(const gfe_c_digital_memory_config& x){
 gfe::digital::MemoryConfig c{x.state_dim,x.channel_count,x.dt,x.leak_rate,gfe::digital::MemoryForm::InputDriven};
 if(x.form==GFE_C_DIGITAL_MEMORY_NEGATIVE_FEEDBACK)c.form=gfe::digital::MemoryForm::NegativeFeedback;
 else if(x.form!=GFE_C_DIGITAL_MEMORY_INPUT_DRIVEN)throw std::invalid_argument("unknown digital memory form");
 return c;
}
gfe::digital::MemoryKernel kernel(const gfe_c_digital_memory_kernel_view& x,std::size_t C){
 if(!x.gamma||!x.weights||x.gamma_size!=C||x.weights_size!=C)throw std::invalid_argument("kernel pointers/sizes invalid");
 return {{x.gamma,x.gamma+C},{x.weights,x.weights+C}};
}
gfe::digital::MemoryState state(const gfe_c_digital_memory_state_view& x,const gfe::digital::MemoryConfig& c){
 if(!c.state_dim||!c.channel_count||!x.u||!x.chi||x.u_size!=c.state_dim||c.channel_count>std::numeric_limits<std::size_t>::max()/c.state_dim||x.chi_size!=c.channel_count*c.state_dim)throw std::invalid_argument("state pointers or dimensions invalid");
 return {{x.u,x.u+x.u_size},{x.chi,x.chi+x.chi_size},x.t};
}
int write_state(const gfe::digital::MemoryState& x,gfe_c_digital_memory_state_mut_view* y,char* e,std::size_t n){
 if(!y||!y->u||!y->chi||!y->u_size||!y->chi_size||!y->t){err(e,n,"output state view contains a null pointer");return GFE_C_STATUS_INVALID_ARGUMENT;}
 *y->u_size=x.u.size(); *y->chi_size=x.chi.size();
 if(y->u_capacity<x.u.size()||y->chi_capacity<x.chi.size()){err(e,n,"output state buffer too small");return GFE_C_STATUS_BUFFER_TOO_SMALL;}
 std::copy(x.u.begin(),x.u.end(),y->u);std::copy(x.chi.begin(),x.chi.end(),y->chi);*y->t=x.t;return GFE_C_STATUS_OK;
}
}
extern "C" int gfe_c_digital_step(const gfe_c_digital_memory_config* c,const gfe_c_digital_memory_kernel_view* k,
 const gfe_c_digital_memory_state_view* s,const double* x,size_t xn,gfe_c_digital_memory_state_mut_view* out,char* e,size_t en){
 try{if(!c||!k||!s||!x||!out)throw std::invalid_argument("required digital step argument is null");
 auto C=cfg(*c);if(xn!=C.state_dim)throw std::invalid_argument("forcing_size must equal state_dim");
 auto K=kernel(*k,C.channel_count);auto S=state(*s,C);std::vector<double>X(x,x+xn);
 return write_state(gfe::digital::step(C,K,S,X),out,e,en);
 }catch(const std::exception& ex){return fail(ex,e,en);}catch(...){err(e,en,"unknown digital step error");return GFE_C_STATUS_RUNTIME_ERROR;}
}
extern "C" int gfe_c_digital_run_sequence(const gfe_c_digital_memory_config* c,const gfe_c_digital_memory_kernel_view* k,
 const gfe_c_digital_memory_state_view* s,const double* xs,size_t steps,size_t xn,gfe_c_digital_memory_state_mut_view* final,
 gfe_c_digital_memory_trajectory_mut_view* traj,gfe_c_digital_memory_diagnostics* diag,char* e,size_t en){
 try{
 if(!c||!k||!s||!final||!diag)throw std::invalid_argument("required digital sequence argument is null");
 auto C=cfg(*c);if(!C.state_dim||steps>std::numeric_limits<std::size_t>::max()/C.state_dim)throw std::invalid_argument("forcing sequence size overflows");
 auto need=steps*C.state_dim;if(xn!=need||(need&&!xs))throw std::invalid_argument("forcing sequence shape or pointer invalid");
 auto K=kernel(*k,C.channel_count);auto S=state(*s,C);std::vector<double>X=need?std::vector<double>(xs,xs+need):std::vector<double>{};
 auto R=gfe::digital::run_sequence(C,K,S,X,steps);
 if(!final->u||!final->chi||!final->u_size||!final->chi_size||!final->t)throw std::invalid_argument("final state output pointers invalid");
 *final->u_size=R.final_state.u.size();*final->chi_size=R.final_state.chi.size();
 bool small=final->u_capacity<R.final_state.u.size()||final->chi_capacity<R.final_state.chi.size();
 std::size_t rows=R.times.size(),perchi=C.channel_count*C.state_dim;
 if(traj){
  if(!traj->t||!traj->u||!traj->chi||!traj->t_size||!traj->u_size||!traj->chi_size)throw std::invalid_argument("trajectory output pointers invalid");
  if(rows>std::numeric_limits<std::size_t>::max()/C.state_dim||rows>std::numeric_limits<std::size_t>::max()/perchi)throw std::invalid_argument("trajectory size overflows");
  *traj->t_size=rows;*traj->u_size=rows*C.state_dim;*traj->chi_size=rows*perchi;
  small=small||traj->t_capacity<rows||traj->u_capacity<rows*C.state_dim||traj->chi_capacity<rows*perchi;
 }
 if(small){err(e,en,"digital sequence output buffer too small");return GFE_C_STATUS_BUFFER_TOO_SMALL;}
 std::copy(R.final_state.u.begin(),R.final_state.u.end(),final->u);std::copy(R.final_state.chi.begin(),R.final_state.chi.end(),final->chi);*final->t=R.final_state.t;
 if(traj){std::copy(R.times.begin(),R.times.end(),traj->t);std::copy(R.u_trajectory.begin(),R.u_trajectory.end(),traj->u);std::copy(R.chi_trajectory.begin(),R.chi_trajectory.end(),traj->chi);}
 diag->steps_executed=R.diagnostics.steps_executed;diag->all_finite=R.diagnostics.all_finite?1:0;
 diag->max_abs_u=R.diagnostics.max_abs_u;diag->max_abs_chi=R.diagnostics.max_abs_chi;return GFE_C_STATUS_OK;
 }catch(const std::exception& ex){return fail(ex,e,en);}catch(...){err(e,en,"unknown digital sequence error");return GFE_C_STATUS_RUNTIME_ERROR;}
}
