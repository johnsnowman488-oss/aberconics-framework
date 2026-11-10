#include "gfe/gfe_c_api.h"
#include "gfe/gfe_digital_memory.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
bool eq(double a,double b){return std::fabs(a-b)<=1e-12*(1+std::max(std::fabs(a),std::fabs(b)));}
void req(bool x,const char* m){if(!x)throw std::runtime_error(m);}
void core_forms(){
 for(auto f:{gfe::digital::MemoryForm::InputDriven,gfe::digital::MemoryForm::NegativeFeedback}){
  gfe::digital::MemoryConfig c{4,3,.05,1.0,f};gfe::digital::MemoryKernel k{{2,.5,.1},{.04,.08,.12}};
  gfe::digital::MemoryState s{{.2,-.1,.7,.05},{.01,.02,.03,.04,-.02,.01,.05,-.01,.03,.02,-.04,.01},.25};
  std::vector<double>x{.5,0,-.25,.1};auto n=gfe::digital::step(c,k,s,x);
  const auto& src=f==gfe::digital::MemoryForm::InputDriven?x:s.u;double sign=f==gfe::digital::MemoryForm::InputDriven?1:-1;
  for(std::size_t d=0;d<4;++d){double fb=0;for(std::size_t ch=0;ch<3;++ch){double chi=s.chi[ch*4+d]+c.dt*(-k.gamma[ch]*s.chi[ch*4+d]+src[d]);req(eq(n.chi[ch*4+d],chi),"chi update mismatch");fb+=k.weights[ch]*chi;}
   req(eq(n.u[d],s.u[d]+c.dt*(-c.leak_rate*s.u[d]+x[d]+sign*fb)),"u update mismatch");}
  req(eq(n.t,.30),"time mismatch");
 }
}
void sequence_and_abi(){
 gfe::digital::MemoryConfig c{3,2,.02,1.25,gfe::digital::MemoryForm::InputDriven};gfe::digital::MemoryKernel k{{1.2,.15},{.1,.2}};
 gfe::digital::MemoryState s{{.1,-.2,.3},std::vector<double>(6),1.0};std::vector<double>x{1,0,0,0,1,0,0,0,1,0,0,0};
 auto r=gfe::digital::run_sequence(c,k,s,x,4);req(r.times.size()==5&&r.u_trajectory.size()==15&&r.chi_trajectory.size()==30,"trajectory sizes");
 auto rep=s;for(std::size_t i=0;i<4;++i)rep=gfe::digital::step(c,k,rep,{x[i*3],x[i*3+1],x[i*3+2]});
 for(int i=0;i<3;++i)req(eq(r.final_state.u[i],rep.u[i]),"sequence parity");
 gfe_c_digital_memory_config cc{2,2,.05,1.0,GFE_C_DIGITAL_MEMORY_INPUT_DRIVEN};
 const double gamma[]{2,.5},w[]{.04,.08},u[]{.2,-.1},chi[]{.01,.02,-.03,.04},force[]{.3,-.2};
 gfe_c_digital_memory_kernel_view kv{gamma,2,w,2};gfe_c_digital_memory_state_view sv{u,2,chi,4,.5};
 double ou[2]{},oc[4]{},ot=0;std::size_t ous=0,ocs=0;gfe_c_digital_memory_state_mut_view ov{ou,2,&ous,oc,4,&ocs,&ot};char e[256]{};
 req(gfe_c_digital_step(&cc,&kv,&sv,force,2,&ov,e,sizeof(e))==GFE_C_STATUS_OK,e);
 req(ous==2&&ocs==4&&eq(ot,.55),"step ABI shape/time");
 const double forces[]{.3,-.2,0,.1};double fu[2]{},fc[4]{},ft=0;std::size_t fus=0,fcs=0;
 gfe_c_digital_memory_state_mut_view fv{fu,2,&fus,fc,4,&fcs,&ft};
 double ts[3]{},tu[6]{},tc[12]{};std::size_t tss=0,tus=0,tcs=0;
 gfe_c_digital_memory_trajectory_mut_view tv{ts,3,&tss,tu,6,&tus,tc,12,&tcs};gfe_c_digital_memory_diagnostics dg{};
 req(gfe_c_digital_run_sequence(&cc,&kv,&sv,forces,2,4,&fv,&tv,&dg,e,sizeof(e))==GFE_C_STATUS_OK,e);
 req(dg.steps_executed==2&&tss==3&&tus==6&&tcs==12&&eq(ts[2],.6),"sequence ABI output");
 double tinyu[1]{},tinyc[1]{},tinyt=0;std::size_t tuz=0,tcz=0;gfe_c_digital_memory_state_mut_view tiny{tinyu,1,&tuz,tinyc,1,&tcz,&tinyt};
 req(gfe_c_digital_step(&cc,&kv,&sv,force,2,&tiny,e,sizeof(e))==GFE_C_STATUS_BUFFER_TOO_SMALL&&tuz==2&&tcz==4,"buffer-too-small contract");
 cc.state_dim=0;
 req(gfe_c_digital_step(&cc,&kv,&sv,force,0,&ov,e,sizeof(e))==GFE_C_STATUS_INVALID_ARGUMENT,"zero state_dim must be rejected safely");
}
}
int main(){try{core_forms();sequence_and_abi();std::cout<<"gfe_digital_memory_test: all checks passed\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
