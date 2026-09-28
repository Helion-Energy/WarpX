/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitIonElectricWork.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Particles/Pusher/UpdateMomentumBoris.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particle.H>
#include <AMReX_Random.H>
#include <AMReX_Print.H>
#include <memory>
using namespace warpx::thermal;
using C=IonElectricWorkComponent;
using P=amrex::ParticleReal;
using R=amrex::Real;
struct Tile {
    std::array<amrex::Gpu::DeviceVector<P>,10> data;
    amrex::Gpu::DeviceVector<std::uint64_t> ids;
    std::array<P,10> initial{};
    IonElectricWorkParticles view;
};
IonWorkPoint rotate(IonWorkPoint a,R theta) {
    return {P(a[0]*std::cos(theta)-a[1]*std::sin(theta)),
            P(a[0]*std::sin(theta)+a[1]*std::cos(theta)),a[2]};
}
void require(bool test,const char* text) { AMREX_ALWAYS_ASSERT_WITH_MESSAGE(test,text); }
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        amrex::ParmParse pp("test");
        std::string name="recorded"; int box_size=4;
        pp.query("case",name); pp.query("box",box_size);
        amrex::Box domain(amrex::IntVect(0),amrex::IntVect(15));
        amrex::RealBox physical({AMREX_D_DECL(0.,0.,0.)},{AMREX_D_DECL(1.,1.,1.)});
        int periodic[AMREX_SPACEDIM]={AMREX_D_DECL(0,0,0)};
#ifdef WARPX_DIM_RZ
        int coord=1;
#else
        int coord=0;
#endif
        amrex::Geometry geom(domain,&physical,coord,periodic);
        amrex::BoxArray cells(domain); cells.maxSize(box_size);
        amrex::DistributionMapping dm(cells);
        std::array<std::unique_ptr<amrex::MultiFab>,3> component,total;
        IonElectricWorkFields fields; fields.filled_ghosts=amrex::IntVect(3);
        for (int d=0;d<3;++d) {
            component[d]=std::make_unique<amrex::MultiFab>(amrex::convert(cells,amrex::IntVect::TheNodeVector()),dm,1,3);
            total[d]=std::make_unique<amrex::MultiFab>(component[d]->boxArray(),dm,1,3);
            fields.component[d]=component[d].get(); fields.total[d]=total[d].get();
        }
        auto fill=[&](R factor) {
            for (int d=0;d<3;++d) {
                for (amrex::MFIter mfi(*component[d]);mfi.isValid();++mfi) {
                    auto c=component[d]->array(mfi), f=total[d]->array(mfi);
                    auto dx=geom.CellSizeArray();
                    amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
#ifdef WARPX_DIM_RZ
                        R const r=i*dx[0], z=j*dx[1];
#else
                        R const r=i*dx[0], z=k*dx[2];
#endif
                        R value=d==0 ? 2000+40*r : (d==1 ? -1500+80*z : 700+55*r);
                        c(i,j,k)=.37*value*factor; f(i,j,k)=value;
                    });
                }
            }
        };
        // No host std::string in a device capture.
        fill(1.);
        std::vector<IonElectricWorkSpecies> species={
            {"D",3.3435837724e-27,PhysConst::q_e},
            {"T",5.0073567446e-27,PhysConst::q_e},
            {"alpha",6.6446573357e-27,2*PhysConst::q_e}};
        std::vector<std::unique_ptr<Tile>> owned;
        std::vector<IonElectricWorkParticles> views;
        R const dt=1e-8;
        std::vector<R> reference(3,0);
        for (amrex::MFIter mfi(*component[0]);mfi.isValid();++mfi) {
            for (int s=0;s<3;++s) {
                auto t=std::make_unique<Tile>();
                t->view.species=s;t->view.grid=mfi.index();t->view.tile=0;t->view.count=1;
                R r=(cells[mfi.index()].smallEnd(0)+1.6)*geom.CellSize(0);
#ifdef WARPX_DIM_RZ
                R z=(cells[mfi.index()].smallEnd(1)+1.4)*geom.CellSize(1);
#else
                R z=(cells[mfi.index()].smallEnd(2)+1.4)*geom.CellSize(2);
#endif
                if (name=="ghosts") { r=cells[mfi.index()].smallEnd(0)*geom.CellSize(0)+.05*geom.CellSize(0); }
                R theta=(name=="rotation" ? 1.81 : .43)+.11*s;
                IonWorkPoint position{P(r),0,P(z)};
                IonWorkPoint e{P(2000+40*r),P(-1500+80*z),P(700+55*r)};
                IonWorkPoint u0{2.1e6,3.2e5,-7.4e5}, b{.03,-.02,.15};
#ifdef WARPX_DIM_RZ
                position=rotate(position,theta);e=rotate(e,theta);u0=rotate(u0,theta);b=rotate(b,theta);
#else
                position[1]=(cells[mfi.index()].smallEnd(1)+1.3)*geom.CellSize(1);
#endif
                if(name=="cooling") { for(auto& u:u0) { u=-u; } }
                auto u1=u0;
                UpdateMomentumBoris(u1[0],u1[1],u1[2],e[0],e[1],e[2],b[0],b[1],b[2],
                    species[s].charge,species[s].mass,dt);
                long double s0=0,s1=0;
                for(int d=0;d<3;++d) {
                    t->initial[d]=position[d];t->initial[3+d]=u0[d];
                    t->initial[6+d]=.5*(u0[d]+u1[d]);
                    s0+=(long double)u0[d]*u0[d];s1+=(long double)u1[d]*u1[d];
                }
                t->initial[9]=1e5*(s+1);
                reference[s]+=R((long double)t->initial[9]*species[s].mass*(s1-s0)/
                    (std::sqrt(1+s1/(PhysConst::c*PhysConst::c))+std::sqrt(1+s0/(PhysConst::c*PhysConst::c))));
                if(name=="reconstruction") { t->initial[2]+=.015; }
                if(name=="invalid_particle") { t->initial[9]=-1; }
                for(int d=0;d<10;++d) {
                    t->data[d].resize(1);
                    amrex::Gpu::copy(amrex::Gpu::hostToDevice,&t->initial[d],&t->initial[d]+1,t->data[d].begin());
                }
                std::uint64_t id=0;amrex::ParticleIDWrapper{id}=1;
                t->ids.resize(1);amrex::Gpu::copy(amrex::Gpu::hostToDevice,&id,&id+1,t->ids.begin());
                t->view.idcpu=t->ids.data();t->view.weight=t->data[9].data();
                for(int d=0;d<3;++d) {
                    t->view.gather_position[d]=t->data[d].data();
                    t->view.old_momentum[d]=t->data[3+d].data();
                    t->view.midpoint_momentum[d]=t->data[6+d].data();
                }
                views.push_back(t->view);owned.push_back(std::move(t));
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(reference.data(),reference.size());
        auto contract=name=="reconstruction" ? IonElectricGatherContract::FinalMidpointReconstruction
                                               : IonElectricGatherContract::RecordedFinalGather;
        if(name=="metadata") { contract=IonElectricGatherContract::Unspecified; }
        if(name=="ghosts") { fields.filled_ghosts=amrex::IntVect(0); }
        if(name=="no_total") { fields.total={}; }
        if(name=="remainder") { fields.remainder=fields.component; }
        if(name=="partial_remainder") { fields.remainder[0]=fields.component[0]; }
        if(name=="invalid_field") { component[0]->setVal(std::numeric_limits<R>::quiet_NaN()); }
        amrex::ResetRandomSeed(314159);
        auto const before=amrex::Random();
        auto result=MeasureImplicitIonElectricWork(geom,fields,species,views,dt,contract);
        auto const after=amrex::Random();
        amrex::ResetRandomSeed(314159);
        require(before==amrex::Random() && after==amrex::Random(),"work evaluator consumed RNG");
        bool const negative=name=="metadata" || name=="ghosts" || name=="invalid_field" || name=="invalid_particle" || name=="partial_remainder";
        require(result.valid!=negative,"valid/invalid result mismatch");
        if(result.valid) {
            require(result.species.size()==3,"all charged species including alpha must be retained");
            auto again=MeasureImplicitIonElectricWork(geom,fields,species,views,dt,contract);
            require(again.species==result.species,"work evaluator replay changed");
            require(again.diagnostics==result.diagnostics,"work diagnostic replay changed");
            for(int s=0;s<3;++s) {
                auto const& x=result.species[s];
                auto const& diagnostic=result.diagnostics[s];
                using D=IonElectricWorkDiagnostic;
                require(diagnostic[D::OldKineticEnergy]>=0 && diagnostic[D::EndpointKineticEnergy]>=0,
                        "negative kinetic energy reference");
                require(diagnostic[D::MomentumRoundoffReference]>0,"missing momentum roundoff reference");
                require(diagnostic[D::ComponentCauchyWorkBound]>=x[C::ComponentWorkAbs]*(1-1e-14),
                        "component work exceeds gathered-field Cauchy bound");
                if(name=="remainder") {
                    require(result.has_remainder_field,"missing optional applied remainder");
                    require(diagnostic[D::RemainderWork]==x[C::ComponentWork] &&
                            diagnostic[D::RemainderWorkAbs]==x[C::ComponentWorkAbs] &&
                            diagnostic[D::NonrelativisticRemainderWork]==x[C::NonrelativisticWork],
                            "same gathered component/remainder work differs");
                    require(diagnostic[D::RemainderCauchyWorkBound]==diagnostic[D::ComponentCauchyWorkBound],
                            "same gathered component/remainder bound differs");
                } else {
                    require(!result.has_remainder_field && diagnostic[D::RemainderWork]==0 &&
                            diagnostic[D::RemainderCauchyWorkBound]==0,"absent remainder has work");
                }
                R const scale=std::abs(reference[s]);
                if(name=="reconstruction") {
                    require(std::abs(x[C::TotalWorkDefect])>1e-5*scale,"updated midpoint erased gather defect");
                } else {
                    require(std::abs(x[C::ComponentWork]-.37*reference[s])<4e-12*scale,"native E gather/Boris work identity");
                    require(std::abs(x[C::KineticChange]-reference[s])<4e-12*scale,"relativistic deltaK oracle");
                    if(name!="no_total") { require(x[C::TotalWorkDefectAbs]<5e-12*scale,"Boris total work closure"); }
                }
            }
            if(name=="derivative") {
                R const h=1e-4;
                fill(std::pow(1+h,-1.5));
                auto plus=MeasureImplicitIonElectricWork(geom,fields,species,views,dt,contract);
                fill(std::pow(1-h,-1.5));
                auto minus=MeasureImplicitIonElectricWork(geom,fields,species,views,dt,contract);
                for(int s=0;s<3;++s) {
                    R const derivative=(plus.species[s][C::ComponentWork]-minus.species[s][C::ComponentWork])/(2*h);
                    require(std::abs(derivative/(-1.5*result.species[s][C::ComponentWork])-1)<2e-8,"temperature-dependent eta directional work derivative");
                }
            }
        }
        for(auto const& t:owned) for(int d=0;d<10;++d) {
            P value;amrex::Gpu::copy(amrex::Gpu::deviceToHost,t->data[d].begin(),t->data[d].end(),&value);
            require(value==t->initial[d],"accepted particle input modified");
        }
        if(name=="cancellation") {
            IonWorkPoint e{2000,-1500,700}, v{2.1e6,3.2e5,-7.4e5}, plus{}, minus{}, negv{};
            R const m=species[0].mass,q=species[0].charge;
            for(int d=0;d<3;++d) {
                R const a=.5*q*dt*e[d]/m;
                plus[d]=v[d]-a; minus[d]=-v[d]-a; negv[d]=-v[d];
            }
            auto a=ComputeIonElectricWork(m,q,1e5,dt,plus,v,e,e,true);
            auto b=ComputeIonElectricWork(m,q,1e5,dt,minus,negv,e,e,true);
            require(a[C::ComponentWork]>0 && b[C::ComponentWork]<0,"signed electric work");
            require(std::abs(a[C::ComponentWork]+b[C::ComponentWork])<1e-13*a[C::ComponentWorkAbs],"signed electric cancellation");
            require(a[C::ComponentWorkAbs]+b[C::ComponentWorkAbs]>a[C::ComponentWorkAbs],"absolute electric work must survive cancellation");
        }
        if(name=="matched_limit") {
            auto defect=[](R h,R tune) {
                R const v0=2.1e6,ve=-3e5,m=3.3435837724e-27;
                R const v1=ve+(v0-ve)*(1+.5*h)/(1-.5*h);
                R const v2=ve+std::exp(-tune*h)*(v1-ve);
                R const electric=.5*m*(v1-v0)*(v1+v0);
                R const bulk=.5*m*(v2-v1)*(v2+v1);
                return std::abs(electric+bulk);
            };
            R const a=defect(.02,1),b=defect(.01,1),c=defect(.005,1);
            require(a/b>7.9 && b/c>7.9,"matched eta/nu midpoint-plus-OU bulk order");
            require(defect(.01,.8)>1e3*b,"independently tuned rate must expose first-order bulk defect");
            amrex::Print()<<"matched ratios "<<a/b<<" "<<b/c<<" tuned defect "<<defect(.01,.8)<<"\n";
        }
        amrex::Print()<<"PASS ion_electric_work "<<name<<" box "<<box_size;
        if(result.valid) { amrex::Print()<<" work "<<result.species[0][C::ComponentWork]<<" defect "<<result.species[0][C::TotalWorkDefect]; }
        amrex::Print()<<"\n";
    }
    amrex::Finalize();
}
