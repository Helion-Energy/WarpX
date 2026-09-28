/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Reduce.H>
#include <iomanip>
#include <string>

using amrex::Real;
using MF = amrex::MultiFab;
using Owned = std::array<std::unique_ptr<MF>, 3>;
using View = ablastr::fields::VectorField;
View view(Owned& f) { return {f[0].get(), f[1].get(), f[2].get()}; }
Real difference(MF const& a, MF const& b) {
    MF d(a.boxArray(),a.DistributionMap(),a.nComp(),a.nGrowVect());
    MF::Copy(d,a,0,0,a.nComp(),a.nGrowVect());
    MF::Subtract(d,b,0,0,a.nComp(),a.nGrowVect());
    return d.norm0(0,a.nComp(),a.nGrowVect());
}
Real checksum(MF const& f) {
    amrex::ReduceOps<amrex::ReduceOpSum> op; amrex::ReduceData<Real> data(op);
    for(amrex::MFIter mfi(f);mfi.isValid();++mfi) {
        auto a=f.const_array(mfi);
        op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            return amrex::GpuTuple<Real>{a(i,j,k)*(1.+.013*i+.007*j+.003*k)}; });
    }
    Real x=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealSum(x);return x;
}
void run(WarpX& w) {
    using warpx::fields::FieldType;
    std::string test="all";bool nodal=false;
    amrex::ParmParse pp("eta_capture");pp.query("case",test);pp.query("nodal",nodal);
#ifdef WARPX_DIM_RZ
    AMREX_ALWAYS_ASSERT(!nodal);
#endif
    auto& m=*w.get_pointer_HybridPICModel();auto const& g=w.Geom(0);
    auto const original_grid=WarpX::grid_type;
    m.m_include_electron_inertia=false;m.m_include_hyper_resistivity_term=false;
    m.m_visc_in_ohms_law=false;m.m_add_external_fields=false;
    m.m_has_per_species_eta=false;m.m_esolve_curlcurl=false;m.m_esolve_tensor=false;
    m.m_pec_conductor_wall_rows=false;m.m_hyper_resistivity_curlcurl=false;
    m.m_include_electron_pressure_term=true;m.m_include_hall_term=true;
    m.m_n_floor=2./PhysConst::q_e;m.m_n_floor_smooth_width=0.;
    auto er=w.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto br=w.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    auto cells=amrex::convert(er[0]->boxArray(),amrex::IntVect(0));auto const& dm=er[0]->DistributionMap();
    Owned E,off,J,Ji,B,ER,EH,referenceJ,referenceJi,referenceB;
    std::array<std::unique_ptr<amrex::iMultiFab>,3> masks;
    for(int c=0;c<3;++c) {
        auto const eba=nodal ? amrex::convert(cells,amrex::IntVect(1)) : er[c]->boxArray();
        auto const bba=nodal ? amrex::convert(cells,amrex::IntVect(1)) : br[c]->boxArray();
        for(auto* f:{&E,&off,&J,&Ji,&ER,&EH,&referenceJ,&referenceJi})
            (*f)[c]=std::make_unique<MF>(eba,dm,1,2);
        B[c]=std::make_unique<MF>(bba,dm,1,2);referenceB[c]=std::make_unique<MF>(bba,dm,1,2);
        masks[c]=std::make_unique<amrex::iMultiFab>(eba,dm,1,2);
    }
    auto const& native_te=m.ElectronTemperatureForSolve(0);
    auto const& native_pe=m.ElectronPressureForSolve(0);
    MF Te(native_te.boxArray(),dm,1,native_te.nGrowVect());
    MF Pe(native_pe.boxArray(),dm,1,native_pe.nGrowVect());
    MF rho(Te.boxArray(),dm,1,2);Pe.setVal(2.);Te.setVal(7000.);
    m.SetElectronThermalTrial(0,Te,Pe);
    std::unique_ptr<FiniteDifferenceSolver> nodal_solver;
    if(nodal) {
        WarpX::grid_type=ablastr::utils::enums::GridType::Collocated;
        m.Ex_IndexType=m.Ey_IndexType=m.Ez_IndexType={1,1,1};
        m.Jx_IndexType=m.Jy_IndexType=m.Jz_IndexType={1,1,1};
        m.Bx_IndexType=m.By_IndexType=m.Bz_IndexType={1,1,1};
        std::array<Real,3> dx{g.CellSize(0),g.CellSize(1),1.};
#if AMREX_SPACEDIM == 3
        dx[2]=g.CellSize(2);
#endif
        nodal_solver=std::make_unique<FiniteDifferenceSolver>(ElectromagneticSolverAlgo::HybridPIC,dx,WarpX::grid_type);
    }
    auto* solver=nodal_solver ? nodal_solver.get() : w.get_pointer_fdtd_solver_fp(0);
    amrex::Parser eta("2+.25*rho+.5*J+t"),eta_te("2+.25*rho+.5*J+.001*Te+t"),constant("2"),hyper(".003");
    eta.registerVariables({"rho","J","t"});eta_te.registerVariables({"rho","J","Te","t"});
    constant.registerVariables({"rho","J","t"});hyper.registerVariables({"rho","B"});
    m.m_eta_h=hyper.compile<2>();m.m_hyper_resistivity_has_B_dependence=false;
    w.sett_new(0,.125);
    int passed=0;
    for(std::string const name:{"constant","rho_j_time","temperature","temperature_refresh","spatial_temperature","inactive","push_relaxation","faraday","subfloor","zero_rho","negative_rho","end_region","hyper","eb_rows"}) {
        if(test!="all" && test!=name && test!="alias" && test!="guards")continue;
        bool const spatial_te=name=="spatial_temperature";
        bool const with_te=name=="temperature"||name=="temperature_refresh"||spatial_te;
        bool const polynomial=name=="hyper";
        bool const masked=name=="eb_rows";
        bool const include=name!="inactive"&&name!="push_relaxation";
        bool const dependent=name!="constant"&&name!="hyper";
        Real const raw=name=="subfloor" ? .25 : name=="zero_rho" ? 0. : name=="negative_rho" ? -.25 : 4.;
        Real const temperature=name=="temperature_refresh" ? 9000. : 7000.;
        rho.setVal(raw);Te.setVal(temperature);
        if(spatial_te)for(amrex::MFIter mfi(Te);mfi.isValid();++mfi) {
            auto a=Te.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {a(i,j,k)=temperature+13.*i+7.*j+3.*k;});
        }
        MF saved_rho(rho.boxArray(),dm,1,rho.nGrowVect()),saved_te(Te.boxArray(),dm,1,Te.nGrowVect()),saved_pe(Pe.boxArray(),dm,1,Pe.nGrowVect());
        MF::Copy(saved_rho,rho,0,0,1,rho.nGrowVect());MF::Copy(saved_te,Te,0,0,1,Te.nGrowVect());MF::Copy(saved_pe,Pe,0,0,1,Pe.nGrowVect());
        m.m_eta=dependent ? eta.compile<3>() : constant.compile<3>();m.m_eta_te=eta_te.compile<4>();
        m.m_resistivity_has_J_dependence=dependent;m.m_resistivity_has_Te_dependence=with_te;
        m.m_include_temperature_relaxation=name=="push_relaxation";
        m.m_include_hyper_resistivity_term=polynomial;
        m.m_holmstrom_vacuum_region=name=="subfloor"||name=="zero_rho"||name=="negative_rho";
        m.m_end_region.width={.3,.3};m.m_end_region.rolloff={0.,0.};
        m.m_end_region.resistivity=name=="end_region" ? 7. : 0.;
        auto const dx=g.CellSizeArray(),lo=g.ProbLoArray();
        for(int c=0;c<3;++c) {
            E[c]->setVal(-123.);off[c]->setVal(-123.);ER[c]->setVal(999.);EH[c]->setVal(888.);
            Ji[c]->setVal(.2*(c+1));B[c]->setVal(.01*(c+1));
            auto const iv=J[c]->ixType().toIntVect();Real const jc=c+3.;
            for(amrex::MFIter mfi(*J[c]);mfi.isValid();++mfi) {
                auto a=J[c]->array(mfi);auto mask=masks[c]->array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    int const ix[3]={i,j,k};
                    Real const z=lo[AMREX_SPACEDIM-1]+(ix[AMREX_SPACEDIM-1]+.5*(1-iv[AMREX_SPACEDIM-1]))*dx[AMREX_SPACEDIM-1];
                    a(i,j,k)=jc*(polynomial ? 1.+.02*z*z : 1.);
                    mask(i,j,k)=masked && i%5==0 && j%3==0 ? 0 : 1;
                });
            }
            MF::Copy(*referenceJ[c],*J[c],0,0,1,2);MF::Copy(*referenceJi[c],*Ji[c],0,0,1,2);MF::Copy(*referenceB[c],*B[c],0,0,1,2);
        }
        if(masked)amrex::ParmParse("warpx").add("eb_implicit_function",std::string("1"));
        auto e=view(E),eo=view(off),j=view(J),ji=view(Ji),b=view(B),capture=view(ER),eh=view(EH);
        if(test=="alias")capture[0]=j[0];
        if(test=="guards") {ER[0]=std::make_unique<MF>(E[0]->boxArray(),dm,1,0);capture=view(ER);}
        solver->HybridPICSolveE(eo,j,ji,b,rho,Pe,masks,0,&m,name=="faraday",include);
#ifndef ETA_CAPTURE_REFERENCE
        solver->HybridPICSolveE(e,j,ji,b,rho,Pe,masks,0,&m,name=="faraday",include,&eh,nullptr,nullptr,&capture);
#else
        // Builds against the immutable pre-change library to prove that the
        // default-null branch preserves the original physical field values.
        solver->HybridPICSolveE(e,j,ji,b,rho,Pe,masks,0,&m,name=="faraday",include,&eh);
#endif
        if(masked)amrex::ParmParse("warpx").remove("eb_implicit_function");
        Real physical_error=0.,mirror_error=0.,input_error=std::max({difference(rho,saved_rho),difference(Te,saved_te),difference(Pe,saved_pe)}),eh_norm=0.,field_sum=0.;
        for(int c=0;c<3;++c) {
            physical_error=std::max(physical_error,difference(*E[c],*off[c]));
            input_error=std::max({input_error,difference(*J[c],*referenceJ[c]),difference(*Ji[c],*referenceJi[c]),difference(*B[c],*referenceB[c])});
            field_sum+=(c+1)*checksum(*E[c]);eh_norm=std::max(eh_norm,EH[c]->norm0());
#ifndef ETA_CAPTURE_REFERENCE
            amrex::ReduceOps<amrex::ReduceOpMax> op;amrex::ReduceData<Real> data(op);
            for(amrex::MFIter mfi(*ER[c]);mfi.isValid();++mfi) {
                auto a=ER[c]->const_array(mfi);auto curr=J[c]->const_array(mfi);auto mask=masks[c]->const_array(mfi);auto valid=mfi.validbox();
                Real const coefficient=dependent ? 2.+.25*raw+.5*std::sqrt(50.)+.125+(with_te ? .001*temperature : 0.) : 2.;
                auto const stagger=ER[c]->ixType().toIntVect();
                bool const active=include||m.m_include_temperature_relaxation;
                op.eval(mfi.fabbox(),data,[=] AMREX_GPU_DEVICE(int i,int jj,int k) {
                    bool applies=valid.contains(amrex::IntVect(AMREX_D_DECL(i,jj,k)))&&active&&mask(i,jj,k)!=0;
#ifdef WARPX_DIM_RZ
                    if(c==1&&i==0)applies=false;
#endif
                    Real t_increment=0.;
                    if(spatial_te) {
                        t_increment=13.*(i+.5*(1-stagger[0]))+7.*(jj+.5*(1-stagger[1]));
#if AMREX_SPACEDIM == 3
                        t_increment+=3.*(k+.5*(1-stagger[2]));
#endif
                    }
                    Real const expected=applies ? (coefficient+.001*t_increment)*curr(i,jj,k) : 0.;
                    return amrex::GpuTuple<Real>{std::abs(a(i,jj,k)-expected)}; });
            }
            Real local=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceRealMax(local);mirror_error=std::max(mirror_error,local);
#endif
        }
        if(name=="end_region") {
            m.m_end_region.resistivity=0.;
            solver->HybridPICSolveE(eo,j,ji,b,rho,Pe,masks,0,&m,false,true);
            Real end_difference=0.;for(int c=0;c<3;++c)end_difference=std::max(end_difference,difference(*E[c],*off[c]));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(end_difference-35.)<1.e-12,"end exclusion test needs its actual nonzero addition");
            m.m_end_region.resistivity=7.;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(physical_error<2.e-12,"capture changed physical field");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mirror_error<2.e-12,"global eta mirror / skipped or ghost row mismatch");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(input_error==0.,"field kernel mutated source inputs");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!polynomial||eh_norm>0.,"hyper exclusion test needs nonzero EH");
        amrex::Print()<<std::setprecision(17)<<"ETA_CAPTURE case="<<name<<" nodal="<<nodal<<" mirror_error="<<mirror_error<<" physical_error="<<physical_error<<" input_error="<<input_error<<" EH_norm="<<eh_norm<<" field_checksum="<<field_sum<<" PASS\n";
        ++passed;
    }
    AMREX_ALWAYS_ASSERT(passed>0);m.ClearElectronThermalTrials();WarpX::grid_type=original_grid;
}
int main(int argc,char** argv) {
    warpx::initialization::initialize_external_libraries(argc,argv);
    { auto& w=WarpX::GetInstance();w.InitData();run(w);WarpX::Finalize(); }
    warpx::initialization::finalize_external_libraries();
}
