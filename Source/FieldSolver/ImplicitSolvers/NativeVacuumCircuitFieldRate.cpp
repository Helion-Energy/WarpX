/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "NativeVacuumCircuitFieldRate.H"
#include "NativeBoundaryFieldRate.H"
#include "Circuit/CircuitCoupler.H"
#include "Circuit/CircuitCoupling.H"
#include "Circuit/Coils/FluxProbes.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <algorithm>
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
using Field=amrex::Array<MF,3>;
using FT=warpx::fields::FieldType;
namespace {
View V(Field& f){return {&f[0],&f[1],&f[2]};}
void Allocate(Field& out,View const& layout){
    for(int c=0;c<3;++c){
        out[c].define(layout[c]->boxArray(),layout[c]->DistributionMap(),1,layout[c]->nGrowVect());
        out[c].setVal(0.);
    }
}
void ScaleAllocation(MF& field,amrex::Real factor){
    for(amrex::MFIter it(field);it.isValid();++it){
        auto const a=field.array(it);
        amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)*=factor;});
    }
}
bool Collective(bool value){amrex::ParallelDescriptor::ReduceBoolAnd(value);return value;}
}
namespace warpx::darwin {
NativeVacuumCircuitFieldRate::NativeVacuumCircuitFieldRate(WarpX& w,CircuitCoupler& c)
    :m_warp(w),m_coupler(c),m_native(w,c)
{
    auto const electric=w.m_fields.get_alldirs(FT::Efield_fp,0);
    Allocate(m_zero,electric);Allocate(m_e,electric);
    Allocate(m_reference,electric);Allocate(m_e_reference,electric);
    Allocate(m_a,w.m_fields.get_alldirs("hybrid_A_fp",0));
    Allocate(m_b,w.m_fields.get_alldirs(FT::Bfield_fp,0));
    Allocate(m_j,w.m_fields.get_alldirs(FT::hybrid_current_fp_plasma,0));
}
bool NativeVacuumCircuitFieldRate::Prepare(amrex::Real time,Mask const& correction,
    Mask const& trace,Options const& options,std::string& error)
{
    m_ready=false;m_coupler.InvalidateNativeEndpointRate();m_columns.clear();m_h.clear();
    auto* subsystem=m_warp.get_pointer_CircuitCoupling();
    auto& model=*m_warp.get_pointer_HybridPICModel();
    if(!Collective(subsystem && subsystem->Coupler()==&m_coupler &&
        model.m_add_external_fields && model.m_external_vector_potential &&
        !options.magnetic && !options.flux_only && std::isfinite(time))){
        error="Native vacuum/circuit response requires an electric, native circuit field context";return false;
    }
    auto const& external=*model.m_external_vector_potential;
    auto const& coils=subsystem->GetCoilSet();
    auto const electric=m_warp.m_fields.get_alldirs(FT::Efield_fp,0);
    auto const magnetic=m_warp.m_fields.get_alldirs(FT::Bfield_fp,0);
    m_h.resize(static_cast<std::size_t>(coils.size())*external.nFields());
    for(int f=0;f<external.nFields();++f){
        for(int c=0;c<3;++c){
            auto const& unit=*m_warp.m_fields.get(external.FieldName(f)+"_Aext",ablastr::fields::Direction{c},0);
            if(!Collective(unit.boxArray()==m_reference[c].boxArray() &&
                unit.DistributionMap()==m_reference[c].DistributionMap() &&
                unit.nGrowVect().allGE(m_reference[c].nGrowVect()))){
                error="Native vacuum/circuit unit field layout changed";return false;
            }
            if (!unit.is_finite(0, 1, unit.nGrowVect())) {
                error="Nonfinite native vacuum/circuit unit field";return false;
            }
            MF::Copy(m_reference[c],unit,0,0,1,m_reference[c].nGrowVect());
            MF::Copy(m_e_reference[c],unit,0,0,1,m_e_reference[c].nGrowVect());
            ScaleAllocation(m_e_reference[c],-1.);
        }
        // A field basis can be arbitrarily normalized by its reference current.
        // Solve a unit-amplitude RHS so the absolute curl tolerance does not
        // depend on that arbitrary normalization, then restore its units.
        amrex::Real amplitude=0.;
        for(auto const& field:m_e_reference){amplitude=std::max(amplitude,field.norm0());}
        if(!std::isfinite(amplitude)) {error="Nonfinite native vacuum/circuit unit field";return false;}
        if(amplitude==0.)amplitude=1.;
        for(auto& field:m_e_reference){ScaleAllocation(field,1./amplitude);}
        // The native core takes PEC wall nodes from the frozen origin.
        // Its reference argument supplies only affine PMC ghost images.
        // Seed only physical PEC nodes; keep plasma and free vacuum baseline zero.
        auto const& geometry=m_warp.Geom(0);
        for(int c=0;c<3;++c){
            m_e[c].setVal(0.);
            auto const domain=amrex::convert(geometry.Domain(),m_e[c].ixType());
            auto const lo=domain.smallEnd(),hi=domain.bigEnd();
            auto const node=m_e[c].ixType().toIntVect();
            auto const low=options.pmc_lo,high=options.pmc_hi;
            amrex::GpuArray<int,AMREX_SPACEDIM> periodic{};
            for(int d=0;d<AMREX_SPACEDIM;++d)periodic[d]=geometry.isPeriodic(d);
            for(amrex::MFIter it(m_e[c]);it.isValid();++it){
                auto const e=m_e[c].array(it);auto const ref=m_e_reference[c].const_array(it);
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    int p[]{i,j,k};bool wall=false;
                    for(int d=0;d<AMREX_SPACEDIM;++d){bool lower=p[d]==lo[d];
#if defined(WARPX_DIM_RZ)
                        if(d==0)lower=false;
#endif
                        wall=wall||(!periodic[d]&&node[d]&&((lower&&!low[d])||(p[d]==hi[d]&&!high[d])));
                    }
#if defined(WARPX_DIM_RZ)
                    if(c==1&&i==0)wall=false;
#endif
                    if(wall)e(i,j,k)=ref(i,j,k);
                });
            }
        }
        auto column=std::make_unique<Field>();Allocate(*column,electric);
        DarwinVacuumAffineResponse core(m_warp,electric,magnetic,options);
        if(!core.Freeze(correction,trace,V(m_e),V(m_e_reference))){
            error="Native vacuum/circuit unit response freeze failed";return false;
        }
        auto result=core.Recover(V(*column));
        if(!result){
            error="Native vacuum/circuit unit response failed: "+std::to_string(static_cast<int>(result.failure));return false;
        }
        for(int c=0;c<3;++c){
            auto& field=(*column)[c];
            for(amrex::MFIter it(field);it.isValid();++it){
                auto const y=field.array(it);auto const active=correction[c]->const_array(it);
                amrex::ParallelFor(it.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){
                    y(i,j,k)=active(i,j,k)?amplitude*y(i,j,k):0.;
                });
            }
            field.OverrideSync(m_warp.Geom(0).periodicity());field.FillBoundary(m_warp.Geom(0).periodicity());
        }
        NativeBoundaryFieldRate(m_warp,V(*column),V(m_reference),V(m_a),V(m_b),V(m_j));
        for(int port=0;port<coils.size();++port){
            m_h[static_cast<std::size_t>(port)*external.nFields()+f]=
                warpx::circuit::DiskFluxLinkage(coils.coil(port),m_b[2]);
        }
        m_columns.push_back(std::move(column));
    }
    if(!m_native.Prepare(time,error,&m_h)){return false;}
    m_ready=true;Apply(V(m_zero),true);error.clear();return true;
}
bool NativeVacuumCircuitFieldRate::Ready() const noexcept{return m_ready&&m_native.Ready();}
void NativeVacuumCircuitFieldRate::Apply(View const& base_electric,bool affine)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Ready(),"Native vacuum/circuit context is stale or unprepared");
    // The prepared port matrix already includes the Y feedback. Its input
    // must be the base completion, not the field completed below.
    m_native.Apply(base_electric,affine);
    auto const* factors=m_coupler.EndpointFieldRates().data();
    for(int c=0;c<3;++c){
        AMREX_ALWAYS_ASSERT(base_electric[c]!=&m_e[c]);
        MF::Copy(m_e[c],*base_electric[c],0,0,1,m_e[c].nGrowVect());
        for(std::size_t f=0;f<m_columns.size();++f){
            auto const& column=(*m_columns[f])[c];
            for(amrex::MFIter it(m_e[c]);it.isValid();++it){
                auto const e=m_e[c].array(it);auto const y=column.const_array(it);
                amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){e(i,j,k)+=factors[f]*y(i,j,k);});
            }
        }
        MF::Copy(m_reference[c],m_native.ElectricReference(c),0,0,1,m_reference[c].nGrowVect());
        ScaleAllocation(m_reference[c],-1.);
    }
    NativeBoundaryFieldRate(m_warp,V(m_e),V(m_reference),V(m_a),V(m_b),V(m_j));
}
}
