/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef NATIVE_RADIAL_POSITIVE_DEPOSIT_ROUNDOFF_H
#define NATIVE_RADIAL_POSITIVE_DEPOSIT_ROUNDOFF_H

#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ShapeFactors.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include <AMReX_Reduce.H>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace radial_fixture {
// Fixture-only bound for TWO deposits of byte-identical particles. It does
// not replace continuity, native endpoint, image or physical charge gates.
// The README derives k and the two-positive-sums relative error bound.
class PositiveDepositRoundoff {
    amrex::Long m_particles_per_fab=0;
    amrex::Long m_donors=0;
    amrex::Long m_operations=0;
    amrex::Real m_relative=0;
public:
    PositiveDepositRoundoff(WarpX& w,amrex::MultiFab const& layout) {
        using Real=amrex::Real;
        static_assert(sizeof(Real)==sizeof(double));
        auto const& g=w.Geom(0);
        AMREX_ALWAYS_ASSERT(w.finestLevel()==0 && WarpX::nox==3 && WarpX::n_rz_azimuthal_modes==1);
        AMREX_ALWAYS_ASSERT(!WarpX::use_filter && !WarpX::do_shared_mem_charge_deposition &&
                            !WarpX::do_single_precision_comms && !g.isAnyPeriodic());
        AMREX_ALWAYS_ASSERT(layout.nComp()==1 && layout.ixType().nodeCentered());
        AMREX_ALWAYS_ASSERT(g.ProbLo(0)==0. && WarpX::field_boundary_lo[0]==FieldBoundaryType::None &&
                            WarpX::field_boundary_hi[0]==FieldBoundaryType::PEC &&
                            WarpX::particle_boundary_hi[0]==ParticleBoundaryType::Reflecting &&
                            WarpX::field_boundary_lo[1]==FieldBoundaryType::PMC &&
                            WarpX::field_boundary_hi[1]==FieldBoundaryType::PMC);
        // A particle contributes at most one positive term to each node in
        // its FAB. Summing counts over all depositing species/tiles therefore
        // bounds every local atomic summation, including excluded alpha ions.
        std::map<int,amrex::Long> counts;
        int valid=1;
        for(auto const& pc:w.GetPartContainer()) {
            if(pc->do_not_deposit || pc->getCharge()==0.)continue;
            AMREX_ALWAYS_ASSERT(pc->getCharge()>0. && !pc->HasiAttrib("ionizationLevel"));
            for(WarpXParIter pti(*pc,0);pti.isValid();++pti) {
                counts[pti.index()]+=pti.numParticles();
                auto const& weight=pti.GetAttribs(PIdx::w);
                std::vector<amrex::ParticleReal> host(pti.numParticles());
                if(!host.empty())amrex::Gpu::dtoh_memcpy(host.data(),weight.dataPtr(),host.size()*sizeof(host[0]));
                for(auto value:host) {
                    Real const q=pc->getCharge()*value;
                    if(!std::isfinite(value) || value<0. || !std::isfinite(q) ||
                       (q!=0. && !std::isnormal(q)))valid=0;
                }
            }
        }
        amrex::ParallelDescriptor::ReduceIntMin(valid);AMREX_ALWAYS_ASSERT(valid);
        for(auto const& entry:counts)m_particles_per_fab=std::max(m_particles_per_fab,entry.second);
        amrex::ParallelDescriptor::ReduceLongMax(m_particles_per_fab);
        m_donors=layout.boxArray().size(); // no periodic images; one term/source FAB/node
        // One axis fold, one inverse-volume operation, and at most four
        // boundary side updates with two multiplications plus one addition.
        m_operations=m_particles_per_fab+m_donors+14;
        long double const eps=std::numeric_limits<Real>::epsilon();
        long double const ku=m_operations*eps;
        AMREX_ALWAYS_ASSERT(ku<0.01L);
        long double const gamma=ku/(1-ku);
        long double const factor=(1+eps)*2*gamma/(1-gamma);
        // Outward rounding of the bound, independent of observed differences.
        m_relative=std::nextafter(static_cast<Real>(factor),std::numeric_limits<Real>::infinity());
    }

    amrex::Real relative_bound() const {return m_relative;}

    // Inspect the same native shape-3 summands, before any error comparison.
    // Normal arithmetic is required; neither negative/cancelling deposits nor
    // underflow are covered by the positive-summation proof.
    void ValidateConstituents(WarpX& w,amrex::MultiFab const& layout,bool endpoint) const {
        using Real=amrex::Real;
        auto const inverse=WarpX::InvCellSize(0);
        Real const invvol=inverse.x*inverse.y*inverse.z;
        Real const largest_denominator=2*MathConst::pi*(w.Geom(0).ProbHi(0)+
            layout.nGrowVect()[0]*w.Geom(0).CellSize(0));
        amrex::ReduceOps<amrex::ReduceOpMin> op;amrex::ReduceData<int> data(op);
        op.eval(1,data,[] AMREX_GPU_DEVICE(int){return amrex::GpuTuple<int>{1};});
        for(auto const& pc:w.GetPartContainer()) {
            if(pc->do_not_deposit || pc->getCharge()==0.)continue;
            for(WarpXParIter pti(*pc,0);pti.isValid();++pti) {
                auto const pos=GetParticlePosition(pti);
                auto const* weight=pti.GetAttribs(PIdx::w).dataPtr();
                auto const* xn=endpoint ? pti.GetAttribs("x_n").dataPtr() : nullptr;
                auto const* yn=endpoint ? pti.GetAttribs("y_n").dataPtr() : nullptr;
                auto const* zn=endpoint ? pti.GetAttribs("z_n").dataPtr() : nullptr;
                auto const origin=WarpX::LowerCorner(layout[pti].box(),0,0.);
                Real const q=pc->getCharge();
                op.eval(pti.numParticles(),data,[=] AMREX_GPU_DEVICE(long i) {
                    amrex::ParticleReal x,y,z;pos(i,x,y,z);
                    if(endpoint){x=2*x-xn[i];y=2*y-yn[i];z=2*z-zn[i];}
                    Real const xr=(std::sqrt(x*x+y*y)-origin.x)*inverse.x;
                    Real const zr=(z-origin.z)*inverse.z;
                    if(!(xr>=0. && zr>=0. && xr<2147483647. && zr<2147483647.))return amrex::GpuTuple<int>{0};
                    Real sx[4],sz[4];Compute_shape_factor<3> shape;shape(sx,xr);shape(sz,zr);
                    Real const wq=q*weight[i]*invvol;
                    bool valid=amrex::Math::isfinite(wq) && wq>=0.;
                    for(int a=0;a<4;++a)for(int b=0;b<4;++b) {
                        Real const term=sx[a]*sz[b]*wq;
                        Real const scaled=term/largest_denominator;
                        valid=valid && amrex::Math::isfinite(term) && term>=0. &&
                            (term==0. || scaled>=std::numeric_limits<Real>::min());
                    }
                    return amrex::GpuTuple<int>{int(valid)};
                });
            }
        }
        int valid=amrex::get<0>(data.value());amrex::ParallelDescriptor::ReduceIntMin(valid);
        AMREX_ALWAYS_ASSERT(valid);
    }


    void Check(amrex::MultiFab const& difference,amrex::MultiFab const& reference,
               std::string const& phase) const {
        using Real=amrex::Real;
        Real const relative=m_relative;
        amrex::ReduceOps<amrex::ReduceOpMax,amrex::ReduceOpMax,
                         amrex::ReduceOpMax,amrex::ReduceOpMax> op;
        amrex::ReduceData<int,Real,Real,Real> data(op);
        op.eval(1,data,[] AMREX_GPU_DEVICE(int){return amrex::GpuTuple<int,Real,Real,Real>{0,0,0,0};});
        for(amrex::MFIter mfi(reference);mfi.isValid();++mfi) {
            auto const a=reference.const_array(mfi),d=difference.const_array(mfi);
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real const ref=a(i,j,k),delta=std::abs(d(i,j,k));
                Real const bound=std::nextafter(relative*ref,std::numeric_limits<Real>::infinity());
                bool const finite=amrex::Math::isfinite(ref) && amrex::Math::isfinite(delta);
                bool const zero=(ref==0. && delta==0.);
                bool const positive=ref>0. && ref>=std::numeric_limits<Real>::min();
                bool const ok=finite && (zero || (positive && delta<=bound));
                return amrex::GpuTuple<int,Real,Real,Real>{int(!ok),delta,
                    ref>0. ? bound : 0.,ref>0. ? delta/bound : (delta==0. ? 0. : 2.)};
            });
        }
        auto const values=data.value();int bad=amrex::get<0>(values);
        Real measurements[3]={amrex::get<1>(values),amrex::get<2>(values),amrex::get<3>(values)};
        amrex::ParallelDescriptor::ReduceIntMax(bad);
        amrex::ParallelDescriptor::ReduceRealMax(measurements,3);
        amrex::Print()<<"NATIVE_DEPOSIT_BOUND phase="<<phase<<" particles_per_fab="<<m_particles_per_fab
            <<" donors="<<m_donors<<" operations="<<m_operations<<" relative_bound="<<m_relative
            <<" difference_max="<<measurements[0]<<" bound_max="<<measurements[1]
            <<" max_node_fraction="<<measurements[2]<<" invalid="<<bad<<std::endl;
        AMREX_ALWAYS_ASSERT(bad==0);
#ifndef AMREX_USE_GPU
        // Preserve the original CPU exact-deposition qualification.
        AMREX_ALWAYS_ASSERT(measurements[0]==0.);
#endif
    }
};
} // namespace radial_fixture
#endif
