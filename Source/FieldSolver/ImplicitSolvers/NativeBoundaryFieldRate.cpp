/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL. */
#include "NativeBoundaryFieldRate.H"
#include "FieldSolver/ImplicitSolvers/DarwinABoundary.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "WarpX.H"
#include <AMReX_FabArrayUtility.H>
using MF=amrex::MultiFab;
using View=ablastr::fields::VectorField;
namespace warpx::darwin {
void NativeBoundaryFieldRate(WarpX& w,View const& electric,View const& reference,
    View potential,View magnetic,View current)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(w.maxLevel()==0 && !EB::enabled() && WarpX::ncomps==1,
        "Native boundary field rate requires one level, no EB and one component");
    for(int c=0;c<3;++c) {
        AMREX_ALWAYS_ASSERT(electric[c] && reference[c] && potential[c] && magnetic[c] && current[c]);
        for(int d=0;d<3;++d) for(auto* output:{potential[c],magnetic[c],current[c]})
            AMREX_ALWAYS_ASSERT(output!=electric[d] && output!=reference[d]);
        for(int d=0;d<3;++d) {
            AMREX_ALWAYS_ASSERT(potential[c]!=magnetic[d] && potential[c]!=current[d] && magnetic[c]!=current[d]);
            if(c!=d)AMREX_ALWAYS_ASSERT(potential[c]!=potential[d] && magnetic[c]!=magnetic[d] && current[c]!=current[d]);
        }
        AMREX_ALWAYS_ASSERT(potential[c]->boxArray()==electric[c]->boxArray() &&
            potential[c]->boxArray()==reference[c]->boxArray() &&
            potential[c]->DistributionMap()==electric[c]->DistributionMap() &&
            potential[c]->DistributionMap()==reference[c]->DistributionMap() &&
            electric[c]->nGrowVect().allGE(potential[c]->nGrowVect()) &&
            reference[c]->nGrowVect().allGE(potential[c]->nGrowVect()));
        MF::Copy(*potential[c],*electric[c],0,0,1,potential[c]->nGrowVect());
        for(amrex::MFIter it(*potential[c]);it.isValid();++it) {
            auto a=potential[c]->array(it);
            amrex::ParallelFor(it.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=-a(i,j,k);});
        }
        magnetic[c]->setVal(0.);current[c]->setVal(0.);
    }
    amrex::GpuArray<int,AMREX_SPACEDIM> pmc_lo{},pmc_hi{};
    bool any_pmc=false;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        pmc_lo[d]=WarpX::field_boundary_lo[d]==FieldBoundaryType::PMC;
        pmc_hi[d]=WarpX::field_boundary_hi[d]==FieldBoundaryType::PMC;
        any_pmc=any_pmc || pmc_lo[d] || pmc_hi[d];
    }
    if(w.get_pointer_HybridPICModel()->m_add_external_fields || any_pmc) {
        auto const& domain=w.Geom(0).Domain();auto const& period=w.Geom(0).periodicity();
        for(int dir=0;dir<3;++dir) {
            MF& A=*potential[dir];MF const& A_bc=*reference[dir];
            for (amrex::MFIter mfi(A, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                amrex::Box tb = mfi.tilebox();
                tb.grow(A.nGrowVect());
                const amrex::Box domain_t = amrex::convert(domain, A.ixType().toIntVect());

                amrex::Array4<amrex::Real> const& a = A.array(mfi);
                amrex::Array4<amrex::Real const> const& abc = A_bc.const_array(mfi);
                amrex::Array4<int const> eb;

                const bool use_eb = false;

                amrex::GpuArray<int, 3> dlo{{0, 0, 0}};
                amrex::GpuArray<int, 3> dhi{{0, 0, 0}};
                amrex::GpuArray<int, 3> per{{1, 1, 1}};
                amrex::GpuArray<int, 3> nodal{{0, 0, 0}};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    dlo[d] = domain_t.smallEnd(d);
                    dhi[d] = domain_t.bigEnd(d);
                    per[d] = period.isPeriodic(d) ? 1 : 0;
                    nodal[d] = A.ixType().nodeCentered(d);
                }

#if defined(WARPX_DIM_RZ)
                bool const on_axis = w.Geom(0).ProbLo(0) == 0.;
#endif
                amrex::ParallelFor(tb,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                   {
#if defined(WARPX_DIM_RZ)
                                       if (on_axis && dir == 1 && i == 0)
                                       {
                                           a(i, j, k) = 0.;
                                           return;
                                       }
#endif
                                       // Embedded conductors: hold A at the gauge zero inside
                                       // masked cells (frozen enclosed flux; the interior field
                                       // stays at B_static).
                                       if (use_eb && eb(i, j, k) == 0)
                                       {
                                           a(i, j, k) = 0.0;
                                           return;
                                       }
                                       // PEC tangential nodes carry the imposed
                                       // drive. PMC tangential nodes are free:
                                       // their response is even. Pin only
                                       // points that lie on a physical wall
                                       // node. A normal component is half a
                                       // cell inside the face; its ghost
                                       // reflection below supplies the wall
                                       // trace.
                                       const int idx[3] = {i, j, k};
                                       bool on_boundary = false;
                                       for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                       {
                                           if (per[d] || !nodal[d])
                                           {
                                               continue;
                                           }
                                           bool lower = idx[d] <= dlo[d] && !pmc_lo[d];
#if defined(WARPX_DIM_RZ)
                                           if (d == 0 && on_axis)
                                           {
                                               lower = false;
                                           }
#endif
                                           if (lower || (idx[d] >= dhi[d] && !pmc_hi[d]))
                                           {
                                               on_boundary = true;
                                           }
                                       }
                                       if (on_boundary)
                                       {
                                           // Clamp the imposed value to the domain edge:
                                           // ghosts continue the wall value rather than the
                                           // (growing) exterior vector potential, so the wall
                                           // ring carries no spurious curl sheet.
                                           int ic[3] = {i, j, k};
                                           for (int d = 0; d < AMREX_SPACEDIM; ++d)
                                           {
                                               if (per[d])
                                               {
                                                   continue;
                                               }
#if defined(WARPX_DIM_RZ)
                                               if (d == 0 && on_axis && ic[d] < dlo[d])
                                               {
                                                   continue;
                                               }
#endif
                                               ic[d] = amrex::Clamp(ic[d], dlo[d], dhi[d]);
                                           }
                                           a(i, j, k) = abc(ic[0], ic[1], ic[2]);
                                       }
                                   });
            }
            A.FillBoundary(w.Geom(0).periodicity());
#if defined(WARPX_DIM_RZ)
            bool const skip_lower_axis = w.Geom(0).ProbLo(0) == 0.;
#else
            bool const skip_lower_axis = false;
#endif
            ApplyDarwinCellCenteredABoundary(A, A_bc, w.Geom(0), skip_lower_axis, nullptr,
                                             pmc_lo, pmc_hi);
        }
#if defined(WARPX_DIM_RZ)
        if(w.Geom(0).ProbLo(0)==0.)w.ApplyFieldBoundaryOnAxis(potential[0],potential[1],potential[2],0);
#endif
    }
    w.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(magnetic,potential,w.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    amrex::Vector<MF*> b{magnetic.begin(),magnetic.end()};
    amrex::FillBoundaryAndSync_nowait(b,w.Geom(0).periodicity());
    amrex::FillBoundaryAndSync_finish(b);
    w.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(current,magnetic,w.GetEBUpdateEFlag()[0],0);
    for(auto* f:current) {
        f->OverrideSync(w.Geom(0).periodicity());f->FillBoundary(w.Geom(0).periodicity());
        ApplyDarwinPMCVectorBoundary(*f,w.Geom(0),pmc_lo,pmc_hi);
    }
}
}
