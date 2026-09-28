/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "MassMatrixDensityProjection.H"
#include "ThermalCurrentRemainder.H"
#include <limits>
#include "ImplicitAbsorbingEndpoint.H"
#include "ImplicitSolver.H"
#include <AMReX_Reduce.H>

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Deposition/ChargeDeposition.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <memory>


namespace {
bool SameEpoch(std::uint64_t value) {
    bool ok=value>0 && value<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max());
    if(!warpx::thermal::remainder::All(ok)) return false;
    amrex::Long lo=static_cast<amrex::Long>(value),hi=lo;
    amrex::ParallelDescriptor::ReduceLongMin(lo);
    amrex::ParallelDescriptor::ReduceLongMax(hi);return lo==hi;
}
bool CopyProjectionFields(amrex::MultiFab& rho,
    ablastr::fields::VectorField const& current,amrex::MultiFab const& source,
    amrex::Array<amrex::MultiFab,3> const& vector)
{
    namespace rem=warpx::thermal::remainder;
    std::array<amrex::MultiFab*,4> out{&rho,current[0],current[1],current[2]};
    std::array<amrex::MultiFab const*,4> in{&source,&vector[0],&vector[1],&vector[2]};
    bool ok=true;
    for(int q=0;q<4;++q) {
        ok=ok && out[q] && in[q]->isDefined();
        if(out[q] && in[q]->isDefined())ok=ok && out[q]->isDefined() &&
            rem::Layout(*out[q],*in[q]) && out[q]->nGrowVect()==in[q]->nGrowVect();
    }
    if(!rem::All(ok))return false;
    for(int q=0;q<4;++q)for(int c=0;c<4;++c) {
        ok=ok && !rem::Overlap(*out[q],*in[c]);
        if(c!=q)ok=ok && !rem::Overlap(*out[q],*out[c]);
    }
    if(!rem::All(ok))return false;
    for(auto const* f:in)if(!f->is_finite(0,1,f->nGrowVect())) return false;
    for(int q=0;q<4;++q)amrex::MultiFab::Copy(*out[q],*in[q],0,0,1,in[q]->nGrowVect());
    return true;
}
}
void MassMatrixDensityProjection::InvalidateCurrentIncrement() noexcept
{
    m_current_increment_ready=false;m_density_increment_ready=false;
    if(m_publication_epoch<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max()))
        ++m_publication_epoch;
}
bool MassMatrixDensityProjection::CopyLinearizationBase(amrex::MultiFab& rho,
    ablastr::fields::VectorField const& current,std::uint64_t expected_base) const
{
    if(!SameEpoch(expected_base) || !warpx::thermal::remainder::All(
        IsCaptured() && expected_base==m_base_epoch))return false;
    return CopyProjectionFields(rho,current,m_density,m_current);
}
bool MassMatrixDensityProjection::CopyRetainedIncrement(amrex::MultiFab& rho,
    ablastr::fields::VectorField const& current,std::uint64_t expected_base,
    std::uint64_t expected_publication) const
{
    if(!SameEpoch(expected_base) || !SameEpoch(expected_publication) ||
        !warpx::thermal::remainder::All(IsCaptured() && m_current_increment_ready &&
        m_density_increment_ready && expected_base==m_base_epoch &&
        expected_publication==m_publication_epoch))return false;
    return CopyProjectionFields(rho,current,m_delta_density,m_delta_current);
}

void MassMatrixDensityProjection::ComputeDivergence (
    WarpX& simulation, int const lev, ablastr::fields::VectorField const& current,
    amrex::MultiFab& divergence)
{
#if defined(WARPX_DIM_RZ)
    auto const& geometry = simulation.Geom(lev);
    auto const dr = geometry.CellSize(0);
    auto const dz = geometry.CellSize(1);
    auto const rmin = geometry.ProbLo(0);
    auto const ilo = geometry.Domain().smallEnd(0);
    amrex::Real const axis_factor = simulation.UseVerboncoeurAxisCorrection() ? 3.0 : 4.0;
    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const jr = current[0]->const_array(mfi);
        auto const jz = current[2]->const_array(mfi);
        auto const out = divergence.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
            amrex::Real const r = rmin+(i-ilo)*dr;
            // Esirkepov is a nearest-neighbor flux difference. At the axis
            // the raw nodal charge is divided by pi*dr/3 (Verboncoeur) or
            // pi*dr/4; Jr at dr/2 is divided by pi*dr. Maxwell's regularized
            // divergence always uses4, so it cannot serve both conventions.
            amrex::Real const radial = r == 0.0 ? axis_factor*jr(i,j,k)/dr
                : ((r+0.5*dr)*jr(i,j,k)-(r-0.5*dr)*jr(i-1,j,k))/(r*dr);
            out(i,j,k) = radial+(jz(i,j,k)-jz(i,j-1,k))/dz;
        });
    }
#else
    simulation.get_pointer_fdtd_solver_fp(lev)->ComputeDivE(current,divergence);
#endif
}

void MassMatrixDensityProjection::Capture (
    amrex::MultiFab const& density, int const component,
    ablastr::fields::VectorField const& current)
{
    BL_PROFILE("MassMatrixDensityProjection::Capture()");
    InvalidateCurrentIncrement();
    if(m_base_epoch<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max()))++m_base_epoch;
    if (!m_density.isDefined() || m_density.boxArray() != density.boxArray() ||
        m_density.DistributionMap() != density.DistributionMap() ||
        m_density.nGrowVect() != density.nGrowVect())
    {
        m_density = amrex::MultiFab(density.boxArray(), density.DistributionMap(),
                                   1, density.nGrowVect());
        m_delta_density = amrex::MultiFab(density.boxArray(), density.DistributionMap(),
                                         1, density.nGrowVect());
        for (int d = 0; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT(current[d]->nComp() == 1);
            m_current[d] = amrex::MultiFab(current[d]->boxArray(),
                current[d]->DistributionMap(), 1, current[d]->nGrowVect());
            m_delta_current[d] = amrex::MultiFab(current[d]->boxArray(),
                current[d]->DistributionMap(), 1, current[d]->nGrowVect());
        }
    }
    amrex::MultiFab::Copy(m_density, density, component, 0, 1, density.nGrowVect());
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(m_current[d], *current[d], 0, 0, 1, current[d]->nGrowVect());
    }
}

void MassMatrixDensityProjection::Apply (
    WarpX& simulation, int const lev, amrex::MultiFab& density, int const component,
    ablastr::fields::VectorField const& current,
    amrex::Real const interval)
{
    BL_PROFILE("MassMatrixDensityProjection::Apply()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(IsCaptured(),
        "Mass-matrix density projection needs a full nonlinear deposit first");
    auto const& geom = simulation.Geom(lev);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(density.nGrowVect()[d] <= geom.Domain().length(d));
    }
    for (int d = 0; d < 3; ++d) {
        // Difference before differentiation: no subtraction of two large
        // divergences and no repeated summation of particle guard deposits.
        amrex::MultiFab::LinComb(m_delta_current[d], 1.0, *current[d], 0,
                                -1.0, m_current[d], 0, 0, 1, current[d]->nGrowVect());
    }
    m_density_increment_ready=false;
    if(m_publication_epoch<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max()))++m_publication_epoch;
    ApplyStoredCurrentIncrement(simulation,lev,density,component,interval,false);
}

void MassMatrixDensityProjection::CaptureCurrentIncrementAndCompose (
    ablastr::fields::VectorField const& current)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(IsCaptured(), "Missing MM nonlinear current base");
    InvalidateCurrentIncrement();
    // Validate every component before writing any of the caller's fields.
    for (int d = 0; d < 3; ++d) {
        AMREX_ALWAYS_ASSERT(current[d] != &m_current[d] &&
            current[d] != &m_delta_current[d] && current[d]->nComp() == 1 &&
            current[d]->boxArray() == m_current[d].boxArray() &&
            current[d]->DistributionMap() == m_current[d].DistributionMap() &&
            current[d]->nGrowVect() == m_current[d].nGrowVect());
    }
    for (int d = 0; d < 3; ++d) {
        amrex::MultiFab::Copy(m_delta_current[d],*current[d],0,0,1,current[d]->nGrowVect());
        amrex::MultiFab::Add(*current[d],m_current[d],0,0,1,current[d]->nGrowVect());
    }
    m_current_increment_ready = true;
}

void MassMatrixDensityProjection::ApplyCurrentIncrement (
    WarpX& simulation, int lev, amrex::MultiFab& density, int component,
    amrex::Real interval)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(IsCaptured() && CurrentIncrementReady(),
        "MM continuity requires a fresh retained current increment");
    ApplyStoredCurrentIncrement(simulation,lev,density,component,interval,true);
}

void MassMatrixDensityProjection::ApplyStoredCurrentIncrement (
    WarpX& simulation, int lev, amrex::MultiFab& density, int component,
    amrex::Real interval, bool difference_first)
{
    m_density_increment_ready=false;
    auto const& geom = simulation.Geom(lev);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT(density.nGrowVect()[d] <= geom.Domain().length(d));
    }
    m_delta_density.setVal(0.0);
    ablastr::fields::VectorField delta_current{
        &m_delta_current[0], &m_delta_current[1], &m_delta_current[2]};
    ComputeDivergence(simulation, lev, delta_current, m_delta_density);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_delta_density, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const base = m_density.const_array(mfi);
        auto const change = m_delta_density.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real const rho0 = base(i, j, k);
                // Preserve the small continuity action before its one final
                // addition to rho0. Keep delta first in max so NaN is not masked.
                // Vacuum and the existing nonnegative-charge
                // constraint are unchanged; constitutive floors are elsewhere.
                change(i, j, k) = rho0 > 0.0
                    ? (difference_first
                        ? std::max(-interval * change(i, j, k), -rho0)
                        : std::max(rho0 - interval * change(i, j, k), amrex::Real(0.0)) - rho0)
                    : 0.0;
            });
    }

    // This is a field correction, not a new particle deposit: OverrideSync
    // and ghost fills are idempotent; SyncRho/charge-wall folding are not.
    m_delta_density.OverrideSync(geom.periodicity());
    m_delta_density.FillBoundary(geom.periodicity());
    auto const domain = amrex::convert(geom.Domain(), m_delta_density.ixType());
    auto const lo = domain.smallEnd();
    auto const hi = domain.bigEnd();
    auto const nodal = m_delta_density.ixType().toIntVect();
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { periodic[d] = geom.isPeriodic(d); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_delta_density); mfi.isValid(); ++mfi) {
        auto const change = m_delta_density.array(mfi);
        auto const box = mfi.fabbox();
        // Gather from the complete interior mirror in every direction. This
        // includes corner ghosts and leaves physical boundary nodes intact.
        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE(int i, int j, [[maybe_unused]] int k) {
                amrex::IntVect const dst(AMREX_D_DECL(i,j,k));
                amrex::IntVect src = dst;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (periodic[d]) { continue; }
                    if (src[d] < lo[d]) { src[d] = 2*lo[d] - (1-nodal[d]) - src[d]; }
                    else if (src[d] > hi[d]) { src[d] = 2*hi[d] + (1-nodal[d]) - src[d]; }
                }
                if (src != dst && box.contains(src)) { change(dst) = change(src); }
            });
    }
    amrex::MultiFab::LinComb(density, 1.0, m_density, 0,
                            1.0, m_delta_density, 0, component, 1, density.nGrowVect());
    // No cached ready bit is restored by rollback. The caller must consume
    // this complete publication under its own native response/base lease.
    if(difference_first && m_current_increment_ready &&
       m_publication_epoch<static_cast<std::uint64_t>(std::numeric_limits<amrex::Long>::max())) {
        ++m_publication_epoch;m_density_increment_ready=true;
    }
}

void MassMatrixDensityProjection::CaptureStepStart (
    amrex::MultiFab const& density, int const component)
{
    InvalidateCurrentIncrement();
    if (!m_step_density.isDefined() || m_step_density.boxArray() != density.boxArray() ||
        m_step_density.DistributionMap() != density.DistributionMap() ||
        m_step_density.nGrowVect() != density.nGrowVect()) {
        m_step_density = amrex::MultiFab(density.boxArray(),density.DistributionMap(),
                                         1,density.nGrowVect());
        m_endpoint_density = amrex::MultiFab(density.boxArray(),density.DistributionMap(),
                                             1,density.nGrowVect());
    }
    amrex::MultiFab::Copy(m_step_density,density,component,0,1,density.nGrowVect());
}

bool MassMatrixDensityProjection::DepositEndpointSubset (
    WarpX& simulation, int const lev, amrex::MultiFab& endpoint_density,
    EndpointSubset const subset)
{
    BL_PROFILE("MassMatrixDensityProjection::DepositEndpointSubset()");
    endpoint_density.setVal(0.0);
    auto const inverse = WarpX::InvCellSize(lev);
    using DecisionOp=amrex::ReduceOps<amrex::ReduceOpMin>;
    using DecisionData=amrex::ReduceData<int>;
    std::unique_ptr<DecisionOp> decisions;
    std::unique_ptr<DecisionData> decision_data;
    if (subset!=EndpointSubset::All) {
        decisions=std::make_unique<DecisionOp>();
        decision_data=std::make_unique<DecisionData>(*decisions);
        decisions->eval(1,*decision_data,[] AMREX_GPU_DEVICE(int) { return amrex::GpuTuple<int>{1}; });
    }
    for (auto const& species : simulation.GetPartContainer()) {
        if (species->do_not_deposit || species->getCharge() == 0.0) { continue; }
        for (WarpXParIter pti(*species,lev); pti.isValid(); ++pti) {
            auto const position = GetParticlePosition<PIdx>(pti);
#if !defined(WARPX_DIM_1D_Z)
            auto const* xn = pti.GetAttribs("x_n").dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
            auto const* yn = pti.GetAttribs("y_n").dataPtr();
#endif
            auto const* zn = pti.GetAttribs("z_n").dataPtr();
            auto& fab = endpoint_density[pti];
            auto const box = fab.box();
            auto const origin = WarpX::LowerCorner(box,lev,0.0);
            auto const low = amrex::lbound(box);
            auto const high = amrex::ubound(box);
            int const reach = WarpX::nox/2+1;
            // Read virtual endpoint positions; the actual particles remain at
            // their nonlinear midpoint, including for the thermal stage and MM.
            auto const endpoint = [=] AMREX_GPU_HOST_DEVICE(long i,
                amrex::ParticleReal& x,amrex::ParticleReal& y,amrex::ParticleReal& z) {
                position(i,x,y,z);
#if !defined(WARPX_DIM_1D_Z)
                x = 2*x-xn[i];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                y = 2*y-yn[i];
#endif
                z = 2*z-zn[i];
                // Never let a virtual endpoint exceed the allocated deposit
                // support even though the stored midpoint passes its own check.
#if defined(WARPX_DIM_RZ)
                amrex::Real const u = (std::hypot(x,y)-origin.x)*inverse.x;
#else
                amrex::Real const u = (x-origin.x)*inverse.x;
#endif
                amrex::Real const v = (z-origin.z)*inverse.z;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
                AMREX_ALWAYS_ASSERT(u >= reach && u <= high.x-low.x-reach);
                AMREX_ALWAYS_ASSERT(v >= reach && v <= high.y-low.y-reach);
#else
                amrex::ignore_unused(u,v,reach,high);
#endif
            };
            auto const* weights = pti.GetAttribs()[PIdx::w].dataPtr();
            amrex::Gpu::DeviceVector<amrex::ParticleReal> selected_weights;
            if (subset!=EndpointSubset::All) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
                selected_weights.resize(pti.numParticles());
                auto* selected=selected_weights.data();
                auto const original=weights;
                auto const boundary=warpx::implicit::MakeAbsorbingEndpointBoundary(
                    simulation.Geom(lev),species->GetParticleBoundaryData(),
                    WarpX::field_boundary_lo,WarpX::field_boundary_hi,
                    simulation.get_pointer_ImplicitSolver() &&
                    !simulation.get_pointer_ImplicitSolver()->ReflectsParticlesInsidePush());
                amrex::GpuArray<const amrex::ParticleReal*,3> old_u{
                    pti.GetAttribs("ux_n").dataPtr(),pti.GetAttribs("uy_n").dataPtr(),pti.GetAttribs("uz_n").dataPtr()};
                amrex::GpuArray<const amrex::ParticleReal*,3> mid_u{
                    pti.GetAttribs(PIdx::ux).dataPtr(),pti.GetAttribs(PIdx::uy).dataPtr(),pti.GetAttribs(PIdx::uz).dataPtr()};
                decisions->eval(pti.numParticles(),*decision_data,[=] AMREX_GPU_DEVICE(long i) {
                    amrex::GpuArray<amrex::ParticleReal,3> x{},u{};
                    endpoint(i,x[0],x[1],x[2]);
                    for (int d=0;d<3;++d) { u[d]=2*mid_u[d][i]-old_u[d][i]; }
                    auto const choice=warpx::implicit::InspectAbsorbingEndpoint(x,u,boundary);
                    using Status=warpx::implicit::AbsorbingEndpointStatus;
                    bool const valid=choice.status==Status::Interior || choice.status==Status::Absorbed;
                    bool const lost=choice.status==Status::Absorbed;
                    selected[i]=(valid && (lost==(subset==EndpointSubset::Absorbed))) ? original[i] : 0;
                    return amrex::GpuTuple<int>{int(valid)};
                });
                weights=selected;
#else
                return false;
#endif
            }
            auto const* ion = species->HasiAttrib("ionizationLevel")
                ? pti.GetiAttribs("ionizationLevel").dataPtr() : nullptr;
            auto const charge = species->getCharge();
            auto dispatch = [&]<int Order>() {
                doChargeDepositionShapeN<Order>(endpoint,weights,ion,fab,
                    pti.numParticles(),inverse,origin,low,charge,WarpX::n_rz_azimuthal_modes);
            };
            if (WarpX::nox == 2) { dispatch.template operator()<2>(); }
            else if (WarpX::nox == 3) { dispatch.template operator()<3>(); }
            else if (WarpX::nox == 4) { dispatch.template operator()<4>(); }
            else { WARPX_ABORT_WITH_MESSAGE("Continuity density requires particle_shape>=2"); }
            if (subset!=EndpointSubset::All) { amrex::Gpu::streamSynchronize(); }
        }
    }
    if (subset!=EndpointSubset::All) {
        bool valid=amrex::get<0>(decision_data->value())!=0;
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        if (!valid) { return false; }
    }
#if defined(WARPX_DIM_RZ)
    simulation.ApplyInverseVolumeScalingToChargeDensity(&endpoint_density,lev);
#endif
    ablastr::fields::MultiLevelScalarField endpoints{&endpoint_density};
    simulation.SyncRho(endpoints,{},{});
    simulation.ApplyRhofieldBoundary(lev,&endpoint_density,PatchType::fine);
    endpoint_density.FillBoundary(simulation.Geom(lev).periodicity());
    return true;
}

void MassMatrixDensityProjection::DepositEndpointAverage (
    WarpX& simulation, int const lev, amrex::MultiFab& density, int const component)
{
    BL_PROFILE("MassMatrixDensityProjection::DepositEndpointAverage()");
    AMREX_ALWAYS_ASSERT(m_step_density.isDefined());
    AMREX_ALWAYS_ASSERT(DepositEndpointSubset(simulation,lev,m_endpoint_density,EndpointSubset::All));
    // Esirkepov is conservative over [n,n+1], so the midpoint density is
    // (rho^n+rho^{n+1})/2. This positive, direct endpoint deposit avoids
    // cancellation-generated density in genuinely empty cells.
    amrex::MultiFab::LinComb(density,0.5,m_step_density,0,
                            0.5,m_endpoint_density,0,component,1,density.nGrowVect());
    amrex::MultiFab::Copy(density,m_step_density,0,0,1,density.nGrowVect());
}
