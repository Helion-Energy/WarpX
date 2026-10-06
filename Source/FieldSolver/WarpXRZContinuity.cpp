/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#ifdef WARPX_DIM_RZ
#include "EmbeddedBoundary/Enabled.H"
#include "BoundaryConditions/WarpX_PEC.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Fields.H"
#include "Parallelization/WarpXSumGuardCells.H"
#include "Utils/TextMsg.H"
#include <AMReX_GpuLaunch.H>
#include <iomanip>

using namespace amrex::literals;
using amrex::MultiFab;
using warpx::fields::FieldType;

void WarpX::ComputeRZContinuityResidual (
    MultiFab& residual, const MultiFab& rho_old, const MultiFab& rho_new,
    const std::array<MultiFab*, 3>& current, const amrex::Real delta_t,
    const bool yee_axis) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(delta_t > 0._rt && n_rz_azimuthal_modes == 1
        && grid_type == GridType::Staggered && Geom(0).ProbLo(0) == 0._rt,
        "RZ continuity residual requires dt>0 and staggered m=0 RZ with r_lo=0");
    const auto dx = Geom(0).CellSizeArray();
    const int axis = Geom(0).Domain().smallEnd(0);
    const amrex::Real axis_factor = yee_axis || !m_verboncoeur_axis_correction
        ? 0.25_rt : 1._rt/3._rt;
    for (amrex::MFIter mfi(residual); mfi.isValid(); ++mfi) {
        const auto out = residual.array(mfi);
        const auto ro = rho_old.const_array(mfi), rn = rho_new.const_array(mfi);
        const auto jr = current[0]->const_array(mfi), jz = current[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real r = static_cast<amrex::Real>(i-axis);
            const amrex::Real divr = i == axis ? jr(i,j,k)/(axis_factor*dx[0])
                : ((r+0.5_rt)*jr(i,j,k)-(r-0.5_rt)*jr(i-1,j,k))/(r*dx[0]);
            out(i,j,k) = (rn(i,j,k)-ro(i,j,k))/delta_t + divr
                + (jz(i,j,k)-jz(i,j-1,k))/dx[1];
        });
    }
}

void WarpX::AuditRZContinuity (const bool filtered)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0
        && evolve_scheme == EvolveScheme::Explicit && !do_current_centering
        && !EB::enabled() && !do_fluid_species
        && n_rz_azimuthal_modes == 1 && grid_type == GridType::Staggered
        && Geom(0).ProbLo(0) == 0._rt,
        "rz_continuity_audit_interval supports single-level explicit hybrid PIC without EB, "
        "current centering or fluid species");
    const auto& period = Geom(0).periodicity();
    auto const& source_rho = *m_fields.get(FieldType::rho_fp, 0);
    auto const source_j = m_fields.get_alldirs(FieldType::current_fp, 0);
    auto rho = std::make_unique<MultiFab>(source_rho.boxArray(),
        source_rho.DistributionMap(), 1, source_rho.nGrowVect());
    MultiFab::Copy(*rho, source_rho, 0, 0, 1, rho->nGrowVect());
    auto current = source_j;
    std::array<std::unique_ptr<MultiFab>, 3> raw_current;
    if (!filtered) {
        // Sum a separate copy of the *local* deposits. Never modify a solver
        // moment or apply filtering to this control. Match SyncCurrentAndRho
        // physical boundary handling, then fill interior/periodic ghosts.
        WarpXSumGuardCells(*rho, period, amrex::min(get_ng_depos_rho(), rho->nGrowVect()));
        for (int c = 0; c < 3; ++c) {
            raw_current[c] = std::make_unique<MultiFab>(source_j[c]->boxArray(),
                source_j[c]->DistributionMap(), 1, source_j[c]->nGrowVect());
            current[c] = raw_current[c].get();
            MultiFab::Copy(*current[c], *source_j[c], 0, 0, 1, current[c]->nGrowVect());
            WarpXSumGuardCells(*current[c], period,
                amrex::min(get_ng_depos_J(), current[c]->nGrowVect()));
        }
        if (UseRZBoundaryCurrent()) {
            FoldRZBoundaryMoments(*rho,current);
            if (m_rz_boundary_current_pending && m_rz_boundary_loss_ready) {
                AddRZAbsorbedCurrent(*m_rz_boundary_total_loss,current);
            }
        } else {
            ApplyRhofieldBoundary(0, rho.get(), PatchType::fine);
            ApplyJfieldBoundary(0, current[0], current[1], current[2], PatchType::fine);
        }
        rho->FillBoundary(period);
        for (auto* j : current) { j->FillBoundary(period); }
    }
    auto& previous = m_rz_continuity_old_rho[filtered ? 1 : 0];
    // Init/restart only seeds the old endpoint. A regrid/load balance resets
    // it as well; comparing different layouts would not be a time derivative.
    const bool same_layout = previous && previous->boxArray() == rho->boxArray()
        && previous->DistributionMap() == rho->DistributionMap();
    if (same_layout && istep[0] % m_rz_continuity_audit_interval == 0) {
        MultiFab residual(rho->boxArray(), rho->DistributionMap(), 1, 0);
        ComputeRZContinuityResidual(residual, *previous, *rho, current, dt[0]);
        const amrex::Real peak = residual.norminf();
        const amrex::Real scale = std::max(previous->norminf(), rho->norminf());
        ComputeRZContinuityResidual(residual, *previous, *rho, current, dt[0], true);
        auto log = amrex::Print();
        log << std::setprecision(17)
            << "RZ_CONTINUITY {\"step\":" << istep[0]
            << ",\"stage\":\"" << (filtered ? "post_sync" : "raw")
            << "\",\"deposition_linf_C_m3_s\":" << peak
            << ",\"yee_linf_C_m3_s\":" << residual.norminf()
            << ",\"dt_s\":" << dt[0] << ",\"rho_scale_C_m3\":" << scale
            << ",\"relative_step_residual\":";
        if (scale > 0._rt) { log << dt[0]*peak/scale; }
        else { log << "null"; }
        log << "}\n";
    }
    previous = std::move(rho);
}

namespace
{
    // Nodal deposition volumes divided by pi*dr^2*dz. Boundary nodes own
    // half a dual cell. These are deposition weights, not the independently
    // clipped geometric volumes used for field-energy diagnostics.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real radial_volume (int i, int nr, amrex::Real axis_volume)
    {
        return i == 0 ? axis_volume : (i == nr ? amrex::Real(nr) : 2._rt*i);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real radial_conductance (int i, int nr, amrex::Real axis_volume)
    {
        auto const left = radial_volume(i,nr,axis_volume);
        auto const right = radial_volume(i+1,nr,axis_volume);
        auto const c = 0.125_rt*(left+right);
        return i == 0 ? amrex::min(c,left) : c;
    }

    // FillBoundary exchanges valid cells; physical ghosts at an MPI seam are
    // outside every valid box and are not exchanged. After each filter pass,
    // reconstruct them from the exchanged interior and a nodal surface-flux
    // field whose endcap nodes ARE valid (and therefore exchanged).
    void fill_physical_moment_ghosts (
        MultiFab& field, MultiFab const* axial_flux, int component,
        amrex::Geometry const& geom)
    {
        int const nr=geom.Domain().bigEnd(0)+1, nz=geom.Domain().bigEnd(1)+1;
        auto const type=field.ixType().toIntVect();
        int const hi_r=nr-(1-type[0]), hi_z=nz-(1-type[1]);
        for (amrex::MFIter mfi(field,false);mfi.isValid();++mfi) {
            auto const a=field.array(mfi);
            auto const flux=axial_flux ? axial_flux->const_array(mfi)
                : amrex::Array4<amrex::Real const>{};
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                if (i>=0 && i<=hi_r && j>=0 && j<=hi_z) { return; }
                int ir=i, iz=j;
                amrex::Real parity=1._rt;
                if (i<0) {
                    ir=-i-(1-type[0]);
                    if (component==0 || component==1) { parity=-1._rt; }
                } else if (i>hi_r) {
                    ir=2*nr-(1-type[0])-i;
                    // The supported radial wall reflects particles: r*Jr_face=0.
                    parity=component==0 ? -(2._rt*ir+1._rt)/(2._rt*i+1._rt)
                        : amrex::Real(ir)/i;
                }
                if (j<0) { iz=-j-(1-type[1]); }
                else if (j>hi_z) { iz=2*nz-(1-type[1])-j; }
                if (component==2 && (j<0 || j>hi_z)) {
                    a(i,j,k)=parity*(2._rt*flux(ir,j<0 ? 0 : nz,k)-a(ir,iz,k));
                } else {
                    a(i,j,k)=parity*a(ir,iz,k);
                }
            });
        }
    }
}

bool WarpX::UseRZBoundaryCurrent () const
{
    return m_rz_continuity_filter && !Geom(0).isPeriodic(1)
        && electromagnetic_solver_id == ElectromagneticSolverAlgo::HybridPIC
        && evolve_scheme == EvolveScheme::Explicit;
}

void WarpX::SyncRZBoundaryMoments ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0 && n_rz_azimuthal_modes == 1
        && grid_type == GridType::Staggered && !do_current_centering && !EB::enabled()
        && Geom(0).ProbLo(0) == 0._rt,
        "Boundary-compatible RZ moments require single-level staggered m=0 hybrid PIC");
    auto& rho = *m_fields.get(FieldType::rho_fp,0);
    auto const current = m_fields.get_alldirs(FieldType::current_fp,0);
    auto const& period = Geom(0).periodicity();
    // A boundary stencil is an operator on the assembled physical moment.
    // Filtering local raw deposits first and folding their ghosts afterwards
    // is a different operator and fails at actual reflection events.
    WarpXSumGuardCells(rho,period,amrex::min(get_ng_depos_rho(),rho.nGrowVect()));
    for (auto* j : current) {
        WarpXSumGuardCells(*j,period,amrex::min(get_ng_depos_J(),j->nGrowVect()));
    }
    FoldRZBoundaryMoments(rho,current);
    if (m_rz_boundary_current_pending && m_rz_boundary_loss_ready) {
        AddRZAbsorbedCurrent(*m_rz_boundary_total_loss,current);
    }
    rho.FillBoundary(period);
    for (auto* j : current) { j->FillBoundary(period); }
    if (use_filter) { FilterRZBoundaryMoments(rho,current); }
}

void WarpX::FilterRZBoundaryMoments (
    MultiFab& rho, const std::array<MultiFab*,3>& current)
{
    int const nr = Geom(0).Domain().bigEnd(0)+1;
    int const nz = Geom(0).Domain().bigEnd(1)+1;
    auto const axis_volume = m_verboncoeur_axis_correction ? 1._rt/3._rt : .25_rt;
    auto const& period = Geom(0).periodicity();
    std::array<MultiFab*,4> fields{&rho,current[0],current[1],current[2]};
    // A=I-DH on nodal density and transverse currents; B=I-HD on the
    // normal current. H vanishes at the physical face, so DB=AD while
    // the surface current itself is retained. The boundary ghost encodes
    // that face flux, e.g. Jz(-1)=2*Jz_face-Jz(0).
    for (int dir=0; dir<2; ++dir) {
        int const passes = bilinear_filter.npass_each_dir[dir];
        for (int pass=0; pass<passes; ++pass) {
            for (int f=0; f<4; ++f) {
                int const component=f-1;
                auto& field=*fields[f];
                MultiFab out(field.boxArray(),field.DistributionMap(),1,field.nGrowVect());
                out.setVal(0._rt);
                bool const normal=(dir==0 && component==0) || (dir==1 && component==2);
                auto const ng=field.nGrowVect();
                for (amrex::MFIter mfi(field,false); mfi.isValid(); ++mfi) {
                    auto const src=field.const_array(mfi);
                    auto const dst=out.array(mfi);
                    auto box=mfi.validbox();
                    box.grow(1-dir,ng[1-dir]);
                    amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        if (dir==0 && normal) {
                            amrex::Real const a=2._rt*i+1._rt;
                            auto const d0=(a*src(i,j,k)-(i==0 ? 0._rt :
                                (a-2._rt)*src(i-1,j,k)))/(i==0 ? axis_volume : 2._rt*i);
                            auto const d1=((a+2._rt)*src(i+1,j,k)-a*src(i,j,k))/(2._rt*(i+1));
                            dst(i,j,k)=src(i,j,k)+radial_conductance(i,nr,axis_volume)/a*(d1-d0);
                        } else if (dir==0) {
                            // The azimuthal component is odd about the axis.
                            if (component==1 && i==0) {
                                dst(i,j,k)=0._rt;
                                return;
                            }
                            auto const v=radial_volume(i,nr,axis_volume);
                            auto const lo=i==0 ? 0._rt : radial_conductance(i-1,nr,axis_volume)
                                *(src(i,j,k)-src(i-1,j,k));
                            auto const hi=i==nr ? 0._rt : radial_conductance(i,nr,axis_volume)
                                *(src(i+1,j,k)-src(i,j,k));
                            dst(i,j,k)=src(i,j,k)+(hi-lo)/v;
                        } else if (normal) {
                            dst(i,j,k)=.5_rt*src(i,j,k)+.25_rt*(src(i,j-1,k)+src(i,j+1,k));
                        } else {
                            auto const lo=j==0 ? 0._rt : .25_rt*(src(i,j,k)-src(i,j-1,k));
                            auto const hi=j==nz ? 0._rt : .25_rt*(src(i,j+1,k)-src(i,j,k));
                            dst(i,j,k)=src(i,j,k)+(hi-lo)/(j==0 || j==nz ? .5_rt : 1._rt);
                        }
                    });
                    // Only ghosts in this sweep direction need rebuilding.
                    // Transverse guards were swept too; this also transports
                    // tangentially filtered surface flux into the next sweep.
                    auto const fab=mfi.fabbox();
                    auto const type=field.ixType().toIntVect();
                    auto const valid=mfi.validbox();
                    bool const at_lo=valid.smallEnd(dir)==0;
                    bool const at_hi=valid.bigEnd(dir)==(dir==0 ? nr : nz)-(normal?1:0);
                    amrex::ParallelFor(fab,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        if (dir==0 && i<0 && at_lo) {
                            int const mirror=type[0] ? -i : -i-1;
                            dst(i,j,k)=(component==0 || component==1 ? -1._rt : 1._rt)
                                *dst(mirror,j,k);
                        } else if (dir==0 && i>nr-(normal?1:0) && at_hi) {
                            int const mirror=2*nr-(normal?1:0)-i;
                            if (normal) {
                                auto const flux=.5_rt*((2._rt*nr-1._rt)*src(nr-1,j,k)
                                    +(2._rt*nr+1._rt)*src(nr,j,k));
                                dst(i,j,k)=(2._rt*flux-(2._rt*mirror+1._rt)*dst(mirror,j,k))
                                    /(2._rt*i+1._rt);
                            } else { dst(i,j,k)=amrex::Real(mirror)/i*dst(mirror,j,k); }
                        } else if (dir==1 && j<0 && at_lo) {
                            int const mirror=normal ? -j-1 : -j;
                            dst(i,j,k)=normal ? src(i,-1,k)+src(i,0,k)-dst(i,mirror,k)
                                : dst(i,mirror,k);
                        } else if (dir==1 && j>nz-(normal?1:0) && at_hi) {
                            int const mirror=2*nz-(normal?1:0)-j;
                            dst(i,j,k)=normal ? src(i,nz-1,k)+src(i,nz,k)-dst(i,mirror,k)
                                : dst(i,mirror,k);
                        }
                    });
                }
                std::unique_ptr<MultiFab> axial_flux;
                if (component==2) {
                    axial_flux=std::make_unique<MultiFab>(rho.boxArray(),
                        rho.DistributionMap(),1,ng);
                    axial_flux->setVal(0._rt);
                    for (amrex::MFIter mfi(*axial_flux,false);mfi.isValid();++mfi) {
                        auto const src=out.const_array(mfi);
                        auto const flux=axial_flux->array(mfi);
                        amrex::ParallelFor(mfi.validbox(),
                            [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                                if (j==0 || j==nz) {
                                    flux(i,j,k)=.5_rt*(src(i,j-1,k)+src(i,j,k));
                                }
                            });
                    }
                    axial_flux->FillBoundary(period);
                }
                MultiFab::Copy(field,out,0,0,1,ng);
                field.FillBoundary(period);
                fill_physical_moment_ghosts(field,axial_flux.get(),component,Geom(0));
            }
        }
    }
}

void WarpX::PrepareRZBoundaryCurrent ()
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level==0 && n_rz_azimuthal_modes==1
        && current_deposition_algo==CurrentDepositionAlgo::Esirkepov
        && particle_boundary_hi[0]==ParticleBoundaryType::Reflecting
        && !do_moving_window && !do_fluid_species && !EB::enabled(),
        "Nonperiodic compatible RZ current requires Esirkepov, a reflecting radial wall, "
        "one m=0 level, and no moving window, fluid species or embedded boundary");
    for (int side=0;side<2;++side) {
        auto const bc=side==0 ? particle_boundary_lo[1] : particle_boundary_hi[1];
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bc==ParticleBoundaryType::Reflecting
            || bc==ParticleBoundaryType::Absorbing || bc==ParticleBoundaryType::Fractional_Absorbing,
            "Compatible nonperiodic RZ current supports reflecting/absorbing/fractional axial particles");
    }
    auto const current=m_fields.get_alldirs(FieldType::current_fp,0);
    auto const& rho=*m_fields.get(FieldType::rho_fp,0);
    auto const loss_ng=amrex::max(amrex::max(rho.nGrowVect(),
        get_ng_depos_J()+amrex::IntVect(1)),
        m_fields.get(FieldType::Efield_fp,ablastr::fields::Direction{2},0)->nGrowVect());
    for (auto& [name,entry] : m_rz_boundary_moments) {
        amrex::ignore_unused(name);
        entry.lost_charge->setVal(0._rt);
        for (auto& j : entry.current) { j->setVal(0._rt); }
    }
    for (auto const& name : mypc->GetSpeciesNames()) {
        auto& pc=mypc->GetParticleContainerFromName(name);
        if (pc.getCharge()==0._rt || pc.do_not_deposit) { continue; }
        auto& entry=m_rz_boundary_moments[name];
        auto deposited=current;
        for (int c=0;c<3;++c) {
            if (!entry.current[c] || entry.current[c]->boxArray()!=current[c]->boxArray()
                || entry.current[c]->DistributionMap()!=current[c]->DistributionMap()) {
                entry.current[c]=std::make_unique<MultiFab>(current[c]->boxArray(),
                    current[c]->DistributionMap(),1,current[c]->nGrowVect());
            }
            entry.current[c]->setVal(0._rt);
            deposited[c]=entry.current[c].get();
        }
        if (!entry.lost_charge || entry.lost_charge->boxArray()!=rho.boxArray()
            || entry.lost_charge->DistributionMap()!=rho.DistributionMap()) {
            entry.lost_charge=std::make_unique<MultiFab>(rho.boxArray(),rho.DistributionMap(),2,loss_ng);
        }
        entry.lost_charge->setVal(0._rt);
        // Keep local extensive deposits. The normal volume scaling, MPI sum,
        // physical fold and compatible filter are applied exactly once later.
        pc.DepositCurrent({deposited},dt[0],-.5_rt*dt[0]);
    }
    m_rz_boundary_current_pending=true;
    m_rz_boundary_loss_ready=false;
}

MultiFab* WarpX::RZBoundaryLossCharge (std::string const& species)
{
    auto const it=m_rz_boundary_moments.find(species);
    return m_rz_boundary_current_pending && it!=m_rz_boundary_moments.end()
        ? it->second.lost_charge.get() : nullptr;
}

void WarpX::FoldRZBoundaryCharge (MultiFab& rho)
{
    // The finite charge cloud of a surviving particle is mirrored into the
    // plasma. Absorbing its center removes that entire mirrored cloud through
    // an explicitly deposited surface current. EM field parity does not set
    // the ion escape current to zero.
    amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM> const lo{
        ParticleBoundaryType::None,ParticleBoundaryType::Reflecting};
    amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM> const hi{
        ParticleBoundaryType::Reflecting,ParticleBoundaryType::Reflecting};
    PEC::ApplyReflectiveBoundarytoRhofield(&rho,field_boundary_lo,field_boundary_hi,
        lo,hi,Geom(0),0,PatchType::fine,ref_ratio);
    for (amrex::MFIter mfi(rho,false);mfi.isValid();++mfi) {
        if (mfi.validbox().smallEnd(0)!=0) { continue; }
        auto box=mfi.fabbox(); box.setBig(0,-1);
        auto const a=rho.array(mfi);
        amrex::ParallelFor(box,rho.nComp(),[=] AMREX_GPU_DEVICE(int i,int j,int k,int n) {
            a(i,j,k,n)=a(-i,j,k,n);
        });
    }
}

void WarpX::FoldRZBoundaryMoments (MultiFab& rho, const std::array<MultiFab*,3>& current)
{
    FoldRZBoundaryCharge(rho);
    amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM> const lo{
        ParticleBoundaryType::None,ParticleBoundaryType::Reflecting};
    amrex::Array<ParticleBoundaryType,AMREX_SPACEDIM> const hi{
        ParticleBoundaryType::Reflecting,ParticleBoundaryType::Reflecting};
    PEC::ApplyReflectiveBoundarytoJfield(current[0],current[1],current[2],
        field_boundary_lo,field_boundary_hi,lo,hi,Geom(0),0,PatchType::fine,ref_ratio);
    ApplyFieldBoundaryOnAxis(current[0],current[1],current[2],0);
}

void WarpX::PrepareRZBoundaryLossDensity ()
{
    if (!m_rz_boundary_current_pending || m_rz_boundary_loss_ready) { return; }
    auto const& rho=*m_fields.get(FieldType::rho_fp,0);
    auto const ng=amrex::max(amrex::max(rho.nGrowVect(),
        get_ng_depos_J()+amrex::IntVect(1)),
        m_fields.get(FieldType::Efield_fp,ablastr::fields::Direction{2},0)->nGrowVect());
    if (!m_rz_boundary_total_loss || m_rz_boundary_total_loss->boxArray()!=rho.boxArray()
        || m_rz_boundary_total_loss->DistributionMap()!=rho.DistributionMap()) {
        m_rz_boundary_total_loss=std::make_unique<MultiFab>(rho.boxArray(),rho.DistributionMap(),2,ng);
    }
    m_rz_boundary_total_loss->setVal(0._rt);
    for (auto& [name,entry] : m_rz_boundary_moments) {
        amrex::ignore_unused(name);
        auto& lost=*entry.lost_charge;
        ApplyInverseVolumeScalingToChargeDensity(&lost,0);
        WarpXSumGuardCells(lost,Geom(0).periodicity(),lost.nGrowVect(),0,lost.nComp());
        FoldRZBoundaryCharge(lost);
        lost.FillBoundary(Geom(0).periodicity());
        MultiFab::Add(*m_rz_boundary_total_loss,lost,0,0,2,ng);
    }
    m_rz_boundary_loss_ready=true;
}

void WarpX::AddRZAbsorbedCurrent (MultiFab const& loss,
    const std::array<MultiFab*,3>& current) const
{
    int const nz=Geom(0).Domain().bigEnd(1)+1;
    int const band=std::min(loss.nGrowVect()[1],nz-1);
    auto const factor=Geom(0).CellSize(1)/dt[0];
    for (amrex::MFIter mfi(*current[2],false);mfi.isValid();++mfi) {
        bool const low=mfi.validbox().smallEnd(1)==0;
        bool const high=mfi.validbox().bigEnd(1)==nz-1;
        if (!low && !high) { continue; }
        auto const source=loss.const_array(mfi);
        auto const jz=current[2]->array(mfi);
        auto const box=mfi.fabbox();
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            amrex::Real value=0._rt;
            if (low && j<band) {
                int const mirror=j<0 ? -j-1 : j;
                amrex::Real inside=0._rt;
                for (int n=mirror+1;n<=band;++n) { inside-=factor*source(i,n,k,0); }
                if (j<0) {
                    auto flux=-.5_rt*factor*source(i,0,k,0);
                    for (int n=1;n<=band;++n) { flux-=factor*source(i,n,k,0); }
                    value+=2._rt*flux-inside;
                } else { value+=inside; }
            }
            if (high && j>=nz-band) {
                int const mirror=j>=nz ? 2*nz-1-j : j;
                amrex::Real inside=0._rt;
                for (int n=nz-band;n<=mirror;++n) { inside+=factor*source(i,n,k,1); }
                if (j>=nz) {
                    auto flux=.5_rt*factor*source(i,nz,k,1);
                    for (int n=nz-band;n<nz;++n) { flux+=factor*source(i,n,k,1); }
                    value+=2._rt*flux-inside;
                } else { value+=inside; }
            }
            jz(i,j,k)+=value;
        });
    }
}
#endif
