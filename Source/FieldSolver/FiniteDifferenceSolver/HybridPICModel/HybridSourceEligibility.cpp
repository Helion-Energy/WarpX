/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "HybridSourceEligibility.H"
#include "HybridPICModel.H"
#include "QdsmcVolumeElement.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"
#include <AMReX_GpuAtomic.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <set>

using namespace amrex::literals;
using warpx::fields::FieldType;
namespace {
amrex::Real Integral (amrex::MultiFab const& f, int comp, int lev)
{
    auto const& g = WarpX::GetInstance().Geom(lev);
    auto const volume = MakeQdsmcVolumeElement(g, f.ixType());
    amrex::MultiFab weighted(f.boxArray(), f.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(weighted); mfi.isValid(); ++mfi) {
        auto const a = f.const_array(mfi);
        auto const b = weighted.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
            b(i,j,k) = a(i,j,k,comp)*volume(i,j,k);
        });
    }
    return weighted.sum_unique(0, false, g.periodicity()); // already global
}
char const* Name (int c) { return c == 0 ? "ohmic" : c == 1 ? "thermal" : "stopping"; }
}

void HybridSourceEligibility::ReadParameters ()
{
    amrex::ParmParse pp("hybrid_pic_model.source_guard");
    pp.query("enabled", m_enabled);
    pp.query("ohmic", m_channels[Ohmic]);
    pp.query("thermal_relaxation", m_channels[Thermal]);
    pp.query("alpha_stopping", m_channels[Stopping]);
    pp.queryarr("background_species", m_background);
    utils::parser::queryWithParser(pp,"minimum_resident_markers",m_minimum);
    utils::parser::queryWithParser(pp,"n_min",m_n_min);
    pp.query("axis_cells",m_axis_cells);
    pp.query("diagnostic_interval",m_diagnostic_interval);
    pp.query("export_interval",m_export_interval);
    pp.query("output_prefix",m_output_prefix);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_minimum > 0 && m_axis_cells >= 0 &&
        std::isfinite(m_n_min) && m_n_min >= 0 && m_diagnostic_interval > 0 &&
        m_export_interval >= 0 && !m_output_prefix.empty(),
        "source_guard: invalid threshold, axis_cells, interval or output_prefix");
    if (m_enabled) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_background.empty() &&
            std::set<std::string>(m_background.begin(),m_background.end()).size() ==
                m_background.size(), "source_guard: background_species must be explicit and unique");
    }
}

void HybridSourceEligibility::Validate (HybridPICModel const& model) const
{
    if (!m_enabled) { return; }
    auto& w = WarpX::GetInstance();
#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("source_guard currently requires RZ geometry");
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(w.maxLevel() == 0 && WarpX::n_rz_azimuthal_modes == 1 && w.Geom(0).ProbLo(0) == 0 &&
        m_axis_cells < w.Geom(0).Domain().length(0) && !w.Geom(0).isPeriodic(0),
        "source_guard requires single-level RZ including the nonperiodic axis");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(model.m_solve_electron_energy_equation &&
        w.evolve_scheme == EvolveScheme::Explicit,
        "source_guard currently requires explicit electron-energy evolution");
    auto& mpc = w.GetPartContainer();
    auto const names = mpc.GetSpeciesNames();
    for (auto const& name : m_background) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(std::find(names.begin(),names.end(),name) != names.end(),
            "source_guard: unknown background species " + name);
        auto const& pc = mpc.GetParticleContainerFromName(name);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pc.getCharge() > 0 && pc.getMass() > 0 &&
            !model.IsRelaxationExcluded(name),
            "source_guard background must be a positive-mass thermal ion: " + name);
    }
    amrex::Print() << "[source_guard] enabled; raw resident cutoff=" << m_minimum
        << "; axis cells=" << m_axis_cells << "; physical n_min=" << m_n_min
        << "; channels=" << m_channels[0] << ',' << m_channels[1] << ',' << m_channels[2]
        << "; linear-contributor Neff is diagnostic, not temperature-estimator Neff\n";
}

void HybridSourceEligibility::BeginBatch (int lev, amrex::MultiFab const& rho,
                                          HybridPICModel const& model)
{
    if (!m_enabled) { return; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_batch,"source_guard: nested source batch");
    Prepare(lev,rho,model);
    m_batch = true;
}

void HybridSourceEligibility::Prepare (int lev, amrex::MultiFab const& rho,
                                       HybridPICModel const& model)
{
    if (!m_enabled || m_batch) { return; }
    ABLASTR_PROFILE("HybridSourceEligibility::Prepare");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0 && rho.ixType().nodeCentered(),
        "source_guard requires a level-zero nodal physical charge density");
    auto& w = WarpX::GetInstance();
    auto& mpc = w.GetPartContainer();
    auto const& g = w.Geom(lev);
    auto const cc = amrex::convert(rho.boxArray(),amrex::IntVect::TheCellVector());
    auto const& dm = rho.DistributionMap();
    int const ns = static_cast<int>(m_background.size());
    if (m_nodes && (m_nodes->boxArray() != rho.boxArray() ||
                    m_nodes->DistributionMap() != dm)) {
        for (auto& d : m_delta) d.reset();
        m_relaxation.clear();
    }
    m_counts = std::make_unique<amrex::MultiFab>(cc,dm,2*ns,1);
    m_contributors = std::make_unique<amrex::MultiFab>(rho.boxArray(),dm,2*ns,1);
    m_counts->setVal(0);
    m_contributors->setVal(0);
    auto const plo = g.ProbLoArray(), dxi = g.InvCellSizeArray();
    for (int s = 0; s < ns; ++s) {
        auto& pc = mpc.GetParticleContainerFromName(m_background[s]);
        amrex::MultiFab resident(pc.ParticleBoxArray(lev),pc.ParticleDistributionMap(lev),2,0);
        amrex::MultiFab weights(amrex::convert(pc.ParticleBoxArray(lev),
            amrex::IntVect::TheNodeVector()),pc.ParticleDistributionMap(lev),2,1);
        resident.setVal(0);
        weights.setVal(0);
        // Scatter: unthreaded tiles and For, not SIMD ParallelFor on CPU.
        for (WarpXParIter pti(pc,lev); pti.isValid(); ++pti) {
            auto const a = resident.array(pti), b = weights.array(pti);
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            auto const pos = GetParticlePosition<PIdx>(pti);
            auto const* weight = pti.GetAttribs(PIdx::w).dataPtr();
            amrex::For(pti.numParticles(), [=] AMREX_GPU_DEVICE(long p) {
                auto const particle = WarpXParticleContainer::ParticleType(ptd,p);
                if (particle.id() <= 0) { return; }
                auto const [i,j,k] = amrex::getParticleCell(particle,plo,dxi).dim3();
                amrex::Gpu::Atomic::AddNoRet(&a(i,j,k,0),1.0_rt);
                amrex::Gpu::Atomic::AddNoRet(&a(i,j,k,1),amrex::Real(weight[p]));
                amrex::ParticleReal x,y,z;
                pos(p,x,y,z);
                int ii=0,jj=0,kk=0;
                amrex::Real shape[AMREX_SPACEDIM][2];
                ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                    x,y,z,plo,dxi,ii,jj,kk,shape);
                for (int c=0;c<(1<<AMREX_SPACEDIM);++c) {
                    int const di=c&1, dj=AMREX_SPACEDIM>1?((c>>1)&1):0;
                    int const dk=AMREX_SPACEDIM>2?((c>>2)&1):0;
                    amrex::Real v=weight[p]*shape[0][di];
#if AMREX_SPACEDIM >= 2
                    v*=shape[1][dj];
#endif
#if AMREX_SPACEDIM == 3
                    v*=shape[2][dk];
#endif
                    amrex::Gpu::Atomic::AddNoRet(&b(ii+di,jj+dj,kk+dk,0),v);
                    amrex::Gpu::Atomic::AddNoRet(&b(ii+di,jj+dj,kk+dk,1),v*v);
                }
            });
        }
        weights.SumBoundary(g.periodicity());
        m_counts->ParallelCopy(resident,0,2*s,2);
        m_contributors->ParallelCopy(weights,0,2*s,2,amrex::IntVect(0),
                                     amrex::IntVect(1),g.periodicity());
    }
    m_counts->FillBoundary(g.periodicity());
    m_contributors->OverrideSync(g.periodicity());
    m_contributors->FillBoundary(g.periodicity());
    auto previous = std::move(m_cells);
    m_cells = std::make_unique<amrex::MultiFab>(cc,dm,3,1);
    m_nodes = std::make_unique<amrex::MultiFab>(rho.boxArray(),dm,3,1);
    m_cells->setVal(0);
    m_nodes->setVal(0);
    amrex::GpuArray<amrex::Real,3> const floors{{
        PhysConst::q_e*std::max({m_n_min,model.m_n_floor,model.m_joule_heating_n_min}),
        PhysConst::q_e*std::max(m_n_min,model.m_n_floor),
        PhysConst::q_e*std::max(m_n_min,model.m_n_floor)}};
    int const minimum=m_minimum, axis=m_axis_cells;
    auto const dom = g.Domain();
    auto const dlo = dom.smallEnd();
    for (amrex::MFIter mfi(*m_cells);mfi.isValid();++mfi) {
        auto const counts=m_counts->const_array(mfi), r=rho.const_array(mfi);
        auto const out=m_cells->array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            int common = i-dlo[0]<axis ? Axis : 0;
            for (int s=0;s<ns;++s) { if (counts(i,j,k,2*s)<minimum) common|=LowCount; }
            for (int ch=0;ch<3;++ch) {
                int reason=common;
                for(int c=0;c<(1<<AMREX_SPACEDIM);++c) {
                    int const di=c&1,dj=AMREX_SPACEDIM>1?((c>>1)&1):0;
                    int const dk=AMREX_SPACEDIM>2?((c>>2)&1):0;
                    amrex::Real const value=r(i+di,j+dj,k+dk);
                    if (!(std::isfinite(value) && value>floors[ch])) { reason|=Vacuum; }
                }
                out(i,j,k,ch)=reason;
            }
        });
    }
    m_cells->FillBoundary(g.periodicity());
    if (previous && (previous->boxArray() != m_cells->boxArray() ||
                     previous->DistributionMap() != m_cells->DistributionMap())) previous.reset();
    amrex::GpuArray<int,AMREX_SPACEDIM> per{};
    for(int d=0;d<AMREX_SPACEDIM;++d) { per[d]=g.isPeriodic(d); }
    for(amrex::MFIter mfi(*m_nodes);mfi.isValid();++mfi) {
        auto const a=m_cells->const_array(mfi);
        auto const b=m_nodes->array(mfi);
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            for(int ch=0;ch<3;++ch) {
                int reason=0;
                for(int c=0;c<(1<<AMREX_SPACEDIM);++c) {
                    int const ii=i-(c&1),jj=j-(AMREX_SPACEDIM>1?((c>>1)&1):0);
                    int const kk=k-(AMREX_SPACEDIM>2?((c>>2)&1):0);
                    amrex::IntVect const cell(AMREX_D_DECL(ii,jj,kk));
                    bool inside=true;
                    for(int d=0;d<AMREX_SPACEDIM;++d) {
                        if(!per[d] && (cell[d]<dom.smallEnd(d)||cell[d]>dom.bigEnd(d))) inside=false;
                    }
                    if(inside) reason|=static_cast<int>(a(ii,jj,kk,ch));
                }
                b(i,j,k,ch)=reason;
            }
        });
    }
    m_nodes->OverrideSync(g.periodicity());
    m_nodes->FillBoundary(g.periodicity());
    ++m_epoch;
    if(w.getistep(0)%m_diagnostic_interval==0) { Coverage(lev,previous.get()); }
    Export(lev,"mask");
}

amrex::MultiFab& HybridSourceEligibility::NewRelaxation (std::string const& s,
                                                        amrex::MultiFab const& te)
{
    auto& p=m_relaxation[s];
    p=std::make_unique<amrex::MultiFab>(te.boxArray(),te.DistributionMap(),3,1);
    p->setVal(0);
    return *p;
}
amrex::MultiFab const& HybridSourceEligibility::Relaxation (std::string const& s) const
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_relaxation.count(s) && m_relaxation_epoch == m_epoch,
        "source_guard: ion relaxation requires same-stage electron coefficients for "+s);
    return *m_relaxation.at(s);
}

void HybridSourceEligibility::BeginChannel (Channel c, amrex::MultiFab const& te)
{
    if(!m_enabled) return;
    m_before[c]=std::make_unique<amrex::MultiFab>(te.boxArray(),te.DistributionMap(),1,0);
    m_delta[c]=std::make_unique<amrex::MultiFab>(te.boxArray(),te.DistributionMap(),1,0);
    amrex::MultiFab::Copy(*m_before[c],te,0,0,1,0);
}
void HybridSourceEligibility::EndChannel (int lev,Channel c,amrex::Real dt,
    amrex::MultiFab const& te,amrex::MultiFab const& capacity,
    amrex::MultiFab const* pedestal,amrex::Real gamma)
{
    if(!m_enabled) return;
    amrex::MultiFab energy(te.boxArray(),te.DistributionMap(),1,0);
    bool const ped=pedestal!=nullptr;
    for(amrex::MFIter mfi(energy);mfi.isValid();++mfi) {
        auto const t=te.const_array(mfi), before=m_before[c]->const_array(mfi);
        auto const rho=capacity.const_array(mfi);
        auto const e=energy.array(mfi), d=m_delta[c]->array(mfi);
        auto const p=ped?pedestal->const_array(mfi):amrex::Array4<amrex::Real const>{};
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            auto const delta=t(i,j,k)-before(i,j,k);
            d(i,j,k)=delta;
            // A skipped vacuum interaction has zero energy even with zero capacity.
            e(i,j,k)=delta==0 ? 0 : delta*(rho(i,j,k)+(ped?p(i,j,k):0))
                *PhysConst::kb/(PhysConst::q_e*(gamma-1));
        });
    }
    m_last_electron_J[c]=Integral(energy,0,lev);
    amrex::Print() << std::setprecision(17) << "[source_guard_energy] step="
        << WarpX::GetInstance().getistep(0) << " epoch=" << m_epoch << " channel=" << Name(c)
        << " electron_J=" << m_last_electron_J[c] << " dt=" << dt
        << " electron_W=" << (dt>0?m_last_electron_J[c]/dt:0) << '\n';
    if(c==Stopping) {
        m_stopping_electron_J += m_last_electron_J[c];
        amrex::Print() << std::setprecision(17) << "[source_guard_stopping] epoch=" << m_epoch
            << " cumulative_signed_particle_J=" << m_stopping_particle_J
            << " cumulative_delivered_electron_J=" << m_stopping_electron_J
            << " residual_or_pending_J=" << m_stopping_particle_J+m_stopping_electron_J
            << " collision_dt=" << m_stopping_dt << '\n';
    }
    m_before[c].reset();
}
void HybridSourceEligibility::RecordOhmic (int lev,amrex::MultiFab const& tally,amrex::Real dt)
{
    auto const declined=Integral(tally,0,lev);
    auto const unresolved=tally.sum_unique(1,false,WarpX::GetInstance().Geom(lev).periodicity());
    m_declined_ohmic_J+=declined;
    m_unevaluable_ohmic_nodes+=unresolved;
    amrex::Print() << std::setprecision(17) << "[source_guard_ohmic] epoch=" << m_epoch
        << " declined_modeled_J=" << declined << " declined_modeled_W=" << (dt>0?declined/dt:0)
        << " unevaluable_species_nodes=" << unresolved
        << " cumulative_declined_J=" << m_declined_ohmic_J << '\n';
}
void HybridSourceEligibility::RecordStoppingParticle (amrex::Real actual, amrex::Real dt)
{
    m_stopping_particle_J+=actual;
    m_stopping_dt=dt;
    amrex::Print() << std::setprecision(17) << "[source_guard_particle_power] channel=stopping epoch="
        << m_epoch << " actual_particle_J=" << actual << " collision_dt=" << dt
        << " actual_particle_W=" << (dt>0?actual/dt:0) << '\n';
}
void HybridSourceEligibility::RecordIon (amrex::Real actual, amrex::Real thermal,
    amrex::Real redirected, amrex::Real dt, bool redirect)
{
    m_last_ion_J=actual;
    amrex::Print() << std::setprecision(17) << "[source_guard_exchange] epoch=" << m_epoch
        << " actual_ion_J=" << actual << " electron_thermal_J=" << m_last_electron_J[Thermal]
        << " thermal_shared_draw_J=" << thermal << " redirect_increment_J=" << redirected
        << " thermal_residual_J=" << thermal+m_last_electron_J[Thermal] << " dt=" << dt
        << " actual_ion_W=" << (dt>0?actual/dt:0)
        << " thermal_shared_draw_W=" << (dt>0?thermal/dt:0)
        << " redirect_increment_W=" << (dt>0?redirected/dt:0)
        << " includes_redirect=" << redirect
        << " closure=stochastic_with_splitting_error\n";
}

void HybridSourceEligibility::Coverage (int lev,amrex::MultiFab const* previous)
{
    auto const& g=WarpX::GetInstance().Geom(lev);
    int const ns=static_cast<int>(m_background.size());
    for(int ch=0;ch<3;++ch) {
        amrex::MultiFab tally(m_cells->boxArray(),m_cells->DistributionMap(),10+2*ns,0);
        for(amrex::MFIter mfi(tally);mfi.isValid();++mfi) {
            auto const a=m_cells->const_array(mfi), count=m_counts->const_array(mfi);
            auto const b=tally.array(mfi);
            bool const old=previous!=nullptr;
            auto const p=old?previous->const_array(mfi):amrex::Array4<amrex::Real const>{};
            auto const vol=MakeQdsmcVolumeElement(g,tally.ixType());
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                int const reason=static_cast<int>(a(i,j,k,ch));
                for(int c=0;c<8;++c) b(i,j,k,c)=reason==c?vol(i,j,k):0;
                b(i,j,k,8)=old && p(i,j,k,ch)!=reason?1:0;
                b(i,j,k,9)=reason!=0?1:0;
                for(int s=0;s<ns;++s) {
                    b(i,j,k,10+2*s)=count(i,j,k,2*s+1);
                    b(i,j,k,11+2*s)=reason!=0?count(i,j,k,2*s+1):0;
                }
            });
        }
        amrex::Print() << "[source_guard_coverage] epoch=" << m_epoch << " channel=" << Name(ch);
        for(int b=0;b<8;++b) {
            auto const value=tally.sum(b);
            amrex::Print() << " reason" << b << "_m3=" << value;
        }
        auto const changes=tally.sum(8), excluded=tally.sum(9);
        amrex::Print() << " changed_cells=" << changes << " excluded_cells=" << excluded;
        for(int s=0;s<ns;++s) {
            auto const total=tally.sum(10+2*s), rejected=tally.sum(11+2*s);
            amrex::Print() << ' ' << m_background[s] << "_inventory=" << total
                << ' ' << m_background[s] << "_excluded_inventory=" << rejected;
        }
        amrex::Print() << '\n';
        amrex::MultiFab nodal(m_nodes->boxArray(),m_nodes->DistributionMap(),1,0);
        for(amrex::MFIter mfi(nodal);mfi.isValid();++mfi) {
            auto const a=m_nodes->const_array(mfi);
            auto const b=nodal.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                b(i,j,k)=a(i,j,k,ch)!=0?1:0;
            });
        }
        auto const v=Integral(nodal,0,lev);
        amrex::Print() << "[source_guard_coverage] epoch=" << m_epoch << " channel=" << Name(ch)
            << " excluded_nodal_volume_m3=" << v << '\n';
    }
    // Disjoint raw-resident count bins. Inventory is a sum of macro weights, not density.
    for(int s=0;s<ns;++s) {
        amrex::MultiFab hist(m_counts->boxArray(),m_counts->DistributionMap(),6,0);
        for(amrex::MFIter mfi(hist);mfi.isValid();++mfi) {
            auto const a=m_counts->const_array(mfi);
            auto const b=hist.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                auto const n=a(i,j,k,2*s);
                int const bin=n==0?0:n<8?1:n<16?2:n<32?3:n<64?4:5;
                for(int c=0;c<6;++c) b(i,j,k,c)=c==bin?1:0;
            });
        }
        amrex::Print() << "[source_guard_counts] epoch=" << m_epoch << " species=" << m_background[s];
        for(int c=0;c<6;++c) { auto const n=hist.sum(c); amrex::Print() << " bin" << c << '=' << n; }
        amrex::Print() << " bins=0,1:7,8:15,16:31,32:63,64+\n";
    }
}

void HybridSourceEligibility::Export (int lev,char const* phase) const
{
    auto const step=WarpX::GetInstance().getistep(0);
    if(m_export_interval<=0 || step%m_export_interval!=0) return;
    auto const dir=m_output_prefix+"_step"+std::to_string(step)+"_epoch"+
        std::to_string(m_epoch)+"_"+phase;
    amrex::UtilCreateDirectory(dir,0755);
    amrex::ParallelDescriptor::Barrier();
    amrex::VisMF::Write(*m_cells,dir+"/cell_reason_ohmic_thermal_stopping");
    amrex::VisMF::Write(*m_nodes,dir+"/node_reason_ohmic_thermal_stopping");
    amrex::VisMF::Write(*m_counts,dir+"/resident_count_weight_by_species");
    amrex::VisMF::Write(*m_contributors,dir+"/linear_sum_a_sum_a2_by_species");
    amrex::ignore_unused(lev);
}

void HybridSourceEligibility::Phase (int lev,char const* phase,HybridPICModel const& model,bool abort)
{
    if(!m_enabled || !m_nodes) return;
    auto& w=WarpX::GetInstance();
    if(!abort && w.getistep(0)%m_diagnostic_interval!=0) return;
    auto const& te=*w.m_fields.get(FieldType::hybrid_electron_temperature_fp,lev);
    auto const& rho=*w.m_fields.get(FieldType::rho_fp,lev);
    if (m_nodes->boxArray() != te.boxArray() || m_nodes->DistributionMap() != te.DistributionMap())
        return; // A regrid/load balance is followed by a fresh source Prepare.
    auto const* pedestal=model.DensityPedestal(lev);
    amrex::MultiFab maximum(te.boxArray(),te.DistributionMap(),1,0);
    amrex::MultiFab::Copy(maximum,te,0,0,1,0);
    // The existing abort is an open-set maximum; preserve its exact selection.
    if(abort) {
        auto const floor=PhysConst::q_e*model.m_n_floor;
        for(amrex::MFIter mfi(maximum);mfi.isValid();++mfi) {
            auto const a=maximum.array(mfi);
            auto const r=rho.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                if(!(r(i,j,k)>floor)) a(i,j,k)=-1;
            });
        }
    }
    auto const index=maximum.maxIndex(0);
    int owner=amrex::ParallelDescriptor::NProcs();
    for(amrex::MFIter mfi(te);mfi.isValid();++mfi) {
        if(mfi.validbox().contains(index)) owner=amrex::ParallelDescriptor::MyProc();
    }
    amrex::ParallelDescriptor::ReduceIntMin(owner);
    int const ns=static_cast<int>(m_background.size()), ncomp=9+3*ns;
    amrex::MultiFab state(te.boxArray(),te.DistributionMap(),ncomp,0);
    bool const ped=pedestal!=nullptr;
    for(amrex::MFIter mfi(state);mfi.isValid();++mfi) {
        auto const out=state.array(mfi);
        auto const t=te.const_array(mfi), r=rho.const_array(mfi);
        auto const p=ped?pedestal->const_array(mfi):amrex::Array4<amrex::Real const>{};
        auto const reasons=m_nodes->const_array(mfi), count=m_counts->const_array(mfi);
        auto const moments=m_contributors->const_array(mfi);
        amrex::GpuArray<amrex::Array4<amrex::Real const>,3> delta{};
        for(int c=0;c<3;++c) if(m_delta[c]) delta[c]=m_delta[c]->const_array(mfi);
        auto const dom=w.Geom(lev).Domain();
        amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            out(i,j,k,0)=t(i,j,k)*PhysConst::kb/PhysConst::q_e;
            out(i,j,k,1)=r(i,j,k)/PhysConst::q_e;
            out(i,j,k,2)=ped?p(i,j,k)/PhysConst::q_e:0;
            for(int c=0;c<3;++c) {
                out(i,j,k,3+c)=reasons(i,j,k,c);
                out(i,j,k,6+c)=delta[c]?delta[c](i,j,k)*PhysConst::kb/PhysConst::q_e:0;
            }
            for(int s=0;s<ns;++s) {
                // Report resident cell on the upper/right side, clipped at physical high faces.
                int const ii=amrex::min(i,dom.bigEnd(0));
                int const jj=AMREX_SPACEDIM>1?amrex::min(j,dom.bigEnd(1)):0;
                int const kk=AMREX_SPACEDIM>2?amrex::min(k,dom.bigEnd(2)):0;
                out(i,j,k,9+3*s)=count(ii,jj,kk,2*s);
                auto const a=moments(i,j,k,2*s), a2=moments(i,j,k,2*s+1);
                out(i,j,k,10+3*s)=a2>0?a*a/a2:0;
                out(i,j,k,11+3*s)=0;
            }
        });
        for(int s=0;s<ns;++s) {
            auto const it=m_relaxation.find(m_background[s]);
            if(it==m_relaxation.end()) continue;
            auto const a=it->second->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                out(i,j,k,11+3*s)=a(i,j,k,2);
            });
        }
    }
    auto box=amrex::grow(amrex::Box(index,index,te.ixType()),abort?2:0);
    box &= amrex::convert(w.Geom(lev).Domain(),te.ixType());
    amrex::DistributionMapping const dm(amrex::Vector<int>{0});
    amrex::MultiFab window(amrex::BoxArray(box),dm,ncomp,0);
    window.ParallelCopy(state,0,0,ncomp);
    if(amrex::ParallelDescriptor::IOProcessor()) {
        amrex::FArrayBox host(box,ncomp,amrex::The_Pinned_Arena());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,window[0].dataPtr(),
            window[0].dataPtr()+window[0].size(),host.dataPtr());
        auto const& g=w.Geom(lev);
        amrex::Print() << std::setprecision(17) << "[source_guard_max] step=" << w.getistep(0)
            << " epoch=" << m_epoch << " Ti_epoch=" << m_relaxation_epoch
            << " phase=" << phase << " rank=" << owner
            << " index=" << index << " r=" << g.ProbLo(0)+(index[0]-g.Domain().smallEnd(0))*g.CellSize(0)
#if AMREX_SPACEDIM >= 2
            << " z=" << g.ProbLo(1)+(index[1]-g.Domain().smallEnd(1))*g.CellSize(1)
#endif
            << " Te_eV=" << host(index,0) << " n_phys=" << host(index,1)
            << " n_ped=" << host(index,2);
        for(int c=0;c<3;++c) amrex::Print() << ' ' << Name(c) << "_delta_eV=" << host(index,6+c);
        for(int s=0;s<ns;++s) amrex::Print() << ' ' << m_background[s]
            << "_resident=" << host(index,9+3*s) << " linear_Neff=" << host(index,10+3*s)
            << " Ti_eV=" << host(index,11+3*s);
        amrex::Print() << '\n';
    }
    if(abort) {
        auto const dir=m_output_prefix+"_abort_step"+std::to_string(w.getistep(0));
        amrex::UtilCreateDirectory(dir,0755);
        amrex::ParallelDescriptor::Barrier();
        amrex::VisMF::Write(window,dir+"/native_neighborhood");
    }
}
