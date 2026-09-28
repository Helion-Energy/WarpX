/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ImplicitIonEndpointTrial.H"
#include <AMReX_ParIter.H>
#include <AMReX_Reduce.H>
#include <cmath>
#include <set>
#include <tuple>
namespace warpx::thermal {
ImplicitIonEndpointTrial::ImplicitIonEndpointTrial (
    const AcceptedIonExchange& layout)
    : m_layout(layout) {
    for (std::size_t s = 0; s < layout.Descriptors().size(); ++s) {
        auto pc = std::make_unique<Container>(
            layout.Geometry(), layout.Distribution(), layout.Cells());
        pc->SetArena(amrex::The_Arena());
        m_particles.push_back(std::move(pc));
    }
}
bool
ImplicitIonEndpointTrial::Refresh (
    const std::vector<ImplicitIonMidpointTile>& inputs) {
    using P = amrex::ParticleReal;
    m_valid = false;
    m_views.clear();
    m_indices.clear();
    for (auto& p : m_particles)
        p->clearParticles();
    auto const& geometry = m_layout.Geometry();
    auto const lo = geometry.ProbLoArray(), hi = geometry.ProbHiArray();
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        periodic[d] = geometry.isPeriodic(d);
    bool ok = true;
    std::set<std::tuple<std::size_t, int, int>> seen;
    for (auto const& t : inputs) {
        ok = ok && t.species < m_particles.size() && t.count >= 0 &&
             t.grid >= 0 && t.grid < m_layout.Cells().size();
        if (t.count == 0)
            continue;
        ok = ok && t.idcpu && t.weight &&
             seen.emplace(t.species, t.grid, t.tile).second;
        for (auto p : t.position)
            ok = ok && p;
        for (int d = 0; d < 3; ++d)
            ok = ok && t.momentum[d] && t.old_position[d] && t.old_momentum[d];
#ifdef WARPX_DIM_RZ
        ok = ok && t.theta;
#endif
        if (t.grid >= 0 && t.grid < m_layout.Cells().size())
            ok = ok && m_layout.Distribution()[t.grid] ==
                           amrex::ParallelDescriptor::MyProc();
    }
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok)
        return false;
    amrex::ReduceOps<amrex::ReduceOpMin,amrex::ReduceOpSum> op;
    amrex::ReduceData<int,amrex::Long> data(op);
    op.eval(1, data,
            [] AMREX_GPU_DEVICE(int) { return amrex::GpuTuple<int,amrex::Long>{1,0}; });
    amrex::Long count_before = 0;
    for (auto const& t : inputs) {
        if (t.count == 0 ||
            m_layout.Descriptors()[t.species].relaxation_excluded)
            continue;
        count_before += t.count;
        auto& tile = m_particles[t.species]->DefineAndReturnParticleTile(
            0, t.grid, t.tile);
        tile.resize(t.count);
        auto out = tile.getParticleTileData();
        op.eval(t.count, data, [=] AMREX_GPU_DEVICE(long n) {
            amrex::GpuArray<P, 3> mid{}, end{};
#ifdef WARPX_DIM_RZ
            mid = {t.position[0][n] * std::cos(t.theta[n]),
                   t.position[0][n] * std::sin(t.theta[n]), t.position[1][n]};
#elif defined(WARPX_DIM_3D)
            mid={t.position[0][n],t.position[1][n],t.position[2][n]};
#else
            return amrex::GpuTuple<int,amrex::Long>{0,0};
#endif
            bool good = amrex::ConstParticleIDWrapper{t.idcpu[n]}.is_valid() &&
                        std::isfinite(t.weight[n]) && t.weight[n] >= 0;
            for (int d = 0; d < 3; ++d) {
                end[d] = 2. * mid[d] - t.old_position[d][n];
                P const u = 2. * t.momentum[d][n] - t.old_momentum[d][n];
                good = good && std::isfinite(end[d]) && std::isfinite(u);
                out.m_rdata[Ux + d][n] = u;
            }
#ifdef WARPX_DIM_RZ
            out.m_rdata[0][n] = std::sqrt(end[0] * end[0] + end[1] * end[1]);
            out.m_rdata[1][n] = end[2];
            out.m_rdata[Theta][n] = std::atan2(end[1], end[0]);
#else
            for(int d=0;d<AMREX_SPACEDIM;++d)out.m_rdata[d][n]=end[d];
            out.m_rdata[Theta][n]=0;
#endif
            warpx::implicit::AbsorbingEndpointResult decision;
            if (t.allow_deterministic_absorption) {
                amrex::GpuArray<P,3> u{out.m_rdata[Ux][n],out.m_rdata[Uy][n],out.m_rdata[Uz][n]};
                decision=warpx::implicit::InspectAbsorbingEndpoint(end,u,t.absorption);
                good=good && (decision.status==warpx::implicit::AbsorbingEndpointStatus::Interior ||
                              decision.status==warpx::implicit::AbsorbingEndpointStatus::Absorbed);
            }
#ifdef WARPX_DIM_RZ
            if (good && decision.reflected_faces) {
                amrex::GpuArray<P,3> const u{out.m_rdata[Ux][n],out.m_rdata[Uy][n],out.m_rdata[Uz][n]};
                auto const image=warpx::implicit::MapRadialEndpoint(end,u,t.absorption.low[0],
                    t.absorption.high[0],t.absorption.native,t.absorption.allow_radial_reflection);
                good=good && image.valid && image.reflected;
                out.m_rdata[0][n]=image.stored[0];
                out.m_rdata[1][n]=image.stored[2];
                out.m_rdata[Theta][n]=image.stored[1];
                for (int d=0;d<3;++d) { out.m_rdata[Ux+d][n]=image.momentum[d]; }
            }
#endif
            bool const absorbed=decision.status==warpx::implicit::AbsorbingEndpointStatus::Absorbed;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                P const x = out.m_rdata[d][n];
                good = good && std::isfinite(x) &&
                       (absorbed || periodic[d] || (x >= lo[d] && x < hi[d]));
            }
            out.m_rdata[Weight][n] = t.weight[n];
            out.m_idcpu[n] = t.idcpu[n];
            if (absorbed) { amrex::ParticleIDWrapper{out.m_idcpu[n]}.make_invalid(); }
            return amrex::GpuTuple<int,amrex::Long>{int(good),amrex::Long(absorbed)};
        });
    }
    auto const decisions=data.value();
    ok = amrex::get<0>(decisions) != 0;
    amrex::Long count_absorbed=amrex::get<1>(decisions);
    amrex::ParallelDescriptor::ReduceBoolAnd(ok);
    if (!ok)
        return false;
    auto const plo = geometry.ProbLoArray(), dxi = geometry.InvCellSizeArray();
    amrex::Long count_after = 0;
    for (std::size_t s = 0; s < m_particles.size(); ++s) {
        auto& pc = *m_particles[s];
        pc.Redistribute();
        for (Container::ParIterType pti(pc, 0); pti.isValid(); ++pti) {
            long const count = pti.numParticles();
            if (count == 0)
                continue;
            count_after += count;
            m_indices.emplace_back(count);
            auto* cell = m_indices.back().data();
            auto pt = pti.GetParticleTile().getParticleTileData();
            amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(long n) {
                cell[n] = amrex::getParticleCell(Container::ParticleType(pt, n),
                                                 plo, dxi);
            });
            auto& soa = pti.GetStructOfArrays();
            IonExchangeParticleView v;
            v.species = s;
            v.grid_index = pti.index();
            v.count = count;
            v.cell = cell;
            v.theta = soa.GetRealData(Theta).data();
            v.weight = soa.GetRealData(Weight).data();
            for (int d = 0; d < 3; ++d)
                v.momentum[d] = soa.GetRealData(Ux + d).data();
            m_views.push_back(v);
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(count_before);
    amrex::ParallelDescriptor::ReduceLongSum(count_after);
    amrex::ParallelDescriptor::ReduceLongSum(count_absorbed);
    m_valid = count_before-count_absorbed == count_after;
    return m_valid;
}
const std::vector<IonExchangeParticleView>&
ImplicitIonEndpointTrial::Views () const {
    AMREX_ALWAYS_ASSERT(m_valid);
    return m_views;
}
} // namespace warpx::thermal
