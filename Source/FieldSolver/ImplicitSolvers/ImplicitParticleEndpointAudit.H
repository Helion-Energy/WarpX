/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_PARTICLE_ENDPOINT_AUDIT_H_
#define WARPX_IMPLICIT_PARTICLE_ENDPOINT_AUDIT_H_
#include "ImplicitParticleRollback.H"
#include "ImplicitAbsorbingEndpoint.H"
#include "ImplicitAxialEndpoint.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

class WarpX;
namespace warpx::implicit {
enum class EndpointIssue {
    NonFinite,
    InvalidIdentity,
    OldOutsideDomain,
    AbsorbingBoundary,
    ThermalBoundary,
    OtherBoundaryAction,
    UnresolvedRadialWall,
    GatherReach,
    CurrentReach,
    EndpointReach,
    ProjectionReach,
    SupportedAbsorption,
    SupportedReflection,
    Count
};
inline constexpr int endpoint_issue_count =
    static_cast<int>(EndpointIssue::Count);
struct EndpointAuditResult {
    amrex::Long particles = 0;
    std::array<amrex::Long, endpoint_issue_count> counts{};
    [[nodiscard]] bool
    ok () const {
        for (int i=0;i<endpoint_issue_count;++i) {
            if (i != static_cast<int>(EndpointIssue::SupportedAbsorption) &&
                i != static_cast<int>(EndpointIssue::SupportedReflection) && counts[i] != 0) {
                return false;
            }
        }
        return true;
    }
    [[nodiscard]] amrex::Long
    count (EndpointIssue issue) const {
        return counts[static_cast<int>(issue)];
    }
    void
    reduceAcrossRanks () {
        amrex::ParallelDescriptor::ReduceLongSum(particles);
        amrex::ParallelDescriptor::ReduceLongSum(counts.data(),
                                                 endpoint_issue_count);
    }
    [[nodiscard]] std::string summary () const;
};
// These are the actual arrays used for the impending operations. Filled ghosts
// are an explicit caller contract; allocated ghosts alone do not establish it.
struct EndpointAuditFields {
    const amrex::MultiFab* endpoint_density = nullptr;
    std::array<const amrex::MultiFab*, 3> current{}, gather_e{}, gather_b{};
    amrex::IntVect gather_filled_ghosts{0};
    amrex::IntVect current_deposit_ghosts{0};
    bool allow_deterministic_absorption = false;
    bool allow_radial_reflection = false;
    bool allow_axial_reflection = false;
};
struct EndpointIndexRange {
    amrex::GpuArray<int, AMREX_SPACEDIM> low{}, high{}, nodal{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> origin{}, inverse{};
};
struct EndpointTileContract {
    EndpointIndexRange rho;
    amrex::GpuArray<EndpointIndexRange, 3> current;
    amrex::GpuArray<EndpointIndexRange, 6> gather;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> domain_low{}, domain_high{};
    amrex::GpuArray<ParticleBoundaryType, AMREX_SPACEDIM> boundary_low{},
        boundary_high{};
    bool deposits = true, gathers = true;
    bool allow_deterministic_absorption = false;
    AbsorbingEndpointBoundary absorption;
    AxialEndpointBoundary axial;
};

// Native host metadata builder shared by the production bridge and fixture.
inline EndpointTileContract
MakeEndpointTileContract (
    amrex::Geometry const& geom, amrex::Box const& particle_tile, int grid,
    EndpointAuditFields const& fields,
    amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM> const& low,
    amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM> const& high) {
    EndpointTileContract c;
    auto range = [&] (amrex::Box const& box) {
        EndpointIndexRange r;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            r.low[d] = box.smallEnd(d);
            r.high[d] = box.bigEnd(d);
            r.nodal[d] = box.ixType().nodeCentered(d);
            r.origin[d] = geom.ProbLo(d) + box.smallEnd(d) * geom.CellSize(d);
            r.inverse[d] = geom.InvCellSize(d);
        }
        return r;
    };
    c.rho = range((*fields.endpoint_density)[grid].box());
    for (int component = 0; component < 3; ++component) {
        auto const& f = *fields.current[component];
        auto box = amrex::convert(particle_tile, f.ixType());
        box.grow(fields.current_deposit_ghosts);
        box &= f[grid].box();
        c.current[component] = range(box);
        for (int family = 0; family < 2; ++family) {
            auto const& g = *(family == 0 ? fields.gather_e[component]
                                          : fields.gather_b[component]);
            auto gb = amrex::convert(particle_tile, g.ixType());
            gb.grow(fields.gather_filled_ghosts);
            gb &= g[grid].box();
            c.gather[3 * family + component] = range(gb);
        }
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        c.domain_low[d] = geom.ProbLo(d);
        c.domain_high[d] = geom.ProbHi(d);
        c.boundary_low[d] = low[d];
        c.boundary_high[d] = high[d];
    }
    return c;
}
// Private axial capability only: restrict the nodal gather band to the
// actual staggered E completion / plasma-B curl rows before MC centering.
// This metadata operation does not change either physical producer.
void RestrictAxialGatherToProduced (WarpX& simulation, int grid,
                                   EndpointTileContract& contract);
EndpointTileContract MakeNativeAxialCurrentContract(WarpX& simulation,
    amrex::Box const& particle_tile,int grid);

namespace endpoint_detail {
using Point = amrex::GpuArray<amrex::ParticleReal, 3>;
using MeshPoint = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;
AMREX_GPU_HOST_DEVICE inline unsigned
bit (EndpointIssue e) {
    return 1u << static_cast<int>(e);
}
AMREX_GPU_HOST_DEVICE inline MeshPoint
mesh (Point const& p) {
#ifdef WARPX_DIM_RZ
    return {std::hypot(p[0], p[1]), p[2]};
#elif defined(WARPX_DIM_3D)
    return {p[0], p[1], p[2]};
#elif defined(WARPX_DIM_XZ)
    return {p[0], p[2]};
#else
    return {p[2]};
#endif
}
AMREX_GPU_HOST_DEVICE inline amrex::Real
envelope (amrex::Real q, amrex::Real origin_grid = 0) {
    return 64 * std::numeric_limits<amrex::ParticleReal>::epsilon() *
           (1 + std::abs(q) + std::abs(origin_grid));
}
// Shape3 first=floor(q)-1 and last=first+3 for nonnegative kernel-local q.
// Enclose the possible one-index ambiguity at a rounded integer. This is a
// deliberately conservative check, including zero-valued endpoint weights.
AMREX_GPU_HOST_DEVICE inline bool
shapeFits (MeshPoint const& a, EndpointIndexRange const& r) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::Real const q =
            (a[d] - r.origin[d]) * r.inverse[d] - .5 * (1 - r.nodal[d]);
        amrex::Real const eps = envelope(q, r.origin[d] * r.inverse[d]);
        int const span = r.high[d] - r.low[d];
        if (!std::isfinite(q) || q - eps < 1 || q + eps > span) {
            return false;
        }
        if (std::floor(q - eps) - 1 < 0 || std::floor(q + eps) + 2 > span) {
            return false;
        }
    }
    return true;
}
// Actual implicit IndexedEsirkepovShape loops use the union of old and endpoint
// nodal supports, shortened one index in the current's cell-centered direction.
AMREX_GPU_HOST_DEVICE inline bool
currentFits (MeshPoint const& a, MeshPoint const& b,
             EndpointIndexRange const& r) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const qa = (a[d] - r.origin[d]) * r.inverse[d];
        auto const qb = (b[d] - r.origin[d]) * r.inverse[d];
        auto const eps = amrex::max(envelope(qa, r.origin[d] * r.inverse[d]),
                                    envelope(qb, r.origin[d] * r.inverse[d]));
        auto const qlo = amrex::min(qa, qb) - eps,
                   qhi = amrex::max(qa, qb) + eps;
        int const span = r.high[d] - r.low[d];
        if (!std::isfinite(qlo) || !std::isfinite(qhi) || qlo < 1 ||
            qhi > span + 1) {
            return false;
        }
        if (std::floor(qlo) - 1 < 0 ||
            std::floor(qhi) + 2 - (1 - r.nodal[d]) > span) {
            return false;
        }
    }
    return true;
}
// Extra certificate for an owned unfolded endpoint. The ordinary producer
// retains all of its own shape/common-support and allocation checks.
AMREX_GPU_HOST_DEVICE inline bool
axialCurrentFits(Point const& endpoint,Point const& momentum,
    Point const& actual_gather,EndpointTileContract const& c,
    AxialEndpointBoundary const& boundary,AxialEndpointImage& image,
    bool require_reflection=true)
{
    image=MapAxialEndpoint(endpoint,momentum,boundary);
    if(!image.valid||(require_reflection&&!image.reflected))return false;
    auto const g=mesh(actual_gather),end=mesh(endpoint);
    for(auto const& band:c.gather)if(!shapeFits(g,band))return false;
    if(!shapeFits(end,c.rho))return false;
    for(auto const& band:c.current)if(!currentFits(end,end,band))return false;
    return true;
}
AMREX_GPU_HOST_DEVICE inline unsigned
inspect (Point const& old, Point const& middle, Point const& old_u,
         Point const& middle_u, bool valid_identity,
         EndpointTileContract const& c) {
    unsigned issues = valid_identity ? 0 : bit(EndpointIssue::InvalidIdentity);
    Point end{};
    for (int d = 0; d < 3; ++d) {
        end[d] = 2 * middle[d] - old[d];
        if (!std::isfinite(old[d]) || !std::isfinite(middle[d]) ||
            !std::isfinite(end[d]) || !std::isfinite(old_u[d]) ||
            !std::isfinite(middle_u[d]) ||
            !std::isfinite(2 * middle_u[d] - old_u[d])) {
            return issues | bit(EndpointIssue::NonFinite);
        }
    }
    auto const x0 = mesh(old), xm = mesh(middle), x1 = mesh(end);
    Point end_u{};
    for (int d=0;d<3;++d) { end_u[d]=2*middle_u[d]-old_u[d]; }
    auto const absorption = c.allow_deterministic_absorption
        ? InspectAbsorbingEndpoint(end,end_u,c.absorption) : AbsorbingEndpointResult{};
    AxialEndpointImage axial;
    if(c.axial.enabled)axial=MapAxialEndpoint(end,end_u,c.axial);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!std::isfinite(x0[d]) || !std::isfinite(xm[d]) ||
            !std::isfinite(x1[d])) {
            return issues | bit(EndpointIssue::NonFinite);
        }
        auto const lo = c.domain_low[d], hi = c.domain_high[d];
        auto const eps = envelope(amrex::max(std::abs(lo), std::abs(hi)));
        if (x0[d] < lo || x0[d] > hi) {
            issues |= bit(EndpointIssue::OldOutsideDomain);
        }
        for (int side = 0; side < 2; ++side) {
            auto const bc = side ? c.boundary_high[d] : c.boundary_low[d];
            if (bc == ParticleBoundaryType::Periodic) {
                continue;
            }
#ifdef WARPX_DIM_RZ
            if (d == 0 && side == 0 && lo == 0 &&
                bc == ParticleBoundaryType::None) {
                continue;
            }
#endif
            bool const touches = side ? x1[d] >= hi - eps : x1[d] <= lo + eps;
            if (!touches) {
                continue;
            }
            if (bc == ParticleBoundaryType::Absorbing ||
                bc == ParticleBoundaryType::Open) {
                if (c.allow_deterministic_absorption &&
                    absorption.status==AbsorbingEndpointStatus::Absorbed &&
                    (absorption.faces & (1u<<(2*d+side)))) {
                    issues |= bit(EndpointIssue::SupportedAbsorption);
                } else {
                    issues |= bit(EndpointIssue::AbsorbingBoundary);
                }
            } else if (bc == ParticleBoundaryType::Thermal) {
                issues |= bit(EndpointIssue::ThermalBoundary);
#ifdef WARPX_DIM_RZ
            } else if (d == 0 && side == 1 &&
                       bc == ParticleBoundaryType::Reflecting) {
                // Only the native endpoint-only image is qualified. In-push
                // segmented reflection/clamps and exact-wall ambiguity remain
                // unsupported; no bounce count is inferred from endpoint data.
                bool const native_image=c.allow_deterministic_absorption &&
                    (absorption.reflected_faces & (1u<<1)) &&
                    (absorption.status==AbsorbingEndpointStatus::Interior ||
                     absorption.status==AbsorbingEndpointStatus::Absorbed);
                issues |= bit(native_image ? EndpointIssue::SupportedReflection
                                          : EndpointIssue::UnresolvedRadialWall);
#endif
#ifdef WARPX_DIM_RZ
            } else if (d==1 && bc==ParticleBoundaryType::Reflecting &&
                       c.axial.enabled && axial.valid && axial.reflected && axial.side==side) {
                issues |= bit(EndpointIssue::SupportedReflection);
#endif
            } else {
                issues |= bit(EndpointIssue::OtherBoundaryAction);
            }
        }
    }
    if (c.gathers) {
        for (auto const& r : c.gather) {
            if (!shapeFits(xm, r)) {
                issues |= bit(EndpointIssue::GatherReach);
            }
        }
    }
    if (c.deposits) {
        for (auto const& r : c.current) {
            if (!currentFits(x0, x1, r)) {
                issues |= bit(EndpointIssue::CurrentReach);
            }
        }
#ifdef WARPX_DIM_RZ
        if(axial.reflected) {
            amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> image{axial.stored[0],axial.stored[2]};
            if(!shapeFits(image,c.rho))issues |= bit(EndpointIssue::EndpointReach);
            for(auto const& r:c.current)if(!currentFits(image,image,r))issues |= bit(EndpointIssue::CurrentReach);
        }
#endif
        if (!shapeFits(x1, c.rho)) {
            issues |= bit(EndpointIssue::EndpointReach);
        }
        // Preserve the stricter existing virtual endpoint projection assertion.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            auto const q = (x1[d] - c.rho.origin[d]) * c.rho.inverse[d];
            auto const eps = envelope(q, c.rho.origin[d] * c.rho.inverse[d]);
            if (!std::isfinite(q) || q - eps < 2 ||
                q + eps > c.rho.high[d] - c.rho.low[d] - 2) {
                issues |= bit(EndpointIssue::ProjectionReach);
            }
        }
    }
    return issues;
}
template <class Position>
EndpointAuditResult
reduceTile (amrex::Long n, Position const& position,
            ParticleStartView const& start,
            std::array<const amrex::ParticleReal*, 3> const& momentum,
            const std::uint64_t* identities,
            EndpointTileContract const& contract) {
    EndpointAuditResult result;
    result.particles = n;
    // Explicit tuple types also compile with CUDA's extended-lambda frontend.
    static_assert(endpoint_issue_count == 13);
    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                     amrex::ReduceOpSum> op;
    amrex::ReduceData<amrex::Long, amrex::Long, amrex::Long, amrex::Long,
                      amrex::Long, amrex::Long, amrex::Long, amrex::Long,
                      amrex::Long, amrex::Long, amrex::Long, amrex::Long,
                      amrex::Long> data(op);
    op.eval(
        n, data,
        [position, start, momentum, identities,
         contract] AMREX_GPU_DEVICE(amrex::Long i) {
            Point old{}, mid{}, un{}, um{};
            position(i, mid[0], mid[1], mid[2]);
            for (int d = 0; d < 3; ++d) {
                old[d] = start.position[d] ? start.position[d][i]
                                           : amrex::ParticleReal(0);
                un[d] = start.momentum[d][i];
                um[d] = momentum[d][i];
            }
            auto const mask =
                inspect(old, mid, un, um,
                        amrex::ConstParticleIDWrapper{identities[i]}.is_valid(),
                        contract);
            return amrex::GpuTuple<amrex::Long, amrex::Long, amrex::Long,
                                   amrex::Long, amrex::Long, amrex::Long,
                                   amrex::Long, amrex::Long, amrex::Long,
                                   amrex::Long, amrex::Long, amrex::Long,
                                   amrex::Long>{
                amrex::Long((mask >> 0) & 1u), amrex::Long((mask >> 1) & 1u),
                amrex::Long((mask >> 2) & 1u), amrex::Long((mask >> 3) & 1u),
                amrex::Long((mask >> 4) & 1u), amrex::Long((mask >> 5) & 1u),
                amrex::Long((mask >> 6) & 1u), amrex::Long((mask >> 7) & 1u),
                amrex::Long((mask >> 8) & 1u), amrex::Long((mask >> 9) & 1u),
                amrex::Long((mask >> 10) & 1u), amrex::Long((mask >> 11) & 1u),
                amrex::Long((mask >> 12) & 1u)};
        });
    auto const values = data.value();
    result.counts = {
        amrex::get<0>(values), amrex::get<1>(values), amrex::get<2>(values),
        amrex::get<3>(values), amrex::get<4>(values), amrex::get<5>(values),
        amrex::get<6>(values), amrex::get<7>(values), amrex::get<8>(values),
        amrex::get<9>(values), amrex::get<10>(values), amrex::get<11>(values),
        amrex::get<12>(values)};
    return result;
}
} // namespace endpoint_detail
// Pure GPU-native local evaluation. The production bridge performs the MPI sum.
template <class Position>
EndpointAuditResult
AuditEndpointTile (amrex::Long n, Position const& position,
                   ParticleStartView const& start,
                   std::array<const amrex::ParticleReal*, 3> const& momentum,
                   const std::uint64_t* identities,
                   EndpointTileContract const& contract) {
    return endpoint_detail::reduceTile(n, position, start, momentum, identities,
                                       contract);
}
// Qualified static, single-level, shape3 Esirkepov/MC kinetic particle bridge.
// Call before endpoint deposit/acceptance. This does not audit intermediate
// Picard iterates retrospectively and never applies any particle boundary
// action.
EndpointAuditResult
AuditImplicitParticleEndpoints (WarpX& simulation, int lev,
                                EndpointAuditFields const& fields);
} // namespace warpx::implicit
#endif
