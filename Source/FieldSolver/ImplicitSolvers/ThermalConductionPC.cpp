/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "ThermalConductionPC.H"
#include "NonlinearSolvers/FlexibleGMRES.H"
#include "ThermalTransportStencil.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Utility.H>
#include <cmath>

namespace warpx::thermal {
using amrex::Real;
namespace {
void
load_specific (amrex::MultiFab& specific, const amrex::MultiFab& correction,
               const amrex::MultiFab& density) {
    for (amrex::MFIter mfi(specific); mfi.isValid(); ++mfi) {
        auto const n = density.const_array(mfi);
        auto const in = correction.const_array(mfi);
        auto const out = specific.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               out(i, j, k) = in(i, j, k) / n(i, j, k);
                           });
    }
}
// NVCC extended device lambdas must have a public enclosing function.
// Keep launch helpers internal to this translation unit, not public PC APIs.
#if !defined(WARPX_DIM_3D)
void
freeze_compression (amrex::MultiFab& compression, const amrex::MultiFab& radial,
                    const amrex::MultiFab& axial, const amrex::Geometry& geom,
                    Real h, Real gm1) {
    auto const dx = geom.CellSizeArray();
#if defined(WARPX_DIM_RZ)
    auto const rlo = geom.ProbLo(0);
#endif
    for (amrex::MFIter mfi(compression); mfi.isValid(); ++mfi) {
        auto const ur = radial.const_array(mfi), uz = axial.const_array(mfi);
        auto const c = compression.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const r = rlo + (i + .5) * dx[0];
                Real const div = ((rlo + (i + 1) * dx[0]) * ur(i + 1, j, k) -
                                  (rlo + i * dx[0]) * ur(i, j, k)) /
                                     (r * dx[0]) +
                                 (uz(i, j + 1, k) - uz(i, j, k)) / dx[1];
                c(i, j, k) = h * gm1 * div;
            });
    }
}

#endif
#if defined(WARPX_DIM_3D)
void
freeze_compression_3d (amrex::MultiFab& compression,
                       const std::array<std::unique_ptr<amrex::MultiFab>,
                                        AMREX_SPACEDIM>& velocity,
                       const amrex::Geometry& geom, Real h, Real gm1) {
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(compression); mfi.isValid(); ++mfi) {
        amrex::GpuArray<amrex::Array4<const Real>, 3> v;
        for (int d = 0; d < 3; ++d) {
            v[d] = velocity[d]->const_array(mfi);
        }
        auto const c = compression.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real div = 0;
            for (int d = 0; d < 3; ++d) {
                int next[3] = {i, j, k};
                ++next[d];
                div +=
                    (v[d](next[0], next[1], next[2]) - v[d](i, j, k)) / dx[d];
            }
            c(i, j, k) = h * gm1 * div;
        });
    }
}
#endif
void
freeze_inverse_reaction (amrex::MultiFab& inverse, const amrex::MultiFab& base,
                         const amrex::MultiFab& density,
                         const amrex::MultiFab& compression,
                         const amrex::Geometry& geom) {
#if defined(WARPX_DIM_RZ)
    auto const dx = geom.CellSizeArray();
#else
    amrex::ignore_unused(geom);
#endif
#if defined(WARPX_DIM_RZ)
    auto const rlo = geom.ProbLo(0);
#endif
    for (amrex::MFIter mfi(inverse); mfi.isValid(); ++mfi) {
        auto const a = base.const_array(mfi), n = density.const_array(mfi),
                   c = compression.const_array(mfi);
        auto const p = inverse.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
#if defined(WARPX_DIM_RZ)
            Real const rn = (rlo + (i + .5) * dx[0]) * n(i, j, k);
#else
            Real const rn=n(i,j,k);
#endif
            p(i, j, k) = amrex::max(std::abs(a(i, j, k)), rn * Real(1.e-12)) +
                         rn * amrex::max(c(i, j, k), Real(0));
        });
    }
}

void
freeze_inverse_faces (
    std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>& inverse,
    const std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>& base,
    const std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>&
        velocity,
    const amrex::MultiFab& density, const amrex::Geometry& geom, Real h,
    Real scale) {
    auto const dx = geom.CellSizeArray();
#if defined(WARPX_DIM_RZ)
    auto const rlo = geom.ProbLo(0);
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bool const periodic = geom.isPeriodic(d);
        int const lo = geom.Domain().smallEnd(d),
                  hi = geom.Domain().bigEnd(d) + 1;
        for (amrex::MFIter mfi(*inverse[d]); mfi.isValid(); ++mfi) {
            auto const n = density.const_array(mfi),
                       u = velocity[d]->const_array(mfi),
                       b = base[d]->const_array(mfi);
            auto const p = inverse[d]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int const index = d == 0 ? i : (d == 1 ? j : k);
                    Real extra = 0;
                    if (periodic || (index > lo && index < hi)) {
                        Real const nf =
                            .5 * (n(i, j, k) +
                                  n(i - (d == 0), j - (d == 1), k - (d == 2)));
#if defined(WARPX_DIM_RZ)
                        Real const rf =
                            rlo + (i + (d == 0 ? Real(0) : Real(.5))) * dx[0];
#else
                    Real const rf=1;
#endif
                        extra = rf * h * scale * std::abs(u(i, j, k)) * dx[d] *
                                nf * .5;
                    }
                    p(i, j, k) = b(i, j, k) + extra;
                });
        }
    }
}

void
load_metric_rhs (amrex::MultiFab& weighted_rhs, const amrex::MultiFab& rhs,
                 const amrex::Geometry& geom) {
#if defined(WARPX_DIM_RZ)
    auto const dr = geom.CellSize(0);
#endif
#if defined(WARPX_DIM_RZ)
    auto const rlo = geom.ProbLo(0);
#endif
    amrex::ignore_unused(geom);
    for (amrex::MFIter mfi(weighted_rhs); mfi.isValid(); ++mfi) {
        auto const in = rhs.const_array(mfi);
        auto const out = weighted_rhs.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                out(i, j, k) = (rlo + (i + 0.5) * dr) * in(i, j, k);
#else
                out(i,j,k)=in(i,j,k);
#endif
            });
    }
}

void
recover_energy (amrex::MultiFab& correction, const amrex::MultiFab& specific,
                const amrex::MultiFab& density) {
    for (amrex::MFIter mfi(correction); mfi.isValid(); ++mfi) {
        auto const n = density.const_array(mfi);
        auto const y = specific.const_array(mfi);
        auto const out = correction.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               out(i, j, k) = n(i, j, k) * y(i, j, k);
                           });
    }
}

#if !defined(WARPX_DIM_3D)
void
add_transport_action (amrex::MultiFab& out, const amrex::MultiFab& state,
                      const amrex::MultiFab& compression,
                      const amrex::MultiFab& radial,
                      const amrex::MultiFab& axial, FrozenTransportGeometry g,
                      bool rows) {
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto const u = state.const_array(mfi), c = compression.const_array(mfi);
        auto const ur = radial.const_array(mfi), uz = axial.const_array(mfi);
        auto const result = out.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real value = 0;
                if (rows) {
                    EmitFrozenTransportRow(
                        g, ur, uz, c(i, j, k), i, j, k,
                        [&] (int ii, int jj, int kk, Real coefficient) {
                            value += coefficient * u(ii, jj, kk);
                        });
                } else {
                    Real const r = g.rlo + (i + .5) * g.dx[0];
                    Real const fl =
                        FrozenCentralFaceFlux(g, 0, i, j, k, ur(i, j, k), u);
                    Real const fr = FrozenCentralFaceFlux(g, 0, i + 1, j, k,
                                                          ur(i + 1, j, k), u);
                    Real const fb =
                        FrozenCentralFaceFlux(g, 1, i, j, k, uz(i, j, k), u);
                    Real const ft = FrozenCentralFaceFlux(g, 1, i, j + 1, k,
                                                          uz(i, j + 1, k), u);
                    value = c(i, j, k) * u(i, j, k) +
                            g.h *
                                ((g.rlo + (i + 1) * g.dx[0]) * fr -
                                 (g.rlo + i * g.dx[0]) * fl) /
                                (r * g.dx[0]) +
                            g.h * (ft - fb) / g.dx[1];
                }
                result(i, j, k) += value;
            });
    }
}
#endif
#if defined(WARPX_DIM_3D)
void
add_transport_action_3d (amrex::MultiFab& out, const amrex::MultiFab& state,
                         const amrex::MultiFab& compression,
                         const std::array<std::unique_ptr<amrex::MultiFab>,
                                          AMREX_SPACEDIM>& velocity,
                         FrozenTransportGeometry g, bool rows) {
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto const u = state.const_array(mfi), c = compression.const_array(mfi);
        auto const result = out.array(mfi);
        amrex::GpuArray<amrex::Array4<const Real>, 3> v;
        for (int d = 0; d < 3; ++d) {
            v[d] = velocity[d]->const_array(mfi);
        }
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real value = 0;
            if (rows) {
                EmitFrozenTransportRow3D(g, v, c(i, j, k), i, j, k,
                                         [&] (int ii, int jj, int kk, Real a) {
                                             value += a * u(ii, jj, kk);
                                         });
            } else {
                value = c(i, j, k) * u(i, j, k);
                for (int d = 0; d < 3; ++d) {
                    int next[3] = {i, j, k};
                    ++next[d];
                    Real const lo =
                        FrozenCentralFaceFlux(g, d, i, j, k, v[d](i, j, k), u);
                    Real const hi = FrozenCentralFaceFlux(
                        g, d, next[0], next[1], next[2],
                        v[d](next[0], next[1], next[2]), u);
                    value += g.h * (hi - lo) / g.dx[d];
                }
            }
            result(i, j, k) += value;
        });
    }
}
#endif
} // namespace

ThermalConductionPC::ThermalConductionPC (EulerianThermalStage& stage,
                                          int cycles)
    : ThermalConductionPC(stage, ThermalPCOptions{cycles}) {}

ThermalConductionPC::ThermalConductionPC (EulerianThermalStage& stage,
                                          ThermalPCOptions options)
    : m_stage(stage), m_options(options),
      m_a(stage.Density().boxArray(), stage.Density().DistributionMap(), 1, 0),
      m_rhs(m_a.boxArray(), m_a.DistributionMap(), 1, 0),
      m_solution(m_a.boxArray(), m_a.DistributionMap(), 1, 1),
      m_scratch(m_a.boxArray(), m_a.DistributionMap(), 1, 0),
      m_density(m_a.boxArray(), m_a.DistributionMap(), 1, 1),
      m_compression(m_a.boxArray(), m_a.DistributionMap(), 1, 0),
      m_transport_state(m_a.boxArray(), m_a.DistributionMap(), 1, 1) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        options.cycles > 0 && options.max_coarsening_level >= 0 &&
            options.max_semicoarsening_level >= 0 &&
            options.semicoarsening_direction >= -1 &&
            options.semicoarsening_direction < AMREX_SPACEDIM &&
            options.transport_inner_max_iterations > 0 &&
            options.transport_inner_relative_tolerance >= 0 &&
            options.transport_inner_relative_tolerance < 1 &&
            std::isfinite(options.transport_upwind_stabilization) &&
            options.transport_upwind_stabilization >= 0,
        "Invalid thermal PC controls");
    auto const& geom = stage.Geometry();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        m_rows.face_scale[d] = 1 / (geom.CellSize(d) * geom.CellSize(d));
        m_rows.domain_low[d] = geom.Domain().smallEnd(d);
        m_rows.domain_high[d] = geom.Domain().bigEnd(d);
        for (int side = 0; side < 2; ++side) {
            auto const bc = stage.Options().boundary[d][side];
            bool const reservoir = bc.kind == BoundaryKind::Reservoir &&
                                   bc.flux_limit == 0 && !bc.drain_only;
            using B = warpx::mhd_pc::ConductionBoundary;
            m_rows.boundary[0][d][side] =
                static_cast<int>(geom.isPeriodic(d) ? B::Periodic
                                 : reservoir        ? B::Dirichlet
                                                    : B::Neumann);
        }
        auto const face = amrex::convert(m_a.boxArray(),
                                         amrex::IntVect::TheDimensionVector(d));
        m_b[d] = std::make_unique<amrex::MultiFab>(face, m_a.DistributionMap(),
                                                   1, 0);
        m_velocity[d] = std::make_unique<amrex::MultiFab>(
            face, m_a.DistributionMap(), 1, 0);
    }
    m_rows.max_order = 2;
    DefineElliptic(m_operator, m_solver);
}

void
ThermalConductionPC::DefineElliptic (
    std::unique_ptr<amrex::MLABecLaplacian>& op,
    std::unique_ptr<amrex::MLMG>& solver) {
    auto const start = amrex::second();
    auto const& geom = m_stage.Geometry();
    amrex::LPInfo info;
    info.setAgglomeration(m_options.agglomeration)
        .setConsolidation(m_options.consolidation)
        .setMaxCoarseningLevel(m_options.max_coarsening_level)
        .setSemicoarsening(m_options.semicoarsening)
        .setMaxSemicoarseningLevel(m_options.max_semicoarsening_level)
        .setSemicoarseningDirection(m_options.semicoarsening_direction);
    // RZ metric is already in a,b,rhs. Never insert it twice.
    info.setMetricTerm(false);
    op = std::make_unique<amrex::MLABecLaplacian>(
        amrex::Vector<amrex::Geometry>{geom},
        amrex::Vector<amrex::BoxArray>{m_a.boxArray()},
        amrex::Vector<amrex::DistributionMapping>{m_a.DistributionMap()}, info);
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> low, high;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int side = 0; side < 2; ++side) {
            using B = warpx::mhd_pc::ConductionBoundary;
            auto const bc = static_cast<B>(m_rows.boundary[0][d][side]);
            (side == 0 ? low : high)[d] =
                bc == B::Periodic    ? amrex::LinOpBCType::Periodic
                : bc == B::Dirichlet ? amrex::LinOpBCType::Dirichlet
                                     : amrex::LinOpBCType::Neumann;
        }
    }
    // Exact half-cell reservoir rows from the tuned MHD thermal block.
    op->setMaxOrder(2);
    op->setDomainBC(low, high);
    op->setLevelBC(0, nullptr);
    op->setScalars(1, 1);
    solver = std::make_unique<amrex::MLMG>(*op);
    solver->setMaxIter(m_options.cycles);
    solver->setFixedIter(m_options.cycles);
    solver->setMaxFmgIter(0);
    solver->setBottomSolver(amrex::BottomSolver::smoother);
    solver->setVerbose(0);
    solver->setBottomVerbose(0);
    ++m_diagnostics.setups;
    m_diagnostics.setup_seconds += amrex::second() - start;
}

bool
ThermalConductionPC::Freeze (const amrex::MultiFab& state) {
    auto const start = amrex::second();
    // Residual and PC coefficient emission share exactly the same face pass.
    if (!m_stage.Residual(state, m_scratch)) {
        m_diagnostics.freeze_seconds += amrex::second() - start;
        return false;
    }
    auto const& geom = m_stage.Geometry();
#if defined(WARPX_DIM_RZ)
    auto const dr = geom.CellSize(0);
#endif
#if defined(WARPX_DIM_RZ)
    auto const rlo = geom.ProbLo(0);
#endif
    auto const h = m_stage.Options().theta * m_stage.Options().dt;
    amrex::MultiFab::Copy(m_density, m_stage.Density(), 0, 0, 1, 0);
#if defined(WARPX_DIM_RZ)
    auto const dz = geom.CellSize(1);
    for (amrex::MFIter mfi(m_a); mfi.isValid(); ++mfi) {
        auto const n = m_density.const_array(mfi);
        auto const a = m_a.array(mfi);
        auto const source_derivative =
            m_stage.SourceDerivative().const_array(mfi);
        auto const wr = m_stage.FaceWallDerivative(0).const_array(mfi);
        auto const wz = m_stage.FaceWallDerivative(1).const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real const r = rlo + (i + 0.5) * dr;
            Real const wall = ((rlo + i * dr) * wr(i, j, k) +
                               (rlo + (i + 1) * dr) * wr(i + 1, j, k)) /
                                  (r * dr) +
                              (wz(i, j, k) + wz(i, j + 1, k)) / dz;
            a(i, j, k) =
                r *
                (n(i, j, k) * (1 - h * source_derivative(i, j, k)) + h * wall);
        });
    }
#else
    auto const dx = geom.CellSizeArray();
    for (amrex::MFIter mfi(m_a); mfi.isValid(); ++mfi) {
        auto const n = m_density.const_array(mfi),
                   dq = m_stage.SourceDerivative().const_array(mfi);
        auto const a = m_a.array(mfi);
        amrex::GpuArray<amrex::Array4<const Real>, 3> wall;
        for (int d = 0; d < 3; ++d) {
            wall[d] = m_stage.FaceWallDerivative(d).const_array(mfi);
        }
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            Real w = 0;
            for (int d = 0; d < 3; ++d) {
                int next[3] = {i, j, k};
                ++next[d];
                w += (wall[d](i, j, k) + wall[d](next[0], next[1], next[2])) /
                     dx[d];
            }
            a(i, j, k) = n(i, j, k) * (1 - h * dq(i, j, k)) + h * w;
        });
    }
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (amrex::MFIter mfi(*m_b[d]); mfi.isValid(); ++mfi) {
            auto const c = m_stage.FaceCoefficient(d).const_array(mfi);
            auto const b = m_b[d]->array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                b(i, j, k) = (rlo + (i + (d == 0 ? Real(0) : Real(0.5))) * dr) *
                             h * c(i, j, k);
#else
                b(i,j,k)=h*c(i,j,k);
#endif
            });
        }
    }
    m_operator->setACoeffs(0, m_a);
    amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> b;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        b[d] = m_b[d].get();
    }
    m_operator->setBCoeffs(0, b);
    FreezeTransport();
    ++m_updates;
    m_diagnostics.freeze_seconds += amrex::second() - start;
    return true;
}

void
ThermalConductionPC::FreezeTransport () {
    auto const& geom = m_stage.Geometry();
    auto const h = m_stage.Options().theta * m_stage.Options().dt;
    auto const gm1 = m_stage.Options().gamma - 1;
    Real speed = 0;
    if (m_options.transport) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::MultiFab::Copy(*m_velocity[d], m_stage.FaceVelocity(d), 0, 0,
                                  1, 0);
            speed = std::max(speed, m_velocity[d]->norm0(0));
        }
    }
    m_transport_active = m_options.transport && speed > 0;
    m_diagnostics.minimum_transport_reaction = 1;
    if (!m_transport_active) {
        m_compression.setVal(0);
        return;
    }
    m_density.FillBoundary(geom.periodicity());
#if !defined(WARPX_DIM_3D)
    freeze_compression(m_compression, *m_velocity[0], *m_velocity[1], geom, h,
                       gm1);
#else
    freeze_compression_3d(m_compression, m_velocity, geom, h, gm1);
#endif
    m_diagnostics.minimum_transport_reaction = 1 + m_compression.min(0);
    if (!m_inverse_operator) {
        m_inverse_a.define(m_a.boxArray(), m_a.DistributionMap(), 1, 0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            m_inverse_b[d] = std::make_unique<amrex::MultiFab>(
                m_b[d]->boxArray(), m_a.DistributionMap(), 1, 0);
        }
        DefineElliptic(m_inverse_operator, m_inverse_solver);
    }
    // Only the inner elliptic inverse uses a positive local surrogate. The
    // signed physical reaction, including compression, stays in the target.
    freeze_inverse_reaction(m_inverse_a, m_a, m_density, m_compression, geom);
    freeze_inverse_faces(m_inverse_b, m_b, m_velocity, m_density, geom, h,
                         m_options.transport_upwind_stabilization);
    m_inverse_operator->setACoeffs(0, m_inverse_a);
    amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM> faces;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
        faces[d] = m_inverse_b[d].get();
    m_inverse_operator->setBCoeffs(0, faces);
}

struct ThermalConductionPC::TransportOps {
    using RT = Real;
    ThermalConductionPC& pc;
    amrex::MultiFab
    makeVecRHS () const {
        return {pc.m_a.boxArray(), pc.m_a.DistributionMap(), 1, 0};
    }
    amrex::MultiFab
    makeVecLHS () const {
        return makeVecRHS();
    }
    void
    setToZero (amrex::MultiFab& x) const {
        x.setVal(0, 0, 1, 0);
    }
    void
    assign (amrex::MultiFab& x, const amrex::MultiFab& y) const {
        amrex::MultiFab::Copy(x, y, 0, 0, 1, 0);
    }
    void
    scale (amrex::MultiFab& x, Real a) const {
        x.mult(a, 0, 1, 0);
    }
    void
    increment (amrex::MultiFab& x, const amrex::MultiFab& y, Real a) const {
        amrex::MultiFab::Saxpy(x, a, y, 0, 0, 1, 0);
    }
    void
    linComb (amrex::MultiFab& x, Real a, const amrex::MultiFab& y, Real b,
             const amrex::MultiFab& z) const {
        amrex::MultiFab::LinComb(x, a, y, 0, b, z, 0, 0, 1, 0);
    }
    Real
    dotProduct (const amrex::MultiFab& x, const amrex::MultiFab& y) const {
        ++pc.m_diagnostics.reductions;
        return amrex::MultiFab::Dot(x, 0, y, 0, 1, 0);
    }
    Real
    norm2 (const amrex::MultiFab& x) const {
        return std::sqrt(dotProduct(x, x));
    }
    void
    apply (amrex::MultiFab& x, const amrex::MultiFab& y) const {
        pc.ApplyOperator(x, y);
    }
    void
    precond (amrex::MultiFab& x, const amrex::MultiFab& y) const {
        pc.ApplyElliptic(x, y);
    }
};

void
ThermalConductionPC::Apply (amrex::MultiFab& correction,
                            const amrex::MultiFab& rhs) {
    AMREX_ALWAYS_ASSERT(m_updates > 0);
    auto const start = amrex::second();
    ++m_diagnostics.applies;
    m_diagnostics.last_inner_status = -1;
    m_diagnostics.last_inner_relative_residual = 0;
    if (!m_transport_active) {
        ApplyElliptic(correction, rhs);
    } else {
        TransportOps ops{*this};
        FlexibleGMRES<amrex::MultiFab, TransportOps> inner;
        inner.define(ops);
        inner.setMaxIters(m_options.transport_inner_max_iterations);
        inner.setRestartLength(m_options.transport_inner_max_iterations);
        // Preserve the original in-place Apply contract. Inner Krylov zeros
        // its output first and revisits the RHS after every solution update.
        std::unique_ptr<amrex::MultiFab> rhs_copy;
        auto const* frozen_rhs = &rhs;
        if (&correction == &rhs) {
            rhs_copy = std::make_unique<amrex::MultiFab>(
                m_a.boxArray(), m_a.DistributionMap(), 1, 0);
            amrex::MultiFab::Copy(*rhs_copy, rhs, 0, 0, 1, 0);
            frozen_rhs = rhs_copy.get();
        }
        Real const rhs_norm = ops.norm2(*frozen_rhs);
        inner.solve(correction, *frozen_rhs,
                    m_options.transport_inner_relative_tolerance, 0);
        m_diagnostics.inner_iterations += inner.getNumIters();
        m_diagnostics.last_inner_status = inner.getStatus();
        m_diagnostics.last_inner_relative_residual =
            rhs_norm > 0 ? inner.getResidualNorm() / rhs_norm : 0;
        if (inner.getStatus() == 2) {
            ++m_diagnostics.breakdowns;
            correction.setVal(0, 0, 1, 0);
        }
    }
    m_diagnostics.apply_seconds += amrex::second() - start;
}

void
ThermalConductionPC::ApplyElliptic (amrex::MultiFab& correction,
                                    const amrex::MultiFab& rhs) {
    AMREX_ALWAYS_ASSERT(m_updates > 0);
    load_metric_rhs(m_rhs, rhs, m_stage.Geometry());
    m_solution.setVal(0);
    ++m_diagnostics.mlmg_solves;
    auto& solver = m_transport_active ? m_inverse_solver : m_solver;
    solver->solve({&m_solution}, {&m_rhs}, 0, 0);
    recover_energy(correction, m_solution, m_density);
}

void
ThermalConductionPC::LoadSpecific (const amrex::MultiFab& correction) {
    AMREX_ALWAYS_ASSERT(m_updates > 0);
    load_specific(m_solution, correction, m_density);
    if (m_transport_active) {
        // Capture before any output write, including an in-place action.
        amrex::MultiFab::Copy(m_transport_state, correction, 0, 0, 1, 0);
        m_transport_state.FillBoundary(m_stage.Geometry().periodicity());
    }
}

void
ThermalConductionPC::ApplyOperator (amrex::MultiFab& out,
                                    const amrex::MultiFab& correction) {
    ++m_diagnostics.operator_applies;
    LoadSpecific(correction);
    m_solver->apply({&m_scratch}, {&m_solution});
#if defined(WARPX_DIM_RZ)
    auto const dr = m_stage.Geometry().CellSize(0);
    auto const rlo = m_stage.Geometry().ProbLo(0);
#endif
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto const value = m_scratch.const_array(mfi);
        auto const result = out.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if defined(WARPX_DIM_RZ)
                result(i, j, k) = value(i, j, k) / (rlo + (i + 0.5) * dr);
#else
                result(i,j,k)=value(i,j,k);
#endif
            });
    }
    AddTransport(out, false);
}

void
ThermalConductionPC::ApplyRows (amrex::MultiFab& out,
                                const amrex::MultiFab& correction) {
    LoadSpecific(correction);
    m_solution.FillBoundary(m_stage.Geometry().periodicity());
#if defined(WARPX_DIM_RZ)
    auto const dr = m_stage.Geometry().CellSize(0);
    auto const rlo = m_stage.Geometry().ProbLo(0);
#endif
    auto const constants = m_rows;
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        warpx::mhd_pc::ConductionStencilArrays arrays;
        arrays.acoef = m_a.const_array(mfi);
        arrays.component[0] = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            arrays.bcoef[d] = m_b[d]->const_array(mfi);
            arrays.box_length[d] = mfi.validbox().length(d);
        }
        auto const y = m_solution.const_array(mfi);
        auto const result = out.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real sum = 0;
                warpx::mhd_pc::EmitConductionRow(
                    constants, arrays, 0, i, j, k,
                    [&] (int ii, int jj, int kk, Real coefficient) {
                        sum += coefficient * y(ii, jj, kk);
                    });
#if defined(WARPX_DIM_RZ)
                result(i, j, k) = sum / (rlo + (i + 0.5) * dr);
#else
                result(i,j,k)=sum;
#endif
            });
    }
    AddTransport(out, true);
}
void
ThermalConductionPC::AddTransport (amrex::MultiFab& out, bool rows) {
    if (!m_transport_active)
        return;
    FrozenTransportGeometry g;
    auto const& geom = m_stage.Geometry();
    g.rlo = geom.ProbLo(0);
    g.h = m_stage.Options().theta * m_stage.Options().dt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        g.low[d] = geom.Domain().smallEnd(d);
        g.high[d] = geom.Domain().bigEnd(d);
        g.periodic[d] = geom.isPeriodic(d);
        g.dx[d] = geom.CellSize(d);
    }
#if !defined(WARPX_DIM_3D)
    add_transport_action(out, m_transport_state, m_compression, *m_velocity[0],
                         *m_velocity[1], g, rows);
#else
    add_transport_action_3d(out, m_transport_state, m_compression, m_velocity,
                            g, rows);
#endif
}

} // namespace warpx::thermal
