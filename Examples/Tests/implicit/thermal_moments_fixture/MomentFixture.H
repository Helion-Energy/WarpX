/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_TEST_THERMAL_MOMENT_FIXTURE_H_
#define WARPX_TEST_THERMAL_MOMENT_FIXTURE_H_
#include "BoundaryConditions/WarpX_PEC.H"
#include "FieldSolver/ImplicitSolvers/KineticThermalMoments.H"
#include "Utils/WarpXConst.H"
#if defined(WARPX_DIM_RZ)
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#else
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#endif
#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace moment_test {
using namespace warpx::thermal;
using amrex::MultiFab;
using amrex::Real;
constexpr Real n0 = 1.e18;
constexpr Real t0 = 4.e5;
constexpr Real gamma_e = 5.0 / 3.0;
using Current = std::array<std::unique_ptr<MultiFab>, 3>;
using Faces = std::array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM>;
inline void
check (bool yes, const char* label) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(yes, label);
}
inline void
near (Real a, Real b, Real tolerance, const char* label) {
    if (!std::isfinite(a) || std::abs(a - b) > tolerance) {
        amrex::Print() << std::setprecision(17) << label << ": " << a << " vs "
                       << b << ", tolerance " << tolerance << '\n';
        amrex::Abort(label);
    }
}
inline amrex::IntVect
yee (int c) {
    auto t = amrex::IntVect::TheNodeVector();
#ifdef WARPX_DIM_RZ
    if (c != 1) {
        t[c / 2] = 0;
    }
#else
    t[c] = 0;
#endif
    return t;
}
inline YeeCurrentView
view (const Current& j) {
    return {j[0].get(), j[1].get(), j[2].get()};
}
inline ThermalFaceView
face_view (const Faces& f) {
    ThermalFaceView a{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        a[d] = f[d].get();
    }
    return a;
}
inline MultiFab
clone (const MultiFab& a) {
    MultiFab b(a.boxArray(), a.DistributionMap(), a.nComp(), a.nGrowVect());
    MultiFab::Copy(b, a, 0, 0, a.nComp(), a.nGrowVect());
    return b;
}
inline Real
difference (const MultiFab& a, const MultiFab& b, bool ghosts = false) {
    auto c = clone(a);
    MultiFab::Subtract(c, b, 0, 0, a.nComp(), ghosts ? a.nGrow() : 0);
    Real e = 0.;
    for (int n = 0; n < c.nComp(); ++n) {
        e = std::max(e, c.norm0(n, ghosts ? c.nGrow() : 0));
    }
    return e;
}
template <class F>
inline void
fill (MultiFab& a, F fun, int component = 0) {
    for (amrex::MFIter mfi(a, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto out = a.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                               out(i, j, k, component) = fun(i, j, k);
                           });
    }
}
struct Context {
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    ThermalMomentOptions opts;
    amrex::Array<FieldBoundaryType, AMREX_SPACEDIM> blo, bhi;
    amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM> plo, phi;
    MultiFab rho, ped, u;
    Current ji, je, jp;
    Context (int max_grid, bool periodic, bool corrected = false,
             bool reflecting = false) {
        amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                          amrex::IntVect(AMREX_D_DECL(15, 15, 15)));
        amrex::RealBox physical({AMREX_D_DECL(0., 0., 0.)},
                                {AMREX_D_DECL(2., 3., 2.5)});
        int periods[AMREX_SPACEDIM] = {
            AMREX_D_DECL(0, int(periodic), int(periodic))};
#ifdef WARPX_DIM_RZ
        int coord = 1;
#else
        int coord = 0;
#endif
        geom = amrex::Geometry(domain, &physical, coord, periods);
        ba = amrex::BoxArray(domain);
        ba.maxSize(max_grid);
        dm = amrex::DistributionMapping(ba);
        opts.number_density_floor = .05 * n0;
        opts.active_density_floor = .1 * n0;
        opts.verboncoeur_axis_correction = corrected;
        opts.temperature_floor_kelvin = 1.;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            for (int s = 0; s < 2; ++s) {
                opts.boundary[d][s] =
                    periods[d] ? MomentBoundary::Periodic : MomentBoundary::PMC;
            }
            blo[d] = bhi[d] = periods[d] ? FieldBoundaryType::Periodic
                                         : FieldBoundaryType::PMC;
            plo[d] = phi[d] = ParticleBoundaryType::Absorbing;
        }
        opts.boundary[0][1] = MomentBoundary::PEC;
        bhi[0] = FieldBoundaryType::PEC;
        if (reflecting) {
            phi[0] = ParticleBoundaryType::Reflecting;
            opts.current_boundary[0][1] = MomentBoundary::PMC;
        }
#ifdef WARPX_DIM_RZ
        opts.boundary[0][0] = MomentBoundary::Axis;
        blo[0] = FieldBoundaryType::None;
#endif
        rho.define(amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm, 2,
                   2);
        ped.define(rho.boxArray(), dm, 1, 2);
        u.define(ba, dm, 1, 2);
        rho.setVal(-6543.);
        ped.setVal(-8765.);
        u.setVal(-1234.);
        for (int c = 0; c < 3; ++c) {
            for (auto* family : {&ji, &je, &jp}) {
                (*family)[c] = std::make_unique<MultiFab>(
                    amrex::convert(ba, yee(c)), dm, 1, 4);
                (*family)[c]->setVal(0.);
            }
        }
        fill(rho, [] AMREX_GPU_DEVICE(int i, int j, int k) {
            return PhysConst::q_e * n0 *
                   (1.3 + .2 * std::sin(.4 * i + .3 * j + .2 * k));
        });
        fill(
            rho,
            [] AMREX_GPU_DEVICE(int i, int j, int k) {
                return PhysConst::q_e * n0 *
                       (1.6 + .1 * std::cos(.2 * i - .3 * j + .1 * k));
            },
            1);
        fill(ped, [] AMREX_GPU_DEVICE(int, int, int) {
            return .2 * n0 * PhysConst::q_e;
        });
        rho.OverrideSync(geom.periodicity());
        ped.OverrideSync(geom.periodicity());
        fill(u, [] AMREX_GPU_DEVICE(int i, int j, int k) {
            return n0 * PhysConst::kb * t0 / (gamma_e - 1) *
                   (1.1 + .1 * std::cos(.2 * i + .4 * j + .15 * k));
        });
    }
    KineticThermalMoments
    adapter () const {
        return {geom, ba, dm, opts};
    }
    MultiFab
    cell (int ng = 0) const {
        return {ba, dm, 1, ng};
    }
    MultiFab
    node (int ng = 1) const {
        return {rho.boxArray(), dm, 1, ng};
    }
    Faces
    faces () const {
        Faces f;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            f[d] = std::make_unique<MultiFab>(
                amrex::convert(ba, amrex::IntVect::TheDimensionVector(d)), dm,
                1, 0);
        }
        return f;
    }
    void
    image_current (Current& j) const {
        for (auto& f : j) {
            f->OverrideSync(geom.periodicity());
            f->FillBoundary(geom.periodicity());
        }
        PEC::ApplyReflectiveBoundarytoJfield(j[0].get(), j[1].get(), j[2].get(),
                                             blo, bhi, plo, phi, geom, 0,
                                             PatchType::fine, {});
        for (auto& f : j) {
            f->OverrideSync(geom.periodicity());
            f->FillBoundary(geom.periodicity());
        }
    }
    void
    manufacture_currents () {
        for (int c = 0; c < 3; ++c) {
            auto ty = yee(c);
            fill(*ji[c], [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const x = i + .5 * (1 - ty[0]);
                return (1.0 + .13 * c) *
                       (.4 + std::sin(.21 * x + .18 * j + .17 * k));
            });
        }
        image_current(ji);
        for (int c = 0; c < 3; ++c) {
            MultiFab::Copy(*je[c], *ji[c], 0, 0, 1, 4);
            je[c]->mult(-.7, 0, 1, 4);
            MultiFab::Copy(*jp[c], *ji[c], 0, 0, 1, 4);
            jp[c]->mult(.3, 0, 1, 4);
        }
    }
    KineticThermalStateView
    state (bool currents = true) const {
        KineticThermalStateView s{rho};
        s.pedestal = &ped;
        s.energy = &u;
        if (currents) {
            s.ion_current = view(ji);
            s.current = view(je);
        }
        return s;
    }
};

inline void
native_divergence (const Context& c, const Current& j, MultiFab& div,
                   [[maybe_unused]] bool verboncoeur = false) {
    // Exact RZ formula of MassMatrixDensityProjection::ComputeDivergence:
    // configured factor 3/4, with production Yee derivative kernels elsewhere.
    // This fixture does not instantiate the full WarpX singleton.
    [[maybe_unused]] Real const axis_factor = verboncoeur ? 3. : 4.;
    auto dx = c.geom.CellSizeArray();
    for (amrex::MFIter mfi(div); mfi.isValid(); ++mfi) {
        auto jr = j[0]->const_array(mfi);
        auto jt = j[1]->const_array(mfi);
        auto jz = j[2]->const_array(mfi);
        auto out = div.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int y, int k) {
                Real cx = 1 / dx[0], cy = 1 / dx[1];
#ifdef WARPX_DIM_RZ
                amrex::ignore_unused(jt);
                out(i, y, k) =
                    (i == 0 ? axis_factor * jr(i, y, k) / dx[0]
                            : CylindricalYeeAlgorithm::DownwardDrr_over_r(
                                  jr, i * dx[0], dx[0], &cx, 1, i, y, k, 0)) +
                    CylindricalYeeAlgorithm::DownwardDz(jz, &cy, 1, i, y, k, 0);
#else
            Real cz=1/dx[2];
            out(i,y,k)=CartesianYeeAlgorithm::DownwardDx(jr,&cx,1,i,y,k)
                +CartesianYeeAlgorithm::DownwardDy(jt,&cy,1,i,y,k)
                +CartesianYeeAlgorithm::DownwardDz(jz,&cz,1,i,y,k);
#endif
            });
    }
}
inline void
cell_divergence (const Context& c, const Faces& f, MultiFab& div) {
    auto dx = c.geom.CellSizeArray();
    for (amrex::MFIter mfi(div); mfi.isValid(); ++mfi) {
        amrex::GpuArray<amrex::Array4<const Real>, AMREX_SPACEDIM> a{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            a[d] = f[d]->const_array(mfi);
        }
        auto out = div.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real value = 0.;
                int p[3] = {i, j, k};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    int high[3] = {i, j, k};
                    ++high[d];
                    Real lo_area = 1., hi_area = 1., volume = dx[d];
#ifdef WARPX_DIM_RZ
                    if (d == 0) {
                        lo_area = i * dx[0];
                        hi_area = (i + 1) * dx[0];
                        volume = (i + .5) * dx[0] * dx[0];
                    }
#endif
                    value += (hi_area * a[d](high[0], high[1], high[2]) -
                              lo_area * a[d](p[0], p[1], p[2])) /
                             volume;
                }
                out(i, j, k) = value;
            });
    }
}
inline Real
integral (const Context& c, const MultiFab& f, bool nodal, int component = 0,
          [[maybe_unused]] bool native_moment = false) {
    auto dx = c.geom.CellSizeArray();
    auto lo = c.geom.Domain().smallEnd(), hi = c.geom.Domain().bigEnd();
    auto periodic = c.geom.isPeriodicArray();
    [[maybe_unused]] bool const corrected_axis =
        c.opts.verboncoeur_axis_correction;
    auto owned = f.OwnerMask(c.geom.periodicity());
    Real total = 0.;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.const_array(mfi);
        auto owner = owned->const_array(mfi);
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<Real> data(op);
        op.eval(
            mfi.validbox(), data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Real> {
                if (!owner(i, j, k)) {
                    return {0.};
                }
                int index[3] = {i, j, k};
                Real v = 1.;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    Real length = dx[d];
#ifdef WARPX_DIM_RZ
                    if (d == 0) {
                        Real low =
                            nodal ? std::max(0., (i - .5) * dx[0]) : i * dx[0];
                        Real high = nodal ? std::min((hi[0] + 1) * dx[0],
                                                     (i + .5) * dx[0])
                                          : (i + 1) * dx[0];
                        v *= MathConst::pi * (high * high - low * low);
                        if (nodal && native_moment) {
                            if (i == 0 && corrected_axis)
                                v *= 4. / 3.;
                            if (i == hi[0] + 1)
                                v *= (hi[0] + 1.) / (hi[0] + .75);
                        }
                        continue;
                    }
#endif
                    if (nodal && !periodic[d] &&
                        (index[d] == lo[d] || index[d] == hi[d] + 1)) {
                        length *= .5;
                    }
                    v *= length;
                }
                return {v * a(i, j, k, component)};
            });
        total += amrex::get<0>(data.value());
    }
    amrex::ParallelDescriptor::ReduceRealSum(total);
    return total;
}
inline Real
surface_flux (const Context& c, const Faces& f) {
    auto dx = c.geom.CellSizeArray();
    auto lo = c.geom.Domain().smallEnd(), hi = c.geom.Domain().bigEnd();
    Real total = 0.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (c.geom.isPeriodic(d)) {
            continue;
        }
        for (amrex::MFIter mfi(*f[d]); mfi.isValid(); ++mfi) {
            auto a = f[d]->const_array(mfi);
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<Real> data(op);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j,
                                         int k) -> amrex::GpuTuple<Real> {
                        int index[3] = {i, j, k};
                        if (index[d] != lo[d] && index[d] != hi[d] + 1) {
                            return {0.};
                        }
                        Real area = 1.;
                        for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                            if (t != d) {
                                area *= dx[t];
                            }
                        }
#ifdef WARPX_DIM_RZ
                        if (d == 0) {
                            area *= 2 * MathConst::pi * i * dx[0];
                        } else {
                            area *= 2 * MathConst::pi * (i + .5) * dx[0];
                        }
#endif
                        return {(index[d] == lo[d] ? -1. : 1.) * area *
                                a(i, j, k)};
                    });
            total += amrex::get<0>(data.value());
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(total);
    return total;
}
void particles (Context& c);
} // namespace moment_test
#endif
