/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "MomentFixture.H"
#include "Particles/Deposition/ChargeDeposition.H"
#include "Particles/Deposition/CurrentDeposition.H"

namespace moment_test {
// Reproduce ONLY inverse-volume scaling/folding, with constants from the
// production WarpXPushFieldsEM.cpp. Deposition and wall/cap images below call
// the actual production kernels. There is no synthetic continuity current.
void
inverse_rz_volume (MultiFab& f, [[maybe_unused]] bool corrected) {
#ifdef WARPX_DIM_RZ
    auto type = f.ixType().toIntVect();
    int const ng = f.nGrow();
    // Geometry is the fixed physical 2 x 3 fixture, dr=2/16.
    constexpr Real dr = 2. / 16.;
    for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
        auto a = f.array(mfi);
        auto box = mfi.fabbox();
        int shift = 1 - type[0];
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (i < 0) {
                return;
            }
            Real radius = (i + .5 * shift) * dr;
            if (i >= 1 - shift && i <= ng - shift &&
                box.contains(amrex::IntVect(-shift - i, j))) {
                // Jr (radially cell centred) is odd. Nodal rho/Jz are even.
                a(i, j, k) += (shift ? -1. : 1.) * a(-shift - i, j, k);
            }
            a(i, j, k) /= radius == 0.
                              ? MathConst::pi * dr * (corrected ? 1. / 3. : .25)
                              : 2 * MathConst::pi * radius;
        });
    }
#else
    amrex::ignore_unused(f);
#endif
}
void
particles (Context& c) {
    auto old = c.node(4), end = c.node(4);
    old.setVal(0.);
    end.setVal(0.);
    for (auto& f : c.ji) {
        f->setVal(0.);
    }
    auto dx = c.geom.CellSizeArray();
    constexpr Real dt = .07;
    // Axis, PEC cloud overlap, both PMC caps, box seams and corner overlaps.
    std::vector<std::array<Real, 6>> trajectory = {
        {.16, .18, .2, .29, .27, .3},
        {15.65, 8.1, 7.7, 15.82, 8.3, 7.8},
        {7.7, .18, .2, 8.15, .3, .25},
        {4.01, 15.7, 15.8, 3.84, 15.83, 15.67},
        {11.85, 7.91, 4.12, 12.1, 8.17, 3.88},
        {.24, 15.77, 15.61, .34, 15.63, 15.73}};
    for (amrex::MFIter mfi(old); mfi.isValid(); ++mfi) {
        std::vector<amrex::ParticleReal> x0, y0, z0, x1, y1, z1, xm, ym, zm,
            theta, w, momentum;
        auto valid = amrex::enclosedCells(mfi.validbox());
        for (std::size_t p = 0; p < trajectory.size(); ++p) {
            auto t = trajectory[p];
            int ix = static_cast<int>(.5 * (t[0] + t[3])),
                iy = static_cast<int>(.5 * (t[1] + t[4]));
#ifdef WARPX_DIM_RZ
            amrex::IntVect owner(ix, iy);
#else
            amrex::IntVect owner(ix, iy, static_cast<int>(.5 * (t[2] + t[5])));
#endif
            if (!valid.contains(owner)) {
                continue;
            }
            x0.push_back(t[0] * dx[0]);
            x1.push_back(t[3] * dx[0]);
            xm.push_back(.5 * (x0.back() + x1.back()));
#ifdef WARPX_DIM_RZ
            y0.push_back(0.);
            y1.push_back(0.);
            ym.push_back(0.);
            z0.push_back(t[1] * dx[1]);
            z1.push_back(t[4] * dx[1]);
#else
            y0.push_back(t[1] * dx[1]);
            y1.push_back(t[4] * dx[1]);
            ym.push_back(.5 * (y0.back() + y1.back()));
            z0.push_back(t[2] * dx[2]);
            z1.push_back(t[5] * dx[2]);
#endif
            zm.push_back(.5 * (z0.back() + z1.back()));
            theta.push_back(0.);
            w.push_back(1. + .2 * p);
            momentum.push_back(0.);
        }
        if (w.empty()) {
            continue;
        }
        auto device = [] (auto const& host) {
            amrex::Gpu::DeviceVector<amrex::ParticleReal> d(host.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(),
                             d.begin());
            return d;
        };
        auto X0 = device(x0), Y0 = device(y0), Z0 = device(z0), X1 = device(x1),
             Y1 = device(y1), Z1 = device(z1), XM = device(xm), YM = device(ym),
             ZM = device(zm), TH = device(theta), W = device(w),
             M = device(momentum);
        GetParticlePosition<PIdx> get_old, get_end, get_mid;
        get_old.m_x = X0.data();
        get_old.m_y = Y0.data();
        get_old.m_z = Z0.data();
        get_end.m_x = X1.data();
        get_end.m_y = Y1.data();
        get_end.m_z = Z1.data();
        get_mid.m_x = XM.data();
        get_mid.m_y = YM.data();
        get_mid.m_z = ZM.data();
#ifdef WARPX_DIM_RZ
        get_old.m_theta = get_end.m_theta = get_mid.m_theta = TH.data();
        amrex::XDim3 inverse{1 / dx[0], 1., 1 / dx[1]};
#else
        amrex::XDim3 inverse{1 / dx[0], 1 / dx[1], 1 / dx[2]};
#endif
        amrex::XDim3 origin{0., 0., 0.};
        amrex::Dim3 lo{0, 0, 0};
        doChargeDepositionShapeN<3>(get_old, W.data(), nullptr, old[mfi],
                                    w.size(), inverse, origin, lo, 1., 1);
        doChargeDepositionShapeN<3>(get_end, W.data(), nullptr, end[mfi],
                                    w.size(), inverse, origin, lo, 1., 1);
        amrex::GpuArray<amrex::GpuArray<double, 2>, AMREX_SPACEDIM> domain{};
        amrex::GpuArray<amrex::GpuArray<bool, 2>, AMREX_SPACEDIM> crop{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            domain[d] = {0., 16.};
            crop[d] = {false, false};
        }
        doChargeConservingDepositionShapeNImplicit<3>(
            X0.data(), Y0.data(), Z0.data(), get_mid, W.data(), M.data(),
            M.data(), M.data(), M.data(), M.data(), M.data(), nullptr,
            c.ji[0]->array(mfi), c.ji[1]->array(mfi), c.ji[2]->array(mfi),
            w.size(), dt, inverse, origin, domain, crop, lo, 1., 1);
        amrex::Gpu::streamSynchronize();
    }
    inverse_rz_volume(old, c.opts.verboncoeur_axis_correction);
    inverse_rz_volume(end, c.opts.verboncoeur_axis_correction);
    inverse_rz_volume(*c.ji[0], c.opts.verboncoeur_axis_correction);
    inverse_rz_volume(*c.ji[2], c.opts.verboncoeur_axis_correction);
    for (auto* f : {&old, &end}) {
        f->SumBoundary(0, 1, f->nGrowVect(), f->nGrowVect(),
                       c.geom.periodicity());
        PEC::ApplyReflectiveBoundarytoRhofield(f, c.blo, c.bhi, c.plo, c.phi,
                                               c.geom, 0, PatchType::fine, {});
        f->OverrideSync(c.geom.periodicity());
        f->FillBoundary(c.geom.periodicity());
    }
    for (auto& f : c.ji) {
        f->SumBoundary(0, 1, f->nGrowVect(), f->nGrowVect(),
                       c.geom.periodicity());
    }
    c.image_current(c.ji);
    auto divergence = c.node(1), residual = c.node(1);
    native_divergence(c, c.ji, divergence, c.opts.verboncoeur_axis_correction);
    MultiFab::LinComb(residual, 1. / dt, end, 0, -1. / dt, old, 0, 0, 1, 0);
    MultiFab::Add(residual, divergence, 0, 0, 1, 0);
    Real const scale = std::max(1., divergence.norm0()),
               node_error = residual.norm0();
    near(node_error, 0., 5.e-13 * scale,
         "real shape-3 Esirkepov nodal continuity");
    auto map = c.adapter();
    auto f = c.faces();
    auto delta = c.cell(), next = c.cell(), before = c.cell();
    map.RestrictNativeMoment(old, 0, before);
    map.RestrictNativeMoment(end, 0, next);
    map.RestrictCurrent(view(c.ji), face_view(f));
    cell_divergence(c, f, delta);
    MultiFab::Saxpy(delta, 1. / dt, next, 0, 0, 1, 0);
    MultiFab::Saxpy(delta, -1. / dt, before, 0, 0, 1, 0);
    near(delta.norm0(), 0., 5.e-13 * scale,
         "real shape-3 Esirkepov mapped cell continuity");
    near(integral(c, next, false) - integral(c, before, false),
         -dt * surface_flux(c, f), 2.e-12,
         "real deposited charge equals physical boundary exchange");
    near(integral(c, next, false), integral(c, end, true, 0, true), 2.e-12,
         "real charge-volume remap accounting");
    if (c.phi[0] == ParticleBoundaryType::Reflecting) {
        near(
            surface_flux(c, f), 0., 2.e-12,
            "actual reflecting radial wall and PMC caps carry zero net charge");
        near(integral(c, next, false), 9., 2.e-12,
             "native quadrature preserves all particle charge");
    } else {
        check(std::abs(surface_flux(c, f)) > .001,
              "real cloud/PEC boundary current is nonzero");
    }
    amrex::Print() << "shape3 nodal_error=" << node_error
                   << " cell_error=" << delta.norm0() << " scale=" << scale
                   << " charge_old=" << integral(c, before, false)
                   << " charge_new=" << integral(c, next, false)
                   << " wall_flux=" << surface_flux(c, f) << '\n';
#ifdef WARPX_DIM_RZ
    if (c.opts.verboncoeur_axis_correction) {
        // Maxwell4 is intentionally the wrong diagnostic for actual particle3.
        native_divergence(c, c.ji, divergence, false);
        MultiFab::LinComb(residual, 1. / dt, end, 0, -1. / dt, old, 0, 0, 1, 0);
        MultiFab::Add(residual, divergence, 0, 0, 1, 0);
        Real const defect = residual.norm0();
        check(defect > .01, "Maxwell4 differs from actual corrected particle3");
        for (amrex::MFIter it(residual); it.isValid(); ++it) {
            auto a = residual.array(it);
            auto jr = c.ji[0]->const_array(it);
            amrex::ParallelFor(it.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                   if (i == 0)
                                       a(i, j, k) -= jr(i, j, k) / dx[0];
                               });
        }
        near(residual.norm0(), 0., 5.e-13 * scale,
             "Maxwell4-particle3 equals Jr/dr");
        amrex::Print() << "wrong_Maxwell4_mismatch=" << defect
                       << " predicted_error=" << residual.norm0() << '\n';
    }
#endif
}
} // namespace moment_test
