/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "MomentFixture.H"

namespace moment_test {
void
closed_faces (Context& c) {
    // Large unrelated adjacent edge currents expose cancellation residue in
    // metric ghost images. A reflecting physical face must be EXACTLY closed,
    // since the thermal equation rejects unspecified inward boundary flow.
    for (int component=0;component<3;++component) {
        fill(*c.ji[component], [] AMREX_GPU_DEVICE(int i,int j,int k) {
            return 834807.5571017149 + 797476.713591479 *
                std::sin(.37*i+.61*j+.29*k);
        });
    }
    auto map=c.adapter(); auto faces=c.faces();
    map.RestrictCurrent(view(c.ji),face_view(faces));
    Real error=0;
    for(int d=0;d<AMREX_SPACEDIM;++d) {
        auto constrained=clone(*faces[d]);
        auto lo=c.geom.Domain().smallEnd(d),hi=c.geom.Domain().bigEnd(d)+1;
        auto low=c.opts.current_boundary[d][0];
        auto high=c.opts.current_boundary[d][1];
        if(low==MomentBoundary::Unspecified)low=c.opts.boundary[d][0];
        if(high==MomentBoundary::Unspecified)high=c.opts.boundary[d][1];
        bool const closed_lo=low==MomentBoundary::PMC || low==MomentBoundary::Axis;
        bool const closed_hi=high==MomentBoundary::PMC;
        for(amrex::MFIter mfi(constrained);mfi.isValid();++mfi) {
            auto a=constrained.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                int x[3]={i,j,k};
                if(!((x[d]==lo && closed_lo) || (x[d]==hi && closed_hi))) a(i,j,k)=0;
            });
        }
        error=std::max(error,constrained.norm0());
    }
    amrex::Print()<<"closed normal face current max="<<std::setprecision(17)<<error<<"\n";
    near(error,0.,0.,"closed normal current must be exact zero");
}
void
continuity (Context& c) {
    c.manufacture_currents();
    auto map = c.adapter();
    auto f = c.faces();
    auto div = c.node(), left = c.cell(), right = c.cell();
    native_divergence(c, c.ji, div, c.opts.verboncoeur_axis_correction);
    map.RestrictCurrent(view(c.ji), face_view(f));
    cell_divergence(c, f, left);
    map.RestrictNativeMoment(div, 0, right);
    Real const err = difference(left, right),
               scale = std::max(1., right.norm0());
    near(err, 0., 3.e-14 * scale,
         "Dcc RJ = Rrho DYee with production Yee and boundary images");
    near(integral(c, left, false), surface_flux(c, f), 3.e-13 * scale,
         "cell metric flux telescopes");
    if (c.phi[0] == ParticleBoundaryType::Reflecting) {
        near(surface_flux(c, f), 0., 3.e-13 * scale,
             "reflecting boundary charge flux is zero");
    } else {
        check(std::abs(surface_flux(c, f)) > .01,
              "PEC boundary charge exchange is retained");
    }
    map.RestrictNodalScalar(c.rho, 1, right);
    near(integral(c, right, false), integral(c, c.rho, true, 1),
         3.e-14 * std::abs(integral(c, right, false)),
         "nodal-to-cell physical volume accounting");
    map.RestrictNativeMoment(c.rho, 1, right);
    near(integral(c, right, false), integral(c, c.rho, true, 1, true),
         3.e-14 * std::abs(integral(c, right, false)),
         "native nodal extensive quadrature accounting");
    fill(c.rho, [] AMREX_GPU_DEVICE(int, int, int) { return 17.; });
    map.RestrictNodalScalar(c.rho, 0, right);
    near(right.min(0), 17., 0., "uniform density map minimum");
    near(right.max(0), 17., 0., "uniform density map maximum");
    map.RestrictNativeMoment(c.rho, 0, right);
    auto native_constant_error = c.cell();
    bool const corrected = c.opts.verboncoeur_axis_correction;
    int const last = c.geom.Domain().bigEnd(0);
    for (amrex::MFIter it(right); it.isValid(); ++it) {
        auto a = right.const_array(it);
        auto e = native_constant_error.array(it);
        amrex::ParallelFor(
            it.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real expected = 17.;
#ifdef WARPX_DIM_RZ
                if (i == 0 && corrected)
                    expected *= 13. / 12.;
                if (i == last)
                    expected *= 1. + 1. / (4. * (2. * last + 1.));
#else
            amrex::ignore_unused(i,corrected,last);
#endif
                e(i, j, k) = a(i, j, k) - expected;
            });
    }
    near(native_constant_error.norm0(), 0., 5.e-14,
         "native constant density has exactly declared axis/wall quadrature "
         "bias");

#ifdef WARPX_DIM_RZ
    // The arithmetic point average is a negative control, especially at axis.
    auto wrong = c.cell();
    for (amrex::MFIter mfi(wrong); mfi.isValid(); ++mfi) {
        auto a = div.const_array(mfi);
        auto b = wrong.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                b(i, j, k) = .25 * (a(i, j, k) + a(i + 1, j, k) +
                                    a(i, j + 1, k) + a(i + 1, j + 1, k));
            });
    }
    check(difference(left, wrong) > 1.e-3,
          "Cartesian point average must fail RZ continuity map");
#endif
    amrex::Print() << "continuity commutator=" << err << " scale=" << scale
                   << " net_PEC_flux=" << surface_flux(c, f) << '\n';
}
void
pressure (Context& c) {
    c.manufacture_currents();
    auto map = c.adapter();
    auto state = c.state();
    state.energy = nullptr;
    // A sharp physical density jump, including negative/empty floor cells.
    fill(c.rho, [] AMREX_GPU_DEVICE(int i, int, int) {
        return PhysConst::q_e * n0 * (i < 4 ? -.01 : (i < 8 ? .03 : 1000.));
    });
    fill(c.ped, [] AMREX_GPU_DEVICE(int i, int, int) {
        return i < 2 ? 0. : PhysConst::q_e * .02 * n0;
    });
    check(map.Evaluate(state), "density evaluation");
    for (amrex::MFIter mfi(c.u); mfi.isValid(); ++mfi) {
        auto n = map.NumberDensity().const_array(mfi);
        auto u = c.u.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                u(i, j, k) = PhysConst::kb * n(i, j, k) * t0 / (gamma_e - 1.);
            });
    }
    state.energy = &c.u;
    check(map.Evaluate(state), "variable density constant T evaluation");
    near(map.CellTemperature().min(0), t0, 1.e-9, "constant T cell min");
    near(map.CellTemperature().max(0), t0, 1.e-9, "constant T cell max");
    near(map.NodalTemperature().min(0), t0, 1.e-9, "constant T node min");
    near(map.NodalTemperature().max(0), t0, 1.e-9, "constant T node max");
    check(map.NumberDensity().min(0) >= c.opts.number_density_floor,
          "effective density floor");
    near(map.ActiveCells().min(0), 0., 0., "empty cells inactive");
    near(map.ActiveCells().max(0), 1., 0.,
         "nonempty cells active with pedestal");
    auto err = c.node(0);
    for (amrex::MFIter mfi(err); mfi.isValid(); ++mfi) {
        auto n = map.NodalNumberDensity().const_array(mfi);
        auto t = map.NodalTemperature().const_array(mfi);
        auto p = map.NodalPressure().const_array(mfi);
        auto e = err.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            e(i, j, k) = p(i, j, k) - PhysConst::kb * n(i, j, k) * t(i, j, k);
        });
    }
    near(err.norm0(), 0., 0., "nodal thermodynamic product exact");
    auto p_reference = clone(map.NodalPressure());
    PEC::ApplyPECtoElectronPressure(&p_reference, c.blo, c.bhi, c.geom, 0,
                                    PatchType::fine, {});
    PEC::ApplyZeroGradientToScalar(&p_reference, c.blo, c.bhi, c.geom, 0,
                                   PatchType::fine, {}, false);
    p_reference.OverrideSync(c.geom.periodicity());
    near(difference(p_reference, map.OhmPressure()), 0., 0.,
         "Ohm pressure matches actual deterministic PEC boundary");
    auto direct = clone(map.FaceElectronVelocity(0));
    state.current = view(c.jp);
    state.current_input = ThermalCurrentInput::TotalPlasma;
    check(map.Evaluate(state), "total plasma current conversion");
    near(difference(direct, map.FaceElectronVelocity(0)), 0.,
         1.e-13 * std::max(1., direct.norm0()),
         "signed Je equals Jp-Ji conversion");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto check_flux = clone(map.FaceElectronCurrent(d));
        for (amrex::MFIter mfi(check_flux); mfi.isValid(); ++mfi) {
            auto n = map.FaceNumberDensity(d).const_array(mfi);
            auto u = map.FaceElectronVelocity(d).const_array(mfi);
            auto je = map.FaceElectronCurrent(d).const_array(mfi);
            auto e = check_flux.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    e(i, j, k) =
                        PhysConst::q_e * n(i, j, k) * u(i, j, k) + je(i, j, k);
                });
        }
        near(check_flux.norm0(), 0., 1.e-14,
             "electron charge sign and face number flux");
    }
    // Temperature interpolation must stay inside its local cell range; an
    // alternating positive field is also a strong non-overshoot control.
    for (amrex::MFIter mfi(c.u); mfi.isValid(); ++mfi) {
        auto n = map.NumberDensity().const_array(mfi);
        auto u = c.u.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                u(i, j, k) = PhysConst::kb * n(i, j, k) *
                             (100. + 900. * ((i + j + k) & 1)) / (gamma_e - 1.);
            });
    }
    check(map.Evaluate(state), "bounded variable T");
    check(map.NodalTemperature().min(0) >= 100. - 1.e-12 &&
              map.NodalTemperature().max(0) <= 1000. + 1.e-12,
          "bounded convex temperature reconstruction");
    auto original_u = clone(c.u);
    c.u.setVal(-1., 0, 1, 0);
    check(!map.Evaluate(state), "negative thermal U rejects without clipping");
    near(c.u.max(0), -1., 0., "rejected input U unchanged");
    MultiFab::Copy(c.u, original_u, 0, 0, 1, 2);
    amrex::Print() << "pressure constant-T, floors, signs, boundedness, actual "
                      "PEC images PASS\n";
}

// Independent interpolation oracle: piecewise nearest-cell physical images,
// periodic communication, then arithmetic cell-to-node interpolation.
void
temperature_nodes (const Context& c, MultiFab& cell, MultiFab& node) {
    cell.FillBoundary(c.geom.periodicity());
    auto dom = c.geom.Domain();
    auto periodic = c.geom.isPeriodicArray();
    for (amrex::MFIter mfi(node); mfi.isValid(); ++mfi) {
        auto a = cell.const_array(mfi);
        auto b = node.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real sum = 0.;
                for (int bits = 0; bits < (1 << AMREX_SPACEDIM); ++bits) {
                    int x[3] = {i - (bits & 1), j - ((bits >> 1) & 1), k};
#if AMREX_SPACEDIM == 3
                    x[2] -= (bits >> 2) & 1;
#endif
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (!periodic[d]) {
                            x[d] = amrex::max(dom.smallEnd(d),
                                              amrex::min(dom.bigEnd(d), x[d]));
                        }
                    }
                    sum += a(x[0], x[1], x[2]);
                }
                b(i, j, k) = sum / static_cast<Real>(1 << AMREX_SPACEDIM);
            });
    }
}
void
derivatives (Context& c) {
    c.manufacture_currents();
    auto base = c.adapter(), plus = c.adapter(), minus = c.adapter();
    auto s = c.state();
    check(base.Evaluate(s), "derivative base");
    auto drho = c.node(), dn = c.cell(1), du = c.cell(2), dT = c.cell(1),
         dTn = c.node(), dP = c.node();
    fill(drho, [] AMREX_GPU_DEVICE(int i, int j, int k) {
        return PhysConst::q_e * n0 *
               (.2 + .03 * std::sin(.5 * i - .1 * j + .3 * k));
    });
    fill(du, [] AMREX_GPU_DEVICE(int i, int j, int k) {
        return PhysConst::kb * n0 * t0 *
               (.1 + .04 * std::cos(.2 * i + .3 * j + .1 * k));
    });
    drho.OverrideSync(c.geom.periodicity());
    base.RestrictNativeMoment(drho, 0, dn);
    dn.mult(1. / PhysConst::q_e, 0, 1, 1);
    for (int direction_case = 0; direction_case < 3; ++direction_case) {
        bool use_u = direction_case != 1, use_rho = direction_case != 0;
        for (amrex::MFIter mfi(dT); mfi.isValid(); ++mfi) {
            auto n = base.NumberDensity().const_array(mfi);
            auto t = base.CellTemperature().const_array(mfi);
            auto nd = dn.const_array(mfi);
            auto ud = du.const_array(mfi);
            auto out = dT.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    out(i, j, k) =
                        (use_u ? (gamma_e - 1.) * ud(i, j, k) /
                                     (PhysConst::kb * n(i, j, k))
                               : 0.) -
                        (use_rho ? t(i, j, k) * nd(i, j, k) / n(i, j, k) : 0.);
                });
        }
        temperature_nodes(c, dT, dTn);
        for (amrex::MFIter mfi(dP); mfi.isValid(); ++mfi) {
            auto n = base.NodalNumberDensity().const_array(mfi);
            auto t = base.NodalTemperature().const_array(mfi);
            auto td = dTn.const_array(mfi);
            auto rd = drho.const_array(mfi);
            auto p = dP.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    p(i, j, k) =
                        PhysConst::kb *
                        (n(i, j, k) * td(i, j, k) +
                         (use_rho ? rd(i, j, k) / PhysConst::q_e * t(i, j, k)
                                  : 0.));
                });
        }
        Real best = 1.;
        for (Real h : {1.e-3, 1.e-4, 1.e-5}) {
            auto rp = clone(c.rho), rm = clone(c.rho), up = clone(c.u),
                 um = clone(c.u);
            if (use_rho) {
                MultiFab::Saxpy(rp, h, drho, 0, 0, 1, 0);
                MultiFab::Saxpy(rm, -h, drho, 0, 0, 1, 0);
            }
            if (use_u) {
                MultiFab::Saxpy(up, h, du, 0, 0, 1, 0);
                MultiFab::Saxpy(um, -h, du, 0, 0, 1, 0);
            }
            KineticThermalStateView sp{rp}, sm{rm};
            sp.pedestal = sm.pedestal = &c.ped;
            sp.energy = &up;
            sm.energy = &um;
            sp.ion_current = sm.ion_current = view(c.ji);
            sp.current = sm.current = view(c.je);
            check(plus.Evaluate(sp) && minus.Evaluate(sm),
                  "directional derivative trials");
            auto numerical = clone(dP);
            MultiFab::LinComb(numerical, 1. / (2 * h), plus.NodalPressure(), 0,
                              -1. / (2 * h), minus.NodalPressure(), 0, 0, 1, 0);
            Real error = difference(numerical, dP) / std::max(1., dP.norm0());
            best = std::min(best, error);
            if (use_rho) {
                auto expected = clone(base.FaceElectronVelocity(0));
                for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
                    auto nf = base.FaceNumberDensity(0).const_array(mfi);
                    auto v = base.FaceElectronVelocity(0).const_array(mfi);
                    auto np = plus.FaceNumberDensity(0).const_array(mfi);
                    auto nm = minus.FaceNumberDensity(0).const_array(mfi);
                    auto a = expected.array(mfi);
                    amrex::ParallelFor(
                        mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            a(i, j, k) = -v(i, j, k) *
                                         (np(i, j, k) - nm(i, j, k)) /
                                         (2 * h * nf(i, j, k));
                        });
                }
                auto nv = clone(expected);
                MultiFab::LinComb(
                    nv, 1. / (2 * h), plus.FaceElectronVelocity(0), 0,
                    -1. / (2 * h), minus.FaceElectronVelocity(0), 0, 0, 1, 0);
                near(difference(nv, expected) / std::max(1., expected.norm0()),
                     0., 3.e-7, "live velocity density denominator derivative");
            }
        }
        near(best, 0., 3.e-9,
             "independent U/rho/mixed pressure directional derivative");
        amrex::Print() << "pressure_Jv direction=" << direction_case
                       << " best_relative_error=" << best << '\n';
    }
    // A whole inactive vacuum region: perturb rho below its floor. The density
    // and pressure derivatives are exactly zero, while U remains a live DOF.
    fill(c.rho, [] AMREX_GPU_DEVICE(int, int, int) {
        return .01 * n0 * PhysConst::q_e;
    });
    fill(c.ped, [] AMREX_GPU_DEVICE(int, int, int) { return 0.; });
    check(base.Evaluate(c.state()), "vacuum base");
    auto saved = clone(base.NodalPressure());
    fill(c.rho, [] AMREX_GPU_DEVICE(int, int, int) {
        return .02 * n0 * PhysConst::q_e;
    });
    check(plus.Evaluate(c.state()), "vacuum perturb");
    near(difference(saved, plus.NodalPressure()), 0., 0.,
         "inactive density floor derivative zero");
    near(plus.ActiveCells().max(0), 1., 0.,
         "pedestal-presence opens nonempty halo at zero pedestal");
    auto no_ped = c.state();
    no_ped.pedestal = nullptr;
    check(plus.Evaluate(no_ped), "inactive no pedestal");
    near(plus.ActiveCells().max(0), 0., 0.,
         "raw density gate remains distinct from effective floor");
    auto vacuum_pressure = clone(plus.NodalPressure());
    c.u.mult(1.125, 0, 1, 0);
    check(plus.Evaluate(no_ped), "inactive U perturbation remains live");
    vacuum_pressure.mult(1.125, 0, 1, 0);
    near(difference(vacuum_pressure, plus.NodalPressure()), 0.,
         4.e-15 * vacuum_pressure.norm0(),
         "inactive floor has live linear U-to-pressure response");
}
void
purity (Context& c) {
    c.manufacture_currents();
    auto r = clone(c.rho), p = clone(c.ped), u = clone(c.u);
    Current snap_i, snap_e, snap_p;
    for (int d = 0; d < 3; ++d) {
        snap_i[d] = std::make_unique<MultiFab>(clone(*c.ji[d]));
        snap_e[d] = std::make_unique<MultiFab>(clone(*c.je[d]));
        snap_p[d] = std::make_unique<MultiFab>(clone(*c.jp[d]));
    }
    auto map = c.adapter();
    auto a = c.state();
    check(map.Evaluate(a), "purity A");
    auto n = clone(map.NumberDensity()), t = clone(map.NodalTemperature()),
         pressure = clone(map.OhmPressure());
    Faces v = c.faces();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        MultiFab::Copy(*v[d], map.FaceElectronVelocity(d), 0, 0, 1, 0);
    }
    auto b = a;
    b.charge_component = 1;
    b.pedestal = nullptr;
    b.current = view(c.jp);
    b.current_input = ThermalCurrentInput::TotalPlasma;
    check(map.Evaluate(b), "purity B");
    check(difference(n, map.NumberDensity()) > .01 * n0,
          "old versus true endpoint density is separate");
    check(map.Evaluate(a), "purity A repeat");
    near(difference(n, map.NumberDensity(), true), 0., 0.,
         "A/B/A density including ghosts");
    near(difference(t, map.NodalTemperature(), true), 0., 0.,
         "A/B/A nodal temperature including ghosts");
    near(difference(pressure, map.OhmPressure(), true), 0., 0.,
         "A/B/A Ohm pressure including ghosts");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        near(difference(*v[d], map.FaceElectronVelocity(d)), 0., 0.,
             "A/B/A face velocity");
    }
    near(difference(r, c.rho, true), 0., 0., "input rho and ghosts unchanged");
    near(difference(p, c.ped, true), 0., 0.,
         "input pedestal and ghosts unchanged");
    near(difference(u, c.u, true), 0., 0., "input U and ghosts unchanged");
    for (int d = 0; d < 3; ++d) {
        near(difference(*snap_i[d], *c.ji[d], true), 0., 0.,
             "input ion current and ghosts unchanged");
        near(difference(*snap_e[d], *c.je[d], true), 0., 0.,
             "input electron current and ghosts unchanged");
        near(difference(*snap_p[d], *c.jp[d], true), 0., 0.,
             "input total current and ghosts unchanged");
    }
    auto density_only = a;
    density_only.energy = nullptr;
    density_only.current = {};
    density_only.ion_current = {};
    check(map.Evaluate(density_only), "density-only endpoint evaluation");
    check(!map.HasPressure() && !map.HasCurrents(),
          "no stale pressure/current marked available");
    amrex::Print() << "A/B/A and all input/ghost purity exact PASS\n";
}

} // namespace moment_test

int
main (int argc, char** argv) {
    using namespace moment_test;
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse("hybrid_pic_model")
            .add("deterministic_pressure_bc", true);
        amrex::ParmParse pp("test");
        int box = 4, periodic = 0, corrected = 0, reflecting = 0;
        std::string test = "continuity";
        pp.query("max_grid_size", box);
        pp.query("periodic", periodic);
        pp.query("case", test);
        pp.query("corrected_axis", corrected);
        pp.query("reflecting", reflecting);
        Context c(box, periodic != 0, corrected != 0, reflecting != 0);
        if (test.rfind("reject_", 0) == 0) {
            if (test == "reject_current_boundary") {
                c.opts.current_boundary[1][0] = MomentBoundary::Periodic;
            } else if (test == "reject_modes") {
                c.opts.azimuthal_modes = 2;
            } else if (test == "reject_floor") {
                c.opts.number_density_floor = 0.;
            } else if (test == "reject_boundary") {
                c.opts.boundary[1][0] = MomentBoundary::Unspecified;
            }
            auto rejected = c.adapter();
            if (test == "reject_layout") {
                KineticThermalStateView s{c.u};
                rejected.Evaluate(s);
            } else if (test == "reject_alias") {
                check(rejected.Evaluate(c.state(false)), "alias seed");
                KineticThermalStateView s{rejected.NodalNumberDensity()};
                rejected.Evaluate(s);
            }
            amrex::Abort(
                "ERROR unsupported fixture did not trigger its specific guard");
        }
        if (test == "closed_faces") {
            closed_faces(c);
        } else if (test == "continuity") {
            continuity(c);
        } else if (test == "pressure") {
            pressure(c);
        } else if (test == "derivatives") {
            derivatives(c);
        } else if (test == "purity") {
            purity(c);
        } else if (test == "particles") {
            particles(c);
        } else {
            amrex::Abort("unknown moment test");
        }
        amrex::Print() << "PASS " << test << " boxes=" << c.ba.size()
                       << " ranks=" << amrex::ParallelDescriptor::NProcs()
                       << '\n';
    }
    amrex::Finalize();
}
