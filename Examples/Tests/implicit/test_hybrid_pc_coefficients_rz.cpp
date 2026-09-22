/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "Initialization/WarpXInit.H"
#include "NonlinearSolvers/HybridPICRZPC.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#ifdef WARPX_DIM_RZ
namespace {
using RT = amrex::Real;
using MF = amrex::MultiFab;
using Op = HybridPICRZOperator;
using Fields = amrex::Array<std::unique_ptr<MF>, 3>;
using ConstFields = amrex::Array<MF const*, 3>;

// Only the PC's adapter interface is mocked. InitData, geometry, field
// registers, HybridPICModel and native PC coefficient kernels are real.
struct MockOps {
    HybridPICModel* model = nullptr;
    MF const* ohm = nullptr;
    MF const* midpoint = nullptr;
    ConstFields magnetic{}, ions{};
    RT theta = .7;
    int ohm_component = 1;
    HybridPICModel const*
    GetHybridPICModel () const {
        return model;
    }
    RT
    GetTheta () const {
        return theta;
    }
    std::pair<MF const*, int>
    GetOhmDensityForPC (int) const {
        return {ohm, ohm_component};
    }
    MF const*
    GetRhoMidForPC (int) const {
        return midpoint;
    }
    ConstFields
    GetBfieldThetaForPC (int) const {
        return magnetic;
    }
    ConstFields
    GetIonCurrentForPC (int) const {
        return ions;
    }
    amrex::Array<FieldBoundaryType, AMREX_SPACEDIM> const&
    GetFieldBoundaryLo () const {
        return WarpX::field_boundary_lo;
    }
    amrex::Array<FieldBoundaryType, AMREX_SPACEDIM> const&
    GetFieldBoundaryHi () const {
        return WarpX::field_boundary_hi;
    }
    amrex::Vector<amrex::Array<MF*, 3>> const*
    GetMassMatricesCoeff () const {
        return nullptr;
    }
};

// All expected values use prescribed analytic fields, never read the PC's
// input arrays or invoke HybridSmoothFloor/the model parsers as an oracle.
struct Parameters {
    RT raw = 4., midpoint = 9., floor = 1., floor_width = 0.;
    RT theta = .7, dt = 2.e-9, time = .23;
    RT te0 = 3000., te_i = 20., te_j = 40.;
    RT pedestal_number = 1.e18, mass = PhysConst::m_e, taper_width = 0.;
    RT density_width = 0., axis_radius = 0., axis_width = 0.;
    RT dr = 0., dz = 0.;
    bool hall = true, pedestal = true, vacuum = false, conductor = false;
    bool bdf = false, recovery = false, flux_only = true, live_mask = false;
    int mask_mode = 0;
    int history = 1;
    amrex::GpuArray<RT, 3> B{{2., 3., 5.}}, Ji{{11., 13., 17.}},
        Jp{{19., 23., 29.}};
};
AMREX_GPU_HOST_DEVICE RT
floor_value (RT rho, Parameters const& p) {
    if (p.floor_width <= 0.) {
        return amrex::max(rho, p.floor);
    }
    RT const delta = rho - p.floor;
    return .5 * (rho + p.floor +
                 std::sqrt(delta * delta + p.floor_width * p.floor_width));
}
AMREX_GPU_HOST_DEVICE RT
gate_value (RT radius, Parameters const& p) {
    RT gate = 1.;
    if (p.vacuum) {
        if (p.density_width <= 0.) {
            // Native hard branch ignores axis_width entirely.
            if (p.raw < p.floor &&
                (p.axis_radius <= 0. || radius < p.axis_radius)) {
                gate = 0.;
            }
        } else {
            RT const density_gate =
                .5 * (1. + std::tanh((p.raw - p.floor) / p.density_width));
            RT radial_mask = 1.;
            if (p.axis_radius > 0.) {
                radial_mask =
                    p.axis_width > 0.
                        ? .5 * (1. - std::tanh((radius - p.axis_radius) /
                                               p.axis_width))
                        : (radius < p.axis_radius ? 1. : 0.);
            }
            gate = 1. - (1. - density_gate) * radial_mask;
        }
    }
    if (p.conductor && p.raw <= 0.) {
        gate = 0.;
    }
    return gate;
}
AMREX_GPU_HOST_DEVICE RT
expected_edge (int i, int j, int c, int slot, Parameters const& p) {
    RT const rindex = i + (c == 0 ? .5 : 0.);
    RT const zindex = j + (c == 2 ? .5 : 0.);
    RT const ped = p.pedestal ? p.pedestal_number * PhysConst::q_e : 0.;
    RT const inverse =
        gate_value(rindex * p.dr, p) / floor_value(p.raw + ped, p);
    RT const hall = p.hall ? p.theta * p.dt * inverse / PhysConst::mu0 : 0.;
    RT const bmag =
        std::sqrt(p.B[0] * p.B[0] + p.B[1] * p.B[1] + p.B[2] * p.B[2]);
    RT const jmag =
        std::sqrt(p.Jp[0] * p.Jp[0] + p.Jp[1] * p.Jp[1] + p.Jp[2] * p.Jp[2]);
    RT const te = p.te0 + p.te_i * rindex + p.te_j * zindex;
    if (slot == Op::RowWeight) {
        RT const density = (p.live_mask ? .3 : .15) * rindex - .4;
        bool const masked =
            p.mask_mode == 2 || (p.mask_mode == 0 && density < p.floor) ||
            (p.mask_mode == 1 && density > 0. && density < p.floor);
        return p.recovery && (!p.flux_only || c == 1) && masked ? 0. : 1.;
    }
    if (slot == Op::Hall) {
        return hall;
    }
    if (slot == Op::InvRho) {
        return inverse;
    }
    if (slot == Op::Eta) {
        return p.theta * p.dt / PhysConst::mu0 *
               (.25 + .125 * p.raw + .05 * jmag + .0002 * te + 2. * p.time);
    }
    if (slot == Op::Hyper) {
        return p.theta * p.dt / PhysConst::mu0 *
               (.02 + .03 * p.raw + .04 * bmag);
    }
    if (slot == Op::Scale) {
        RT const l2 = 4. / (p.dr * p.dr) + 4. / (p.dz * p.dz);
        return amrex::max(hall * bmag, 1. / l2);
    }
    return 0.; // No mass response in this bounded coefficient-adapter fixture.
}
AMREX_GPU_HOST_DEVICE RT
expected_node (int slot, Parameters const& p) {
    if (slot < 3) {
        return p.B[slot];
    }
    if (slot < 6) {
        int const c = slot - 3;
        return p.theta * p.dt * (p.Ji[c] - (p.hall ? p.Jp[c] : 0.));
    }
    if (p.midpoint <= 0.) {
        return 0.;
    }
    RT const gamma = p.bdf && p.history >= 2 ? (2. * p.theta + 1.) / 2. : 1.;
    RT const taper =
        p.taper_width > 0.
            ? .5 * (1. + std::tanh((p.midpoint - p.floor) / p.taper_width))
            : 1.;
    RT const ped = p.pedestal ? p.pedestal_number * PhysConst::q_e : 0.;
    return gamma * p.mass * taper /
           (PhysConst::mu0 * PhysConst::q_e * floor_value(p.midpoint + ped, p));
}

template <class Expected>
void
check (MF const& mf, int component, amrex::Geometry const& geom,
       std::string const& label, Expected expected) {
    // Exclude two physical layers; these tests isolate coefficient assembly,
    // while the independent operator fixture owns BC/parity verification.
    amrex::Box interior = amrex::surroundingNodes(geom.Domain());
    interior.grow(-2);
    MF error(mf.boxArray(), mf.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
        auto e = error.array(mfi);
        auto a = mf.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            if (!interior.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                e(i, j, k) = 0.;
                return;
            }
            RT const target = expected(i, j, k);
            RT const value = a(i, j, k, component);
            // NaN comparisons would otherwise evade a max-error test.
            bool const finite = std::isfinite(value) && std::isfinite(target);
            RT const scale = target == 0. ? 1. : std::abs(target);
            e(i, j, k) = finite ? std::abs(value - target) / scale : 1.e100;
        });
    }
    RT const error_max = error.norminf();
    amrex::Print() << "RZ_PC_COEFF " << label << " relative_error=" << error_max
                   << "\n";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(error_max < 2.e-12, label.c_str());
}
} // namespace
#endif

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
#ifdef WARPX_DIM_RZ
        // Start from inputs_test_rz_recovery_probe. Allocate a real uniform
        // pedestal at InitData, then test both enabled and disabled access.
        amrex::ParmParse hp_inputs("hybrid_pic_model");
        hp_inputs.add("density_pedestal", 1);
        hp_inputs.add("density_pedestal_profile(x,y,z)", std::string("1.e18"));
        amrex::ParmParse("pc_hybrid_pic").add("inner_max", 0);
        amrex::ParmParse("pc_hybrid_pic").add("vacuum_rows", 0);
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto& hp = *sim.get_pointer_HybridPICModel();
        using warpx::fields::FieldType;
        auto const& geom = sim.Geom(0);
        auto const* registered_rho = sim.m_fields.get(FieldType::rho_fp, 0);
        MF ohm(registered_rho->boxArray(), registered_rho->DistributionMap(), 3,
               2);
        MF mid(registered_rho->boxArray(), registered_rho->DistributionMap(), 4,
               2);
        Fields B, Ji;
        MockOps ops;
        ops.model = &hp;
        ops.ohm = &ohm;
        ops.midpoint = &mid;
        for (int c = 0; c < 3; ++c) {
            auto const* b =
                sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0)[c];
            auto const* e =
                sim.m_fields.get_alldirs(FieldType::Efield_fp, 0)[c];
            B[c] =
                std::make_unique<MF>(b->boxArray(), b->DistributionMap(), 1, 2);
            Ji[c] =
                std::make_unique<MF>(e->boxArray(), e->DistributionMap(), 1, 2);
            ops.magnetic[c] = B[c].get();
            ops.ions[c] = Ji[c].get();
            // These must not be added to the mock's already-total B.
            sim.m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0)[c]
                ->setVal(101. + 2. * c);
        }
        WarpXSolverVec state;
        state.Define(&sim, "Efield_fp");
        state.zero();
        HybridPICRZPC<WarpXSolverVec, MockOps> pc;
        pc.Define(state, &ops);
        amrex::Parser eta("0.25+0.125*rho+0.05*J+0.0002*Te+2*t");
        eta.registerVariables({"rho", "J", "Te", "t"});
        amrex::Parser hyper("0.02+0.03*rho+0.04*B");
        hyper.registerVariables({"rho", "B"});
        hp.m_eta_expression =
            "coefficient_fixture"; // Nonzero selector; no reparsing.
        hp.m_eta_te = eta.compile<4>();
        hp.m_resistivity_has_Te_dependence = true;
        hp.m_resistivity_has_J_dependence = true;
        hp.m_eta_h = hyper.compile<2>();
        hp.m_include_hyper_resistivity_term = true;
        hp.m_hyper_resistivity_has_B_dependence = true;
        auto& te =
            *sim.m_fields.get(FieldType::hybrid_electron_temperature_fp, 0);
        auto const plasma =
            sim.m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, 0);
        Parameters p;
        p.dr = geom.CellSize(0);
        p.dz = geom.CellSize(1);
        auto run = [&] (std::string const& name) {
            ohm.setVal(-777.);
            ohm.setVal(p.raw, 1, 1, 2);
            mid.setVal(999.);
            mid.setVal(p.midpoint, 2, 1, 2);
            ops.theta = p.theta;
            for (int c = 0; c < 3; ++c) {
                B[c]->setVal(p.B[c]);
                Ji[c]->setVal(p.Ji[c]);
                plasma[c]->setVal(p.Jp[c]);
            }
            hp.m_n_floor = p.floor / PhysConst::q_e;
            hp.m_n_floor_smooth_width = p.floor_width / p.floor;
            hp.m_include_electron_inertia = true;
            hp.m_electron_inertia_mass = p.mass;
            hp.m_electron_inertia_floor_taper = p.taper_width / p.floor;
            hp.m_electron_inertia_bdf2 = p.bdf;
            hp.m_inertia_history_levels = p.history;
            hp.m_include_hall_term = p.hall;
            hp.m_density_pedestal = p.pedestal;
            hp.m_holmstrom_vacuum_region = p.vacuum;
            hp.m_holmstrom_transition_width = p.density_width / p.floor;
            hp.m_holmstrom_axis_radius = p.axis_radius;
            hp.m_holmstrom_axis_rolloff = p.axis_width;
            hp.m_pec_conductor_wall_rows = p.conductor;
            sim.sett_new(0, p.time);
            pc.CurTime(p.time + 7.);
            pc.CurTimeStep(p.dt);
            // CurTime intentionally differs: native Ohm uses gett_new(0).
            for (amrex::MFIter mfi(te); mfi.isValid(); ++mfi) {
                auto a = te.array(mfi);
                auto const par = p;
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        a(i, j, k) = par.te0 + par.te_i * i + par.te_j * j;
                    });
            }
            pc.vacuum_rows = p.recovery;
            hp.m_darwin_vacuum_recovery_components =
                p.flux_only ? "flux" : "all";
            hp.m_darwin_vacuum_recovery_frozen_mask = !p.live_mask;
            hp.m_darwin_vacuum_recovery_density_fraction = 1.;
            hp.m_darwin_vacuum_recovery_mask = p.mask_mode == 2   ? "global"
                                               : p.mask_mode == 1 ? "transition"
                                                                  : "vacuum";
            for (int live = 0; live < 2; ++live) {
                auto* mask = live
                                 ? sim.m_fields.get(FieldType::rho_fp, 0)
                                 : sim.m_fields.get("hybrid_rho_vacmask_fp", 0);
                mask->setVal(777.);
                for (amrex::MFIter mfi(*mask); mfi.isValid(); ++mfi) {
                    auto a = mask->array(mfi);
                    amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(
                                                         int i, int j, int k) {
                        a(i, j, k, 0) = (live ? .3 : .15) * i - .4;
                    });
                }
            }
            pc.Update(state);
            auto const par = p;
            auto const& l = *pc.op.levels[0];
            for (int n = 0; n < Op::NN; ++n) {
                check(*l.nodes, n, geom, name + "/node" + std::to_string(n),
                      [=] AMREX_GPU_HOST_DEVICE(int, int, int) {
                          return expected_node(n, par);
                      });
            }
            for (int c = 0; c < 3; ++c) {
                for (int n = 0; n < Op::NC; ++n) {
                    check(*l.coeff[c], n, geom,
                          name + "/edge" + std::to_string(c) + "/slot" +
                              std::to_string(n),
                          [=] AMREX_GPU_HOST_DEVICE(int i, int j, int) {
                              return expected_edge(i, j, c, n, par);
                          });
                }
            }
        };
        run("base_total_B_component_density_Kelvin_J_time_pedestal");
        p.pedestal = false;
        run("pedestal_disabled");
        p.pedestal = true;
        p.raw = .7;
        p.midpoint = .8;
        p.floor_width = .2;
        p.taper_width = .3;
        run("different_density_smooth_floor_taper");
        p.raw = 4.;
        p.midpoint = 9.;
        p.floor_width = 0.;
        p.taper_width = 0.;
        p.bdf = true;
        for (RT theta : {.5, .7, 1.}) {
            p.theta = theta;
            for (int history : {1, 2}) {
                p.history = history;
                run("theta" + std::to_string(theta) + "_history" +
                    std::to_string(history));
            }
        }
        p.bdf = false;
        p.history = 2;
        run("history_present_bdf_disabled");
        p.hall = false;
        run("Hall_off_keeps_ion_motional_response");
        p.hall = true;
        p.raw = .4;
        p.midpoint = 9.;
        p.vacuum = true;
        p.axis_radius = 7.25 * p.dr;
        p.axis_width = 1.3 * p.dr;
        p.density_width = 0.;
        run("hard_density_gate_ignores_axis_rolloff");
        p.density_width = .25;
        run("smooth_density_and_axis_masks");
        p.axis_radius = 0.;
        run("smooth_global_density_gate");
        p.vacuum = false;
        p.conductor = true;
        p.raw = 0.;
        p.midpoint = 0.;
        run("raw_zero_conductor_and_inertia");
        p.raw = 4.;
        p.midpoint = 9.;
        p.recovery = true;
        for (int mode = 0; mode < 3; ++mode) {
            p.mask_mode = mode;
            for (bool live : {false, true}) {
                p.live_mask = live;
                for (bool flux : {false, true}) {
                    p.flux_only = flux;
                    run("Darwin_replaced_rows_mode" + std::to_string(mode) +
                        "_live" + std::to_string(live) + "_flux" +
                        std::to_string(flux));
                }
            }
        }
        amrex::Print() << "RZ_PC_COEFFICIENT_ADAPTER PASS\n";
        // No field solve/particle push is performed. All state is fixture-local
        // to this process. Parser lifetimes extend through every Update.
        amrex::Gpu::streamSynchronize();
        WarpX::Finalize();
#else
        amrex::Abort("This coefficient fixture requires an RZ build");
#endif
    }
    warpx::initialization::finalize_external_libraries();
}
