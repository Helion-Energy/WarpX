/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/ImplicitSolvers/DarwinThermalAdvance.H"
#include "FieldSolver/ImplicitSolvers/ImplicitSolver.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

using Real = amrex::Real;
using MF = amrex::MultiFab;
using Vec = WarpXSolverVec;
using ablastr::fields::Direction;
using warpx::fields::FieldType;
using warpx::thermal::ThermalNonlinearOperator;
using warpx::thermal::ThermalSolveOptions;
using warpx::thermal::ThermalSolveResult;
namespace {
constexpr const char* energy_name = "hybrid_electron_energy_fp";
long particle_calls = 0, stage_calls = 0;
struct ProbeComplete {};
MF
Clone (const MF& x, int component = 0) {
    MF y(x.boxArray(), x.DistributionMap(), 1, 0);
    MF::Copy(y, x, component, 0, 1, 0);
    return y;
}
struct Moments {
    std::vector<MF> fields;
    static Moments
    Capture (WarpX& sim) {
        Moments s;
        for (int c = 0; c < 3; ++c)
            s.fields.push_back(Clone(
                *sim.m_fields.get(FieldType::current_fp, Direction{c}, 0)));
        auto const& rho = *sim.m_fields.get(FieldType::rho_fp, 0);
        s.fields.push_back(Clone(rho, rho.nComp() / 2));
        auto const& model = *sim.get_pointer_HybridPICModel();
        s.fields.push_back(Clone(model.ElectronTemperatureForSolve(0)));
        s.fields.push_back(Clone(model.ElectronPressureForSolve(0)));
        return s;
    }
    static Moments
    Difference (const Moments& a, const Moments& b, Real factor = 1.) {
        Moments s;
        for (std::size_t i = 0; i < a.fields.size(); ++i) {
            s.fields.push_back(Clone(a.fields[i]));
            MF::Subtract(s.fields.back(), b.fields[i], 0, 0, 1, 0);
            s.fields.back().mult(factor);
        }
        return s;
    }
    std::vector<Real>
    Norms () const {
        std::vector<Real> n;
        for (auto const& f : fields)
            n.push_back(f.norminf());
        return n;
    }
};
struct Evaluation {
    Vec residual;
    Moments moments;
    long pushes = 0, stages = 0;
};
Evaluation
Evaluate (ThermalNonlinearOperator& op, const Vec& state, bool probe) {
    Evaluation result;
    result.residual.Define(state);
    auto const p = particle_calls, m = stage_calls;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        op.Residual(result.residual, state, 1, probe),
        "Actual runtime rejected the directional trial");
    result.pushes = particle_calls - p;
    result.stages = stage_calls - m;
    result.moments = Moments::Capture(*state.getWarpX());
    return result;
}
Vec
Difference (const Vec& a, const Vec& b, Real factor = 1.) {
    Vec x;
    x.Define(a);
    x.linComb(factor, a, -factor, b);
    return x;
}
void
Freeze (ThermalNonlinearOperator& op, const Vec& state) {
    ThermalSolveResult stats;
    AMREX_ALWAYS_ASSERT(op.Freeze(state, 0, false, stats));
}
void
Values (std::ostream& out, const std::vector<Real>& x) {
    out << '[';
    for (std::size_t i = 0; i < x.size(); ++i) {
        AMREX_ALWAYS_ASSERT(std::isfinite(x[i]));
        if (i)
            out << ',';
        out << x[i];
    }
    out << ']';
}
void
DirectionFields (Vec& de, Vec& du, const Vec& anchor) {
    de.Define(anchor);
    du.Define(anchor);
    de.zero();
    du.zero();
    auto& sim = *anchor.getWarpX();
    auto const dx = sim.Geom(0).CellSizeArray(), lo = sim.Geom(0).ProbLoArray();
    Real const R = sim.Geom(0).ProbLength(0), L = sim.Geom(0).ProbLength(1),
               pi = std::acos(-1.);
    bool curlfree = false;
    Real amplitude = 1000.;
    amrex::ParmParse pp("runtime_probe");
    pp.query("curlfree", curlfree);
    pp.query("electric_amplitude", amplitude);
    for (int c = 0; c < 3; ++c) {
        auto& f = *de.getArrayVec()[0][c];
        auto const node = f.ixType().toIntVect();
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto const a = f.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(
                                                   int i, int j, int k) {
                Real const r = (i + .5 * (1 - node[0])) * dx[0] / R,
                           z = (lo[1] + (j + .5 * (1 - node[1])) * dx[1]) / L;
                if (curlfree)
                    a(i, j, k) = c == 0 ? amplitude * std::sin(pi * r) : 0.;
                else
                    a(i, j, k) =
                        amplitude *
                        (c == 0 ? std::sin(pi * r) * std::cos(2 * pi * z)
                                : (c == 1 ? .7 * r * (1 - r) * (1 - r) *
                                                std::sin(2 * pi * z)
                                          : .3 * std::cos(.5 * pi * r) *
                                                std::sin(2 * pi * z)));
            });
        }
    }
    auto& u = du.getMultiFabBlock(energy_name, 0);
    auto const& base = anchor.getMultiFabBlock(energy_name, 0);
    for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
        auto const a = u.array(mfi);
        auto const b = base.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Real const r = (i + .5) * dx[0] / R,
                           z = (lo[1] + (j + .5) * dx[1]) / L;
                a(i, j, k) = .2 * b(i, j, k) *
                             (1 + .3 * std::cos(pi * r) * std::cos(2 * pi * z));
            });
    }
}
void
Run (ThermalNonlinearOperator& op, Vec& solution) {
    AMREX_ALWAYS_ASSERT(
        dynamic_cast<warpx::thermal::DarwinThermalAdvance*>(&op));
    AMREX_ALWAYS_ASSERT(solution.blockNames().size() == 2 &&
                        solution.hasMultiFabBlock(energy_name));
    Vec anchor;
    anchor.Copy(solution);
    Vec de, du, dm;
    DirectionFields(de, du, anchor);
    dm.Define(anchor);
    dm.linComb(1., de, 1., du);
    Real base_offset = 0.;
    amrex::ParmParse("runtime_probe")
        .query("base_electric_offset", base_offset);
    if (base_offset)
        anchor.increment(de, base_offset);
    auto accepted_energy = Clone(*anchor.getWarpX()->m_fields.get(
        FieldType::hybrid_electron_energy_fp, 0));
    auto base = Evaluate(op, anchor, false);
    Freeze(op, anchor);
    auto zero = Evaluate(op, anchor, true);
    std::ofstream file;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        file.open("PROBES.jsonl");
        file << std::setprecision(17) << std::scientific;
    }
    file << "{\"kind\":\"anchor\",\"block_names\":[\"E\",\"U\"],\"moment_"
            "names\":[\"Jr\",\"Jtheta\",\"Jz\",\"rho\",\"Te_K\",\"Pe_Pa\"],"
            "\"residual_norms\":";
    Values(file, base.residual.blockNorms(false));
    file << ",\"mm_zero_difference\":";
    Values(file, Difference(zero.residual, base.residual).blockNorms(false));
    file << ",\"moment_mm_zero_difference\":";
    Values(file, Moments::Difference(zero.moments, base.moments).Norms());
    file << ",\"full_pushes\":" << base.pushes
         << ",\"full_stage_calls\":" << base.stages
         << ",\"probe_pushes\":" << zero.pushes
         << ",\"probe_stage_calls\":" << zero.stages << "}\n";
    std::vector<Real> eps{.1,      .05,   .025,  .0125, .00625,
                          .003125, 1.e-3, 1.e-4, 1.e-5};
    std::array<Vec, 3> last_full, last_mm;
    std::array<const Vec*, 3> directions{&de, &du, &dm};
    const char* names[3] = {"E", "U", "mixed"};
    for (int d = 0; d < 3; ++d) {
        Vec previous_full, previous_mm;
        for (Real h : eps) {
            Vec plus, minus;
            plus.Copy(anchor);
            minus.Copy(anchor);
            plus.increment(*directions[d], h);
            minus.increment(*directions[d], -h);
            auto fp = Evaluate(op, plus, false),
                 fm = Evaluate(op, minus, false);
            auto df = Difference(fp.residual, fm.residual, 1 / (2 * h));
            auto mf = Moments::Difference(fp.moments, fm.moments, 1 / (2 * h));
            auto fresh = Evaluate(op, anchor, false);
            Freeze(op, anchor);
            auto mp = Evaluate(op, plus, true), mm = Evaluate(op, minus, true);
            auto da = Difference(mp.residual, mm.residual, 1 / (2 * h));
            auto ma = Moments::Difference(mp.moments, mm.moments, 1 / (2 * h));
            auto back_mm = Evaluate(op, anchor, true),
                 back_full = Evaluate(op, anchor, false);
            Vec even;
            even.Define(anchor);
            even.linComb(1., fp.residual, 1., fm.residual);
            even.increment(base.residual, -2.);
            file << "{\"kind\":\"direction\",\"direction\":\"" << names[d]
                 << "\",\"epsilon\":" << h;
            file << ",\"full_derivative\":";
            Values(file, df.blockNorms(false));
            file << ",\"mm_derivative\":";
            Values(file, da.blockNorms(false));
            file << ",\"derivative_difference\":";
            Values(file, Difference(da, df).blockNorms(false));
            file << ",\"full_moments\":";
            Values(file, mf.Norms());
            file << ",\"mm_moments\":";
            Values(file, ma.Norms());
            file << ",\"moment_derivative_difference\":";
            Values(file, Moments::Difference(ma, mf).Norms());
            file << ",\"full_even_remainder\":";
            Values(file, even.blockNorms(false));
            file << ",\"full_ABA\":";
            Values(file, Difference(back_full.residual, base.residual)
                             .blockNorms(false));
            file << ",\"mm_ABA\":";
            Values(
                file,
                Difference(back_mm.residual, fresh.residual).blockNorms(false));
            if (previous_full.IsDefined()) {
                file << ",\"full_refinement_change\":";
                Values(file, Difference(df, previous_full).blockNorms(false));
                file << ",\"mm_refinement_change\":";
                Values(file, Difference(da, previous_mm).blockNorms(false));
            }
            file << ",\"full_pushes\":" << fp.pushes + fm.pushes
                 << ",\"full_stage_calls\":" << fp.stages + fm.stages
                 << ",\"probe_pushes\":" << mp.pushes + mm.pushes
                 << ",\"probe_stage_calls\":" << mp.stages + mm.stages << "}\n";
            file.flush();
            previous_full.Copy(df);
            previous_mm.Copy(da);
            last_full[d].Copy(df);
            last_mm[d].Copy(da);
            amrex::Print() << "PROBE " << names[d] << " h=" << h
                           << " full=" << df.norm2()
                           << " MMgap=" << Difference(da, df).norm2() << "\n";
        }
    }
    auto lf = Difference(last_full[2], last_full[0]);
    lf.increment(last_full[1], -1.);
    auto lm = Difference(last_mm[2], last_mm[0]);
    lm.increment(last_mm[1], -1.);
    file << "{\"kind\":\"linearity\",\"full\":";
    Values(file, lf.blockNorms(false));
    file << ",\"mm\":";
    Values(file, lm.blockNorms(false));
    file << "}\n";
    auto const& accepted = *anchor.getWarpX()->m_fields.get(
        FieldType::hybrid_electron_energy_fp, 0);
    auto unchanged = Clone(accepted);
    MF::Subtract(unchanged, accepted_energy, 0, 0, 1, 0);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(unchanged.norminf() == 0.,
                                     "Runtime probes modified accepted U");
    op.RestoreInput(solution);
    file.close();
    amrex::Print() << "PASS RUNTIME_FOUR_BLOCK directional probes completed "
                      "before acceptance\n";
}
} // namespace
// GNU/Itanium link wrappers instrument the real native entry points. The two
// member wrappers make the existing implicit this argument explicit; all calls
// are forwarded unchanged. No class layout, header or production TU is changed.
extern "C" void
RealParticlePush (WarpX*, Real, bool, PositionPushType, MomentumPushType,
                  ImplicitOptions const*) asm("__real__"
                                              "ZN5WarpX23PushParticlesandDeposi"
                                              "tEdb16PositionPushType16Momentum"
                                              "PushTypePK15ImplicitOptions");
extern "C" void WrappedParticlePush (
    WarpX*, Real, bool, PositionPushType, MomentumPushType,
    ImplicitOptions const*) asm("__wrap__"
                                "ZN5WarpX23PushParticlesandDepositEdb16Position"
                                "PushType16MomentumPushTypePK15ImplicitOption"
                                "s");
extern "C" void
WrappedParticlePush (WarpX* self, Real t, bool skip, PositionPushType x,
                     MomentumPushType v, ImplicitOptions const* options) {
    ++particle_calls;
    RealParticlePush(self, t, skip, x, v, options);
}
extern "C" void RealStage (ImplicitSolver*, Real, int,
                           bool) asm("__real__ZN14ImplicitSolver8PreRHSOpEdib");
extern "C" void
WrappedStage (ImplicitSolver*, Real, int,
              bool) asm("__wrap__ZN14ImplicitSolver8PreRHSOpEdib");
extern "C" void
WrappedStage (ImplicitSolver* self, Real time, int iteration, bool probe) {
    ++stage_calls;
    RealStage(self, time, iteration, probe);
}
extern "C" ThermalSolveResult
WrappedSolve (ThermalNonlinearOperator&, Vec&, ThermalSolveOptions const&) asm(
    "__wrap__ZN5warpx7thermal18SolveThermalSystemERNS0_"
    "24ThermalNonlinearOperatorER14WarpXSolverVecRKNS0_19ThermalSolveOptionsE");
extern "C" ThermalSolveResult
WrappedSolve (ThermalNonlinearOperator& op, Vec& state,
              ThermalSolveOptions const&) {
    Run(op, state);
    throw ProbeComplete{};
}
int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    bool complete = false;
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        try {
            sim.Evolve(1);
        } catch (ProbeComplete const&) {
            complete = true;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            complete, "Did not intercept actual coupled runtime solver");
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
