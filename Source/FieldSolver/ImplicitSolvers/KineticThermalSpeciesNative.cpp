/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "EmbeddedBoundary/Enabled.H"
#include "KineticThermalSpeciesDeposition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/SubcycledParticleContainer.H"
#include "WarpX.H"
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>

namespace warpx::thermal {
namespace {
void
CheckNative (KineticThermalSpecies const& context,
             MultiParticleContainer& particles) {
    auto const& w = WarpX::GetInstance();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !EB::enabled() && w.finestLevel() == 0 &&
            WarpX::n_rz_azimuthal_modes == 1 && WarpX::ncomps == 1 &&
            !WarpX::do_shared_mem_current_deposition &&
            WarpX::grid_type == GridType::Staggered,
        "KineticThermalSpecies native bridge requires single-level "
        "m0/no-EB/non-shared Yee deposition");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::do_single_precision_comms ==
                context.Options().single_precision_comms &&
            context.Ghosts().allGE(w.get_ng_depos_J()) &&
            context.Ghosts().allGE(w.get_ng_depos_rho()),
        "KineticThermalSpecies native communication/guard mismatch");
    auto const& g = w.Geom(0);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        g.Domain() == context.Geometry().Domain(),
        "KineticThermalSpecies native geometry mismatch");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            g.ProbLo(d) == context.Geometry().ProbLo(d) &&
                g.ProbHi(d) == context.Geometry().ProbHi(d) &&
                g.isPeriodic(d) == context.Geometry().isPeriodic(d),
            "KineticThermalSpecies native geometry mismatch");
    }
    auto species = DescribeNativeThermalSpecies(particles);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(species.size() ==
                                         context.Descriptors().size(),
                                     "KineticThermalSpecies native context "
                                     "must include every charged species");
    for (std::size_t s = 0; s < species.size(); ++s) {
        auto const& expected = context.Descriptors()[s];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            species[s].name == expected.name &&
                species[s].charge_number == expected.charge_number &&
                species[s].mass == expected.mass,
            "KineticThermalSpecies native charged-species order/metadata "
            "mismatch");
        auto& pc = particles.GetParticleContainerFromName(expected.name);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            dynamic_cast<PhysicalParticleContainer*>(&pc) &&
                !dynamic_cast<SubcycledParticleContainer*>(&pc) &&
                !pc.DoFieldIonization(),
            "KineticThermalSpecies native subcycled/variable-charge species "
            "unsupported");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc.ParticleBoxArray(0) == context.Cells() &&
                pc.ParticleDistributionMap(0) == context.Distribution(),
            "KineticThermalSpecies native particle layout mismatch");
    }
}
void
CheckImplicitCurrent (WarpXParticleContainer& pc) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Direct,
        "KineticThermalSpecies native current requires qualified direct or "
        "Esirkepov trajectory");
    auto names = pc.GetRealSoANames();
    for (auto const* n : {"ux_n", "uy_n", "uz_n"}) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::find(names.begin(), names.end(), n) != names.end(),
            "KineticThermalSpecies native current requires saved old momenta");
    }
#ifdef WARPX_DIM_RZ
    for (auto const* n : {"x_n", "y_n", "z_n"}) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::find(names.begin(), names.end(), n) != names.end(),
            "KineticThermalSpecies native current requires saved old "
            "positions");
    }
#endif
    // A final midpoint cannot reconstruct the actual piecewise suborbit
    // current.
    if (pc.HasiAttrib("nsuborbits")) {
        int invalid = 0;
        for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
            auto p = pti.GetiAttribs("nsuborbits").dataPtr();
            amrex::ReduceOps<amrex::ReduceOpMax> op;
            amrex::ReduceData<int> data(op);
            using Tuple = decltype(data)::Type;
            op.eval(
                pti.numParticles(), data,
                [=] AMREX_GPU_DEVICE(int i) -> Tuple { return {p[i] > 1}; });
            invalid = std::max(invalid, amrex::get<0>(data.value()));
        }
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !invalid, "KineticThermalSpecies overlay current requires supplied "
                      "suborbit trajectories");
    }
}
} // namespace
std::vector<KineticSpeciesDescriptor>
DescribeNativeThermalSpecies (const MultiParticleContainer& pc) {
    std::vector<KineticSpeciesDescriptor> out;
    for (auto const& name : pc.GetSpeciesNames()) {
        auto const& s = pc.GetParticleContainerFromName(name);
        if (s.getCharge() == 0) {
            continue;
        }
        KineticSpeciesDescriptor d;
        d.name = name;
        d.charge_number = s.getCharge() / PhysConst::q_e;
        d.mass = s.getMass();
        out.push_back(d);
    }
    return out;
}
bool
FreezeNativeOldSpeciesTemperature (KineticThermalSpecies& context,
                                   MultiParticleContainer& pc) {
    CheckNative(context, pc);
    auto& w = WarpX::GetInstance();
    return context.FreezeOldTemperature(
        [&] (std::size_t s, SpeciesVariancePass pass,
             const SpeciesVarianceFields& f) {
            auto& particles =
                pc.GetParticleContainerFromName(context.Descriptors()[s].name);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                particles.getTemperatureDepositionFlag(),
                "KineticThermalSpecies relaxation requires native temperature "
                "deposition enabled");
            if (particles.do_not_deposit) {
                return true;
            }
            for (WarpXParIter pti(particles, 0); pti.isValid(); ++pti) {
                auto const& weight = pti.GetAttribs(PIdx::w);
                auto const& ux = pti.GetAttribs(PIdx::ux);
                auto const& uy = pti.GetAttribs(PIdx::uy);
                auto const& uz = pti.GetAttribs(PIdx::uz);
                auto position = GetParticlePosition<PIdx>(pti);
                auto tile = pti.tilebox();
                tile.grow(w.get_ng_depos_J());
                auto const inverse = WarpX::InvCellSize(0),
                           origin = WarpX::LowerCorner(tile, 0, 0.);
                auto lower = amrex::lbound(tile);
                auto deposit = [&]<int Order>() {
                    DepositSpeciesTemperatureShapeN<Order>(
                        f, pti, position, weight.dataPtr(), ux.dataPtr(),
                        uy.dataPtr(), uz.dataPtr(), pti.numParticles(), inverse,
                        origin, lower, pass);
                };
                switch (WarpX::nox) {
                case 1:
                    deposit.template operator()<1>();
                    break;
                case 2:
                    deposit.template operator()<2>();
                    break;
                case 3:
                    deposit.template operator()<3>();
                    break;
                case 4:
                    deposit.template operator()<4>();
                    break;
                default:
                    amrex::Abort(
                        "KineticThermalSpecies unsupported native shape order");
                }
            }
            return true;
        },
        [&] (std::size_t, const SpeciesMutableVector& fields) {
            if (WarpX::use_filter) {
                w.ApplyFilterMF({fields}, 0);
            }
        });
}
bool
RefreshNativeThermalSpecies (KineticThermalSpecies& context,
                             MultiParticleContainer& pc, amrex::Real dt,
                             MaterializedSpeciesTrial state) {
    CheckNative(context, pc);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(dt) && dt > 0,
        "KineticThermalSpecies needs actual positive implicit deposition dt");
    return context.RefreshMaterializedTrial(
        state, [&] (std::size_t s, amrex::MultiFab& rho,
                    const SpeciesMutableVector& current) {
            auto& particles =
                pc.GetParticleContainerFromName(context.Descriptors()[s].name);
            // Current particle positions are the materialized trial midpoint.
            // This raw density intentionally differs from endpoint-averaged
            // physical rho.
            ablastr::fields::MultiLevelScalarField charge{&rho};
            particles.DepositCharge(charge, true, true, false, false);
            if (!current[0]) {
                return true;
            }
            CheckImplicitCurrent(particles);
            // The public vector overload hardcodes Explicit; use the real
            // implicit trajectory deposition entry point, with immutable saved
            // xn/un.
            for (WarpXParIter pti(particles, 0); pti.isValid(); ++pti) {
                particles.DepositCurrent(
                    pti, pti.GetAttribs(PIdx::w), pti.GetAttribs(PIdx::ux),
                    pti.GetAttribs(PIdx::uy), pti.GetAttribs(PIdx::uz), nullptr,
                    current[0], current[1], current[2], 0, pti.numParticles(),
                    0, 0, 0, dt, 0., PushType::Implicit);
            }
            return true;
        });
}
} // namespace warpx::thermal
