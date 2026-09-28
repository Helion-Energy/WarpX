/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "NativeInstantaneousForce.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "WarpX.H"

using amrex::MultiFab;
using ablastr::fields::VectorField;
using warpx::fields::FieldType;
namespace {
void CheckLayout(const MultiFab& source, const MultiFab& output)
{
    AMREX_ALWAYS_ASSERT(source.boxArray() == output.boxArray() &&
        source.DistributionMap() == output.DistributionMap() &&
        source.nComp() == output.nComp() && source.nGrowVect() == output.nGrowVect());
}
}
namespace warpx::darwin {
std::unique_ptr<thermal::EulerianDissipation> MakeNativeInstantaneousDissipation(WarpX& sim)
{
    using namespace thermal;
    auto const& model = *sim.get_pointer_HybridPICModel();
    if (!model.m_include_electron_viscosity && !model.m_include_hyper_resistivity_term) {
        return nullptr;
    }
    TrialDissipationOptions options;
    options.viscosity = model.m_include_electron_viscosity;
    options.viscous = model.ViscosityPointParams(0);
    options.eta_h = model.m_eta_h;
    options.eta_h_depends_on_B = model.m_hyper_resistivity_has_B_dependence;
    options.transformed_electric_solve = model.m_esolve_tensor || model.m_esolve_curlcurl;
    if (model.m_include_hyper_resistivity_term) {
        options.hyper = model.m_hyper_resistivity_curlcurl ? HyperMode::AmpereCurlCurl :
            (model.m_hyper_res_curl_curl ? HyperMode::InteriorCurlCurl : HyperMode::Laplacian);
    }
    auto const geometry = model.ElectronThermalGeometry();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int side = 0; side < 2; ++side) {
            auto const boundary = side ? WarpX::field_boundary_hi[d] : WarpX::field_boundary_lo[d];
            auto& selected = options.boundary[d][side];
            if (geometry.isPeriodic(d)) { selected = DissipationBoundary::Periodic; }
#if defined(WARPX_DIM_RZ)
            else if (d == 0 && side == 0) { selected = DissipationBoundary::Axis; }
#endif
            else if (boundary == FieldBoundaryType::PEC) { selected = DissipationBoundary::PEC; }
            else if (boundary == FieldBoundaryType::PMC) { selected = DissipationBoundary::PMC; }
            else { amrex::Abort("Native force dissipation needs axis/PEC/PMC/periodic boundaries"); }
        }
    }
    return std::make_unique<EulerianDissipation>(
        geometry, sim.boxArray(0), sim.DistributionMap(0), options);
}

bool CaptureNativeInstantaneousForce(WarpX& sim, const MultiFab& charge,
    const VectorField& full, const VectorField& nores,
    thermal::EulerianDissipation* dissipation)
{
    auto const& model = *sim.get_pointer_HybridPICModel();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sim.maxLevel() == 0 && !model.m_esolve_tensor &&
        !model.m_esolve_curlcurl && !model.m_pec_conductor_wall_rows &&
        !model.m_has_per_species_eta,
        "Native force capture requires one algebraic level without conductor/species overlays");
    auto const electric = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    auto const current = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
    auto const magnetic = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    std::array<MultiFab, 3> saved;
    for (int c = 0; c < 3; ++c) {
        CheckLayout(*electric[c], *full[c]); CheckLayout(*electric[c], *nores[c]);
        for (int d = 0; d < 3; ++d) {
            AMREX_ALWAYS_ASSERT(full[c] != electric[d] && nores[c] != electric[d] &&
                full[c] != nores[d]);
        }
        saved[c].define(electric[c]->boxArray(), electric[c]->DistributionMap(),
                        electric[c]->nComp(), electric[c]->nGrowVect());
        MultiFab::Copy(saved[c], *electric[c], 0, 0, electric[c]->nComp(), electric[c]->nGrowVect());
    }
    // Both wrappers use precisely the same registered output, as ComputeRHS does.
    model.HybridPICSolveE(electric, current, magnetic, charge, sim.GetEBUpdateEFlag()[0],
                         0, false, true, dissipation);
    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(*full[c], *electric[c], 0, 0, electric[c]->nComp(), electric[c]->nGrowVect());
    }
    model.HybridPICSolveE(electric, current, magnetic, charge, sim.GetEBUpdateEFlag()[0],
                         0, false, false, dissipation);
    bool valid = true;
    for (int c = 0; c < 3; ++c) {
        MultiFab::Copy(*nores[c], *electric[c], 0, 0, electric[c]->nComp(), electric[c]->nGrowVect());
        MultiFab::Copy(*electric[c], saved[c], 0, 0, electric[c]->nComp(), electric[c]->nGrowVect());
        valid = full[c]->is_finite(0, full[c]->nComp(), full[c]->nGrow()) && valid;
        valid = nores[c]->is_finite(0, nores[c]->nComp(), nores[c]->nGrow()) && valid;
    }
    return valid;
}

void ApplyNativeInstantaneousForce(const VectorField& electric, const VectorField& full,
    const VectorField& nores, bool enabled)
{
    if (!enabled) { return; }
    for (int c = 0; c < 3; ++c) {
        CheckLayout(*electric[c], *full[c]); CheckLayout(*electric[c], *nores[c]);
        AMREX_ALWAYS_ASSERT(electric[c] != full[c] && electric[c] != nores[c]);
        MultiFab::Subtract(*electric[c], *full[c], 0, 0,
                           electric[c]->nComp(), electric[c]->nGrowVect());
        MultiFab::Add(*electric[c], *nores[c], 0, 0,
                      electric[c]->nComp(), electric[c]->nGrowVect());
    }
}
}
