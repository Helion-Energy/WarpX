/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ThetaImplicitMHD.H"
#include "ThetaImplicitMHD_K.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

/* The global energy audit of the theta-implicit MHD solver
 * (implicit_mhd.energy_audit_file): every quantity here is a READ of the
 * state or of scratch registers recomputed at the accepted theta state;
 * the state itself is never written. See the member documentation in
 * ThetaImplicitMHD.H for the column definitions and the closure. */

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
    /* Dual volume of a field component at its own staggering: the
     * FieldEnergy reduced-diagnostic convention (2 pi r; pi dr/4 on the
     * axis; half weights at the nodal ends of a box, which sums shared
     * nodes once and gives the domain's nodal boundary its half cell).
     * Returned in m^3 for RZ and in m (per unit area) for 1D. */
    struct FieldDualVolume
    {
        int nodal_r = 0;
        int nodal_z = 0;
        int lo_r = 0;
        int hi_r = 0;
        int lo_z = 0;
        int hi_z = 0;
        amrex::Real rmin = 0.0;
        amrex::Real dr = 0.0;
        amrex::Real dz = 0.0;

        FieldDualVolume (const amrex::MultiFab& mf, const amrex::Box& validbox,
                         const amrex::Geometry& geom)
        {
            const amrex::IntVect nodal = mf.ixType().toIntVect();
#if defined(WARPX_DIM_RZ)
            nodal_r = nodal[0];
            nodal_z = nodal[1];
            lo_r = validbox.smallEnd(0);
            hi_r = validbox.bigEnd(0);
            lo_z = validbox.smallEnd(1);
            hi_z = validbox.bigEnd(1);
            dr = geom.CellSize(0);
            dz = geom.CellSize(1);
            rmin = geom.ProbLo(0) + lo_r * dr + (nodal_r ? 0.0_rt : 0.5_rt * dr);
#else
            nodal_z = nodal[0];
            lo_z = validbox.smallEnd(0);
            hi_z = validbox.bigEnd(0);
            dz = geom.CellSize(0);
#endif
        }

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        operator() (const int i, const int j) const noexcept
        {
#if defined(WARPX_DIM_RZ)
            const amrex::Real r = rmin + (i - lo_r) * dr;
            amrex::Real v = 2.0_rt * MathConst::pi * r;
            if (r == 0.0_rt) {
                v = MathConst::pi * dr / 4.0_rt;
            } else if (nodal_r && i == lo_r) {
                v *= 0.5_rt;
            }
            if (nodal_r && i == hi_r && r != 0.0_rt) { v *= 0.5_rt; }
            if (nodal_z && j == lo_z) { v *= 0.5_rt; }
            if (nodal_z && j == hi_z) { v *= 0.5_rt; }
            return v * dr * dz;
#else
            amrex::ignore_unused(j);
            amrex::Real v = dz;
            if (nodal_z && i == lo_z) { v *= 0.5_rt; }
            if (nodal_z && i == hi_z) { v *= 0.5_rt; }
            return v;
#endif
        }
    };

    /* Cell volume of the cell-centered fluid grid. */
    struct CellVolume
    {
        amrex::Real rlo = 0.0;
        amrex::Real dr = 0.0;
        amrex::Real dz = 0.0;
        explicit CellVolume (const amrex::Geometry& geom)
        {
#if defined(WARPX_DIM_RZ)
            rlo = geom.ProbLo(0);
            dr = geom.CellSize(0);
            dz = geom.CellSize(1);
#else
            dz = geom.CellSize(0);
#endif
        }
        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
        operator() (const int i) const noexcept
        {
#if defined(WARPX_DIM_RZ)
            return 2.0_rt * MathConst::pi * (rlo + (i + 0.5_rt) * dr) * dr * dz;
#else
            amrex::ignore_unused(i);
            return dz;
#endif
        }
    };

    amrex::Real all_reduce_sum (amrex::Real value)
    {
        amrex::ParallelAllReduce::Sum(value,
                                      amrex::ParallelContext::CommunicatorSub());
        return value;
    }

    /* sum_cells f(i,j,k,comp) dV over the valid cells of a cell-centered
     * MultiFab. */
    amrex::Real cell_sum (const amrex::MultiFab& mf, const int comp,
                          const amrex::Geometry& geom)
    {
        const CellVolume volume(geom);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const auto arr = mf.const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                               return {arr(i, j, k, comp) * volume(i)};
                           });
        }
        return all_reduce_sum(amrex::get<0>(reduce_data.value(reduce_op)));
    }

    /* Kinetic energy sum_cells |m|^2 / (2 max(rho, floor)) dV. */
    amrex::Real kinetic_sum (const amrex::MultiFab& density,
                             const amrex::MultiFab& momentum,
                             const amrex::Real density_floor,
                             const amrex::Geometry& geom)
    {
        const CellVolume volume(geom);
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
            const auto rho = density.const_array(mfi);
            const auto mom = momentum.const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                               amrex::Real m2 = 0.0_rt;
                               for (int c = 0; c < 3; ++c) {
                                   m2 += mom(i, j, k, c) * mom(i, j, k, c);
                               }
                               return {0.5_rt * m2 /
                                       std::max(rho(i, j, k), density_floor) *
                                       volume(i)};
                           });
        }
        return all_reduce_sum(amrex::get<0>(reduce_data.value(reduce_op)));
    }

    /* sum_cells (a - b - c) dV (the Newton defect of one block) and
     * sum_cells c dV (the rhs block sum), both per unit theta later. */
    std::pair<amrex::Real, amrex::Real>
    defect_and_rhs_sum (const amrex::MultiFab& state, const amrex::MultiFab& old_state,
                        const amrex::MultiFab& rhs, const amrex::Geometry& geom)
    {
        const CellVolume volume(geom);
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
            const auto u = state.const_array(mfi);
            const auto u_old = old_state.const_array(mfi);
            const auto r = rhs.const_array(mfi);
            reduce_op.eval(mfi.validbox(), reduce_data,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                               const amrex::Real dv = volume(i);
                               return {(u(i, j, k) - u_old(i, j, k) - r(i, j, k)) * dv,
                                       r(i, j, k) * dv};
                           });
        }
        auto sums = reduce_data.value(reduce_op);
        return {all_reduce_sum(amrex::get<0>(sums)), all_reduce_sum(amrex::get<1>(sums))};
    }
} // namespace

ThetaImplicitMHD::EnergyAuditTotals
ThetaImplicitMHD::EnergyAuditFluidTotals (const WarpXSolverVec& state) const
{
    EnergyAuditTotals totals;
    const amrex::Geometry& geom = m_WarpX->Geom(0);
    const amrex::MultiFab& density = state.getMultiFabBlock(MassDensityName, 0);
    totals.mass = cell_sum(density, 0, geom);
    totals.ue = cell_sum(state.getMultiFabBlock(ElectronEnergyName, 0), 0, geom);
    totals.ei = cell_sum(state.getMultiFabBlock(IonEnergyName, 0), 0, geom);
    if (m_ion_closure == "dual_energy") {
        totals.ui = cell_sum(state.getMultiFabBlock(IonInternalEnergyName, 0), 0, geom);
    }
    totals.ke = kinetic_sum(density, state.getMultiFabBlock(MomentumDensityName, 0),
                            m_mass_density_floor, geom);
    return totals;
}

amrex::Real ThetaImplicitMHD::EnergyAuditDualVolumeTotal (const amrex::MultiFab& mf) const
{
    const amrex::Geometry& geom = m_WarpX->Geom(0);
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box box = mfi.validbox();
        const FieldDualVolume weight(mf, box, geom);
        reduce_op.eval(box, reduce_data,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                           amrex::ignore_unused(k);
                           return {weight(i, j)};
                       });
    }
    return all_reduce_sum(amrex::get<0>(reduce_data.value(reduce_op)));
}

amrex::Real ThetaImplicitMHD::EnergyAuditFieldDot (
    const std::array<const amrex::MultiFab*, 3>& a,
    const std::array<const amrex::MultiFab*, 3>& b,
    const std::array<const amrex::MultiFab*, 3>* a_add) const
{
    const amrex::Geometry& geom = m_WarpX->Geom(0);
    amrex::Real total = 0.0_rt;
    for (int component = 0; component < 3; ++component) {
        const amrex::MultiFab& mf_a = *a[component];
        const amrex::MultiFab& mf_b = *b[component];
        const amrex::MultiFab* const mf_add = a_add ? (*a_add)[component] : nullptr;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            mf_a.ixType() == mf_b.ixType() &&
                (mf_add == nullptr || mf_add->ixType() == mf_a.ixType()),
            "ThetaImplicitMHD energy audit: field dot product between "
            "different staggerings");
        amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(mf_a); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.validbox();
            const FieldDualVolume weight(mf_a, box, geom);
            const auto arr_a = mf_a.const_array(mfi);
            const auto arr_b = mf_b.const_array(mfi);
            const auto arr_add = mf_add ? mf_add->const_array(mfi)
                                        : amrex::Array4<amrex::Real const>{};
            const bool has_add = mf_add != nullptr;
            reduce_op.eval(box, reduce_data,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                               amrex::Real va = arr_a(i, j, k);
                               if (has_add) { va += arr_add(i, j, k); }
                               return {va * arr_b(i, j, k) * weight(i, j)};
                           });
        }
        total += amrex::get<0>(reduce_data.value(reduce_op));
    }
    return all_reduce_sum(total);
}

void ThetaImplicitMHD::EnergyAuditBeginStep (const int step)
{
    m_energy_audit_step =
        !m_energy_audit_file.empty() && m_energy_audit_interval > 0 &&
        ((step + 1) % m_energy_audit_interval == 0);
    if (!m_energy_audit_step) { return; }
    using ablastr::fields::Direction;
    // Bfield_fp holds the TOTAL field at t^n here (the externals were added
    // back at the end of the previous step, or by the initialization).
    std::array<const amrex::MultiFab*, 3> b_total{};
    for (int component = 0; component < 3; ++component) {
        const amrex::MultiFab& field =
            *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{component}, 0);
        if (m_energy_audit_b_total_old[component] == nullptr) {
            m_energy_audit_b_total_old[component] = std::make_unique<amrex::MultiFab>(
                field.boxArray(), field.DistributionMap(), field.nComp(),
                field.nGrowVect());
        }
        amrex::MultiFab::Copy(*m_energy_audit_b_total_old[component], field, 0, 0,
                              field.nComp(), field.nGrowVect());
        b_total[component] = m_energy_audit_b_total_old[component].get();
    }
    m_energy_audit.wb_total_old =
        EnergyAuditFieldDot(b_total, b_total) / (2.0_rt * PhysConst::mu0);
    if (m_hybrid_pic_model->m_add_external_fields) {
        // B_ext(t^n): the stored externals before this step's theta refresh.
        std::array<const amrex::MultiFab*, 3> b_ext{};
        for (int component = 0; component < 3; ++component) {
            const amrex::MultiFab& field = *m_WarpX->m_fields.get(
                FieldType::hybrid_B_fp_external, Direction{component}, 0);
            if (m_energy_audit_b_ext_old[component] == nullptr) {
                m_energy_audit_b_ext_old[component] = std::make_unique<amrex::MultiFab>(
                    field.boxArray(), field.DistributionMap(), field.nComp(),
                    field.nGrowVect());
            }
            amrex::MultiFab::Copy(*m_energy_audit_b_ext_old[component], field, 0, 0,
                                  field.nComp(), field.nGrowVect());
            b_ext[component] = m_energy_audit_b_ext_old[component].get();
        }
        m_energy_audit.wb_ext_old =
            EnergyAuditFieldDot(b_ext, b_ext) / (2.0_rt * PhysConst::mu0);
    } else {
        m_energy_audit.wb_ext_old = 0.0_rt;
    }
}

void ThetaImplicitMHD::EnergyAuditSnapshotLedgers ()
{
    if (m_energy_audit_file.empty()) { return; }
    EnergyAuditStep& a = m_energy_audit;
    a.prev_floor_supply = m_floor_supplied_energy;
    a.prev_pinned = m_pinned_defect_energy;
    a.prev_halo_relax = m_halo_relaxation_energy;
    a.prev_eater = m_eater_removed_energy;
    a.prev_wall_energy = m_shaped_wall_energy;
    a.prev_absorb_energy = m_absorbed_wall_energy;
    a.prev_pedestal_e = m_halo_pedestal_energy_e;
    a.prev_pedestal_i = m_halo_pedestal_energy_i;
}

void ThetaImplicitMHD::EnergyAuditRecordOldState ()
{
    if (!m_energy_audit_step) { return; }
    m_energy_audit.old_fluid = EnergyAuditFluidTotals(m_state_old);
    std::array<const amrex::MultiFab*, 3> b_resp{};
    for (int component = 0; component < 3; ++component) {
        b_resp[component] = m_state_old.getArrayVec()[0][component];
    }
    m_energy_audit.wb_resp_old =
        EnergyAuditFieldDot(b_resp, b_resp) / (2.0_rt * PhysConst::mu0);
    if (m_energy_audit.domain_volume == 0.0_rt) {
        const CellVolume volume(m_WarpX->Geom(0));
        const amrex::Box& domain = m_WarpX->Geom(0).Domain();
        amrex::Real v = 0.0_rt;
        for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
#if defined(WARPX_DIM_RZ)
            v += volume(i) * domain.length(1);
#else
            v += volume(i);
#endif
        }
        m_energy_audit.domain_volume = v;
        // The reduction weights of the three E staggerings (the header's
        // weight-consistency line): the FieldEnergy dual volumes sum to
        // the domain volume for a cell-centered-in-r staggering and to the
        // domain volume plus pi dr^2 L_z / 4 for a nodal-in-r one (the
        // axis disk pi dr^2/4 and the half-weighted outer node together
        // exceed the exact dual annuli by one axis disk).
        for (int component = 0; component < 3; ++component) {
            m_energy_audit.dual_volume_total[component] = EnergyAuditDualVolumeTotal(
                *m_WarpX->m_fields.get(FieldType::Efield_fp,
                                       ablastr::fields::Direction{component}, 0));
        }
    }
}

void ThetaImplicitMHD::EnergyAuditThetaStage (const amrex::Real start_time)
{
    if (!m_energy_audit_step) { return; }
    using ablastr::fields::Direction;
    const amrex::Real theta_time = start_time + m_theta * m_dt;
    const amrex::Geometry& geom = m_WarpX->Geom(0);
    const bool add_external = m_hybrid_pic_model->m_add_external_fields;

    // 1. Save Efield_fp as the solve left it.
    for (int component = 0; component < 3; ++component) {
        const amrex::MultiFab& field =
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{component}, 0);
        if (m_energy_audit_e_saved[component] == nullptr) {
            m_energy_audit_e_saved[component] = std::make_unique<amrex::MultiFab>(
                field.boxArray(), field.DistributionMap(), field.nComp(),
                field.nGrowVect());
        }
        amrex::MultiFab::Copy(*m_energy_audit_e_saved[component], field, 0, 0,
                              field.nComp(), field.nGrowVect());
    }

    // 2. Recompute the theta-state scratch exactly as the residual did
    //    (the AccumulateAbsorbedWallLedger pattern; UpdateWarpXFields has
    //    already refilled the theta-stage B and the fluid fields).
    const auto magnetic_field =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0);
    m_hybrid_pic_model->CalculatePlasmaCurrent(magnetic_field,
                                               m_WarpX->GetEBUpdateEFlag());
    if (m_z_neumann) {
        for (int direction = 0; direction < 3; ++direction) {
            amrex::MultiFab& current_component = *m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{direction}, 0);
            ApplyNeumannZDomainGhosts(current_component, 1);
            if (direction == 2) {
                ApplyMirrorZLoDomainGhosts(current_component, {-1, -1, -1});
            }
        }
    }
    FillCellCenteredElectromagneticFields();
    ComputeFaceFluxes(theta_time);
    // Edge resistivity registers on the E staggerings (see the member),
    // armed for the audit's own Ohm assembly only.
    for (int component = 0; component < 3; ++component) {
        const amrex::MultiFab& reference =
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{component}, 0);
        if (m_energy_audit_eta[component] == nullptr) {
            m_energy_audit_eta[component] = std::make_unique<amrex::MultiFab>(
                reference.boxArray(), reference.DistributionMap(), 2, 0);
        }
        m_energy_audit_eta[component]->setVal(0.0_rt);
        if (m_energy_audit_eh[component] == nullptr) {
            m_energy_audit_eh[component] = std::make_unique<amrex::MultiFab>(
                reference.boxArray(), reference.DistributionMap(), 2, 0);
        }
        m_energy_audit_eh[component]->setVal(0.0_rt);
        // Ohm's-law component registers (see the member): five per E
        // component; 1D packs E_x and E_y into register 0.
#if defined(WARPX_DIM_1D_Z)
        constexpr int ohm_register_components = 10;
#else
        constexpr int ohm_register_components = 5;
#endif
        if (m_energy_audit_ohm[component] == nullptr) {
            m_energy_audit_ohm[component] = std::make_unique<amrex::MultiFab>(
                reference.boxArray(), reference.DistributionMap(),
                ohm_register_components, 0);
        }
        m_energy_audit_ohm[component]->setVal(0.0_rt);
    }
    m_energy_audit_capture = true;
    AssembleOhmElectricField(theta_time, true);
    m_energy_audit_capture = false;
    if (m_joule_ohm_current && m_include_joule_heating) {
        FillCellCenteredOhmElectricField();
    }

    // 3. The fluid RHS at the accepted theta state with the registers armed.
    const amrex::MultiFab& density = *m_WarpX->m_fields.get(MassDensityName, 0);
    if (m_energy_audit_register == nullptr) {
        m_energy_audit_register = std::make_unique<amrex::MultiFab>(
            density.boxArray(), density.DistributionMap(),
            static_cast<int>(EnergyAuditRegister::count), 0);
    }
    m_energy_audit_register->setVal(0.0_rt);
    if (!m_energy_audit_rhs.IsDefined()) {
        m_energy_audit_rhs.Define(m_state);
    }
    m_energy_audit_capture = true;
    ComputeFluidRHS(m_energy_audit_rhs, theta_time);
    m_energy_audit_capture = false;

    // 4. Reductions.
    EnergyAuditStep& audit = m_energy_audit;
    const amrex::Real dt = m_dt;
    for (int c = 0; c < static_cast<int>(EnergyAuditRegister::count); ++c) {
        audit.src[c] = dt * cell_sum(*m_energy_audit_register, c, geom);
    }
    {
        const std::array<const char*, 4> names = {MassDensityName, ElectronEnergyName,
                                                  IonEnergyName, IonInternalEnergyName};
        for (int b = 0; b < 4; ++b) {
            audit.newton[b] = 0.0_rt;
            audit.rhs_sum[b] = 0.0_rt;
            if (b == 3 && m_ion_closure != "dual_energy") { continue; }
            const auto sums = defect_and_rhs_sum(
                m_state.getMultiFabBlock(names[b], 0),
                m_state_old.getMultiFabBlock(names[b], 0),
                m_energy_audit_rhs.getMultiFabBlock(names[b], 0), geom);
            audit.newton[b] = sums.first / m_theta;
            audit.rhs_sum[b] = sums.second / m_theta;
        }
    }

    // Field-to-fluid transfer dt sum E_ohm . J with E_ohm = E_resp + E_ext.
    {
        std::array<const amrex::MultiFab*, 3> e_resp{};
        std::array<const amrex::MultiFab*, 3> e_ext{};
        std::array<const amrex::MultiFab*, 3> j_plasma{};
        for (int component = 0; component < 3; ++component) {
            e_resp[component] =
                m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{component}, 0);
            j_plasma[component] = m_WarpX->m_fields.get(
                FieldType::hybrid_current_fp_plasma, Direction{component}, 0);
            e_ext[component] =
                add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external,
                                                     Direction{component}, 0)
                             : nullptr;
        }
        audit.ej = dt * EnergyAuditFieldDot(e_resp, j_plasma,
                                            add_external ? &e_ext : nullptr);

        // The field's resistive dissipation dt sum eta J_stage . J^theta
        // with the resistivity the advance used (register component 0:
        // vacuum boost and band override included) and with the un-boosted
        // user eta (component 1). J_stage is the resistive-stage current
        // when resistive_theta differs from theta, else J^theta.
        {
            const bool stage = (m_resistive_theta != m_theta);
            amrex::Real res_field = 0.0_rt;
            amrex::Real res_user = 0.0_rt;
            for (int component = 0; component < 3; ++component) {
                const amrex::MultiFab& j_theta_mf = *j_plasma[component];
                const amrex::MultiFab& j_stage_mf =
                    stage ? *m_WarpX->m_fields.get(ResistiveStageCurrentName,
                                                   Direction{component}, 0)
                          : j_theta_mf;
#if defined(WARPX_DIM_1D_Z)
                // Ex and Ey share the z-nodal register (component 0); the
                // cell-centered Ez never enters the 1D resistive curl-curl.
                if (component == 2) { continue; }
                const amrex::MultiFab& eta_use = *m_energy_audit_eta[0];
#else
                const amrex::MultiFab& eta_use = *m_energy_audit_eta[component];
#endif
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    eta_use.ixType() == j_theta_mf.ixType(),
                    "ThetaImplicitMHD energy audit: resistivity register staggering");
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                for (amrex::MFIter mfi(j_theta_mf); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    const FieldDualVolume weight(j_theta_mf, box, geom);
                    const auto eta_arr = eta_use.const_array(mfi);
                    const auto jt = j_theta_mf.const_array(mfi);
                    const auto js = j_stage_mf.const_array(mfi);
                    reduce_op.eval(box, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                            const amrex::Real jj = js(i, j, k) * jt(i, j, k) * weight(i, j);
                            return {eta_arr(i, j, k, 0) * jj, eta_arr(i, j, k, 1) * jj};
                        });
                }
                auto sums = reduce_data.value(reduce_op);
                res_field += amrex::get<0>(sums);
                res_user += amrex::get<1>(sums);
            }
            audit.res_field = dt * all_reduce_sum(res_field);
            audit.res_user = dt * all_reduce_sum(res_user);
        }

        // Hyper-resistive dissipation dt sum E_H . J^theta (E_H registered
        // per edge by the assembly; 1D: x and y share register 0).
        {
            amrex::Real res_hyper = 0.0_rt;
            for (int component = 0; component < 3; ++component) {
                const amrex::MultiFab& j_theta_mf = *j_plasma[component];
#if defined(WARPX_DIM_1D_Z)
                if (component == 2) { continue; }
                const amrex::MultiFab& eh = *m_energy_audit_eh[0];
                const int eh_comp = component;
#else
                const amrex::MultiFab& eh = *m_energy_audit_eh[component];
                const int eh_comp = 0;
#endif
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    eh.ixType() == j_theta_mf.ixType(),
                    "ThetaImplicitMHD energy audit: hyper register staggering");
                amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                for (amrex::MFIter mfi(j_theta_mf); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    const FieldDualVolume weight(j_theta_mf, box, geom);
                    const auto eh_arr = eh.const_array(mfi);
                    const auto jt = j_theta_mf.const_array(mfi);
                    reduce_op.eval(box, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                            return {eh_arr(i, j, k, eh_comp) * jt(i, j, k) * weight(i, j)};
                        });
                }
                res_hyper += amrex::get<0>(reduce_data.value(reduce_op));
            }
            audit.res_hyper = dt * all_reduce_sum(res_hyper);
        }

        // Ohm's-law component works dt sum E_c . J^theta from the edge
        // registers: the induction (ideal) part as assembled, the plain
        // cell-mean ideal EMF at the same edges, the Hall EMF, the
        // electron-inertia field and the UCT corner dissipation (1D: E_x
        // and E_y packed into register 0; the cell-centered 1D E_z carries
        // no current).
        {
            amrex::Real works[5] = {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
            for (int component = 0; component < 3; ++component) {
                const amrex::MultiFab& j_theta_mf = *j_plasma[component];
#if defined(WARPX_DIM_1D_Z)
                if (component == 2) { continue; }
                const amrex::MultiFab& reg = *m_energy_audit_ohm[0];
                const int offset = 5 * component;
#else
                const amrex::MultiFab& reg = *m_energy_audit_ohm[component];
                const int offset = 0;
#endif
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    reg.ixType() == j_theta_mf.ixType(),
                    "ThetaImplicitMHD energy audit: Ohm component register staggering");
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                                 amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                    reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                for (amrex::MFIter mfi(j_theta_mf); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    const FieldDualVolume weight(j_theta_mf, box, geom);
                    const auto reg_arr = reg.const_array(mfi);
                    const auto jt = j_theta_mf.const_array(mfi);
                    reduce_op.eval(box, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                            const amrex::Real jj = jt(i, j, k) * weight(i, j);
                            return {reg_arr(i, j, k, offset) * jj,
                                    reg_arr(i, j, k, offset + 1) * jj,
                                    reg_arr(i, j, k, offset + 2) * jj,
                                    reg_arr(i, j, k, offset + 3) * jj,
                                    reg_arr(i, j, k, offset + 4) * jj};
                        });
                }
                const auto sums = reduce_data.value(reduce_op);
                works[0] += amrex::get<0>(sums);
                works[1] += amrex::get<1>(sums);
                works[2] += amrex::get<2>(sums);
                works[3] += amrex::get<3>(sums);
                works[4] += amrex::get<4>(sums);
            }
            audit.ind_work = dt * all_reduce_sum(works[0]);
            audit.ind_cc_work = dt * all_reduce_sum(works[1]);
            audit.hall_work = dt * all_reduce_sum(works[2]);
            audit.inertia_work = dt * all_reduce_sum(works[3]);
            audit.corner_diss_work = dt * all_reduce_sum(works[4]);
        }

        // The circuit's electromagnetic power into the domain: the coil
        // current sheet lies INSIDE the domain (curl B_ext != 0 there), so
        // the drive enters the field energy as -dt sum E_ohm . J_coil, not
        // through the boundary Poynting flux. J_coil = curl B_ext^theta /
        // mu0 with the solver's own Ampere operator (the discrete curl
        // CalculatePlasmaCurrent applies to the response field), into the
        // audit's scratch; split by the E it works against (E_ext: the
        // vacuum field's own energy; E_resp: the work against the
        // plasma-induced field, the plasma's load on the circuit).
        audit.circuit_in = 0.0_rt;
        audit.circuit_in_ext = 0.0_rt;
        audit.circuit_in_plasma = 0.0_rt;
        if (add_external) {
            ablastr::fields::VectorField j_coil{};
            std::array<const amrex::MultiFab*, 3> j_coil_const{};
            for (int component = 0; component < 3; ++component) {
                const amrex::MultiFab& reference = *j_plasma[component];
                if (m_energy_audit_j_coil[component] == nullptr) {
                    m_energy_audit_j_coil[component] = std::make_unique<amrex::MultiFab>(
                        reference.boxArray(), reference.DistributionMap(),
                        reference.nComp(), reference.nGrowVect());
                }
                m_energy_audit_j_coil[component]->setVal(0.0_rt);
                j_coil[component] = m_energy_audit_j_coil[component].get();
                j_coil_const[component] = m_energy_audit_j_coil[component].get();
            }
            const ablastr::fields::VectorField b_ext =
                m_WarpX->m_fields.get_alldirs(FieldType::hybrid_B_fp_external, 0);
            m_WarpX->get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(
                j_coil, b_ext, m_WarpX->GetEBUpdateEFlag()[0], 0);
            audit.circuit_in_ext = -dt * EnergyAuditFieldDot(e_ext, j_coil_const);
            audit.circuit_in_plasma = -dt * EnergyAuditFieldDot(e_resp, j_coil_const);
            audit.circuit_in = audit.circuit_in_ext + audit.circuit_in_plasma;
        }
    }

    // Poynting outflow through the domain faces (exact discrete form, see
    // the header) for the TOTAL fields (E_resp + E_ext, B_resp + B_ext) and
    // for the external fields alone (the external field's own identity),
    // and the fluid export per face and per block.
    for (auto& v : audit.poynt_out) { v = 0.0_rt; }
    for (auto& v : audit.poynt_out_ext) { v = 0.0_rt; }
    for (auto& row : audit.export_domain) { for (auto& v : row) { v = 0.0_rt; } }
    const amrex::Box& domain = geom.Domain();
    constexpr int flux_mass = FaceFluxComponent::mass;
    constexpr int flux_ue = FaceFluxComponent::electron_energy;
    constexpr int flux_ei = FaceFluxComponent::ion_energy;
    constexpr int flux_ui = FaceFluxComponent::ion_internal_energy;
    const bool dual = m_ion_closure == "dual_energy";
    const amrex::Real inverse_mu0 = 1.0_rt / PhysConst::mu0;
#if defined(WARPX_DIM_RZ)
    {
        const amrex::Real rlo = geom.ProbLo(0);
        const amrex::Real dr = geom.CellSize(0);
        const amrex::Real dz = geom.CellSize(1);
        const int nr = domain.bigEnd(0) + 1;   // nodal index of the r_hi face
        const int nz = domain.bigEnd(1) + 1;   // nodal index of the z_hi face
        const amrex::Real r_hi = rlo + nr * dr;
        const bool z_periodic = geom.isPeriodic(1);
        const amrex::MultiFab& e_r = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0);
        const amrex::MultiFab& e_t = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, 0);
        const amrex::MultiFab& e_z = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{2}, 0);
        const amrex::MultiFab& b_r = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0);
        const amrex::MultiFab& b_t = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, 0);
        const amrex::MultiFab& b_z = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, 0);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            e_r.ixType().toIntVect() == amrex::IntVect(0, 1) &&
                e_t.ixType().toIntVect() == amrex::IntVect(1, 1) &&
                e_z.ixType().toIntVect() == amrex::IntVect(1, 0) &&
                b_r.ixType().toIntVect() == amrex::IntVect(1, 0) &&
                b_t.ixType().toIntVect() == amrex::IntVect(0, 0) &&
                b_z.ixType().toIntVect() == amrex::IntVect(0, 1),
            "ThetaImplicitMHD energy audit: unexpected RZ Yee staggering");
        const amrex::MultiFab* const ext_e_r = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{0}, 0) : nullptr;
        const amrex::MultiFab* const ext_e_t = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{1}, 0) : nullptr;
        const amrex::MultiFab* const ext_e_z = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{2}, 0) : nullptr;
        const amrex::MultiFab* const ext_b_r = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{0}, 0) : nullptr;
        const amrex::MultiFab* const ext_b_t = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{1}, 0) : nullptr;
        const amrex::MultiFab* const ext_b_z = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{2}, 0) : nullptr;
        const amrex::MultiFab& flux_r = *m_WarpX->m_fields.get(FaceFluxRName, 0);
        const amrex::MultiFab& flux_z = *m_WarpX->m_fields.get(FaceFluxZName, 0);

        // The face sums for one (E, B) pair, each given as a base MultiFab
        // per component plus an optional addend (nullptr = none): out[0] =
        // r_hi outflow, out[1] = z_lo outflow, out[2] = z_hi outflow, all
        // as dt x power.
        auto poynting_faces = [&] (const amrex::MultiFab* er, const amrex::MultiFab* et,
                                   const amrex::MultiFab* ez, const amrex::MultiFab* aer,
                                   const amrex::MultiFab* aet, const amrex::MultiFab* aez,
                                   const amrex::MultiFab* br, const amrex::MultiFab* bt,
                                   const amrex::MultiFab* bz, const amrex::MultiFab* abr,
                                   const amrex::MultiFab* abt, const amrex::MultiFab* abz,
                                   amrex::Real* out) {
            const bool ext = (aer != nullptr);
            // r_hi face: sum over nodal j (E_theta Bbar_z, half weights at the
            // nodal box ends) and over cell-centered j (E_z Btilde_theta).
            {
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
                amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                for (amrex::MFIter mfi(*et); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();      // nodal r, nodal z
                    if (box.bigEnd(0) != nr) { continue; }
                    amrex::Box face = box;
                    face.setSmall(0, nr);
                    const int jlo = box.smallEnd(1);
                    const int jhi = box.bigEnd(1);
                    const auto et_a = et->const_array(mfi);
                    const auto ez_a = ez->const_array(mfi);
                    const auto bz_a = bz->const_array(mfi);
                    const auto bt_a = bt->const_array(mfi);
                    const auto eet = ext ? aet->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto eez = ext ? aez->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto ebz = ext ? abz->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto ebt = ext ? abt->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const amrex::Real r_in = rlo + (nr - 0.5_rt) * dr;
                    const amrex::Real r_out = rlo + (nr + 0.5_rt) * dr;
                    reduce_op.eval(face, reduce_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                            amrex::Real e_theta = et_a(i, j, k);
                            amrex::Real bz_in = bz_a(i - 1, j, k);
                            amrex::Real bz_out = bz_a(i, j, k);
                            if (ext) {
                                e_theta += eet(i, j, k);
                                bz_in += ebz(i - 1, j, k);
                                bz_out += ebz(i, j, k);
                            }
                            amrex::Real wj = 1.0_rt;
                            if (j == jlo || j == jhi) { wj = 0.5_rt; }
                            const amrex::Real nodal_term =
                                wj * e_theta * 0.5_rt * (bz_in + bz_out);
                            amrex::Real cc_term = 0.0_rt;
                            if (j < jhi) {
                                amrex::Real e_zed = ez_a(i, j, k);
                                amrex::Real bt_in = bt_a(i - 1, j, k);
                                amrex::Real bt_out = bt_a(i, j, k);
                                if (ext) {
                                    e_zed += eez(i, j, k);
                                    bt_in += ebt(i - 1, j, k);
                                    bt_out += ebt(i, j, k);
                                }
                                cc_term = e_zed * 0.5_rt * (r_in * bt_in + r_out * bt_out) / r_hi;
                            }
                            return {nodal_term, cc_term};
                        });
                }
                auto sums = reduce_data.value(reduce_op);
                const amrex::Real area = 2.0_rt * MathConst::pi * r_hi * dz;
                out[0] = dt * area * inverse_mu0 *
                         all_reduce_sum(amrex::get<0>(sums) - amrex::get<1>(sums));
            }
            out[1] = 0.0_rt;
            out[2] = 0.0_rt;
            if (!z_periodic) {
                for (int side = 0; side < 2; ++side) {
                    const int jf = (side == 0) ? domain.smallEnd(1) : nz;
                    const amrex::Real outward = (side == 0) ? -1.0_rt : 1.0_rt;
                    // (E x B)_z = E_r B_theta - E_theta B_r on the face.
                    amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
                    amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
                    using ReduceTuple = typename decltype(reduce_data)::Type;
                    for (amrex::MFIter mfi(*er); mfi.isValid(); ++mfi) {
                        const amrex::Box box = mfi.validbox();   // cc r, nodal z
                        if (jf < box.smallEnd(1) || jf > box.bigEnd(1)) { continue; }
                        amrex::Box face = box;
                        face.setSmall(1, jf);
                        face.setBig(1, jf);
                        const int ilo = box.smallEnd(0);
                        const int ihi = box.bigEnd(0);   // cc r: nodal E_theta reaches ihi+1
                        const auto er_a = er->const_array(mfi);
                        const auto et_a = et->const_array(mfi);
                        const auto br_a = br->const_array(mfi);
                        const auto bt_a = bt->const_array(mfi);
                        const auto eer = ext ? aer->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                        const auto eet = ext ? aet->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                        const auto ebr = ext ? abr->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                        const auto ebt = ext ? abt->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                        reduce_op.eval(face, reduce_data,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                                amrex::Real e_rad = er_a(i, j, k);
                                amrex::Real bt_in = bt_a(i, j - 1, k);
                                amrex::Real bt_out = bt_a(i, j, k);
                                if (ext) {
                                    e_rad += eer(i, j, k);
                                    bt_in += ebt(i, j - 1, k);
                                    bt_out += ebt(i, j, k);
                                }
                                const amrex::Real r_cc = rlo + (i + 0.5_rt) * dr;
                                const amrex::Real cc_term =
                                    r_cc * e_rad * 0.5_rt * (bt_in + bt_out);
                                amrex::Real nodal_term = 0.0_rt;
                                for (int n = 0; n < 2; ++n) {
                                    const int in = i + n;
                                    if (n == 1 && i != ihi) { continue; }
                                    const amrex::Real r_n = rlo + in * dr;
                                    amrex::Real w = 1.0_rt;
                                    if (in == ilo || in == ihi + 1) { w = 0.5_rt; }
                                    amrex::Real e_theta = et_a(in, j, k);
                                    amrex::Real br_in = br_a(in, j - 1, k);
                                    amrex::Real br_out = br_a(in, j, k);
                                    if (ext) {
                                        e_theta += eet(in, j, k);
                                        br_in += ebr(in, j - 1, k);
                                        br_out += ebr(in, j, k);
                                    }
                                    nodal_term += w * r_n * e_theta * 0.5_rt * (br_in + br_out);
                                }
                                return {cc_term, nodal_term};
                            });
                    }
                    auto sums = reduce_data.value(reduce_op);
                    out[1 + side] =
                        outward * dt * 2.0_rt * MathConst::pi * dr * inverse_mu0 *
                        all_reduce_sum(amrex::get<0>(sums) - amrex::get<1>(sums));
                }
            }
        };
        // total fields: E_resp + E_ext, B_resp + B_ext
        poynting_faces(&e_r, &e_t, &e_z, ext_e_r, ext_e_t, ext_e_z,
                       &b_r, &b_t, &b_z, ext_b_r, ext_b_t, ext_b_z, audit.poynt_out);
        // the external field alone (its own Faraday identity)
        if (add_external) {
            poynting_faces(ext_e_r, ext_e_t, ext_e_z, nullptr, nullptr, nullptr,
                           ext_b_r, ext_b_t, ext_b_z, nullptr, nullptr, nullptr,
                           audit.poynt_out_ext);
        }
        // r_hi fluid export (cell-centered j faces of the r-flux register).
        {
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (amrex::MFIter mfi(flux_r); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox();
                if (box.bigEnd(0) != nr) { continue; }
                amrex::Box face = box;
                face.setSmall(0, nr);
                const auto f = flux_r.const_array(mfi);
                reduce_op.eval(face, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                        return {f(i, j, k, flux_mass), f(i, j, k, flux_ue),
                                f(i, j, k, flux_ei), dual ? f(i, j, k, flux_ui) : 0.0_rt};
                    });
            }
            auto sums = reduce_data.value(reduce_op);
            const amrex::Real area = 2.0_rt * MathConst::pi * r_hi * dz;
            audit.export_domain[0][0] = dt * area * all_reduce_sum(amrex::get<0>(sums));
            audit.export_domain[1][0] = dt * area * all_reduce_sum(amrex::get<1>(sums));
            audit.export_domain[2][0] = dt * area * all_reduce_sum(amrex::get<2>(sums));
            audit.export_domain[3][0] = dt * area * all_reduce_sum(amrex::get<3>(sums));
        }
        if (!z_periodic) {
            for (int side = 0; side < 2; ++side) {
                const int jf = (side == 0) ? domain.smallEnd(1) : nz;
                const amrex::Real outward = (side == 0) ? -1.0_rt : 1.0_rt;
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> flux_op;
                amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> flux_data(flux_op);
                using FluxTuple = typename decltype(flux_data)::Type;
                for (amrex::MFIter mfi(flux_z); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();   // cc r, nodal z
                    if (jf < box.smallEnd(1) || jf > box.bigEnd(1)) { continue; }
                    amrex::Box face = box;
                    face.setSmall(1, jf);
                    face.setBig(1, jf);
                    const auto f = flux_z.const_array(mfi);
                    flux_op.eval(face, flux_data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> FluxTuple {
                            const amrex::Real r_cc = rlo + (i + 0.5_rt) * dr;
                            return {r_cc * f(i, j, k, flux_mass), r_cc * f(i, j, k, flux_ue),
                                    r_cc * f(i, j, k, flux_ei),
                                    dual ? r_cc * f(i, j, k, flux_ui) : 0.0_rt};
                        });
                }
                auto fsums = flux_data.value(flux_op);
                const amrex::Real factor = outward * dt * 2.0_rt * MathConst::pi * dr;
                audit.export_domain[0][1 + side] = factor * all_reduce_sum(amrex::get<0>(fsums));
                audit.export_domain[1][1 + side] = factor * all_reduce_sum(amrex::get<1>(fsums));
                audit.export_domain[2][1 + side] = factor * all_reduce_sum(amrex::get<2>(fsums));
                audit.export_domain[3][1 + side] = factor * all_reduce_sum(amrex::get<3>(fsums));
            }
        }
        // Shaped-wall stair deposition per block (signs INTO the wall).
        for (auto& v : audit.export_wall) { v = 0.0_rt; }
        if (m_wall_mask.GetThermalBC() != ImplicitMHDWallMask::ThermalBC::none) {
            const int* const AMREX_RESTRICT fm = m_wall_mask.FirstMaskedCellCentered();
            const int mask_z_lo = -m_wall_mask.GhostCells();
            const int mask_z_hi = m_wall_mask.AxialCells() - 1 + m_wall_mask.GhostCells();
            const amrex::Real two_pi = 2.0_rt * MathConst::pi;
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            const amrex::Box face_domain_r = amrex::convert(domain, flux_r.ixType().toIntVect());
            const auto owner_r = flux_r.OwnerMask(geom.periodicity());
            for (amrex::MFIter mfi(flux_r); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox() & face_domain_r;
                if (box.isEmpty()) { continue; }
                const auto f = flux_r.const_array(mfi);
                const auto own = owner_r->const_array(mfi);
                reduce_op.eval(box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                        const int jc = std::max(mask_z_lo, std::min(mask_z_hi, j));
                        const bool left_masked = (i - 1 >= fm[jc]);
                        const bool right_masked = (i >= fm[jc]);
                        if (left_masked == right_masked || !own(i, j, k)) {
                            return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                        }
                        const amrex::Real sign = right_masked ? 1.0_rt : -1.0_rt;
                        const amrex::Real area = two_pi * (rlo + i * dr) * dz;
                        return {sign * area * f(i, j, k, flux_mass),
                                sign * area * f(i, j, k, flux_ue),
                                sign * area * f(i, j, k, flux_ei),
                                dual ? sign * area * f(i, j, k, flux_ui) : 0.0_rt};
                    });
            }
            const amrex::Box face_domain_z = amrex::convert(domain, flux_z.ixType().toIntVect());
            const auto owner_z = flux_z.OwnerMask(geom.periodicity());
            for (amrex::MFIter mfi(flux_z); mfi.isValid(); ++mfi) {
                const amrex::Box box = mfi.validbox() & face_domain_z;
                if (box.isEmpty()) { continue; }
                const auto f = flux_z.const_array(mfi);
                const auto own = owner_z->const_array(mfi);
                reduce_op.eval(box, reduce_data,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                        const int jl = std::max(mask_z_lo, std::min(mask_z_hi, j - 1));
                        const int jr = std::max(mask_z_lo, std::min(mask_z_hi, j));
                        const bool left_masked = (i >= fm[jl]);
                        const bool right_masked = (i >= fm[jr]);
                        if (left_masked == right_masked || !own(i, j, k)) {
                            return {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                        }
                        const amrex::Real sign = right_masked ? 1.0_rt : -1.0_rt;
                        const amrex::Real area = two_pi * (rlo + (i + 0.5_rt) * dr) * dr;
                        return {sign * area * f(i, j, k, flux_mass),
                                sign * area * f(i, j, k, flux_ue),
                                sign * area * f(i, j, k, flux_ei),
                                dual ? sign * area * f(i, j, k, flux_ui) : 0.0_rt};
                    });
            }
            auto sums = reduce_data.value(reduce_op);
            audit.export_wall[0] = dt * all_reduce_sum(amrex::get<0>(sums));
            audit.export_wall[1] = dt * all_reduce_sum(amrex::get<1>(sums));
            audit.export_wall[2] = dt * all_reduce_sum(amrex::get<2>(sums));
            audit.export_wall[3] = dt * all_reduce_sum(amrex::get<3>(sums));
        }
    }
#elif defined(WARPX_DIM_1D_Z)
    {
        const bool z_periodic = geom.isPeriodic(0);
        for (auto& v : audit.export_wall) { v = 0.0_rt; }
        if (!z_periodic) {
            const int nz = domain.bigEnd(0) + 1;
            const amrex::MultiFab& e_x = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0);
            const amrex::MultiFab& e_y = *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{1}, 0);
            const amrex::MultiFab& b_x = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, 0);
            const amrex::MultiFab& b_y = *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, 0);
            const amrex::MultiFab* const ext_e_x = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{0}, 0) : nullptr;
            const amrex::MultiFab* const ext_e_y = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{1}, 0) : nullptr;
            const amrex::MultiFab* const ext_b_x = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{0}, 0) : nullptr;
            const amrex::MultiFab* const ext_b_y = add_external ? m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{1}, 0) : nullptr;
            const amrex::MultiFab& flux_z = *m_WarpX->m_fields.get(FaceFluxZName, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                e_x.ixType().toIntVect() == amrex::IntVect(1) &&
                    b_x.ixType().toIntVect() == amrex::IntVect(0),
                "ThetaImplicitMHD energy audit: unexpected 1D Yee staggering");
            for (int side = 0; side < 2; ++side) {
                const int jf = (side == 0) ? domain.smallEnd(0) : nz;
                const amrex::Real outward = (side == 0) ? -1.0_rt : 1.0_rt;
                amrex::Real poynting = 0.0_rt;
                amrex::Real export_blocks[4] = {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                for (amrex::MFIter mfi(e_x); mfi.isValid(); ++mfi) {
                    const amrex::Box box = mfi.validbox();
                    if (jf < box.smallEnd(0) || jf > box.bigEnd(0)) { continue; }
                    const auto ex = e_x.const_array(mfi);
                    const auto ey = e_y.const_array(mfi);
                    const auto bx = b_x.const_array(mfi);
                    const auto by = b_y.const_array(mfi);
                    const auto f = flux_z.const_array(mfi);
                    amrex::Real e_1 = 0.0_rt, e_2 = 0.0_rt, bx_mean = 0.0_rt, by_mean = 0.0_rt;
                    amrex::Real fl[4] = {0.0_rt, 0.0_rt, 0.0_rt, 0.0_rt};
                    // host-side gather of one face value (tiny box)
                    amrex::Gpu::DeviceVector<amrex::Real> dvals(8, 0.0_rt);
                    amrex::Real* const dptr = dvals.data();
                    const auto eex = add_external ? ext_e_x->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto eey = add_external ? ext_e_y->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto ebx = add_external ? ext_b_x->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const auto eby = add_external ? ext_b_y->const_array(mfi) : amrex::Array4<amrex::Real const>{};
                    const bool ext = add_external;
                    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
                        amrex::Real vex = ex(jf, 0, 0), vey = ey(jf, 0, 0);
                        amrex::Real vbx_in = bx(jf - 1, 0, 0), vbx_out = bx(jf, 0, 0);
                        amrex::Real vby_in = by(jf - 1, 0, 0), vby_out = by(jf, 0, 0);
                        if (ext) {
                            vex += eex(jf, 0, 0); vey += eey(jf, 0, 0);
                            vbx_in += ebx(jf - 1, 0, 0); vbx_out += ebx(jf, 0, 0);
                            vby_in += eby(jf - 1, 0, 0); vby_out += eby(jf, 0, 0);
                        }
                        dptr[0] = vex; dptr[1] = vey;
                        dptr[2] = 0.5_rt * (vbx_in + vbx_out);
                        dptr[3] = 0.5_rt * (vby_in + vby_out);
                        dptr[4] = f(jf, 0, 0, flux_mass);
                        dptr[5] = f(jf, 0, 0, flux_ue);
                        dptr[6] = f(jf, 0, 0, flux_ei);
                        dptr[7] = dual ? f(jf, 0, 0, flux_ui) : 0.0_rt;
                    });
                    amrex::Gpu::streamSynchronize();
                    std::vector<amrex::Real> hvals(8);
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost, dvals.begin(), dvals.end(), hvals.begin());
                    e_1 = hvals[0]; e_2 = hvals[1]; bx_mean = hvals[2]; by_mean = hvals[3];
                    for (int b = 0; b < 4; ++b) { fl[b] = hvals[4 + b]; }
                    poynting += (e_1 * by_mean - e_2 * bx_mean) * inverse_mu0;
                    for (int b = 0; b < 4; ++b) { export_blocks[b] += fl[b]; }
                }
                audit.poynt_out[1 + side] = outward * dt * all_reduce_sum(poynting);
                for (int b = 0; b < 4; ++b) {
                    audit.export_domain[b][1 + side] = outward * dt * all_reduce_sum(export_blocks[b]);
                }
            }
        }
    }
#endif

    // 5. Restore Efield_fp.
    for (int component = 0; component < 3; ++component) {
        amrex::MultiFab& field =
            *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{component}, 0);
        amrex::MultiFab::Copy(field, *m_energy_audit_e_saved[component], 0, 0,
                              field.nComp(), field.nGrowVect());
    }
}

void ThetaImplicitMHD::EnergyAuditFinishStage (const int stage)
{
    if (!m_energy_audit_step) { return; }
    using ablastr::fields::Direction;
    EnergyAuditStep& audit = m_energy_audit;
    switch (stage) {
    case 0: audit.stage_raw = EnergyAuditFluidTotals(m_state); break;
    case 1: audit.stage_eater = EnergyAuditFluidTotals(m_state); break;
    case 2: audit.stage_efloor = EnergyAuditFluidTotals(m_state); break;
    case 3: audit.stage_eirestore = EnergyAuditFluidTotals(m_state); break;
    case 4: audit.stage_sync = EnergyAuditFluidTotals(m_state); break;
    case 5: {
        audit.final_fluid = EnergyAuditFluidTotals(m_state);
        // Bfield_fp holds the TOTAL field at t^{n+1} here; the state block
        // the plasma response.
        std::array<const amrex::MultiFab*, 3> b_total{};
        std::array<const amrex::MultiFab*, 3> b_total_old{};
        std::array<const amrex::MultiFab*, 3> b_resp{};
        for (int component = 0; component < 3; ++component) {
            b_total[component] =
                m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{component}, 0);
            b_total_old[component] = m_energy_audit_b_total_old[component].get();
            b_resp[component] = m_state.getArrayVec()[0][component];
        }
        const amrex::Real inverse_2mu0 = 1.0_rt / (2.0_rt * PhysConst::mu0);
        const amrex::Real new_new = EnergyAuditFieldDot(b_total, b_total);
        const amrex::Real old_old = EnergyAuditFieldDot(b_total_old, b_total_old);
        const amrex::Real new_old = EnergyAuditFieldDot(b_total, b_total_old);
        audit.wb_total_new = new_new * inverse_2mu0;
        audit.wb_resp_new = EnergyAuditFieldDot(b_resp, b_resp) * inverse_2mu0;
        // sum B^theta . dB / mu0 with B^theta = (1 - theta) B^n + theta B^{n+1}
        // and sum |dB|^2 / mu0, from the three dot products.
        const amrex::Real db2 = new_new - 2.0_rt * new_old + old_old;
        audit.b_theta_dot_db =
            ((1.0_rt - m_theta) * (new_old - old_old) + m_theta * (new_new - new_old)) /
            PhysConst::mu0;
        audit.theta_diss = (m_theta - 0.5_rt) * db2 / PhysConst::mu0;
        if (m_hybrid_pic_model->m_add_external_fields) {
            std::array<const amrex::MultiFab*, 3> b_ext{};
            std::array<const amrex::MultiFab*, 3> b_ext_old{};
            for (int component = 0; component < 3; ++component) {
                b_ext[component] = m_WarpX->m_fields.get(
                    FieldType::hybrid_B_fp_external, Direction{component}, 0);
                b_ext_old[component] = m_energy_audit_b_ext_old[component].get();
            }
            const amrex::Real ext_nn = EnergyAuditFieldDot(b_ext, b_ext);
            const amrex::Real ext_oo = EnergyAuditFieldDot(b_ext_old, b_ext_old);
            const amrex::Real ext_no = EnergyAuditFieldDot(b_ext, b_ext_old);
            audit.wb_ext_new = ext_nn * inverse_2mu0;
            audit.theta_diss_ext =
                (m_theta - 0.5_rt) * (ext_nn - 2.0_rt * ext_no + ext_oo) / PhysConst::mu0;
        } else {
            audit.wb_ext_new = 0.0_rt;
            audit.theta_diss_ext = 0.0_rt;
        }
        break;
    }
    default: break;
    }
}

void ThetaImplicitMHD::EnergyAuditWriteRow (const amrex::Real end_time, const int step)
{
    if (!m_energy_audit_step) { return; }
    EnergyAuditStep& a = m_energy_audit;
    const amrex::Real dt = m_dt;
    using R = EnergyAuditRegister;
    const bool dual = m_ion_closure == "dual_energy";

    // Energy changes over the step (final state vs step-old state).
    const amrex::Real dwb_total = a.wb_total_new - a.wb_total_old;
    const amrex::Real dwb_resp = a.wb_resp_new - a.wb_resp_old;
    const amrex::Real dwb_ext = a.wb_ext_new - a.wb_ext_old;
    const amrex::Real due = a.final_fluid.ue - a.old_fluid.ue;
    const amrex::Real dei = a.final_fluid.ei - a.old_fluid.ei;
    const amrex::Real dui = a.final_fluid.ui - a.old_fluid.ui;
    const amrex::Real dke = a.final_fluid.ke - a.old_fluid.ke;
    const amrex::Real dmass = a.final_fluid.mass - a.old_fluid.mass;
    const amrex::Real dw_total = dwb_total + due + dei;

    // Inflows (positive into the domain) and wall deposition (positive
    // into the wall).
    const amrex::Real poynt_in = -(a.poynt_out[0] + a.poynt_out[1] + a.poynt_out[2]);
    auto face_sum = [] (const amrex::Real* v) { return v[0] + v[1] + v[2]; };
    const amrex::Real fluid_in_mass = -face_sum(a.export_domain[0]);
    const amrex::Real fluid_in_e = -face_sum(a.export_domain[1]);
    const amrex::Real fluid_in_i = -face_sum(a.export_domain[2]);
    const amrex::Real fluid_in_ui = -face_sum(a.export_domain[3]);
    const amrex::Real wall_e = a.export_wall[1];
    const amrex::Real wall_i = a.export_wall[2];
    const amrex::Real wall_ui = a.export_wall[3];
    const amrex::Real wall_mass = a.export_wall[0];

    // End-of-step stage differences.
    const amrex::Real eater_e = a.stage_eater.ue - a.stage_raw.ue;
    const amrex::Real eater_i = a.stage_eater.ei - a.stage_raw.ei;
    const amrex::Real eater_ui = a.stage_eater.ui - a.stage_raw.ui;
    const amrex::Real eater_mass = a.stage_eater.mass - a.stage_raw.mass;
    const amrex::Real floor_e = a.stage_efloor.ue - a.stage_eater.ue;
    const amrex::Real floor_ei = a.stage_eirestore.ei - a.stage_efloor.ei;
    const amrex::Real sync_ei = a.stage_sync.ei - a.stage_eirestore.ei;
    const amrex::Real sync_floor_ui = a.stage_sync.ui - a.stage_eirestore.ui;
    // anything after the sync stage (should be zero: copies and BCs only)
    const amrex::Real post_e = a.final_fluid.ue - a.stage_sync.ue;
    const amrex::Real post_i = a.final_fluid.ei - a.stage_sync.ei;

    // Between-step injections (previous audited step's final vs this
    // step's start), only when the previous audited step is step - 1.
    amrex::Real inject_e = 0.0_rt, inject_i = 0.0_rt, inject_mass = 0.0_rt;
    if (a.have_prev_final && a.prev_step == step - 1) {
        inject_e = a.old_fluid.ue - a.prev_final.ue;
        inject_i = a.old_fluid.ei - a.prev_final.ei;
        inject_mass = a.old_fluid.mass - a.prev_final.mass;
    }

    // Fluid RHS self-checks: sum rhs dV / theta must equal
    // dt [-export_domain - export_wall + sum sources] per block.
    const amrex::Real src_e = a.src[R::pw_e] + a.src[R::joule_e] + a.src[R::equil] -
                              a.src[R::relax_e] + a.src[R::fcs_e];
    const amrex::Real src_i = a.src[R::lorentz] + a.src[R::pw_i] + a.src[R::joule_i] -
                              a.src[R::drain_ei] - a.src[R::equil] - a.src[R::relax_ei] +
                              a.src[R::fcs_ei];
    const amrex::Real src_ui = a.src[R::pdv_ui] + a.src[R::visc_ui] + a.src[R::joule_i] -
                               a.src[R::drain_ui] - a.src[R::equil] - a.src[R::relax_ui] +
                               a.src[R::fcs_ui];
    const amrex::Real fluxw_mass = a.rhs_sum[0] - (fluid_in_mass - wall_mass + a.src[R::fcs_mass]);
    const amrex::Real fluxw_e = a.rhs_sum[1] - (fluid_in_e - wall_e + src_e);
    const amrex::Real fluxw_i = a.rhs_sum[2] - (fluid_in_i - wall_i + src_i);
    const amrex::Real fluxw_ui = dual ? a.rhs_sum[3] - (fluid_in_ui - wall_ui + src_ui) : 0.0_rt;

    // Field identity: sum B^theta . dB / mu0 = -dt (poynting_out + E.J) +
    // circuit_in + defect (the coil current sheet inside the domain feeds
    // the field through E . J_coil).
    const amrex::Real poynt_out_total = -poynt_in;
    const amrex::Real faraday_defect =
        a.b_theta_dot_db + poynt_out_total + a.ej - a.circuit_in;
    // Exchange mismatch: what the field lost minus what the fluid received.
    const amrex::Real exchange = a.ej - (a.src[R::lorentz] + a.src[R::joule_e] + a.src[R::joule_i]);
    const amrex::Real pw_pair = a.src[R::pw_e] + a.src[R::pw_i];

    // The code's own ledgers, per step (cumulative counters differenced
    // against the previous row; the wall ledger's counter is compared
    // with the audit's own stair sum in the tests).
    // (the prev_* snapshots are refreshed EVERY step by
    // EnergyAuditSnapshotLedgers, so these are one-step deltas at any
    // energy_audit_interval)
    const amrex::Real d_ledger_floor_supply = m_floor_supplied_energy - a.prev_floor_supply;
    const amrex::Real d_ledger_pinned = m_pinned_defect_energy - a.prev_pinned;
    const amrex::Real d_ledger_halo_relax = m_halo_relaxation_energy - a.prev_halo_relax;
    const amrex::Real d_ledger_eater = m_eater_removed_energy - a.prev_eater;
    const amrex::Real d_ledger_wall = m_shaped_wall_energy - a.prev_wall_energy;
    const amrex::Real d_ledger_pedestal_e = m_halo_pedestal_energy_e - a.prev_pedestal_e;
    const amrex::Real d_ledger_pedestal_i = m_halo_pedestal_energy_i - a.prev_pedestal_i;
    // The external field's own Faraday identity: dW_B_ext = circuit_in_ext
    // - poynt_out_ext - theta_diss_ext for an external field whose stored
    // E_ext is the exact time derivative of its stored B_ext (segment-driven
    // or linear-in-time scales); the defect is what is not.
    const amrex::Real poynt_out_ext_total =
        a.poynt_out_ext[0] + a.poynt_out_ext[1] + a.poynt_out_ext[2];
    const amrex::Real ext_defect =
        dwb_ext - a.circuit_in_ext + poynt_out_ext_total + a.theta_diss_ext;
    const amrex::Real lorentz_withheld =
        a.src[R::lorentz_unweighted] - a.src[R::lorentz];
    // The magnetic-force work the physical-share weight / band-cells mask
    // withheld from the fluid rows (lorentz_force_current = physical,
    // lorentz_force_band_cells); exactly zero when both are off, so every
    // expression below that subtracts it is bit-identical then.
    const amrex::Real force_withheld = a.src[R::force_withheld];
    const amrex::Real res_boost = a.res_field - a.res_user;
    // The Ohm's-law component split of the exchange remainder (see the
    // header): exchange_rest2 = ideal_mismatch + hall_work + inertia_work +
    // ohm_rest exactly, ideal_mismatch = stagger_mismatch + recon_work +
    // corner_diss_work exactly.
    const amrex::Real exchange_rest2 =
        exchange - a.res_field + a.src[R::joule_e] + a.src[R::joule_i] - a.res_hyper -
        lorentz_withheld - force_withheld;
    const amrex::Real ideal_mismatch =
        a.ind_work - a.src[R::lorentz_unweighted] - force_withheld;
    const amrex::Real stagger_mismatch =
        a.ind_cc_work - a.src[R::lorentz_unweighted] - force_withheld;
    const amrex::Real recon_work = a.ind_work - a.ind_cc_work - a.corner_diss_work;
    const amrex::Real ohm_rest =
        a.ej - (a.ind_work + a.hall_work + a.inertia_work + a.res_field + a.res_hyper);
    const amrex::Real exchange_rest3 =
        exchange_rest2 - (ideal_mismatch + a.hall_work + a.inertia_work + ohm_rest);

    // Closure residuals. dW_total spans THIS step (the state after the
    // between-step pedestal raise -> the end of the step), so the raise
    // (inject_*) is reported but enters neither residual; the energy since
    // the first row is dW_total_cum + inject_cum. 'booked': the physical
    // fluxes, the audit's wall sums, the code's own ledger bookings (floor
    // supply, pinned defect, halo relaxation outlet, eater) and the mock
    // vacuum resistivity's dissipation res_boost as a named design term --
    // what remains is the unintended loss. 'full': every measured term.
    const amrex::Real resid_booked =
        dw_total - (poynt_in + a.circuit_in + fluid_in_e + fluid_in_i) + wall_e + wall_i -
        eater_e - eater_i - d_ledger_floor_supply - d_ledger_pinned + d_ledger_halo_relax +
        res_boost + force_withheld;
    const amrex::Real resid_full =
        dw_total - (poynt_in + a.circuit_in + fluid_in_e + fluid_in_i) + wall_e + wall_i -
        eater_e - eater_i -
        (a.src[R::fcs_e] + a.src[R::fcs_ei]) + (a.src[R::relax_e] + a.src[R::relax_ei]) +
        a.theta_diss - faraday_defect + exchange - pw_pair +
        a.src[R::drain_ei] - a.newton[1] - a.newton[2] - floor_e - floor_ei -
        sync_ei - post_e - post_i - fluxw_e - fluxw_i;

    a.cum_resid_booked += resid_booked;
    a.cum_resid_full += resid_full;
    a.cum_poynt_in += poynt_in;
    a.cum_circuit_in += a.circuit_in;
    a.cum_res_field += a.res_field;
    a.cum_res_user += a.res_user;
    a.cum_res_hyper += a.res_hyper;
    a.cum_lorentz_withheld += lorentz_withheld;
    a.cum_force_withheld += force_withheld;
    a.cum_poynt_in_ext += -poynt_out_ext_total;
    a.cum_theta_diss_ext += a.theta_diss_ext;
    a.cum_ext_defect += ext_defect;
    a.cum_ideal_mismatch += ideal_mismatch;
    a.cum_stagger_mismatch += stagger_mismatch;
    a.cum_recon_work += recon_work;
    a.cum_corner_diss_work += a.corner_diss_work;
    a.cum_hall_work += a.hall_work;
    a.cum_inertia_work += a.inertia_work;
    a.cum_ohm_rest += ohm_rest;
    a.cum_fluid_in += fluid_in_e + fluid_in_i;
    a.cum_wall += wall_e + wall_i;
    a.cum_ej += a.ej;
    a.cum_theta_diss += a.theta_diss;
    a.cum_exchange += exchange;
    a.cum_pwpair += pw_pair;
    a.cum_faraday += faraday_defect;
    a.cum_newton += a.newton[1] + a.newton[2];
    a.cum_floor += floor_e + floor_ei;
    a.cum_sync += sync_ei;
    a.cum_eater += eater_e + eater_i;
    a.cum_inject += inject_e + inject_i;
    a.cum_fluxw += fluxw_e + fluxw_i;
    a.cum_dw_total += dw_total;

    const amrex::Real thermal = a.final_fluid.ue + (dual ? a.final_fluid.ui
                                                          : a.final_fluid.ei - a.final_fluid.ke);
    auto safe_frac = [] (const amrex::Real num, const amrex::Real den) {
        return den != 0.0_rt ? num / den : 0.0_rt;
    };

    // Ledger deltas as the code books them (cumulative counters).
    const amrex::Real ledger_wall = m_shaped_wall_energy;
    const amrex::Real ledger_absorb = m_absorbed_wall_energy;
    const amrex::Real ledger_friction = m_wall_friction_energy_dropped;
    const amrex::Real ledger_floor_supply = m_floor_supplied_energy;
    const amrex::Real ledger_pinned = m_pinned_defect_energy;
    const amrex::Real ledger_halo_relax = m_halo_relaxation_energy;
    const amrex::Real ledger_eater = m_eater_removed_energy;
    const amrex::Real ledger_pedestal_e = m_halo_pedestal_energy_e;
    const amrex::Real ledger_pedestal_i = m_halo_pedestal_energy_i;

    std::vector<std::pair<const char*, amrex::Real>> columns = {
        {"step", static_cast<amrex::Real>(step + 1)},
        {"time", end_time},
        {"dt", dt},
        // totals at t^{n+1}
        {"W_B_resp", a.wb_resp_new},
        {"W_B_tot", a.wb_total_new},
        {"W_B_ext", a.wb_ext_new},
        {"U_e", a.final_fluid.ue},
        {"E_i", a.final_fluid.ei},
        {"U_i", a.final_fluid.ui},
        {"KE", a.final_fluid.ke},
        {"mass", a.final_fluid.mass},
        {"U_e_bg", PedestalChangeOfVariables() ? m_pedestal_electron_energy * a.domain_volume : 0.0_rt},
        {"U_i_bg", PedestalChangeOfVariables() ? m_pedestal_ion_internal * a.domain_volume : 0.0_rt},
        // changes over the step
        {"dW_B_tot", dwb_total},
        {"dW_B_resp", dwb_resp},
        {"dW_B_ext", dwb_ext},
        {"dU_e", due},
        {"dE_i", dei},
        {"dU_i", dui},
        {"dKE", dke},
        {"dmass", dmass},
        {"dW_total", dw_total},
        // boundary fluxes (positive = into the domain)
        {"poynt_in_rhi", -a.poynt_out[0]},
        {"poynt_in_zlo", -a.poynt_out[1]},
        {"poynt_in_zhi", -a.poynt_out[2]},
        {"poynt_in", poynt_in},
        {"poynt_in_ext_rhi", -a.poynt_out_ext[0]},
        {"poynt_in_ext_zlo", -a.poynt_out_ext[1]},
        {"poynt_in_ext_zhi", -a.poynt_out_ext[2]},
        {"poynt_in_ext", -poynt_out_ext_total},
        {"circuit_in", a.circuit_in},
        {"circuit_in_ext", a.circuit_in_ext},
        {"circuit_in_plasma", a.circuit_in_plasma},
        {"fluid_in_mass_rhi", -a.export_domain[0][0]},
        {"fluid_in_mass_zlo", -a.export_domain[0][1]},
        {"fluid_in_mass_zhi", -a.export_domain[0][2]},
        {"fluid_in_e_rhi", -a.export_domain[1][0]},
        {"fluid_in_e_zlo", -a.export_domain[1][1]},
        {"fluid_in_e_zhi", -a.export_domain[1][2]},
        {"fluid_in_i_rhi", -a.export_domain[2][0]},
        {"fluid_in_i_zlo", -a.export_domain[2][1]},
        {"fluid_in_i_zhi", -a.export_domain[2][2]},
        {"fluid_in_ui", fluid_in_ui},
        // shaped-wall stair deposition (positive = into the wall)
        {"wall_mass", wall_mass},
        {"wall_e", wall_e},
        {"wall_i", wall_i},
        {"wall_ui", wall_ui},
        // field-fluid exchange and the volume sources as deposited
        {"EJ", a.ej},
        {"lorentz", a.src[R::lorentz]},
        {"lorentz_raw", a.src[R::lorentz_raw]},
        {"joule_e", a.src[R::joule_e]},
        {"joule_i", a.src[R::joule_i]},
        {"pw_e", a.src[R::pw_e]},
        {"pw_i", a.src[R::pw_i]},
        {"pw_pair", pw_pair},
        {"pdv_ui", a.src[R::pdv_ui]},
        {"visc_ui", a.src[R::visc_ui]},
        {"equil", a.src[R::equil]},
        {"drain_ei", a.src[R::drain_ei]},
        {"drain_ui", a.src[R::drain_ui]},
        {"cov_shift_pw_e", a.src[R::cov_shift_pw_e]},
        {"cov_shift_pdv_ui", a.src[R::cov_shift_pdv_ui]},
        {"relax_e", a.src[R::relax_e]},
        {"relax_i", a.src[R::relax_ei]},
        {"relax_ui", a.src[R::relax_ui]},
        {"fcs_mass", a.src[R::fcs_mass]},
        {"fcs_e", a.src[R::fcs_e]},
        {"fcs_i", a.src[R::fcs_ei]},
        {"fcs_ui", a.src[R::fcs_ui]},
        {"exchange", exchange},
        {"res_field", a.res_field},
        {"res_user", a.res_user},
        {"res_boost", res_boost},
        {"exchange_rest", exchange - a.res_field + a.src[R::joule_e] + a.src[R::joule_i]},
        {"res_hyper", a.res_hyper},
        {"lorentz_unweighted", a.src[R::lorentz_unweighted]},
        {"lorentz_withheld", lorentz_withheld},
        {"force_withheld", force_withheld},
        {"exchange_rest2", exchange_rest2},
        // the Ohm's-law component split of exchange_rest2
        {"ideal_edge_work", a.ind_work},
        {"ideal_cc_work", a.ind_cc_work},
        {"ideal_mismatch", ideal_mismatch},
        {"stagger_mismatch", stagger_mismatch},
        {"recon_work", recon_work},
        {"corner_diss_work", a.corner_diss_work},
        {"hall_work", a.hall_work},
        {"inertia_work", a.inertia_work},
        {"ohm_rest", ohm_rest},
        {"exchange_rest3", exchange_rest3},
        // scheme terms
        {"theta_diss", a.theta_diss},
        {"theta_diss_ext", a.theta_diss_ext},
        {"faraday_defect", faraday_defect},
        {"ext_defect", ext_defect},
        {"newton_mass", a.newton[0]},
        {"newton_e", a.newton[1]},
        {"newton_i", a.newton[2]},
        {"newton_ui", a.newton[3]},
        {"fluxw_mass", fluxw_mass},
        {"fluxw_e", fluxw_e},
        {"fluxw_i", fluxw_i},
        {"fluxw_ui", fluxw_ui},
        // end-of-step restorations (signed additions)
        {"eater_mass", eater_mass},
        {"eater_e", eater_e},
        {"eater_i", eater_i},
        {"eater_ui", eater_ui},
        {"floor_e", floor_e},
        {"floor_Ei", floor_ei},
        {"sync_Ei", sync_ei},
        {"sync_floor_Ui", sync_floor_ui},
        {"post_e", post_e},
        {"post_i", post_i},
        {"inject_mass", inject_mass},
        {"inject_e", inject_e},
        {"inject_i", inject_i},
        // the code's own ledgers: this step and cumulative
        {"ledger_wall_energy", d_ledger_wall},
        {"ledger_floor_supply", d_ledger_floor_supply},
        {"ledger_pinned_defect", d_ledger_pinned},
        {"ledger_halo_relax", d_ledger_halo_relax},
        {"ledger_eater_energy", d_ledger_eater},
        {"ledger_pedestal_e", d_ledger_pedestal_e},
        {"ledger_pedestal_i", d_ledger_pedestal_i},
        {"ledger_wall_energy_cum", ledger_wall},
        {"ledger_absorb_energy_cum", ledger_absorb},
        {"ledger_friction_cum", ledger_friction},
        {"ledger_floor_supply_cum", ledger_floor_supply},
        {"ledger_pinned_defect_cum", ledger_pinned},
        {"ledger_halo_relax_cum", ledger_halo_relax},
        {"ledger_eater_energy_cum", ledger_eater},
        {"ledger_pedestal_e_cum", ledger_pedestal_e},
        {"ledger_pedestal_i_cum", ledger_pedestal_i},
        // closure
        {"resid_booked", resid_booked},
        {"resid_full", resid_full},
        {"resid_booked_cum", a.cum_resid_booked},
        {"resid_full_cum", a.cum_resid_full},
        {"poynt_in_cum", a.cum_poynt_in},
        {"circuit_in_cum", a.cum_circuit_in},
        {"fluid_in_cum", a.cum_fluid_in},
        {"wall_cum", a.cum_wall},
        {"EJ_cum", a.cum_ej},
        {"theta_diss_cum", a.cum_theta_diss},
        {"exchange_cum", a.cum_exchange},
        {"res_field_cum", a.cum_res_field},
        {"res_user_cum", a.cum_res_user},
        {"res_hyper_cum", a.cum_res_hyper},
        {"lorentz_withheld_cum", a.cum_lorentz_withheld},
        {"force_withheld_cum", a.cum_force_withheld},
        {"poynt_in_ext_cum", a.cum_poynt_in_ext},
        {"theta_diss_ext_cum", a.cum_theta_diss_ext},
        {"ext_defect_cum", a.cum_ext_defect},
        {"ideal_mismatch_cum", a.cum_ideal_mismatch},
        {"stagger_mismatch_cum", a.cum_stagger_mismatch},
        {"recon_work_cum", a.cum_recon_work},
        {"corner_diss_work_cum", a.cum_corner_diss_work},
        {"hall_work_cum", a.cum_hall_work},
        {"inertia_work_cum", a.cum_inertia_work},
        {"ohm_rest_cum", a.cum_ohm_rest},
        {"pw_pair_cum", a.cum_pwpair},
        {"faraday_defect_cum", a.cum_faraday},
        {"newton_cum", a.cum_newton},
        {"floor_cum", a.cum_floor},
        {"sync_Ei_cum", a.cum_sync},
        {"eater_cum", a.cum_eater},
        {"inject_cum", a.cum_inject},
        {"fluxw_cum", a.cum_fluxw},
        {"dW_total_cum", a.cum_dw_total},
        {"frac_resid_booked_cum_of_poynt_in", safe_frac(a.cum_resid_booked, a.cum_poynt_in)},
        {"frac_resid_booked_cum_of_delivered", safe_frac(a.cum_resid_booked, a.cum_poynt_in + a.cum_circuit_in)},
        {"frac_resid_booked_cum_of_thermal", safe_frac(a.cum_resid_booked, thermal)},
        {"frac_resid_full_cum_of_thermal", safe_frac(a.cum_resid_full, thermal)},
    };

    amrex::Print().SetPrecision(10)
        << "MHD energy audit: step " << step + 1 << " dW_total [J] = " << dw_total
        << " poynt_in = " << poynt_in << " circuit_in = " << a.circuit_in
        << " fluid_in = " << fluid_in_e + fluid_in_i
        << " wall = " << wall_e + wall_i << " EJ = " << a.ej
        << " theta_diss = " << a.theta_diss << " exchange = " << exchange
        << " res_boost = " << res_boost << " ext_defect = " << ext_defect
        << " sync_Ei = " << sync_ei << " resid_booked = " << resid_booked
        << " resid_full = " << resid_full
        << " | cum booked = " << a.cum_resid_booked
        << " full = " << a.cum_resid_full << "\n";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream file(m_energy_audit_file,
                           m_energy_audit_started ? std::ios::app : std::ios::trunc);
        if (!m_energy_audit_started) {
            file.precision(17);
            file << "# ThetaImplicitMHD global energy audit: one row per audited step "
                    "(implicit_mhd.energy_audit_interval = " << m_energy_audit_interval
                 << "); all energies in J"
#if defined(WARPX_DIM_RZ)
                    " (full 2 pi volumes)"
#else
                    " (per unit cross-section)"
#endif
                    ", mass in kg; theta = " << m_theta
                 << "; ion_closure = " << m_ion_closure
                 << "; domain volume = " << a.domain_volume << "\n";
            file << "# Weights: the field sums use the FieldEnergy dual volumes (2 pi r dr dz; pi dr^2/4 dz on "
                    "the axis; half weights at the nodal ends), which total"
#if defined(WARPX_DIM_RZ)
                    " E_r " << a.dual_volume_total[0] << ", E_theta " << a.dual_volume_total[1]
                 << ", E_z " << a.dual_volume_total[2]
                 << " (a nodal-in-r staggering sums to the domain volume + pi dr^2 L_z/4, a cell-centered-in-r "
                    "one to the domain volume exactly); the fluid sums use the cell volumes (= the domain volume)\n";
#else
                    " E_x " << a.dual_volume_total[0] << ", E_y " << a.dual_volume_total[1]
                 << ", E_z " << a.dual_volume_total[2] << " (all = the domain length)\n";
#endif
            file << "# Conventions: totals are at the end of the step; d* are changes over the step; "
                    "poynt_in_* / fluid_in_* are positive INTO the domain (Poynting = E_ohm^theta x "
                    "B_tot^theta with the external inductive E and B included, the exact summation-by-parts "
                    "face term of the Yee curl pair); circuit_in = -dt sum E_ohm^theta . J_coil with J_coil = "
                    "curl B_ext^theta/mu0 (the coil current sheet inside the domain: the circuit's electromagnetic "
                    "power into the domain; _ext / _plasma = the parts against E_ext and against the plasma-response "
                    "E); wall_* are positive into the shaped wall; EJ = dt sum "
                    "E_ohm^theta . J^theta (the field's loss to the fluid); the source columns are dt x the "
                    "RHS volume sources as deposited; theta_diss = (theta - 1/2) sum |dB_tot|^2/mu0; "
                    "faraday_defect = sum B^theta . dB/mu0 + dt (poynt_out + EJ) - circuit_in; res_field = dt sum eta_used "
                    "J_stage . J^theta (the field's resistive dissipation with the resistivity the advance used: vacuum "
                    "boost + wall-band override), res_user = the same with the un-boosted user eta, res_boost = res_field - "
                    "res_user (the mock resistivity's cost), exchange_rest = exchange - res_field + joule_e + joule_i (the "
                    "non-resistive exchange mismatch; res_hyper = dt sum E_H . J^theta the hyper-resistive "
                    "dissipation, lorentz_withheld = the gated magnetic-force work the Holmstrom vacuum switch "
                    "withholds, force_withheld = the magnetic-force work the physical-share weight "
                    "(lorentz_force_current = physical) / band-cells mask (lorentz_force_band_cells) withholds "
                    "from the fluid rows (wall_live x (1 - w) x u . (j_plasma x B), ungated; zero when off), "
                    "exchange_rest2 = exchange_rest - res_hyper - lorentz_withheld - force_withheld the "
                    "remainder, decomposed by the Ohm's-law components: exchange_rest2 = ideal_mismatch + "
                    "hall_work + inertia_work + ohm_rest EXACTLY, where ideal_edge_work = dt sum E_ind . J^theta "
                    "with E_ind the induction (ideal) EMF as assembled (the face Riemann/central induction flux "
                    "for E_r and E_z, the UCT corner average + dissipation for E_theta), ideal_cc_work = the same "
                    "with the plain two-cell (four-cell at a corner) mean of the cell-centered -(u x B), "
                    "ideal_mismatch = ideal_edge_work - lorentz_unweighted - force_withheld (the edge ideal work vs the "
                    "cell-centered stress work the momentum kernel would deposit unweighted: zero in the collocated 1D pairing, "
                    "the RZ pairing defect), stagger_mismatch = ideal_cc_work - lorentz_unweighted - force_withheld (its "
                    "plain-mean part: staggering, r-weights, axis), recon_work = ideal_edge_work - ideal_cc_work "
                    "- corner_diss_work (reconstruction and UCT upwind weighting), corner_diss_work = the UCT "
                    "corner dissipation's work (zero under the central flux), hall_work = dt sum E_Hall . J^theta, "
                    "inertia_work = dt sum E_inertia . J^theta, ohm_rest = EJ - (ideal_edge_work + hall_work + "
                    "inertia_work + res_field + res_hyper) = the work of the post-assembly boundary projections "
                    "(domain field BCs, wall projection; zero when none touches an edge with J != 0), "
                    "exchange_rest3 = exchange_rest2 - (ideal_mismatch + hall_work + inertia_work + ohm_rest) "
                    "identically zero -- the column arithmetic's own check); the recast Ohm's law has no grad p_e "
                    "term (the electron pressure acts through the fluid work pair), so there is no gradpe_work; "
                    "poynt_in_ext*, theta_diss_ext, ext_defect = the external field's own identity "
                    "(dW_B_ext = circuit_in_ext - poynt_out_ext - theta_diss_ext + ext_defect; exact for a "
                    "segment-driven or linear-in-time scale); theta_diss includes theta_diss_ext (the prescribed "
                    "field's quadrature term, not a scheme loss); inject_* = the between-step pedestal raise, "
                    "reported, in neither residual (the energy since the first row is dW_total_cum + inject_cum); "
                    "res_boost is booked in resid_booked as the mock vacuum resistivity's DISSIPATION (a design "
                    "term), force_withheld likewise as the withheld mock-current force work (a design term); exchange = EJ - (lorentz + "
                    "joule_e + joule_i); newton_* = sum (U^theta - U^n - rhs) dV/theta; fluxw_* = sum rhs "
                    "dV/theta - dt(-export - wall + sources); eater_*/floor_*/sync_* are the end-of-step "
                    "restorations (signed additions); inject_* = between-step changes.\n";
            file << "# resid_booked = dW_total - (poynt_in + circuit_in + fluid_in_e + fluid_in_i) + wall_e + wall_i - "
                    "eater_e - eater_i - ledger_floor_supply - ledger_pinned_defect + ledger_halo_relax + res_boost "
                    "+ force_withheld   (the code's own bookings + the mock dissipation and the withheld mock-force "
                    "work as design terms)\n";
            file << "# resid_full = dW_total - (poynt_in + circuit_in + fluid_in_e + fluid_in_i) + wall_e + wall_i - "
                    "eater_e - eater_i - (fcs_e + fcs_i) + (relax_e + relax_i) + "
                    "theta_diss - faraday_defect + exchange - pw_pair + drain_ei - newton_e - newton_i - "
                    "floor_e - floor_Ei - sync_Ei - post_e - post_i - fluxw_e - fluxw_i   (every measured term)\n";
            file << "# columns:";
            for (const auto& column : columns) { file << " " << column.first; }
            file << "\n";
        }
        m_energy_audit_started = true;
        file.precision(17);
        for (const auto& column : columns) { file << column.second << " "; }
        file << "\n";
    }

    a.have_prev_final = true;
    a.prev_step = step;
    a.prev_final = a.final_fluid;
}
