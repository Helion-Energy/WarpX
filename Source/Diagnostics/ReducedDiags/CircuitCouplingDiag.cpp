/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CircuitCouplingDiag.H"

#include "Circuit/CircuitCoupling.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <fstream>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

CircuitCouplingDiag::CircuitCouplingDiag (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
#if !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "the CircuitCoupling reduced diagnostic is implemented for RZ "
        "geometry");
#endif

    // Sized from the inputs (the WarpX instance may not exist yet).
    const amrex::ParmParse pp_circuit("circuit");
    std::vector<std::string> coil_names;
    pp_circuit.queryarr("coils", coil_names);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!coil_names.empty(),
        "the CircuitCoupling reduced diagnostic requires circuit.coils");

    const auto n = static_cast<int>(coil_names.size());
    m_data.resize(3 * static_cast<std::size_t>(n) + 2, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
                          std::ofstream::out};
        int c = 0;
        ofs << "#";
        ofs << "[" << c++ << "]step()";
        ofs << m_sep << "[" << c++ << "]time(s)";
        for (const auto& name : coil_names) {
            ofs << m_sep << "[" << c++ << "]s_" << name << "()";
            ofs << m_sep << "[" << c++ << "]dsdt_" << name << "(1/s)";
            ofs << m_sep << "[" << c++ << "]lambda_" << name << "(Wb*A)";
        }
        ofs << m_sep << "[" << c++ << "]P_circuit(W)";
        ofs << m_sep << "[" << c++ << "]P_field(W)";
        ofs << "\n";
        ofs.close();
    }
}

void
CircuitCouplingDiag::ComputeDiags (const int step)
{
    if (!DoDiags(step)) { return; }
#if defined(WARPX_DIM_RZ)
    using namespace warpx::circuit;
    auto& warpx = WarpX::GetInstance();

    auto* manager = warpx.get_pointer_CircuitCoupling();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(manager != nullptr,
        "the CircuitCoupling reduced diagnostic requires circuit.coils");
    auto* hybrid = warpx.get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        hybrid != nullptr && hybrid->m_add_external_fields,
        "the CircuitCoupling reduced diagnostic requires the hybrid solver "
        "with external fields");
    auto& ext = *hybrid->m_external_vector_potential;
    const CircuitCoupler* coupler = manager->Coupler();

    const amrex::Real t_new = warpx.gett_new(0);
    const amrex::Real dt = warpx.getdt(0);
    const amrex::Real t_old = t_new - dt;

    const CoilSet& coils = manager->GetCoilSet();
    int idx = 0;
    amrex::Real p_circuit = 0.0_rt;
    for (int ic = 0; ic < coils.size(); ++ic) {
        const Coil& c = coils.coil(ic);
        const amrex::Real s_new = ext.GetScale(c.field_name, t_new);
        const amrex::Real s_old = ext.GetScale(c.field_name, t_old);
        const amrex::Real dsdt = (s_new - s_old) / dt;
        const amrex::Real lam =
            coupler ? coupler->CoilLinkageOr(c.name, 0.0_rt) : 0.0_rt;
        m_data[idx++] = s_new;
        m_data[idx++] = dsdt;
        m_data[idx++] = lam;
        p_circuit -= dsdt * lam;
    }
    m_data[idx++] = p_circuit;

    // Field side of the double entry: Int J_plasma . E_ext dV on the
    // plasma current of the step's final Ohm solve.
    const std::array<const amrex::MultiFab*, 3> J = {
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{0}, 0),
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{1}, 0),
        warpx.m_fields.get(FieldType::hybrid_current_fp_plasma,
                           ablastr::fields::Direction{2}, 0)};
    const std::array<const amrex::MultiFab*, 3> E = {
        warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                           ablastr::fields::Direction{0}, 0),
        warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                           ablastr::fields::Direction{1}, 0),
        warpx.m_fields.get(FieldType::hybrid_E_fp_external,
                           ablastr::fields::Direction{2}, 0)};
    m_data[idx++] = CouplingPowerIntegral(J, E);
#else
    amrex::ignore_unused(step);
#endif
}
