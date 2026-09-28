/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SOLVER_LAYOUT_FIXTURE_H_
#define WARPX_SOLVER_LAYOUT_FIXTURE_H_

// Standalone fixture facade only. The tests compile the production
// solver-vector, DOF and field-register sources. This replaces only the full
// application's geometry/registry accessor; it does not replace any vector or
// DOF operation. Production translation units must also be compiled with the
// real WarpX.H.
#include "Fields.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <map>
#include <memory>
#include <string>

class WarpX
{
  public:
    ablastr::fields::MultiFabRegister m_fields;
    amrex::Geometry geometry;
    mutable std::map<std::string, std::unique_ptr<amrex::iMultiFab>> masks;

    const amrex::Geometry&
    Geom (int) const
    {
        return geometry;
    }

    const amrex::iMultiFab*
    getFieldDotMaskPointer (warpx::fields::FieldType type, int lev,
                            ablastr::fields::Direction dir) const
    {
        using warpx::fields::FieldType;
        const auto key = amrex::getEnumNameString(type) + std::to_string(int(dir));
        auto& mask = masks[key];
        if (!mask) {
            const auto* field =
                type == FieldType::phi_fp ? m_fields.get(type, lev) : m_fields.get(type, dir, lev);
            mask = field->OwnerMask(geometry.periodicity());
        }
        return mask.get();
    }
};
#endif
