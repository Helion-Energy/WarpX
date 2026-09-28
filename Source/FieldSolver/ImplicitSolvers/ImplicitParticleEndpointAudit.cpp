/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "ImplicitParticleEndpointAudit.H"
#include "ImplicitSolver.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"
#include <algorithm>
#include <sstream>

namespace warpx::implicit {
void RestrictAxialGatherToProduced (WarpX& simulation, int grid,
                                   EndpointTileContract& contract)
{
#ifdef WARPX_DIM_RZ
    // This private owner already requires shape3 MC and order-2 centering.
    // Keep that precondition explicit here rather than infer a new stencil.
    bool const supported=WarpX::nox==3 &&
        WarpX::field_gathering_algo==GatheringAlgo::MomentumConserving &&
        WarpX::field_centering_nox==2 && WarpX::field_centering_noz==2;
    auto const& geom=simulation.Geom(0);
    int const z=1;
    // Exact width selected by Theta's DarwinParticleCurlGrow for the fixed
    // private stencil: max_d(shape_order/2 + centering_order[d]/2).
    int const curl_grow=std::max(1,WarpX::nox/2+
        std::max(WarpX::field_centering_nox/2,WarpX::field_centering_noz/2));
    for(int family=0;family<2;++family)for(int component=0;component<3;++component){
        auto& band=contract.gather[3*family+component];
        if(!supported){band.high[z]=band.low[z]-1;continue;}
        auto const type=family==0?warpx::fields::FieldType::Efield_fp:
                                   warpx::fields::FieldType::Bfield_fp;
        auto const& source=*simulation.m_fields.get(type,
            ablastr::fields::Direction{component},0);
        int const nodal=source.ixType().nodeCentered(z);
        // E's PMC completion fills ng_fieldgather. B's plasma contribution
        // is reset then filled only on CurlEvaluationBox; FillBoundary cannot
        // produce a physical ghost beyond that domain-clipped curl band.
        int const width=family==0?simulation.get_ng_fieldgather()[z]:curl_grow;
        int low=std::max(source[grid].box().smallEnd(z),geom.Domain().smallEnd(z)-width);
        int high=std::min(source[grid].box().bigEnd(z),geom.Domain().bigEnd(z)+nodal+width);
        // warpx_interp to nodal: a cell source for output j consumes
        // [j-order/2,j+order/2-1]; a nodal source is copied at j.
        if(!nodal){low+=WarpX::field_centering_noz/2;high-=WarpX::field_centering_noz/2-1;}
        band.low[z]=std::max(band.low[z],low);
        band.high[z]=std::min(band.high[z],high);
        band.origin[z]=geom.ProbLo(z)+band.low[z]*geom.CellSize(z);
    }
#else
    amrex::ignore_unused(simulation,grid);
    for(auto& band:contract.gather)band.high[0]=band.low[0]-1;
#endif
}

EndpointTileContract MakeNativeAxialCurrentContract(WarpX& simulation,
    amrex::Box const& particle_tile,int grid)
{
    EndpointAuditFields fields;
    fields.endpoint_density=simulation.m_fields.get(warpx::fields::FieldType::rho_fp,0);
    fields.gather_filled_ghosts=simulation.get_ng_fieldgather();
    fields.current_deposit_ghosts=simulation.get_ng_depos_J();
    for(int c=0;c<3;++c){auto const d=ablastr::fields::Direction{c};
        fields.current[c]=simulation.m_fields.get(warpx::fields::FieldType::current_fp,d,0);
        fields.gather_e[c]=simulation.m_fields.get(warpx::fields::FieldType::Efield_aux,d,0);
        fields.gather_b[c]=simulation.m_fields.get(warpx::fields::FieldType::Bfield_aux,d,0);}
    auto out=MakeEndpointTileContract(simulation.Geom(0),particle_tile,grid,fields,
        WarpX::particle_boundary_lo,WarpX::particle_boundary_hi);
    RestrictAxialGatherToProduced(simulation,grid,out);
    return out;
}

std::string
EndpointAuditResult::summary () const {
    static constexpr std::array<const char*, endpoint_issue_count> names = {
        "nonfinite",
        "invalid_identity",
        "old_outside_domain",
        "absorbing_boundary",
        "thermal_boundary",
        "other_boundary_action",
        "unresolved_radial_wall",
        "gather_reach",
        "current_reach",
        "endpoint_reach",
        "projection_reach",
        "supported_absorption",
        "supported_reflection"};
    std::ostringstream s;
    s << "particles=" << particles;
    for (int i = 0; i < endpoint_issue_count; ++i) {
        if (counts[i]) {
            s << " " << names[i] << "=" << counts[i];
        }
    }
    return s.str();
}
EndpointAuditResult
AuditImplicitParticleEndpoints (WarpX& simulation, int lev,
                                EndpointAuditFields const& fields) {
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        lev == 0 && simulation.finestLevel() == 0 && WarpX::nox == 3 &&
            WarpX::current_deposition_algo ==
                CurrentDepositionAlgo::Esirkepov &&
            WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving &&
            !simulation.getdo_moving_window() &&
            simulation.m_v_galilean[0] == 0 &&
            simulation.m_v_galilean[1] == 0 && simulation.m_v_galilean[2] == 0,
        "Endpoint audit requires static single-level shape3 Esirkepov/MC "
        "fields");
    auto const& geom = simulation.Geom(lev);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(geom.Domain().smallEnd() ==
                                         amrex::IntVect(0),
                                     "Endpoint audit requires the native "
                                     "zero-origin WarpX mesh index domain");
#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        geom.ProbLo(0) == 0 && WarpX::n_rz_azimuthal_modes == 1,
        "Endpoint audit requires the axis-inclusive m=0 RZ domain");
#endif
    auto validate = [&] (const amrex::MultiFab* f, amrex::IntVect const& ghosts,
                         amrex::IntVect const& type) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f && f->ixType().toIntVect() == type &&
                f->boxArray() ==
                    amrex::convert(simulation.boxArray(lev), type) &&
                f->DistributionMap() == simulation.DistributionMap(lev) &&
                ghosts.allGE(amrex::IntVect(0)) && f->nGrowVect().allGE(ghosts),
            "Endpoint audit field layout or qualified ghost extent is "
            "inconsistent");
    };
    auto const nodal = amrex::IntVect::TheNodeVector();
    validate(fields.endpoint_density, amrex::IntVect(0), nodal);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        simulation.get_ng_fieldgather().allGE(fields.gather_filled_ghosts) &&
            simulation.get_ng_depos_J().allGE(fields.current_deposit_ghosts),
        "Endpoint audit ghost contract exceeds the native "
        "communicated/deposited extent");
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bool const periodic = geom.isPeriodic(d);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (WarpX::particle_boundary_lo[d] ==
             ParticleBoundaryType::Periodic) == periodic &&
                (WarpX::particle_boundary_hi[d] ==
                 ParticleBoundaryType::Periodic) == periodic,
            "Endpoint audit requires consistent particle/geometry periodicity");
    }
    for (int component = 0; component < 3; ++component) {
        auto type = nodal;
#ifdef WARPX_DIM_RZ
        if (component != 1) {
            type[component / 2] = 0;
        }
#else
        type[component] = 0;
#endif
        validate(fields.current[component], fields.current_deposit_ghosts,
                 type);
        validate(fields.gather_e[component], fields.gather_filled_ghosts,
                 nodal);
        validate(fields.gather_b[component], fields.gather_filled_ghosts,
                 nodal);
    }
    EndpointAuditResult result;
    amrex::Gpu::synchronize();
    for (auto const& pc : simulation.GetPartContainer()) {
        for (WarpXParIter it(*pc, lev); it.isValid(); ++it) {
            ParticleStartView start;
            start.position = {it.GetAttribs("x_n").dataPtr(),
                              it.GetAttribs("y_n").dataPtr(),
                              it.GetAttribs("z_n").dataPtr()};
            start.momentum = {it.GetAttribs("ux_n").dataPtr(),
                              it.GetAttribs("uy_n").dataPtr(),
                              it.GetAttribs("uz_n").dataPtr()};
            auto c = MakeEndpointTileContract(
                geom, it.tilebox(), it.index(), fields,
                WarpX::particle_boundary_lo, WarpX::particle_boundary_hi);
            if(fields.allow_axial_reflection)RestrictAxialGatherToProduced(simulation,it.index(),c);
            c.allow_deterministic_absorption=fields.allow_deterministic_absorption;
            if (c.allow_deterministic_absorption) {
                c.absorption=MakeAbsorbingEndpointBoundary(geom,pc->GetParticleBoundaryData(),
                    WarpX::field_boundary_lo,WarpX::field_boundary_hi,
                    fields.allow_radial_reflection && simulation.get_pointer_ImplicitSolver() &&
                    !simulation.get_pointer_ImplicitSolver()->ReflectsParticlesInsidePush());
            }
            c.axial=MakeAxialEndpointBoundary(geom,pc->GetParticleBoundaryData(),
                WarpX::field_boundary_lo,WarpX::field_boundary_hi,
                fields.allow_axial_reflection && simulation.get_pointer_ImplicitSolver() &&
                !simulation.get_pointer_ImplicitSolver()->ReflectsParticlesInsidePush());
            c.deposits = !pc->do_not_deposit && pc->getCharge() != 0;
            // Conservatively qualify the gather range for every kinetic
            // species.
            c.gathers = true;
            auto const local = AuditEndpointTile(
                it.numParticles(), GetParticlePosition(it), start,
                {it.GetAttribs(PIdx::ux).dataPtr(),
                 it.GetAttribs(PIdx::uy).dataPtr(),
                 it.GetAttribs(PIdx::uz).dataPtr()},
                it.GetStructOfArrays().GetIdCPUData().data(), c);
            result.particles += local.particles;
            for (int j = 0; j < endpoint_issue_count; ++j) {
                result.counts[j] += local.counts[j];
            }
        }
    }
    result.reduceAcrossRanks();
    return result;
#else
    amrex::ignore_unused(simulation, lev, fields);
    WARPX_ABORT_WITH_MESSAGE(
        "Endpoint audit is qualified only in RZ and Cartesian 3D");
    return {};
#endif
}
} // namespace warpx::implicit
