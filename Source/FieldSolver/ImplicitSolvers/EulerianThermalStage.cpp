/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "EulerianThermalStage.H"
#include "ThermalCurrentRemainder.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConduction.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalConduction3D.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalWall.H"
#include "Utils/WarpXConst.H"
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "EulerianThermalStageUtils.H"

namespace warpx::thermal {
using amrex::Real;
using stage_detail::check_layout;
using stage_detail::fill_images;
using stage_detail::finite;

EulerianThermalStage::EulerianThermalStage (
    const amrex::Geometry& geometry, const amrex::MultiFab& old_energy,
    const amrex::MultiFab& number_density,
    const amrex::MultiFab& magnetic_field, const amrex::MultiFab& source,
    ThermalStageOptions options)
    : EulerianThermalStage(geometry, old_energy, number_density, number_density,
                           number_density, magnetic_field, source, {},
                           std::move(options)) {}

EulerianThermalStage::EulerianThermalStage (
    const amrex::Geometry& geometry, const amrex::MultiFab& old_energy,
    const amrex::MultiFab& old_density, const amrex::MultiFab& stage_density,
    const amrex::MultiFab& endpoint_density,
    const amrex::MultiFab& magnetic_field, const amrex::MultiFab& source,
    ThermalFaceContext faces, ThermalStageOptions options)
    : m_lifetime(options.lifetime ? options.lifetime : std::make_shared<ThermalStageLifetime>()),
      m_instance(m_lifetime->NewInstance()),
      m_geometry(geometry), m_options(std::move(options)),
      m_old(old_energy.boxArray(), old_energy.DistributionMap(), 1, 3),
      m_old_density(old_energy.boxArray(), old_energy.DistributionMap(), 1, 3),
      m_density(old_energy.boxArray(), old_energy.DistributionMap(), 1, 3),
      m_endpoint_density(old_energy.boxArray(), old_energy.DistributionMap(), 1,
                         0),
      m_magnetic(old_energy.boxArray(), old_energy.DistributionMap(), 3, 3),
      m_source(old_energy.boxArray(), old_energy.DistributionMap(), 1, 0),
      m_stage(old_energy.boxArray(), old_energy.DistributionMap(), 1, 3),
      m_kappa(old_energy.boxArray(), old_energy.DistributionMap(), 2, 3),
      m_live_source(old_energy.boxArray(), old_energy.DistributionMap(), 1, 0),
      m_source_derivative(old_energy.boxArray(), old_energy.DistributionMap(),
                          1, 0),
      m_conduction_active(old_energy.boxArray(), old_energy.DistributionMap(),
                          1, 3),
      m_parallel_parser(m_options.kappa_parallel.c_str()),
      m_perpendicular_parser(m_options.kappa_perpendicular.c_str()) {
#if !defined(WARPX_DIM_RZ) && !defined(WARPX_DIM_3D)
    amrex::Abort("EulerianThermalStage requires RZ or Cartesian 3D geometry");
#endif
    check_layout(old_energy, old_energy, 1);
    check_layout(old_density, old_energy, 1);
    check_layout(magnetic_field, old_energy, 3);
    check_layout(source, old_energy, 1);
    auto const& o = m_options;
    bool const valid_options =
        std::isfinite(o.dt) && o.dt > 0 && std::isfinite(o.theta) &&
        o.theta >= 0.5 && o.theta <= 1 && std::isfinite(o.conduction_theta) &&
        o.conduction_theta >= 0.5 && o.conduction_theta <= 1 &&
        std::isfinite(o.gamma) && o.gamma > 1 &&
        std::isfinite(o.temperature_floor_ev) && o.temperature_floor_ev >= 0 &&
        std::isfinite(o.magnetic_floor) && o.magnetic_floor > 0 &&
        std::isfinite(o.magnetic_floor * o.magnetic_floor) &&
        o.magnetic_floor * o.magnetic_floor > 0 && std::isfinite(o.time) &&
        std::isfinite(o.limiter_width) && o.limiter_width > 0 &&
        std::isfinite(o.free_streaming_fraction) &&
        o.free_streaming_fraction >= 0 &&
        std::isfinite(o.conductivity.isotropic_B) &&
        (o.conductivity.isotropic_B <= 0 ||
         (std::isfinite(o.conductivity.isotropic_B *
                        o.conductivity.isotropic_B) &&
          o.conductivity.isotropic_B * o.conductivity.isotropic_B > 0)) &&
        (o.order == 2 || o.order == 4) &&
        (o.cross_mode == 0 || o.cross_mode == 1) &&
        o.transport.fluid_reconstruction >= 0 &&
        o.transport.fluid_reconstruction <= 5 &&
        std::isfinite(o.transport.reconstruction_kappa) &&
        o.transport.reconstruction_kappa > 0 &&
        std::isfinite(o.transport.density_floor) &&
        o.transport.density_floor >= 0 &&
        std::isfinite(o.transport.central_dissipation_entropy) &&
        o.transport.central_dissipation_entropy >= 0 &&
        o.transport.central_dissipation_entropy <= 1;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        valid_options, "Invalid Eulerian thermal stage parameters");
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 1 && !geometry.isPeriodic(0) &&
            geometry.ProbLo(0) >= 0 &&
            geometry.Domain().smallEnd() == amrex::IntVect(0) &&
            old_energy.boxArray().minimalBox() == geometry.Domain() &&
            old_energy.boxArray().numPts() == geometry.Domain().numPts(),
        "Eulerian thermal stage requires a complete zero-indexed RZ domain");
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.Coord() == 0 &&
            geometry.Domain().smallEnd() == amrex::IntVect(0) &&
            old_energy.boxArray().minimalBox() == geometry.Domain() &&
            old_energy.boxArray().numPts() == geometry.Domain().numPts(),
        "Eulerian thermal stage requires a complete zero-indexed Cartesian "
        "domain");
#endif
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int side = 0; side < 2; ++side) {
            auto const& bc = o.boundary[d][side];
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                bc.kind != BoundaryKind::LegUnsupported,
                "Eulerian thermal leg/source coupling has not been derived; "
                "unsupported");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                bc.kind == BoundaryKind::Adiabatic ||
                    bc.kind == BoundaryKind::PrescribedFlux ||
                    bc.kind == BoundaryKind::Reservoir ||
                    bc.kind == BoundaryKind::Leg,
                "Unknown Eulerian thermal boundary kind");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                (bc.flux_limit == 0 && !bc.drain_only) ||
                    bc.kind == BoundaryKind::Reservoir ||
                    bc.kind == BoundaryKind::Leg,
                "Wall caps and drain gates require a reservoir or physical "
                "leg");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(bc.value),
                                             "Nonfinite thermal BC");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                !geometry.isPeriodic(d) || bc.kind == BoundaryKind::Adiabatic,
                "A periodic thermal direction cannot also carry a physical "
                "wall law");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                (bc.kind != BoundaryKind::Reservoir &&
                 bc.kind != BoundaryKind::Leg) ||
                    bc.value >= o.temperature_floor_ev,
                "Reservoir temperature must be at least the endpoint "
                "floor");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(bc.leg_length) && bc.leg_length > 0 &&
                    std::isfinite(bc.flux_limit) && bc.flux_limit >= 0 &&
                    std::isfinite(bc.inflow_temperature_ev) &&
                    (bc.inflow_temperature_ev < 0 ||
                     bc.inflow_temperature_ev >= o.temperature_floor_ev) &&
                    (bc.cap_form == WallCapForm::FreeStreaming ||
                     bc.cap_form == WallCapForm::Sonic) &&
                    (bc.flux_limit == 0 || bc.cap_form != WallCapForm::Sonic ||
                     (std::isfinite(bc.ion_mass) && bc.ion_mass > 0)),
                "Invalid thermal physical wall/leg parameters");
        }
        auto const face_boxes = amrex::convert(
            old_energy.boxArray(), amrex::IntVect::TheDimensionVector(d));
        m_flux[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
        m_coefficient[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
        m_velocity[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
        m_speed[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
        m_advection[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
        m_wall_derivative[d] = std::make_unique<amrex::MultiFab>(
            face_boxes, old_energy.DistributionMap(), 1, 0);
    }
#if defined(WARPX_DIM_RZ)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.ProbLo(0) != 0 ||
            o.boundary[0][0].kind == BoundaryKind::Adiabatic,
        "The RZ axis has zero physical area and must be adiabatic");
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        finite(old_energy) && finite(old_density) && old_density.min(0) > 0,
        "Thermal old context requires finite fields and positive n");
    amrex::MultiFab::Copy(m_old, old_energy, 0, 0, 1, 0);
    amrex::MultiFab::Copy(m_old_density, old_density, 0, 0, 1, 0);
    fill_images(m_old, m_geometry);
    fill_images(m_old_density, m_geometry);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        RefreshContext(old_density, old_density, magnetic_field, source),
        "Thermal context requires finite fields and positive n");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Admissible(m_old),
                                     "Step-old U violates thermal floor");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !faces.number_flux[0] || (o.transport.fluid_reconstruction == 0 &&
                                  o.transport.central_dissipation_entropy == 0),
        "Conserved electron number-flux thermal transport requires "
        "reconstruction=none and penalty=0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        RefreshContext(stage_density, endpoint_density, magnetic_field, source,
                       faces),
        "Thermal context requires finite fields and positive n/valid face "
        "velocity");
    m_parallel_parser.registerVariables({"n", "Te", "t"});
    m_perpendicular_parser.registerVariables({"n", "Te", "t"});
    if (o.use_compiled_conductivity) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            o.parallel_executor && o.perpendicular_executor,
            "Borrowed thermal conductivity requires two valid executors");
        m_parallel = o.parallel_executor;
        m_perpendicular = o.perpendicular_executor;
    } else {
        m_parallel = m_parallel_parser.compile<3>();
        m_perpendicular = m_perpendicular_parser.compile<3>();
    }
}

EulerianThermalStage::~EulerianThermalStage() {
    m_lifetime->Invalidate(m_instance);
}
bool EulerianThermalStage::InvalidateEvaluation() noexcept {
    return m_lifetime->Invalidate(m_instance);
}
bool EulerianThermalStage::CompleteEvaluation() noexcept {
    if(!m_instance || m_lifetime->m_instance!=m_instance || m_lifetime->m_exhausted)return false;
    m_lifetime->m_complete=true;return true;
}
ThermalStageReceipt EulerianThermalStage::CompletedReceipt() const noexcept {
    return {m_lifetime,m_instance,m_lifetime->m_evaluation};
}

bool
EulerianThermalStage::RefreshContext (const amrex::MultiFab& stage_density,
                                      const amrex::MultiFab& endpoint_density,
                                      const amrex::MultiFab& magnetic_field,
                                      const amrex::MultiFab& source,
                                      ThermalFaceContext faces) {
    if(!InvalidateEvaluation())return false;
    bool paired=false;
    if(m_options.retain_current_remainder) {
        int count=0;for(auto* p:faces.number_flux_remainder)count+=p!=nullptr;
        int minimum=count,maximum=count;amrex::ParallelDescriptor::ReduceIntMin(minimum);amrex::ParallelDescriptor::ReduceIntMax(maximum);
        if(minimum!=maximum || (count!=0 && count!=AMREX_SPACEDIM))return false;
        paired=count!=0;
    } else {
        for(auto const* low:faces.number_flux_remainder)AMREX_ALWAYS_ASSERT(low==nullptr);
    }
    if(paired) {
        bool valid=remainder::ArithmeticSupported() && m_options.transport.fluid_reconstruction==0 && m_options.transport.central_dissipation_entropy==0;
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            valid=valid && faces.number_flux[d] && !faces.velocity[d] && !faces.dissipation_speed[d];
            if(faces.number_flux[d])valid=valid && remainder::Layout(*faces.number_flux[d],*m_velocity[d]) &&
                remainder::Layout(*faces.number_flux_remainder[d],*m_velocity[d]) &&
                !remainder::Overlap(*faces.number_flux_remainder[d],*faces.number_flux[d]);
        }
        if(!remainder::All(valid))return false;
        for(int d=0;d<AMREX_SPACEDIM;++d)if(!finite(*faces.number_flux_remainder[d]))return false;
    }
    check_layout(stage_density, m_old, 1);
    check_layout(endpoint_density, m_old, 1);
    check_layout(magnetic_field, m_old, 3);
    check_layout(source, m_old, 1);
    if (!finite(stage_density) || !finite(endpoint_density) ||
        !finite(magnetic_field) || !finite(source) ||
        stage_density.min(0) <= 0 || endpoint_density.min(0) <= 0) {
        return false;
    }
    if (faces.conduction_active) {
        auto const& mask = *faces.conduction_active;
        if (mask.nComp() != 1 || mask.hasEBFabFactory() ||
            mask.boxArray() != m_old.boxArray() ||
            mask.DistributionMap() != m_old.DistributionMap() ||
            !finite(mask)) {
            return false;
        }
        amrex::ReduceOps<amrex::ReduceOpMin> op;
        amrex::ReduceData<int> data(op);
        using Tuple = decltype(data)::Type;
        for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
            auto const a = mask.const_array(mfi);
            op.eval(mfi.validbox(), data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                        return {a(i, j, k) == 0 || a(i, j, k) == 1};
                    });
        }
        int valid = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceIntMin(valid);
        if (!valid) {
            return false;
        }
    }
    bool const velocities = faces.velocity[0] != nullptr;
    bool const number_flux = faces.number_flux[0] != nullptr;
    bool const transport = velocities || number_flux;
    if (number_flux && (m_options.transport.fluid_reconstruction != 0 ||
                        m_options.transport.central_dissipation_entropy != 0)) {
        return false; // kinetic continuity has no MHD density-penalty channel
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if ((faces.velocity[d] != nullptr) != velocities ||
            (faces.number_flux[d] != nullptr) != number_flux ||
            (faces.dissipation_speed[d] && !transport)) {
            return false;
        }
        for (auto const* face : {faces.velocity[d], faces.dissipation_speed[d],
                                 faces.number_flux[d]}) {
            if (face && (face->nComp() != 1 || face->hasEBFabFactory() ||
                         face->boxArray() != m_velocity[d]->boxArray() ||
                         face->DistributionMap() != m_old.DistributionMap() ||
                         !finite(*face))) {
                return false;
            }
        }
        if (faces.dissipation_speed[d] &&
            faces.dissipation_speed[d]->min(0) < 0) {
            return false;
        }
    }
    amrex::MultiFab::Copy(m_density, stage_density, 0, 0, 1, 0);
    amrex::MultiFab::Copy(m_endpoint_density, endpoint_density, 0, 0, 1, 0);
    amrex::MultiFab::Copy(m_magnetic, magnetic_field, 0, 0, 3, 0);
    amrex::MultiFab::Copy(m_source, source, 0, 0, 1, 0);
    fill_images(m_density, m_geometry);
    fill_images(m_magnetic, m_geometry);
    if (faces.conduction_active) {
        amrex::MultiFab::Copy(m_conduction_active, *faces.conduction_active, 0,
                              0, 1, 0);
        fill_images(m_conduction_active, m_geometry);
    } else {
        m_conduction_active.setVal(1);
    }
    m_transport = transport;
    m_remainder_context=paired;m_remainder_ready=false;
    if(paired) {
        if(!m_residual_remainder)m_residual_remainder=std::make_unique<amrex::MultiFab>(m_old.boxArray(),m_old.DistributionMap(),1,0);
        for(int d=0;d<AMREX_SPACEDIM;++d) {
            if(!m_velocity_remainder[d])m_velocity_remainder[d]=std::make_unique<amrex::MultiFab>(m_velocity[d]->boxArray(),m_old.DistributionMap(),1,0);
            if(!m_advection_remainder[d])m_advection_remainder[d]=std::make_unique<amrex::MultiFab>(m_advection[d]->boxArray(),m_old.DistributionMap(),1,0);
        }
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (number_flux) {
            for (amrex::MFIter mfi(*m_velocity[d]); mfi.isValid(); ++mfi) {
                auto const n = m_density.const_array(mfi);
                auto const flux = faces.number_flux[d]->const_array(mfi);
                auto const v = m_velocity[d]->array(mfi);
                auto const fl=paired?faces.number_flux_remainder[d]->const_array(mfi):amrex::Array4<Real const>{};
                auto const vl=paired?m_velocity_remainder[d]->array(mfi):amrex::Array4<Real>{};
                amrex::ParallelFor(
                    mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Real const nf =
                            0.5 * (n(i, j, k) +
                                   n(i - (d == 0), j - (d == 1), k - (d == 2)));
                        v(i, j, k) = flux(i, j, k) / nf;
                        if(paired)vl(i,j,k)=remainder::WithHigh(remainder::Divide({flux(i,j,k),fl(i,j,k)},nf),v(i,j,k)).lo;
                    });
            }
            m_velocity[d]->OverrideSync(m_geometry.periodicity());
            if(paired)m_velocity_remainder[d]->OverrideSync(m_geometry.periodicity());
        } else if (velocities) {
            amrex::MultiFab::Copy(*m_velocity[d], *faces.velocity[d], 0, 0, 1,
                                  0);
            // Synchronize duplicated face ownership before either divergence.
            m_velocity[d]->OverrideSync(m_geometry.periodicity());
            if(paired)m_velocity_remainder[d]->OverrideSync(m_geometry.periodicity());
        } else {
            m_velocity[d]->setVal(0);
        }
        if (faces.dissipation_speed[d]) {
            amrex::MultiFab::Copy(*m_speed[d], *faces.dissipation_speed[d], 0,
                                  0, 1, 0);
            m_speed[d]->OverrideSync(m_geometry.periodicity());
        } else {
            amrex::MultiFab::Copy(*m_speed[d], *m_velocity[d], 0, 0, 1, 0);
            m_speed[d]->abs(0, 1, 0);
        }
    }
    return true;
}

bool
EulerianThermalStage::Admissible (const amrex::MultiFab& stage) const {
    check_layout(stage, m_old, 1);
    if (!finite(stage)) {
        return false;
    }
    auto const theta = m_options.theta;
    auto const floor_e =
        PhysConst::q_e * m_options.temperature_floor_ev / (m_options.gamma - 1);
    amrex::ReduceOps<amrex::ReduceOpMin> op;
    amrex::ReduceData<Real> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(stage); mfi.isValid(); ++mfi) {
        auto const u = stage.const_array(mfi);
        auto const old = m_old.const_array(mfi);
        auto const n = m_endpoint_density.const_array(mfi);
        auto const ns = m_density.const_array(mfi);
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    Real const stage_specific = u(i, j, k) / ns(i, j, k);
                    Real const endpoint =
                        (u(i, j, k) - (1 - theta) * old(i, j, k)) / theta;
                    Real const margin = endpoint / n(i, j, k) - floor_e;
                    return {
                        std::isfinite(stage_specific) && stage_specific > 0 &&
                                std::isfinite(endpoint) && std::isfinite(margin)
                            ? margin
                            : Real(-1)};
                });
    }
    Real margin = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealMin(margin);
    return std::isfinite(margin) && (floor_e == 0 ? margin > 0 : margin >= 0);
}

amrex::Real
EulerianThermalStage::StepBound (const amrex::MultiFab& stage,
                                 const amrex::MultiFab& direction) const {
    check_layout(direction, m_old, 1);
    auto const theta = m_options.theta;
    auto const floor_e =
        PhysConst::q_e * m_options.temperature_floor_ev / (m_options.gamma - 1);
    amrex::ReduceOps<amrex::ReduceOpMin> op;
    amrex::ReduceData<Real> data(op);
    using Tuple = decltype(data)::Type;
    for (amrex::MFIter mfi(stage); mfi.isValid(); ++mfi) {
        auto const u = stage.const_array(mfi);
        auto const v = direction.const_array(mfi);
        auto const old = m_old.const_array(mfi);
        auto const n = m_endpoint_density.const_array(mfi);
        op.eval(mfi.validbox(), data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                    Real const margin = u(i, j, k) -
                                        (1 - theta) * old(i, j, k) -
                                        theta * n(i, j, k) * floor_e;
                    return {v(i, j, k) < 0
                                ? amrex::max(
                                      Real(0),
                                      amrex::min(Real(1), margin / -v(i, j, k)))
                                : Real(1)};
                });
    }
    Real bound = amrex::get<0>(data.value());
    amrex::ParallelDescriptor::ReduceRealMin(bound);
    return bound;
}

void
EulerianThermalStage::Endpoint (const amrex::MultiFab& stage,
                                amrex::MultiFab& endpoint) const {
    check_layout(endpoint, m_old, 1);
    auto const theta = m_options.theta;
    for (amrex::MFIter mfi(endpoint); mfi.isValid(); ++mfi) {
        auto const u = stage.const_array(mfi);
        auto const old = m_old.const_array(mfi);
        auto const out = endpoint.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j,
                                                                int k) {
            out(i, j, k) = (u(i, j, k) - (1 - theta) * old(i, j, k)) / theta;
        });
    }
}

} // namespace warpx::thermal
