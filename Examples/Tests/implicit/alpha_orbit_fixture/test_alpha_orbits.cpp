/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitIonElectricWork.H"
#include "FieldSolver/ImplicitSolvers/MassMatrixDensityProjection.H"
#include "Initialization/WarpXInit.H"
#include "Particles/Gather/ImplicitGatherSafety.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/ImplicitFinalGather.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

using Real = amrex::Real;
using PR = amrex::ParticleReal;
using warpx::fields::FieldType;
constexpr int count = 12;
constexpr int width = 15;
using State = std::array<Real, width>;
using States = std::array<State, count>;
constexpr Real pi = 3.141592653589793238462643383279502884;

States
Collect (PhysicalParticleContainer& pc, bool pushed = true) {
    States result{};
    for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
        auto const np = pti.numParticles();
        auto pos = GetParticlePosition(pti);
        amrex::Gpu::DeviceVector<PR> xyz(3 * np);
        auto* a = xyz.data();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long p) {
            pos(p, a[3 * p], a[3 * p + 1], a[3 * p + 2]);
        });
        amrex::Gpu::HostVector<PR> hx(3 * np);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, xyz.begin(), xyz.end(),
                         hx.begin());
        std::array<amrex::Gpu::HostVector<PR>, 7> values;
        char const* names[] = {"ux",
                               "uy",
                               "uz",
                               "w",
                               "implicit_final_gather_x",
                               "implicit_final_gather_y",
                               "implicit_final_gather_z"};
        for (int c = 0; c < 7; ++c) {
            values[c].assign(np, 0.);
            if (!pushed && c >= 4) {
                continue;
            }
            auto const* src = pti.GetAttribs(names[c]).dataPtr();
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, src, src + np,
                             values[c].begin());
        }
        amrex::Gpu::HostVector<int> tags(np), iters(np, 0);
        for (auto pair : {std::pair{"orbit_tag", &tags},
                          std::pair{"alpha_orbit_picard_iterations", &iters}}) {
            if (!pushed && pair.second == &iters) {
                continue;
            }
            auto const* src = pti.GetiAttribs(pair.first).dataPtr();
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, src, src + np,
                             pair.second->begin());
        }
        for (long p = 0; p < np; ++p) {
            int const tag = tags[p];
            AMREX_ALWAYS_ASSERT(tag >= 0 && tag < count);
            auto& row = result[tag];
            for (int d = 0; d < 3; ++d) {
                row[d] = hx[3 * p + d];
                row[3 + d] = values[d][p];
            }
            row[6] = values[3][p];
            row[7] = pti.index();
            row[8] = amrex::ParallelDescriptor::MyProc();
            row[9] = iters[p];
            for (int d = 0; d < 3; ++d) {
                row[10 + d] = values[4 + d][p];
            }
            row[13] = 1.;
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(result[0].data(), count * width);
    for (auto const& row : result) {
        AMREX_ALWAYS_ASSERT(row[13] == 1.);
    }
    return result;
}

void
DepositCharge (WarpX& w, amrex::MultiFab& rho) {
    ablastr::fields::MultiLevelScalarField r{&rho};
    w.GetPartContainer().DepositCharge(r, 0.);
    w.SyncRho(r, {}, {});
    w.ApplyRhofieldBoundary(0, &rho, PatchType::fine);
    rho.FillBoundary(w.Geom(0).periodicity());
}

Real
Kinetic (State const& p, Real mass) {
    Real const u2 = p[3] * p[3] + p[4] * p[4] + p[5] * p[5];
    return mass * u2 / (1 + std::sqrt(1 + u2 / (PhysConst::c * PhysConst::c)));
}

int
main (int argc, char** argv) {
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& w = WarpX::GetInstance();
        w.InitData();
        auto& pc = dynamic_cast<PhysicalParticleContainer&>(
            w.GetPartContainer().GetParticleContainerFromName("alpha"));
        pc.AddIntComp("orbit_tag", 1);
        pc.AddIntComp("alpha_orbit_picard_iterations", 0);
        AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == 0);
        AMREX_ALWAYS_ASSERT(
            WarpX::nox == 3 &&
            WarpX::field_gathering_algo == GatheringAlgo::MomentumConserving &&
            WarpX::current_deposition_algo ==
                CurrentDepositionAlgo::Esirkepov &&
            WarpX::particle_pusher_algo == ParticlePusherAlgo::Boris);
        pc.ValidateImplicitIonElectricWorkCapture();
        amrex::ParmParse pp("orbit");
        std::string kind = "gyro";
        int per_period = 64;
        pp.query("case", kind);
        pp.query("steps_per_period", per_period);
        AMREX_ALWAYS_ASSERT(kind == "gyro" || kind == "circular" ||
                            kind == "drift");
#ifdef WARPX_DIM_RZ
        AMREX_ALWAYS_ASSERT(kind != "drift");
#endif
        Real const mass = pc.getMass(), charge = pc.getCharge();
        Real const c = PhysConst::c, energy = 3.5e6 * PhysConst::q_e;
        Real const gamma = 1 + energy / (mass * c * c),
                   speed_u = c * std::sqrt((gamma - 1) * (gamma + 1));
        Real const speed = speed_u / gamma, B = 10.;
        Real const omega = charge * B / (gamma * mass), period = 2 * pi / omega;
        Real const dt = period / per_period, rho = speed / omega,
                   radius = 10 * rho;
        Real const radial_coefficient =
            speed * B / radius -
            gamma * mass * speed * speed / (charge * radius * radius);
        int const steps = 4 * per_period;
        w.setdt(dt);
        AMREX_ALWAYS_ASSERT(charge == 2 * PhysConst::q_e && per_period >= 16);
        amrex::Vector<PR> x(count), y(count), z(count), ux(count), uy(count),
            uz(count);
        amrex::Vector<amrex::Vector<PR>> weight(1, amrex::Vector<PR>(count));
        // Runtime integer attributes can precede this fixture's orbit tag.
        // AddNParticles initializes integer arrays by component index.
        int const num_int = pc.NumIntComps();
        int const orbit_tag = pc.GetIntCompIndex("orbit_tag");
        AMREX_ALWAYS_ASSERT(orbit_tag >= 0 && orbit_tag < num_int);
        amrex::Vector<amrex::Vector<int>> attributes_int(
            num_int, amrex::Vector<int>(count, 0));
        for (int p = 0; p < count; ++p) {
            Real const angle = .17 + .43 * p;
            if (kind == "gyro") {
                Real const center = p < 2 ? 0. : (p < 7 ? .249 : .499);
                Real const theta = .31 + .21 * p;
                ux[p] = speed_u * std::cos(angle);
                uy[p] = speed_u * std::sin(angle);
                uz[p] = 0.;
                x[p] = center * std::cos(theta) - uy[p] / (gamma * omega);
                y[p] = center * std::sin(theta) + ux[p] / (gamma * omega);
                z[p] = -.26 + .13 * (p % 5);
            } else if (kind == "circular") {
                x[p] = radius * std::cos(angle);
                y[p] = radius * std::sin(angle);
                z[p] = -.26 + .13 * (p % 5);
                ux[p] = speed_u * std::sin(angle);
                uy[p] = -speed_u * std::cos(angle);
                uz[p] = 0.;
            } else {
                x[p] = -.26 + .13 * (p % 5);
                y[p] = -.26 + .052 * p;
                z[p] = -.4 + .05 * p;
                ux[p] = 0.;
                uy[p] = -.25 * speed_u;
                uz[p] = std::sqrt(15.) * .25 * speed_u;
            }
            weight[0][p] = 1. + p;
            attributes_int[orbit_tag][p] = p;
        }
        pc.AddNParticles(0, count, x, y, z, ux, uy, uz, 1, weight, num_int, attributes_int, 0);
        pc.Redistribute();
        auto E = w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto Bf = w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto Ea = w.m_fields.get_alldirs(FieldType::Efield_aux, 0);
        auto Ba = w.m_fields.get_alldirs(FieldType::Bfield_aux, 0);
        auto const& geom = w.Geom(0);
        auto const dx = geom.CellSizeArray(), lo = geom.ProbLoArray();
        for (int component = 0; component < 6; ++component) {
            auto& field = *(component < 3 ? E[component] : Bf[component - 3]);
            auto stag = field.ixType().toIntVect();
            bool const radial = kind == "circular", drift = kind == "drift";
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto f = field.array(mfi);
                amrex::ParallelFor(
                    mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        Real value = component == 5 ? B : 0.;
                        if (radial && component == 0) {
                            value = radial_coefficient *
                                    (lo[0] + (i + .5 * (1 - stag[0])) * dx[0]);
                        }
#ifdef WARPX_DIM_3D
                        if (radial && component == 1) {
                            value = radial_coefficient *
                                    (lo[1] + (j + .5 * (1 - stag[1])) * dx[1]);
                        }
#endif
                        if (drift && component == 0) {
                            value = .25 * speed * B;
                        }
                        f(i, j, k) = value;
                    });
            }
        }
        // Same native centering as the actual implicit MC gather, using the
        // exact prescribed polynomial fields in their allocated source guards.
        w.UpdateAuxiliaryData();
        std::array<amrex::MultiFab, 3> zero;
        warpx::thermal::IonElectricWorkFields work_fields;
        for (int d = 0; d < 3; ++d) {
            zero[d].define(Ea[d]->boxArray(), Ea[d]->DistributionMap(), 1,
                           Ea[d]->nGrowVect());
            zero[d].setVal(0.);
            work_fields.component[d] = &zero[d];
            work_fields.total[d] = Ea[d];
        }
        work_fields.filled_ghosts = w.get_ng_fieldgather();
        auto const& charge_layout = *w.m_fields.get(FieldType::rho_fp, 0);
        auto make_rho = [&] {
            return amrex::MultiFab(charge_layout.boxArray(),
                                   charge_layout.DistributionMap(), 1,
                                   charge_layout.nGrowVect());
        };
        auto old_rho = make_rho(), end_rho = make_rho(),
             divergence = make_rho();
        auto current = w.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto initial = Collect(pc, false), previous = initial;
        Real initial_inventory = 0.;
        for (auto const& p : initial) {
            initial_inventory += p[6] * Kinetic(p, mass);
        }
        Real max_energy = 0., max_position = 0., max_velocity = 0.,
             max_guiding = 0., max_phase = 0.;
        Real max_discrete_phase = 0., max_potential_energy = 0.,
             max_continuity = 0., max_work_defect = 0.;
        Real max_gather_midpoint_gap = 0., max_flight_cells = 0.,
             min_boundary_distance = 1.e300;
        long total_iterations = 0, migrations = 0, rank_migrations = 0;
        int max_iterations = 0, min_iterations = 10000;
        std::array<long, 51> iteration_histogram{};
        std::array<Real, count> phase{};
        std::ofstream trajectory;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            trajectory.open("trajectory.csv");
            trajectory << std::setprecision(17);
            trajectory << "step,tag,time,x,y,z,ux,uy,uz,kinetic,phase,inner_"
                          "iterations\n";
        }
        ImplicitOptions options;
        options.max_particle_iterations = 50;
        options.particle_tolerance = 1.e-10;
        auto const dinv = WarpX::InvCellSize(0);
        for (int step = 0; step < steps; ++step) {
            bool const check =
                (step == 0 || step == steps / 3 || step == steps - 1);
            if (check) {
                DepositCharge(w, old_rho);
                for (auto* j : current) {
                    j->setVal(0.);
                }
            }
            w.SaveParticlesAtImplicitStepStart();
            for (WarpXParIter pti(pc, 0); pti.isValid(); ++pti) {
                auto* count_scratch =
                    pti.GetiAttribs("alpha_orbit_picard_iterations").dataPtr();
                amrex::ParallelFor(
                    pti.numParticles(),
                    [=] AMREX_GPU_DEVICE(long p) { count_scratch[p] = 0; });
                long unconverged = 0;
                amrex::Gpu::DeviceVector<long> indices;
                amrex::Gpu::DeviceVector<PR> saved_weights;
                pc.ImplicitPushXP(pti, &(*Ea[0])[pti], &(*Ea[1])[pti],
                                  &(*Ea[2])[pti], &(*Ba[0])[pti],
                                  &(*Ba[1])[pti], &(*Ba[2])[pti], &options,
                                  Ea[0]->nGrowVect(), 0, pti.numParticles(), 0,
                                  0, dt, unconverged, indices, saved_weights);
                AMREX_ALWAYS_ASSERT(unconverged == 0);
                // Validate the actual final captured gather with the same pure
                // native support checker used before every pusher grid read.
                auto box = amrex::grow(pti.tilebox(), Ea[0]->nGrowVect());
                auto origin = WarpX::LowerCorner(box, 0, 0.);
                auto lower = amrex::lbound(box);
                amrex::GpuArray<amrex::Array4<Real const>, 6> fields;
                amrex::GpuArray<amrex::IndexType, 6> types;
                for (int d = 0; d < 3; ++d) {
                    fields[d] = Ea[d]->const_array(pti);
                    fields[3 + d] = Ba[d]->const_array(pti);
                    types[d] = Ea[d]->ixType();
                    types[3 + d] = Ba[d]->ixType();
                }
                auto const* gx =
                    pti.GetAttribs("implicit_final_gather_x").dataPtr();
                auto const* gy =
                    pti.GetAttribs("implicit_final_gather_y").dataPtr();
                auto const* gz =
                    pti.GetAttribs("implicit_final_gather_z").dataPtr();
                auto const np = pti.numParticles();
                auto bad = amrex::Reduce::Sum<long>(
                    np, [=] AMREX_GPU_DEVICE(long p) -> long {
                        return !warpx::particles::CheckMomentumGather(
                            gx[p], gy[p], gz[p], fields, types, dinv, origin,
                            lower, 1, 3);
                    });
                AMREX_ALWAYS_ASSERT(bad == 0);
                if (check) {
                    pc.DepositCurrent(
                        pti, pti.GetAttribs(PIdx::w), pti.GetAttribs(PIdx::ux),
                        pti.GetAttribs(PIdx::uy), pti.GetAttribs(PIdx::uz),
                        nullptr, current[0], current[1], current[2], 0, np, 0,
                        0, 0, dt, 0., PushType::Implicit);
                }
            }
            if (check) {
                using namespace warpx::thermal;
                auto work = MeasureNativeImplicitIonElectricWork(
                    w, 0, dt, work_fields,
                    NativeIonWorkContract::
                        FullBorisGatherWithoutAdditionalMomentumChanges);
                AMREX_ALWAYS_ASSERT(work.valid && work.species.size() == 1 &&
                                    work.particles == count);
                auto const& l = work.species[0];
                AMREX_ALWAYS_ASSERT(
                    l[IonElectricWorkComponent::ComponentWork] == 0. &&
                    l[IonElectricWorkComponent::ComponentWorkAbs] == 0.);
                max_work_defect =
                    std::max(max_work_defect,
                             l[IonElectricWorkComponent::TotalWorkDefectAbs] /
                                 initial_inventory);
#ifdef WARPX_DIM_RZ
                w.ApplyInverseVolumeScalingToCurrentDensity(
                    current[0], current[1], current[2], 0);
#endif
                w.SyncCurrent("current_fp");
                w.ApplyJfieldBoundary(0, current[0], current[1], current[2],
                                      PatchType::fine);
                for (auto* j : current) {
                    j->FillBoundary(geom.periodicity());
                    AMREX_ALWAYS_ASSERT(j->is_finite());
                }
            }
            pc.FinishImplicitParticleUpdate(w.m_fields, 0, (step + 1) * dt, dt);
            auto const state = Collect(pc);
            Real const time = (step + 1) * dt;
            for (int p = 0; p < count; ++p) {
                auto const& a = state[p];
                auto const& b = initial[p];
                auto const& old = previous[p];
                auto const iterations = int(a[9]);
                AMREX_ALWAYS_ASSERT(iterations >= 0 && iterations <= 50);
                ++iteration_histogram[iterations];
                total_iterations += iterations;
                max_iterations = std::max(max_iterations, iterations);
                min_iterations = std::min(min_iterations, iterations);
                Real expected_x = b[0], expected_y = b[1], expected_z = b[2];
                Real expected_ux = b[3], expected_uy = b[4], expected_uz = b[5];
                Real expected_angle = 0.;
                if (kind == "gyro") {
                    Real const angle = omega * time, cs = std::cos(angle),
                               sn = std::sin(angle);
                    expected_ux = b[3] * cs + b[4] * sn;
                    expected_uy = b[4] * cs - b[3] * sn;
                    expected_x +=
                        (b[3] * sn + b[4] * (1 - cs)) / (gamma * omega);
                    expected_y +=
                        (b[4] * sn - b[3] * (1 - cs)) / (gamma * omega);
                    expected_angle = -angle;
                    Real const gx = a[0] + a[4] / (gamma * omega) -
                                    (b[0] + b[4] / (gamma * omega));
                    Real const gy = a[1] - a[3] / (gamma * omega) -
                                    (b[1] - b[3] / (gamma * omega));
                    max_guiding =
                        std::max(max_guiding, std::hypot(gx, gy) / rho);
                } else if (kind == "circular") {
                    Real const angle = speed * time / radius,
                               cs = std::cos(angle), sn = std::sin(angle);
                    expected_x = b[0] * cs + b[1] * sn;
                    expected_y = b[1] * cs - b[0] * sn;
                    expected_ux = b[3] * cs + b[4] * sn;
                    expected_uy = b[4] * cs - b[3] * sn;
                    expected_angle = -angle;
                } else {
                    expected_x += time * b[3] / gamma;
                    expected_y += time * b[4] / gamma;
                    expected_z += time * b[5] / gamma;
                }
                Real const du = std::hypot(
                    std::hypot(a[3] - expected_ux, a[4] - expected_uy),
                    a[5] - expected_uz);
                Real const dr =
                    std::hypot(std::hypot(a[0] - expected_x, a[1] - expected_y),
                               a[2] - expected_z);
                max_position = std::max(
                    max_position, dr / (kind == "circular" ? radius : rho));
                max_velocity = std::max(max_velocity, du / speed_u);
                max_energy = std::max(
                    max_energy,
                    std::abs(Kinetic(a, mass) - Kinetic(b, mass)) / energy);
                Real const potential_a = -.5 * charge * radial_coefficient *
                                         (a[0] * a[0] + a[1] * a[1]);
                Real const potential_b = -.5 * charge * radial_coefficient *
                                         (b[0] * b[0] + b[1] * b[1]);
                if (kind == "circular") {
                    max_potential_energy =
                        std::max(max_potential_energy,
                                 std::abs(Kinetic(a, mass) - Kinetic(b, mass) +
                                          potential_a - potential_b) /
                                     (energy + std::abs(potential_b)));
                }
                if (kind != "drift") {
                    phase[p] += std::atan2(old[3] * a[4] - old[4] * a[3],
                                           old[3] * a[3] + old[4] * a[4]);
                    max_phase = std::max(max_phase,
                                         std::abs(phase[p] - expected_angle));
                    if (kind == "gyro") {
                        max_discrete_phase =
                            std::max(max_discrete_phase,
                                     std::abs(phase[p] +
                                              (step + 1) * 2 *
                                                  std::atan(.5 * omega * dt)));
                    }
                }
#ifdef WARPX_DIM_RZ
                Real radial = std::hypot(a[0], a[1]);
                Real boundary = std::min(
                    geom.ProbHi(0) - radial,
                    std::min(a[2] - geom.ProbLo(1), geom.ProbHi(1) - a[2]));
                amrex::RealVect location(radial, a[2]);
#else
                Real boundary = 1.e300;
                for (int d = 0; d < 3; ++d) {
                    boundary = std::min({boundary, a[d] - geom.ProbLo(d),
                                         geom.ProbHi(d) - a[d]});
                }
                amrex::RealVect location(a[0], a[1], a[2]);
#endif
                AMREX_ALWAYS_ASSERT(boundary > 0.);
                min_boundary_distance =
                    std::min(min_boundary_distance, boundary);
                auto cell = geom.CellIndex(&location[0]);
                int const grid = int(a[7]);
                if (!pc.ParticleBoxArray(0)[grid].contains(cell)) {
                    ++migrations;
                    for (int g = 0; g < pc.ParticleBoxArray(0).size(); ++g)
                        if (pc.ParticleBoxArray(0)[g].contains(cell)) {
                            rank_migrations +=
                                pc.ParticleDistributionMap(0)[g] != int(a[8]);
                            break;
                        }
                }
                for (int d = 0; d < 3; ++d) {
                    max_gather_midpoint_gap =
                        std::max(max_gather_midpoint_gap,
                                 std::abs(a[10 + d] - .5 * (old[d] + a[d])));
                }
#ifdef WARPX_DIM_RZ
                max_flight_cells =
                    std::max({max_flight_cells,
                              std::abs(std::hypot(a[0], a[1]) -
                                       std::hypot(old[0], old[1])) /
                                  geom.CellSize(0),
                              std::abs(a[2] - old[2]) / geom.CellSize(1)});
#else
                for (int d = 0; d < 3; ++d) {
                    max_flight_cells =
                        std::max(max_flight_cells,
                                 std::abs(a[d] - old[d]) / geom.CellSize(d));
                }
#endif
                if (trajectory && (step % (std::max(1, steps / 64)) == 0 ||
                                   step == steps - 1)) {
                    trajectory << step + 1 << ',' << p << ',' << time;
                    for (int d = 0; d < 6; ++d) {
                        trajectory << ',' << a[d];
                    }
                    trajectory << ',' << Kinetic(a, mass) << ',' << phase[p]
                               << ',' << iterations << '\n';
                }
            }
            previous = state;
            pc.Redistribute();
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles() == count);
            if (check) {
                DepositCharge(w, end_rho);
                MassMatrixDensityProjection::ComputeDivergence(w, 0, current,
                                                               divergence);
                divergence.mult(dt, 0, 1, 0);
                amrex::MultiFab::Add(divergence, end_rho, 0, 0, 1, 0);
                amrex::MultiFab::Subtract(divergence, old_rho, 0, 0, 1, 0);
                Real const defect = divergence.norminf() /
                                    std::max(old_rho.norminf(), Real(1.e-100));
                max_continuity = std::max(max_continuity, defect);
            }
        }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream final("endpoints.hex");
            final << std::hexfloat;
            for (int p = 0; p < count; ++p) {
                final << p;
                for (int d = 0; d < 7; ++d) {
                    final << ' ' << previous[p][d];
                }
                final << '\n';
            }
        }
        amrex::Print()
            << std::setprecision(17) << "ALPHA_ORBIT_RESULT {\"case\":\""
            << kind << "\",\"ranks\":" << amrex::ParallelDescriptor::NProcs()
            << ",\"boxes\":" << w.boxArray(0).size()
            << ",\"steps_per_period\":" << per_period << ",\"steps\":" << steps
            << ",\"mass_kg\":" << mass << ",\"charge_C\":" << charge
            << ",\"gamma0\":" << gamma << ",\"u0_m_per_s\":" << speed_u
            << ",\"v0_m_per_s\":" << speed << ",\"B_T\":" << B
            << ",\"omega_rad_per_s\":" << omega << ",\"larmor_m\":" << rho
            << ",\"dt_s\":" << dt << ",\"final_time_s\":" << steps * dt
            << ",\"max_energy_relative\":" << max_energy
            << ",\"max_position_relative\":" << max_position
            << ",\"max_velocity_relative\":" << max_velocity
            << ",\"max_guiding_center_relative\":" << max_guiding
            << ",\"max_phase_error_rad\":" << max_phase
            << ",\"max_discrete_phase_defect_rad\":" << max_discrete_phase
            << ",\"max_total_energy_relative\":" << max_potential_energy
            << ",\"max_continuity_relative\":" << max_continuity
            << ",\"max_electric_work_defect_over_initial_K\":"
            << max_work_defect
            << ",\"zero_component_work_exact\":true,\"max_gather_midpoint_gap_"
               "m\":"
            << max_gather_midpoint_gap
            << ",\"max_endpoint_flight_cells\":" << max_flight_cells
            << ",\"min_boundary_distance_m\":" << min_boundary_distance
            << ",\"inner_iterations_min\":" << min_iterations
            << ",\"inner_iterations_max\":" << max_iterations
            << ",\"inner_iterations_total\":" << total_iterations
            << ",\"box_crossings\":" << migrations
            << ",\"rank_crossings\":" << rank_migrations
            << ",\"inner_histogram\":[";
        for (int i = 0; i <= 50; ++i) {
            amrex::Print() << (i ? "," : "") << iteration_histogram[i];
        }
        amrex::Print()
            << "],\"unconverged_particles\":0,\"final_gather_support_valid\":"
               "true,\"endpoint_inside\":true,\"native_current_samples\":3";
        for (auto item :
             {std::pair{"aux_ghosts", Ea[0]->nGrowVect()},
              std::pair{"gather_ghosts", w.get_ng_fieldgather()},
              std::pair{"current_ghosts", current[0]->nGrowVect()}}) {
            amrex::Print() << ",\"" << item.first << "\":[";
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                amrex::Print() << (d ? "," : "") << item.second[d];
            }
            amrex::Print() << "]";
        }
        amrex::Print() << ",\"cell_size_m\":[";
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Print() << std::setprecision(17) << (d ? "," : "")
                           << geom.CellSize(d);
        }
        amrex::Print() << "]}\n";
    }
    WarpX::ResetInstance();
    warpx::initialization::finalize_external_libraries();
}
