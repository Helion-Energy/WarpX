/* Exact identity against the separately pinned final MHD header. */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/EulerianThermalTransport.H"
#include "ThetaImplicitMHD_K.H"
#include <AMReX.H>
#include <AMReX_Print.H>
#include <cmath>
int
main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    namespace t = warpx::thermal::mhd_transport;
    namespace m = theta_implicit_mhd;
    for (int mode = 1; mode <= 5; ++mode) {
        t::TransportParameters a;
        a.fluid_reconstruction = mode;
        m::FluxParameters b{};
        b.fluid_reconstruction = mode;
        b.reconstruction_kappa = 0.01;
        for (int i = 0; i < 500; ++i) {
            double q[4];
            for (int j = 0; j < 4; ++j) {
                q[j] = 2 + std::sin(0.37 * (i + j)) +
                       0.2 * std::cos(0.81 * (2 * i - j));
            }
            double tl, tr, ml, mr;
            t::reconstruct_face_pair(q[0], q[1], q[2], q[3], a, tl, tr);
            m::reconstruct_face_pair(q[0], q[1], q[2], q[3], b, ml, mr);
            AMREX_ALWAYS_ASSERT(tl == ml && tr == mr);
        }
    }
    for (double velocity : {-3., 0., 2.}) {
        for (double penalty : {0., 0.25, 1.}) {
            m::CellState left{}, right{};
            left.density = right.density = 1;
            left.electron_energy = 2;
            right.electron_energy = 3;
            left.electron_velocity_normal = right.electron_velocity_normal =
                velocity;
            left.electron_pressure = right.electron_pressure = 1;
            left.ion_pressure = right.ion_pressure = 1;
            left.fast_speed = right.fast_speed = 2;
            left.wave_speed = right.wave_speed = 3;
            m::FluxParameters p{};
            p.mu0 = 1;
            p.central_dissipation_entropy = penalty;
            p.central_dissipation_entropy_flow = true;
            p.central_dissipation_flow_kappa = 0;
            auto const expected = m::central_flux(left, right, 1, 2, p);
            auto const actual = t::CentralElectronFlux(
                2, 3, velocity, std::abs(velocity), penalty);
            AMREX_ALWAYS_ASSERT(actual.energy == expected.electron_energy);
            AMREX_ALWAYS_ASSERT(actual.velocity == expected.electron_velocity);
        }
    }
    amrex::Print() << "PASS pinned MHD reconstruction and electron "
                      "central/compression identities\n";
    amrex::Finalize();
}
