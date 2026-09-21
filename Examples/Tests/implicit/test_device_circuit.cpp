/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Circuit/DeviceCircuit.H"
#include "Circuit/DeviceScaleView.H"
#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main (int argc, char** argv)
{
    amrex::Initialize(argc,argv);
    {
        std::string mode = "valid";
        amrex::ParmParse pp;
        pp.query("mode",mode);
        std::vector<double> p0{2.,-3.}, g{.5,-1.,2.,.25}, iref{2.,-1.}, entry{.8,.2};
        std::vector<double> guard0{1.,0.}, guardg{.1,.2,0.,0.};
        std::vector<int32_t> signs{1,0};
        WarpxCircuitAffineViewV1 packet{};
        packet.struct_bytes=sizeof(packet); packet.scalar_kind=WARPX_CIRCUIT_AFFINE_F64;
        packet.guard_kind=WARPX_CIRCUIT_AFFINE_SIGN_GUARD_V1;
        packet.token=1; packet.n_port=2; packet.n_guard=2;
        packet.t0_sim=3.; packet.t1_sim=3.2; packet.guard_relative_band=1.e-3;
        packet.p0=p0.data(); packet.g=g.data(); packet.i_ref=iref.data();
        packet.entry_current=entry.data(); packet.guard_v0=guard0.data();
        packet.guard_g=guardg.data(); packet.guard_sign=signs.data();
        if (mode == "sign") { guard0[0]=.1; guardg[0]=0.; guardg[1]=-1.; }
        if (mode == "band") { guard0[0]=3.1001; guardg[0]=0.; guardg[1]=-1.; }
        if (mode == "entry") { entry[1]=.4; }
        if (mode == "interval") { packet.t1_sim=3.3; }
        warpx::circuit::DeviceCircuit circuit;
        circuit.Prepare(packet,3.,.2,.5,1.e-14,.25,{2,0},{-.2,7.,.4},
                        {-.2,9.,.4},{.1,-.4},{6.,-2.},{.03,-.2},
                        {.1,.2,.3,-.2,.1,.4},{1,1},iref);
        std::vector<double> total{2.3,-1.9};
        if (mode == "nonfinite") { total[1]=std::numeric_limits<double>::quiet_NaN(); }
        amrex::Gpu::DeviceVector<double> device_total(2);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,total.begin(),total.end(),device_total.begin());
        circuit.Evaluate(device_total);
        if (mode == "convergence") { circuit.RequireConverged(); }
        std::vector<double> endpoint,emf,response;
        circuit.ReadAccepted(endpoint,emf,response);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mode == "valid", "A required refusal did not occur");
        auto close = [](double a, double b) { return std::abs(a-b) < 2.e-13; };
        AMREX_ALWAYS_ASSERT(close(emf[0],.1475) && close(emf[1],3.1));
        AMREX_ALWAYS_ASSERT(close(response[0],.4) && close(response[1],-3.));
        AMREX_ALWAYS_ASSERT(close(endpoint[2],-.513125) && close(endpoint[0],1.93));
        AMREX_ALWAYS_ASSERT(endpoint[1] == 7.);
        // Both external-field consumers use this view: B and boundary A use
        // Value; the induced E field uses -Slope. Check on the execution device.
        warpx::circuit::DeviceScaleView view{circuit.Start(),circuit.End(),3.,3.2-3.};
        amrex::Gpu::DeviceVector<double> sampled(2);
        auto* data=sampled.data();
        amrex::ParallelFor(1,[=] AMREX_GPU_DEVICE (int) noexcept {
            data[0]=view.Value(2,3.1); data[1]=-view.Slope(2);
        });
        std::vector<double> host(2);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,sampled.begin(),sampled.end(),host.begin());
        AMREX_ALWAYS_ASSERT(close(host[0],-.0565625) && close(host[1],4.565625));
        circuit.ResetTrial(); circuit.Evaluate(device_total);
        std::vector<double> replay,again,unused;
        circuit.ReadAccepted(replay,again,unused);
        AMREX_ALWAYS_ASSERT(replay == endpoint && again == emf);
        // A constant response map converges in two iterations; unmeasured
        // ports have exactly zero EMF, even with nonzero frozen filter memory.
        std::fill(g.begin(),g.end(),0.);
        circuit.Prepare(packet,3.,.2,.5,1.e-14,.25,{2,0},{-.2,7.,.4},
                        {-.2,9.,.4},{.1,-.4},{6.,-2.},{.03,-.2},
                        {.1,.2,.3,-.2,.1,.4},{1,0},iref);
        circuit.Evaluate(device_total); circuit.Evaluate(device_total);
        circuit.RequireConverged(); circuit.ReadAccepted(endpoint,emf,response);
        AMREX_ALWAYS_ASSERT(endpoint[2] == 1. && endpoint[0] == 3. && emf[1] == 0.);
        amrex::Print() << "DEVICE_CIRCUIT_PASS\n";
    }
    amrex::Finalize();
}
