/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/BraginskiiConductivity.H"
#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>
#include <cmath>
#include <limits>
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcConductionReport.H"
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
    // Independent NRL/Braginskii reference table, SI, CODATA2018 mass.
    amrex::GpuArray<amrex::GpuArray<amrex::Real,10>,17> const cases = {{
        {3e+21, 1000, 0, 1, 10, 8351480.4861879144, 8351480.4861879144, 8351480.4861879144, 0, 10},
        {3e+21, 1000, 0, 100000, 10, 8351480.4861879144, 8351480.4861879144, 8351480.4861879144, 0, 10},
        {3e+21, 1000, 1.5679832320191904e-13, 1, 10, 8351480.4861879107, 8351480.4861879144, 8351480.4861879107, 1.0000000000000002e-08, 10},
        {3e+21, 1000, 1.5679832320191904e-13, 100000, 10, 8351480.4861879107, 8351480.4861879144, 8351480.4861879144, 1.0000000000000002e-08, 10},
        {3e+21, 1000, 1.5679832320191904e-05, 1, 10, 2239633.2221670267, 8351480.4861879144, 2239633.2221670267, 1.0000000000000002, 10},
        {3e+21, 1000, 1.5679832320191904e-05, 100000, 10, 2239633.2221670267, 8351480.4861879144, 8351480.4861879144, 1.0000000000000002, 10},
        {3e+21, 1000, 0.0015679832320191905, 1, 10, 1230.5259654193351, 8351480.4861879144, 1230.5259654193351, 100.00000000000001, 10},
        {3e+21, 1000, 0.0015679832320191905, 100000, 10, 1230.5259654193351, 8351480.4861879144, 8351480.4861879144, 100.00000000000001, 10},
        {3e+21, 1000, 0.15679832320191905, 1, 10, 0.1232030933118151, 8351480.4861879144, 0.1232030933118151, 10000.000000000004, 10},
        {3e+21, 1000, 0.15679832320191905, 100000, 10, 0.1232030933118151, 8351480.4861879144, 12320.30933118151, 10000.000000000004, 10},
        {3e+21, 1000, 1.5679832320191903e+95, 1, 10, 1.23203108384794e-193, 8351480.4861879144, 1.23203108384794e-193, 1.0000000000000002e+100, 10},
        {3e+21, 1000, 1.5679832320191903e+95, 100000, 10, 1.23203108384794e-193, 8351480.4861879144, 1.23203108384794e-188, 1.0000000000000002e+100, 10},
        {1e+18, 1, 0.01, 100000, 0, 9.7475391707279877e-05, 0.28754674211864012, 0.28754674211864012, 65.875636040964224, 9.1844894420357264},
        {1e+18, 1000, 0.01, 100000, 0, 5.7525590582035659e-06, 4886122.6962882942, 0.57525590582035657, 1119388.2358763712, 17.092244721017863},
        {9.9999999999999992e+22, 0.10000000000000001, 0, 100000, 0, 0.0041757402430939589, 0.0041757402430939589, 0.0041757402430939589, 0, 2},
        {0, 1000, 1, 100000, 10, 0, 0, 0, 0, 0},
        {3e+21, 0, 1, 100000, 10, 0, 0, 0, 0, 0}
    }};
    amrex::Gpu::DeviceVector<amrex::Real> output(17*5);
    auto* out=output.data();
    amrex::ParallelFor(17,[=] AMREX_GPU_DEVICE(int i){
        auto const s=cases[i];auto const c=EvaluateBraginskiiConductivity(s[0],s[1],s[2],s[3],s[4]);
        out[5*i]=c.perpendicular;out[5*i+1]=c.parallel_physical;out[5*i+2]=c.parallel_used;
        out[5*i+3]=c.hall;out[5*i+4]=c.coulomb_log;
    });
    amrex::Gpu::HostVector<amrex::Real> actual(17*5);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,output.begin(),output.end(),actual.begin());
    double worst=0;
    for(int i=0;i<17;++i)for(int j=0;j<5;++j){
        auto const expected=cases[i][5+j], got=actual[5*i+j];
        double const error=expected==0?std::abs(got):std::abs(got/expected-1);
        worst=std::max(worst,error);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(got) && error<3.e-7,"Physical coefficient reference mismatch");
    }
    amrex::Print()<<"PHYSICAL_COEFFICIENTS cases=17 max_relative_error="<<worst<<" PASS\n";
    amrex::Gpu::DeviceVector<amrex::Real> extreme(5);
    auto* values = extreme.data();
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int) {
        auto const c = EvaluateBraginskiiConductivity(1.e-280, 8000, 10, 1.e5, 10);
        values[0] = c.perpendicular; values[1] = c.parallel_physical;
        values[2] = c.parallel_used; values[3] = c.hall;
        values[4] = c.coulomb_log;
    });
    amrex::Gpu::HostVector<amrex::Real> result(5);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, extreme.begin(), extreme.end(), result.begin());
    long double const tau = 3.44e11L * 8000.L * std::sqrt(8000.L) / (1.e-280L * 10.L);
    long double const hall = 1.602176634e-19L * 10.L * tau / 9.1093837015e-31L;
    long double const base = 1.e-280L * 1.380649e-23L * 1.602176634e-19L * 8000.L * tau / 9.1093837015e-31L;
    long double const inverse2 = (1.L / hall) * (1.L / hall);
    long double const perpendicular = base * inverse2 * (4.664L + 11.92L * inverse2) /
                                      (1.L + 14.79L * inverse2 + 3.7703L * inverse2 * inverse2);
    AMREX_ALWAYS_ASSERT(hall > std::numeric_limits<amrex::Real>::max());
    AMREX_ALWAYS_ASSERT(result[0] == static_cast<amrex::Real>(perpendicular));
    AMREX_ALWAYS_ASSERT(result[2] == static_cast<amrex::Real>(1.e5L * perpendicular));
    AMREX_ALWAYS_ASSERT(std::isfinite(result[1]) &&
        std::abs(result[1] / static_cast<amrex::Real>(base * 11.92L / 3.7703L) - 1) < 3.e-7);
    AMREX_ALWAYS_ASSERT(result[3] == std::numeric_limits<amrex::Real>::infinity() && result[4] == 10);
    QdsmcConductionReport report;
    report.coefficient_min = {1,2,3,4,5,result[3]};
    report.coefficient_max = {6,7,8,9,10,result[3]};
    report.EncodeHallOverflow();
    AMREX_ALWAYS_ASSERT(report.hall_min_overflow && report.hall_max_overflow &&
                        report.coefficient_min[5] == std::numeric_limits<amrex::Real>::max() &&
                        report.coefficient_max[5] == std::numeric_limits<amrex::Real>::max());
    for (int j=0; j<5; ++j) {
        AMREX_ALWAYS_ASSERT(report.coefficient_min[j] == j+1 && report.coefficient_max[j] == j+6);
    }
    report.coefficient_min[5] = std::numeric_limits<amrex::Real>::quiet_NaN();
    report.coefficient_max[5] = -std::numeric_limits<amrex::Real>::infinity();
    report.coefficient_max[3] = std::numeric_limits<amrex::Real>::infinity();
    report.EncodeHallOverflow();
    AMREX_ALWAYS_ASSERT(!report.hall_min_overflow && !report.hall_max_overflow &&
                        std::isnan(report.coefficient_min[5]) &&
                        report.coefficient_max[5] == -std::numeric_limits<amrex::Real>::infinity() &&
                        report.coefficient_max[3] == std::numeric_limits<amrex::Real>::infinity());
    amrex::Print()<<"HALL_OVERFLOW coefficients-and-report PASS\n";
    }
    amrex::Finalize();
}
