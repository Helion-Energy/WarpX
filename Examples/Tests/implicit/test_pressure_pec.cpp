// Manufactured PEC pressure extension, including intersecting walls and ghosts.
#include "BoundaryConditions/WarpX_PEC.H"
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace amrex;
        ParmParse pp("hybrid_pic_model");
        if (!pp.contains("deterministic_pressure_bc")) {
            pp.add("deterministic_pressure_bc", true);
        }
        Box const domain(IntVect(0), IntVect(7));
        RealBox const physical({0.,0.,0.}, {1.,1.,1.});
        int const periodic[3] = {0,0,1};
        Geometry const geom(domain, &physical, 0, periodic);
        BoxArray boxes(domain);
        boxes.maxSize(4);
        boxes.convert(IntVect(1));
        DistributionMapping const dm(boxes);
        MultiFab pressure(boxes, dm, 1, 2);
        MultiFab error(boxes, dm, 1, 2);
        Array<FieldBoundaryType,3> const bc = {
            FieldBoundaryType::PEC, FieldBoundaryType::PEC, FieldBoundaryType::Periodic};
        // Exact extension of q(i)=i*i on nodes0..8: boundary nodes copy
        // nodes1/7; two ghosts mirror nodes1/2 and7/6, respectively.
        GpuArray<Real,13> const q = {4.,1.,1.,1.,4.,9.,16.,25.,36.,49.,49.,49.,36.};
        for (int repeat=0; repeat<20; ++repeat) {
            pressure.setVal(-999.);
            for (MFIter mfi(pressure); mfi.isValid(); ++mfi) {
                auto const a = pressure.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    a(i,j,k) = i*i + 10*j*j + 100*(k%8);
                });
            }
            PEC::ApplyPECtoElectronPressure(&pressure, bc, bc, geom, 0,
                                            PatchType::fine, Vector<IntVect>{});
            for (MFIter mfi(error); mfi.isValid(); ++mfi) {
                auto const a = pressure.const_array(mfi);
                auto const e = error.array(mfi);
                ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    e(i,j,k) = a(i,j,k) - (q[i+2] + 10*q[j+2] + 100*((k+8)%8));
                });
            }
            Real const maximum = error.norm0(0,2);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(maximum == 0.,
                "PEC pressure face/corner/ghost manufactured extension differs");
        }
        Print() << "PEC pressure manufactured extension passed20 repetitions\n";
    }
    amrex::Finalize();
}
