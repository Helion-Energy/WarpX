/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Initialization/WarpXInit.H"
#include "WarpX.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <array>
#include <memory>
#include <vector>

int main (int argc, char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    {
        auto& simulation=WarpX::GetInstance();
        simulation.InitData();
        auto& ext=*simulation.get_pointer_HybridPICModel()->m_external_vector_potential;
        using warpx::fields::FieldType;
        std::array<FieldType,2> const kinds{FieldType::hybrid_E_fp_external,FieldType::hybrid_B_fp_external};
        std::array<std::unique_ptr<amrex::MultiFab>,6> reference;
        double const t0=3.e-9,dt=2.e-9;
        std::vector<double> start(ext.nFields(),0.),end(ext.nFields(),0.);
        start[0]=.7; end[0]=-.2;
        amrex::Gpu::DeviceVector<double> ds(start.size()),de(end.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,start.begin(),start.end(),ds.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,end.begin(),end.end(),de.begin());
        for(double fraction : {0.,.5,1.}) {
            ext.SetScale(ext.FieldName(0),start[0],end[0],t0,t0+dt);
            ext.UpdateHybridExternalFields(t0+fraction*dt,dt);
            for(int k=0;k<2;++k)for(int d=0;d<3;++d) {
                auto const& f=*simulation.m_fields.get(kinds[k],ablastr::fields::Direction{d},0);
                auto& r=reference[3*k+d];
                if(!r)r=std::make_unique<amrex::MultiFab>(f.boxArray(),f.DistributionMap(),1,f.nGrowVect());
                amrex::MultiFab::Copy(*r,f,0,0,1,f.nGrowVect());
            }
            // Poison the inactive host mirror: the device branch must read
            // its own scale for both fields, while other parser fields stay live.
            ext.SetScale(ext.FieldName(0),999.,1000.,t0,t0+dt);
            ext.SetDeviceScaleSegments(ds.data(),de.data(),t0,(t0+dt)-t0,{0});
            ext.UpdateHybridExternalFields(t0+fraction*dt,dt);
            for(int k=0;k<2;++k)for(int d=0;d<3;++d) {
                auto const& f=*simulation.m_fields.get(kinds[k],ablastr::fields::Direction{d},0);
                auto& r=*reference[3*k+d];
                double const scale=r.norminf(0,r.nGrow());
                amrex::MultiFab::Subtract(r,f,0,0,1,f.nGrowVect());
                AMREX_ALWAYS_ASSERT(r.norminf(0,r.nGrow()) <= 2.e-14*std::max(1.,scale));
            }
            ext.ClearDeviceScaleSegments();
        }
        amrex::Print()<<"DEVICE_EXTERNAL_FIELDS_PASS\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
