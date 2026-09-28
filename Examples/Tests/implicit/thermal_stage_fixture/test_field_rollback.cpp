/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "FieldSolver/ImplicitSolvers/ImplicitFieldRollback.H"
#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        int max_box=4; amrex::ParmParse("test").query("max_grid_size",max_box);
        amrex::Box domain(amrex::IntVect(0),amrex::IntVect(7));
        amrex::BoxArray boxes(domain);boxes.maxSize(max_box);
        amrex::DistributionMapping dm(boxes);
        ablastr::fields::MultiFabRegister fields;
        auto* u=fields.alloc_init("energy",0,boxes,dm,2,amrex::IntVect(3),0.);
        auto faces=amrex::convert(boxes,amrex::IntVect::TheDimensionVector(0));
        auto* e=fields.alloc_init("electric",ablastr::fields::Direction{0},0,
                                  faces,dm,1,amrex::IntVect(2),0.);
        fields.alias_init("energy_alias","energy",0);
        auto fill=[](amrex::MultiFab& f,amrex::Real shift) {
            for(amrex::MFIter mfi(f);mfi.isValid();++mfi) {
                auto a=f.array(mfi);int const box=mfi.index();
                amrex::ParallelFor(mfi.fabbox(),f.nComp(),[=] AMREX_GPU_DEVICE(int i,int j,int k,int c) {
                    a(i,j,k,c)=shift+i+16*j+256*k+4096*c+32768*box;
                });
            }
        };
        auto same=[](const amrex::MultiFab& a,const amrex::MultiFab& b) {
            amrex::MultiFab delta(a.boxArray(),a.DistributionMap(),a.nComp(),a.nGrowVect());
            amrex::MultiFab::LinComb(delta,1.,a,0,-1.,b,0,0,a.nComp(),a.nGrowVect());
            for(int c=0;c<a.nComp();++c) {AMREX_ALWAYS_ASSERT(delta.norminf(c,1,a.nGrowVect())==0);}
        };
        fill(*u,1.);fill(*e,-2.);
        amrex::MultiFab old_u(u->boxArray(),dm,u->nComp(),u->nGrowVect());
        amrex::MultiFab old_e(e->boxArray(),dm,e->nComp(),e->nGrowVect());
        amrex::MultiFab::Copy(old_u,*u,0,0,u->nComp(),u->nGrowVect());
        amrex::MultiFab::Copy(old_e,*e,0,0,e->nComp(),e->nGrowVect());
        warpx::implicit::FieldRollback saved;saved.Capture(fields);
        AMREX_ALWAYS_ASSERT(saved.Open() && saved.Size()==3);
        fill(*u,900.);fill(*e,-700.);
        // A lazily allocated scratch field cannot alter captured fields.
        fields.alloc_init("new_trial_scratch",0,boxes,dm,1,amrex::IntVect(0),77.);
        AMREX_ALWAYS_ASSERT(saved.Restore(fields));same(*u,old_u);same(*e,old_e);
        fill(*u,88.);AMREX_ALWAYS_ASSERT(saved.Restore(fields));same(*u,old_u);
        // Erasing one entry rejects collectively BEFORE restoration of others.
        fill(*e,700.);amrex::MultiFab changed(e->boxArray(),dm,e->nComp(),e->nGrowVect());
        amrex::MultiFab::Copy(changed,*e,0,0,e->nComp(),e->nGrowVect());
        fields.erase("energy_alias",0);
        AMREX_ALWAYS_ASSERT(!saved.Restore(fields));same(*e,changed);
        saved.Discard();AMREX_ALWAYS_ASSERT(!saved.Open());
        saved.Capture(fields);fill(*e,200.);AMREX_ALWAYS_ASSERT(saved.Restore(fields));same(*e,changed);
        saved.Discard();saved.Capture(fields);fill(*e,300.);AMREX_ALWAYS_ASSERT(saved.Restore(fields));same(*e,changed);
        saved.Discard();
        // Reconstructible tangent storage may grow during an implicit trial.
        // A default snapshot still rejects that change; only an explicitly
        // omitted scratch object is exempt, with all physical rows restored.
        auto* tangent=fields.alloc_init("mass_matrix_scratch",0,boxes,dm,9,amrex::IntVect(1),3.);
        saved.Capture(fields);
        *tangent=amrex::MultiFab(boxes,dm,25,amrex::IntVect(1));tangent->setVal(6.);
        fill(*e,444.);amrex::MultiFab::Copy(changed,*e,0,0,e->nComp(),e->nGrowVect());
        AMREX_ALWAYS_ASSERT(!saved.CanRestore(fields) && !saved.Restore(fields));same(*e,changed);
        saved.Discard();
        saved.Capture(fields,{tangent});
        *tangent=amrex::MultiFab(boxes,dm,49,amrex::IntVect(1));tangent->setVal(9.);
        fill(*e,555.);AMREX_ALWAYS_ASSERT(saved.CanRestore(fields) && saved.Restore(fields));same(*e,changed);
        AMREX_ALWAYS_ASSERT(tangent->nComp()==49 && tangent->min(0)==9. && tangent->max(0)==9.);
        saved.Discard();saved.Capture(fields,{tangent});
        // Changing a captured physical array is still a collective no-write
        // rejection even when the tangent scratch list is supplied.
        *u=amrex::MultiFab(boxes,dm,3,amrex::IntVect(3));u->setVal(88.);
        fill(*e,666.);amrex::MultiFab::Copy(changed,*e,0,0,e->nComp(),e->nGrowVect());
        AMREX_ALWAYS_ASSERT(!saved.CanRestore(fields) && !saved.Restore(fields));same(*e,changed);
        saved.Discard();
        amrex::Print()<<"FIELD_ROLLBACK PASS: valid/ghost, component, alias, repeat, no-partial topology rejection, recapture, explicit scratch resize\n";
    }
    amrex::Finalize();
}
