/* Native RZ stopping-event boundary/operator prerequisite. Observation only. */
#include "NativeBytes.H"
#include "FieldSolver/ImplicitSolvers/DarwinABoundary.H"
#include "FieldSolver/ImplicitSolvers/DarwinInitialRateSchur.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/QdsmcVolumeElement.H"
#include "Initialization/WarpXInit.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <fstream>
#include <iomanip>

using Real = amrex::Real;
using Vector = std::array<amrex::MultiFab,3>;
using View = ablastr::fields::VectorField;
using ConstView = ablastr::fields::ConstVectorField;
using warpx::fields::FieldType;

View Mutable (Vector& v) { return {&v[0],&v[1],&v[2]}; }
ConstView Const (Vector const& v) { return {&v[0],&v[1],&v[2]}; }
void Synchronize (Vector& v, amrex::Geometry const& g)
{
    for (auto& f:v) { f.OverrideSync(g.periodicity()); f.FillBoundary(g.periodicity()); }
}
void Initialize (Vector& out, View const& model)
{
    for (int c=0;c<3;++c) {
        out[c].define(model[c]->boxArray(),model[c]->DistributionMap(),1,model[c]->nGrowVect());
        out[c].setVal(0.);
    }
}
void Boundary (WarpX& sim, Vector& a, Vector const& zero)
{
    auto const& g=sim.Geom(0);
    amrex::GpuArray<int,AMREX_SPACEDIM> lo{},hi{};
    lo[1]=hi[1]=!g.isPeriodic(1);
    for (int c=0;c<3;++c) {
        auto const domain=amrex::convert(g.Domain(),a[c].ixType());
        int const high=domain.bigEnd(0); bool const nodal=a[c].ixType().nodeCentered(0);
        for (amrex::MFIter mfi(a[c]);mfi.isValid();++mfi) {
            auto const value=a[c].array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                if ((c==1 && i==0) || (nodal && i>=high)) { value(i,j,k)=0.; }
            });
        }
        a[c].FillBoundary(g.periodicity());
        ApplyDarwinCellCenteredABoundary(a[c],zero[c],g,true,nullptr,lo,hi);
    }
    auto v=Mutable(a); sim.ApplyFieldBoundaryOnAxis(v[0],v[1],v[2],0);
}
void Dump (char const* label, Vector const& fields, amrex::Geometry const& g)
{
    int const rank=amrex::ParallelDescriptor::MyProc();
    std::ofstream out(std::string(label)+"-rank"+std::to_string(rank)+".txt");
    out<<std::setprecision(17);
    for (int c=0;c<3;++c) {
        auto const& f=fields[c];auto owner=f.OwnerMask(g.periodicity());
        auto const measure=MakeQdsmcVolumeElement(g,f.ixType());
        for (amrex::MFIter mfi(f);mfi.isValid();++mfi) {
            auto const box=mfi.validbox();
            amrex::FArrayBox values(f[mfi].box(),1,amrex::The_Pinned_Arena());
            amrex::IArrayBox owned((*owner)[mfi].box(),1,amrex::The_Pinned_Arena());
            amrex::Gpu::dtoh_memcpy(values.dataPtr(),f[mfi].dataPtr(),f[mfi].size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(owned.dataPtr(),(*owner)[mfi].dataPtr(),(*owner)[mfi].size()*sizeof(int));
            auto const a=values.const_array(); auto const mask=owned.const_array();
            for (int j=box.smallEnd(1);j<=box.bigEnd(1);++j) {
                for (int i=box.smallEnd(0);i<=box.bigEnd(0);++i) {
                    if (mask(i,j,0)) {
                        out<<c<<' '<<i<<' '<<j<<' '<<f.ixType()[0]<<' '<<f.ixType()[1]
                           <<' '<<a(i,j,0)<<' '<<measure(i,j,0)<<'\n';
                    }
                }
            }
        }
    }
}
Real Dot (Vector const& a, Vector const& b, amrex::Geometry const& g)
{
    Real result=0.;
    for (int c=0;c<3;++c) {
        auto owner=a[c].OwnerMask(g.periodicity());auto const volume=MakeQdsmcVolumeElement(g,a[c].ixType());
        amrex::ReduceOps<amrex::ReduceOpSum> op;amrex::ReduceData<Real> data(op);
        using Tuple=typename decltype(data)::Type;
        for (amrex::MFIter mfi(a[c]);mfi.isValid();++mfi) {
            auto const x=a[c].const_array(mfi),y=b[c].const_array(mfi);auto const mask=owner->const_array(mfi);
            op.eval(mfi.validbox(),data,[=] AMREX_GPU_DEVICE(int i,int j,int k)->Tuple {
                return {mask(i,j,k)?x(i,j,k)*y(i,j,k)*volume(i,j,k):0.};
            });
        }
        result+=amrex::get<0>(data.value());
    }
    amrex::ParallelDescriptor::ReduceRealSum(result);return result;
}
void Run (WarpX& sim)
{
    auto const& native=sim.Geom(0);int periodic[2]={0,int(native.isPeriodic(1))};
    amrex::RealBox real(native.ProbLo(),native.ProbHi());
    amrex::Geometry g(native.Domain(),&real,1,periodic);
    auto const E=sim.m_fields.get_alldirs(FieldType::Efield_fp,0);
    auto const B=sim.m_fields.get_alldirs(FieldType::Bfield_fp,0);
    std::vector<amrex::MultiFab const*> accepted;
    for (int c=0;c<3;++c) { accepted.push_back(E[c]);accepted.push_back(B[c]); }
    accepted.push_back(sim.m_fields.get(FieldType::rho_fp,0));
    auto const before=FieldBytes(accepted);auto const particles=ParticleBytes(sim);
    Vector a,b,current,zero,longitudinal,transverse,bt,ct;
    for (auto* f:{&a,&current,&zero,&longitudinal,&transverse,&ct}) { Initialize(*f,E); }
    Initialize(b,B);Initialize(bt,B);
    auto const dx=g.CellSizeArray();bool const per=periodic[1];
    for (int c=0;c<3;++c) {
        auto const t=a[c].ixType();
        for (amrex::MFIter mfi(a[c]);mfi.isValid();++mfi) {
            auto const out=a[c].array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real const r=(i+.5*(1-t[0]))*dx[0],z=(j+.5*(1-t[1]))*dx[1];
                Real const phase=(per?2.:1.)*MathConst::pi*z;
                Real const radial=1.-r*r;
                out(i,j,k)=c==0?r*radial*std::cos(phase):
                    c==1?r*radial*radial*std::cos(phase):radial*radial*std::sin(phase);
            });
        }
    }
    Synchronize(a,g);Boundary(sim,a,zero);
    auto av=Mutable(a),bv=Mutable(b),cv=Mutable(current);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(bv,av,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    Synchronize(b,g);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(cv,bv,sim.GetEBUpdateEFlag()[0],0);
    Synchronize(current,g);
    warpx::darwin::InitialRateSchurOptions options;
    using BC=warpx::darwin::InitialRateBoundary;
    options.compatible_yee=true;options.relative_tolerance=1.e-12;options.absolute_tolerance=0.;
    options.lower={BC::Axis,per?BC::Periodic:BC::PMC};
    options.upper={BC::PEC,per?BC::Periodic:BC::PMC};options.output_ghosts=E[0]->nGrowVect();
    warpx::darwin::DarwinInitialRateSchur projection(g,sim.boxArray(0),sim.DistributionMap(0),options);
    amrex::MultiFab one(amrex::convert(sim.boxArray(0),amrex::IntVect(1)),sim.DistributionMap(0),1,1);
    one.setVal(1.);AMREX_ALWAYS_ASSERT(projection.Freeze(one));
    auto const result=projection.Correct(Const(a),Const(zero));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(result.converged,"original native RZ projection gate");
    auto const correction=projection.CorrectionField();
    for (int c=0;c<3;++c) {
        amrex::MultiFab::Copy(longitudinal[c],*correction[c],0,0,1,0);
        amrex::MultiFab::LinComb(transverse[c],1.,a[c],0,-1.,longitudinal[c],0,0,1,0);
    }
    Synchronize(transverse,g);Boundary(sim,transverse,zero);
    auto tv=Mutable(transverse),tbv=Mutable(bt),tcv=Mutable(ct);
    sim.get_pointer_fdtd_solver_fp(0)->ComputeCurlA(tbv,tv,sim.GetEBUpdateBFlag()[0],0,
        DarwinPMCCurlGrow(WarpX::field_boundary_lo,WarpX::field_boundary_hi));
    Synchronize(bt,g);
    sim.get_pointer_fdtd_solver_fp(0)->CalculateCurrentAmpere(tcv,tbv,sim.GetEBUpdateEFlag()[0],0);
    Synchronize(ct,g);
    Dump("A",a,g);Dump("B",b,g);Dump("C",current,g);
    Dump("PL",longitudinal,g);Dump("PT",transverse,g);Dump("BT",bt,g);Dump("CT",ct,g);
    Real const magnetic=Dot(b,b,g)/PhysConst::mu0,curl_work=Dot(a,current,g);
    Real const orthogonal=Dot(transverse,longitudinal,g);
    AMREX_ALWAYS_ASSERT(FieldBytes(accepted)==before && ParticleBytes(sim)==particles);
    amrex::Print()<<std::setprecision(17)<<"RZ_EVENT_OPERATOR periodic="<<per<<" n="<<g.Domain().length(0)
        <<" mu0="<<PhysConst::mu0<<" magnetic_norm="<<magnetic<<" curl_work="<<curl_work
        <<" curl_difference="<<magnetic-curl_work<<" projected_orthogonality="<<orthogonal
        <<" scalar_residual="<<result.residual<<" scalar_target="<<result.target
        <<" exact_fields_particles_rng=1\n";
}
int main (int argc,char** argv)
{
    warpx::initialization::initialize_external_libraries(argc,argv);
    { auto& sim=WarpX::GetInstance();sim.InitData();Run(sim);WarpX::Finalize(); }
    warpx::initialization::finalize_external_libraries();
}
