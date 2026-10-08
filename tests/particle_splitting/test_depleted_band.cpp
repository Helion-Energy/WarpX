/* Private depleted-band conservation regression. License: BSD-3-Clause-LBNL. */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "WarpX.H"
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>
using namespace amrex::literals;

int main(int argc, char* argv[]) try
{
    std::vector<std::string> args{argv[0], "max_step=0", "warpx.verbose=0",
        "amr.max_level=0", "amr.max_grid_size=8", "amr.blocking_factor=8",
        "algo.maxwell_solver=yee", "algo.particle_shape=3", "warpx.use_filter=0",
        "amrex.the_arena_is_managed=0", "amrex.the_arena_init_size=0",
        "amrex.signal_handling=0", "amrex.throw_exception=1",
        "geometry.dims=RZ", "amr.n_cell=16 16", "geometry.prob_lo=0 0", "geometry.prob_hi=1 1",
        "boundary.field_lo=none pec", "boundary.field_hi=pec pec",
        "boundary.particle_lo=none reflecting", "boundary.particle_hi=reflecting reflecting",
        "particles.species_names=ions_D ions_T"};
    for (const std::string name : {"ions_D", "ions_T"}) {
        for (const std::string option : {".charge=q_e", ".injection_style=NUniformPerCell",
             ".profile=constant", ".density=0", ".momentum_distribution_type=at_rest",
             ".num_particles_per_cell_each_dim=1 1 1", ".hybrid_split_target_ppc=5"}) {
            args.push_back(name+option);
        }
    }
    args.emplace_back("ions_D.mass=2*m_p");
    args.emplace_back("ions_T.mass=3*m_p");
    for (int i=1; i<argc; ++i) { args.emplace_back(argv[i]); }
    std::vector<char*> ptrs;
    ptrs.reserve(args.size());
    for (auto& arg:args) { ptrs.push_back(arg.data()); }
    const int nargs=static_cast<int>(ptrs.size());
    char** pargs=ptrs.data();
    warpx::initialization::initialize_external_libraries(nargs,pargs);
    {
        auto& warpx=WarpX::GetInstance();
        warpx.InitData();
        for (int isp=0; isp<2; ++isp) {
            auto& pc=dynamic_cast<PhysicalParticleContainer&>(warpx.GetPartContainer().GetParticleContainer(isp));
            const std::string name=isp==0 ? "ions_D" : "ions_T";
            int target=0;
            const amrex::ParmParse pp(name);
            pp.get("hybrid_split_target_ppc", target);
            AMREX_ALWAYS_ASSERT(target>=3);
            pc.AddRealComp("inherited_real",true);
            pc.AddIntComp("parent_index",true);
            amrex::MultiFab rho(amrex::convert(pc.ParticleBoxArray(0),amrex::IntVect::TheNodeVector()),
                               pc.ParticleDistributionMap(0),1,1);
            rho.setVal(1);
            for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
                const auto a=rho.array(mfi);
                amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    if (i>=12 && j<=5) { a(i,j,k)=4; }
                });
            }
            pc.SplitDepletedBand(rho,.5_rt,1.5_rt,0); // empty species
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles()==0);
            // Axis, radial wall, both axial walls, grid/rank interfaces,
            // exactly/above target, excluded density band, and an empty cell.
            const std::array<double,12> rs{1e-10,.9999999999,.2,.3,.5000000001,.49,.65,.7,.875,.9,0,.5};
            const std::array<double,12> zs{1e-10,.6,.9999999999,1e-10,.49,.5000000001,.65,.7,.25,.9,.3125,.5625};
            const std::array<int,12> count{1,2,3,target-1,1+isp,2,target,target+1,1,0,1,1};
            amrex::Vector<amrex::ParticleReal> x,y,z,ux,uy,uz;
            amrex::Vector<amrex::Vector<amrex::ParticleReal>> weight(1);
            std::vector<int> expected_count,cell_r,cell_z;
            std::vector<double> r0,theta0;
            for (int cell=0; cell<12; ++cell) {
                for (int n=0; n<count[cell]; ++n) {
                    const int p=static_cast<int>(x.size());
                    const double theta=cell>=10 ? 0 : .13*(p%11);
                    x.push_back(rs[cell]*std::cos(theta)); y.push_back(rs[cell]*std::sin(theta));
                    z.push_back(zs[cell]);
                    ux.push_back(1e5*(p+1)); uy.push_back((p%2?1:-1)*2e5); uz.push_back(3e5);
                    weight[0].push_back(4.0+.25*p);
                    const double radius=std::hypot(x.back(),y.back());
                    r0.push_back(radius);theta0.push_back(theta);
                    cell_r.push_back(static_cast<int>(std::floor(radius*16)));
                    cell_z.push_back(static_cast<int>(std::floor(z.back()*16)));
                    expected_count.push_back(cell==8 || count[cell]>=target ? 1 : (target+count[cell]-1)/count[cell]);
                }
            }
            const int valid_parents=static_cast<int>(x.size());
            // An invalid marker in the one-parent axis cell must neither count
            // towards PPC nor be resurrected by splitting.
            x.push_back(x[0]); y.push_back(y[0]); z.push_back(z[0]);
            ux.push_back(1e5*(valid_parents+1)); uy.push_back(0); uz.push_back(0); weight[0].push_back(1000);
            pc.AddNParticles(0,static_cast<int>(x.size()),x,y,z,ux,uy,uz,1,weight,0,{},0,-1);
            for (WarpXParIter pti(pc,0); pti.isValid(); ++pti) {
                const auto a=pti.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(int i) {
                    const int parent=static_cast<int>(a.m_rdata[PIdx::ux][i]/1e5_prt)-1;
                    a.m_runtime_rdata[0][i]=.125_prt+a.m_rdata[PIdx::ux][i];
                    a.m_runtime_idata[0][i]=parent;
                    if (parent==valid_parents) { a.m_idcpu[i]=amrex::ParticleIdCpus::Invalid; }
                });
            }
            amrex::Gpu::streamSynchronize();
            pc.SplitDepletedBand(rho,.5_rt,1.5_rt,0);
            pc.Redistribute();
            std::vector<int> observed(valid_parents,0);
            std::vector<double> moments(8*valid_parents,0);
            const auto tol=1024*std::numeric_limits<amrex::ParticleReal>::epsilon();
            std::map<std::pair<int,int>,int> occupancy;
            std::set<std::uint64_t> unique_local_ids;
            for (WarpXParIter pti(pc,0); pti.isValid(); ++pti) {
                const auto& soa=pti.GetStructOfArrays();
                const auto np=pti.numParticles();
                const std::array<int,8> comps{PIdx::r,PIdx::theta,PIdx::z,PIdx::w,PIdx::ux,PIdx::uy,PIdx::uz,
                                              pc.GetRealCompIndex("inherited_real")};
                std::array<amrex::Gpu::PinnedVector<amrex::ParticleReal>,8> values;
                for (int c=0;c<8;++c) {
                    values[c].resize(np);
                    const auto& v=soa.GetRealData(comps[c]);
                    amrex::Gpu::copy(amrex::Gpu::deviceToHost,v.begin(),v.end(),values[c].begin());
                }
                amrex::Gpu::PinnedVector<int> parent(np);
                const auto& parent_device=soa.GetIntData(pc.GetIntCompIndex("parent_index"));
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,parent_device.begin(),parent_device.end(),parent.begin());
                amrex::Gpu::PinnedVector<std::uint64_t> ids(np);
                const auto& id_device=soa.GetIdCPUData();
                amrex::Gpu::copy(amrex::Gpu::deviceToHost,id_device.begin(),id_device.end(),ids.begin());
                for (long i=0;i<np;++i) {
                    const int p=parent[i];
                    AMREX_ALWAYS_ASSERT(p>=0 && p<valid_parents);
                    AMREX_ALWAYS_ASSERT(amrex::ConstParticleIDWrapper{ids[i]}.is_valid());
                    AMREX_ALWAYS_ASSERT(unique_local_ids.insert(ids[i]).second);
                    const double r=values[0][i], theta=values[1][i], zz=values[2][i], w=values[3][i];
                    AMREX_ALWAYS_ASSERT(r>=0 && r<1 && zz>=0 && zz<1);
                    AMREX_ALWAYS_ASSERT(static_cast<int>(std::floor(r*16))==cell_r[p]);
                    AMREX_ALWAYS_ASSERT(static_cast<int>(std::floor(zz*16))==cell_z[p]);
                    AMREX_ALWAYS_ASSERT(std::abs(theta-theta0[p])<tol);
                    AMREX_ALWAYS_ASSERT(values[4][i]==ux[p] && values[5][i]==uy[p] && values[6][i]==uz[p]);
                    AMREX_ALWAYS_ASSERT(values[7][i]==.125_prt+ux[p]);
                    AMREX_ALWAYS_ASSERT(std::abs(w-weight[0][p]/expected_count[p])<tol*weight[0][p]);
                    ++observed[p]; ++occupancy[{cell_r[p],cell_z[p]}];
                    moments[8*p]+=w;
                    moments[8*p+1]+=w*values[4][i];moments[8*p+2]+=w*values[5][i];moments[8*p+3]+=w*values[6][i];
                    moments[8*p+4]+=w*(values[4][i]*values[4][i]+values[5][i]*values[5][i]+values[6][i]*values[6][i]);
                    moments[8*p+5]+=w*r*std::cos(theta);moments[8*p+6]+=w*r*std::sin(theta);moments[8*p+7]+=w*zz;
                }
            }
            amrex::ParallelDescriptor::ReduceIntSum(observed.data(),valid_parents);
            amrex::ParallelDescriptor::ReduceRealSum(moments.data(),static_cast<int>(moments.size()));
            long expected_total=0;
            for (int p=0;p<valid_parents;++p) {
                AMREX_ALWAYS_ASSERT(observed[p]==expected_count[p]);expected_total+=expected_count[p];
                const std::array<double,8> expected{1,ux[p],uy[p],uz[p],ux[p]*ux[p]+uy[p]*uy[p]+uz[p]*uz[p],x[p],y[p],z[p]};
                for (int c=0;c<8;++c) {
                    const double wanted=weight[0][p]*expected[c];
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(moments[8*p+c]-wanted)<=tol*std::max(1.0,std::abs(wanted)),
                                                      "Split must preserve each parent's weight, momentum, energy and centroid");
                }
            }
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles()==expected_total);
            // At fixed state, the occupied-cell floor is satisfied in one call.
            // A second call must not grow the population or lose any weight.
            const double total_weight=pc.sumParticleWeight(false);
            const double total_energy=pc.sumParticleEnergy(false);
            pc.SplitDepletedBand(rho,.5_rt,1.5_rt,0);pc.Redistribute();
            AMREX_ALWAYS_ASSERT(pc.TotalNumberOfParticles()==expected_total);
            AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleWeight(false)-total_weight)<=tol*total_weight);
            AMREX_ALWAYS_ASSERT(std::abs(pc.sumParticleEnergy(false)-total_energy)<=tol*total_energy);
            amrex::Print()<<"Depleted-band boundaries, PPC floor, no-repeat growth, attributes and moments: PASS species="
                          <<name<<" target="<<target<<" parents="<<valid_parents<<" final="<<expected_total<<"\n";
        }
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
    return 0;
} catch (const std::exception& e) {
    amrex::Abort(e.what());
}
