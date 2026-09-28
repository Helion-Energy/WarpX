/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef NATIVE_RADIAL_ENDPOINT_MOMENT_DIAGNOSTIC_H
#define NATIVE_RADIAL_ENDPOINT_MOMENT_DIAGNOSTIC_H
#include "FieldSolver/ImplicitSolvers/ImplicitIonEndpointTrial.H"
#include "Particles/MultiParticleContainer.H"
#include <AMReX_BoxIterator.H>
#include <array>
#include <cstring>
#include <iomanip>
#include <map>
#include <cmath>
#include <limits>

namespace radial_fixture {
// Complete source inputs exposed by the REAL private endpoint trial. No private
// ids accessor exists: compare every tuple byte as a multiset, while the main
// fixture separately checks the actual native endpoint image by particle id.
class EndpointSourceTuples {
    using Record=std::array<std::uint64_t,9>;
    std::vector<Record> m_records;
    static std::uint64_t bits(amrex::ParticleReal x) {
        static_assert(sizeof(x)==sizeof(std::uint64_t));
        std::uint64_t value;std::memcpy(&value,&x,sizeof(value));return value;
    }
    static amrex::ParticleReal value(std::uint64_t x) {
        amrex::ParticleReal r;std::memcpy(&r,&x,sizeof(r));return r;
    }
    template<class T> static std::vector<T> host(T const* p,long count) {
        std::vector<T> result(count);
        if(count)amrex::Gpu::dtoh_memcpy(result.data(),p,count*sizeof(T));
        return result;
    }
    void Append(warpx::thermal::IonExchangeParticleView const& v) {
        auto cells=host(v.cell,v.count);
        auto theta=host(v.theta,v.count),weight=host(v.weight,v.count);
        std::array<std::vector<amrex::ParticleReal>,3> momentum;
        for(int d=0;d<3;++d)momentum[d]=host(v.momentum[d],v.count);
        for(long i=0;i<v.count;++i)m_records.push_back({v.species,std::uint64_t(v.grid_index),
            std::uint64_t(cells[i][0]),std::uint64_t(cells[i][1]),bits(theta[i]),bits(weight[i]),
            bits(momentum[0][i]),bits(momentum[1][i]),bits(momentum[2][i])});
    }
    using Cell=std::array<std::uint64_t,3>;
    struct SumBound {
        std::uint64_t count=0;
        long double absolute=0;
        amrex::Real bound=0;
        bool valid=true;
    };
    static long double up(long double x) {
        return std::nextafter(x,std::numeric_limits<long double>::infinity());
    }
    static long double down(long double x) {return std::nextafter(x,0.L);}
    static long double gamma(std::uint64_t depth,long double eps) {
        long double const product=up(depth*eps);
        return up(product/down(1-product));
    }
    std::map<Cell,SumBound> Bounds(int component) const {
        using Real=amrex::Real;
        int const species=component/7,channel=component%7;
        std::map<Cell,SumBound> sums;
        // One binade of margin prevents host-product rounding from admitting
        // a device underflow/overflow edge; such extreme inputs are unsupported.
        auto normal=[](long double x) {
            return std::isfinite(x) && (x==0 ||
                (x>=2*std::numeric_limits<Real>::min() && x<=std::numeric_limits<Real>::max()/2));
        };
        for(auto const& r:m_records) {
            if(r[0]!=std::uint64_t(species))continue;
            auto& sum=sums[{r[1],r[2],r[3]}];++sum.count;
            long double term=value(r[5]);sum.valid=sum.valid && normal(term);
            if(channel) {
                long double const u=std::abs(static_cast<long double>(value(r[6+(channel-1)%3])));
                term*=u;sum.valid=sum.valid && normal(term);
                if(channel>=4){term*=u;sum.valid=sum.valid && normal(term);}
            }
            sum.absolute+=term;sum.valid=sum.valid && normal(sum.absolute);
        }
        for(auto& entry:sums) {
            auto& sum=entry.second;
            auto const depth=sum.count+2; // BOTH product roundings plus the atomic sum
            long double const eps=std::numeric_limits<Real>::epsilon();
            long double const eps_host=std::numeric_limits<long double>::epsilon();
            sum.valid=sum.valid && depth*eps<.01L;
            if(!sum.valid || sum.absolute==0)continue;
            long double const gd=gamma(depth,eps),gh=gamma(depth,eps_host);
            long double const upper=up(sum.absolute/down(1-gh));
            // Two independently rounded product+sum paths, then subtraction.
            long double const bound=up(up(up(1+eps)*up(2*gd))*upper);
            sum.valid=normal(bound) && std::isfinite(upper) &&
                upper<down(std::numeric_limits<Real>::max()/(1+gd));
            if(sum.valid)sum.bound=std::nextafter(static_cast<Real>(bound),std::numeric_limits<Real>::infinity());
        }
        return sums;
    }
public:
    static EndpointSourceTuples Private(warpx::thermal::ImplicitIonEndpointTrial const& trial) {
        amrex::Gpu::synchronize();EndpointSourceTuples result;
        for(auto const& v:trial.Views())result.Append(v);
        std::sort(result.m_records.begin(),result.m_records.end());return result;
    }
    static EndpointSourceTuples Native(WarpX& w,std::vector<warpx::thermal::KineticSpeciesDescriptor> const& descriptors) {
        amrex::Gpu::synchronize();EndpointSourceTuples result;std::size_t species=0;
        auto const plo=w.Geom(0).ProbLoArray(),dxi=w.Geom(0).InvCellSizeArray();
        for(auto const& pc:w.GetPartContainer()) {
            if(pc->getCharge()==0)continue;
            auto const s=species++;if(descriptors[s].relaxation_excluded)continue;
            for(WarpXParIter pti(*pc,0);pti.isValid();++pti) {
                amrex::Gpu::DeviceVector<amrex::IntVect> storage(pti.numParticles());auto* cells=storage.data();
                auto const data=pti.GetParticleTile().getParticleTileData();
                amrex::ParallelFor(pti.numParticles(),[=] AMREX_GPU_DEVICE(long i) {
                    cells[i]=amrex::getParticleCell(WarpXParticleContainer::ParticleType(data,i),plo,dxi);});
                warpx::thermal::IonExchangeParticleView v;v.species=s;v.grid_index=pti.index();
                v.count=pti.numParticles();v.cell=cells;v.theta=pti.GetAttribs(PIdx::theta).dataPtr();
                v.weight=pti.GetAttribs(PIdx::w).dataPtr();
                v.momentum={pti.GetAttribs(PIdx::ux).dataPtr(),pti.GetAttribs(PIdx::uy).dataPtr(),pti.GetAttribs(PIdx::uz).dataPtr()};
                amrex::Gpu::synchronize();result.Append(v);
            }
        }
        std::sort(result.m_records.begin(),result.m_records.end());return result;
    }
    void RequireEqual(EndpointSourceTuples const& other,std::string const& phase) const {
        amrex::Long changed=std::abs(amrex::Long(m_records.size())-amrex::Long(other.m_records.size()));
        for(std::size_t n=0;n<std::min(m_records.size(),other.m_records.size());++n) {
            if(m_records[n]!=other.m_records[n]) {
                if(changed==0) {
                    amrex::AllPrint()<<"ENDPOINT_TUPLE_FIRST_MISMATCH phase="<<phase<<" tuple="<<n<<std::endl;
                    for(int c=0;c<9;++c)amrex::AllPrint()<<" tuple_component="<<c<<" private_bits="
                        <<m_records[n][c]<<" native_bits="<<other.m_records[n][c]<<std::endl;
                }
                ++changed;
            }
        }
        amrex::Long count=m_records.size();amrex::ParallelDescriptor::ReduceLongSum(count);
        amrex::ParallelDescriptor::ReduceLongSum(changed);
        amrex::Print()<<"ENDPOINT_SOURCE_TUPLES phase="<<phase<<" count="<<count
            <<" bytes_per_tuple="<<sizeof(Record)<<" changed="<<changed<<std::endl;
        AMREX_ALWAYS_ASSERT(changed==0);
    }
    void CheckMomentSummation(amrex::MultiFab const& p,amrex::MultiFab const& n,
                              std::string const& phase) const {
        using Real=amrex::Real;
        for(int c=0;c<p.nComp();++c) {
            auto const bounds=Bounds(c);int bad=0;
            amrex::Long count=0;Real measurements[3]={0.,0.,0.};
            for(auto const& entry:bounds) {
                bad=std::max(bad,int(!entry.second.valid));
                count=std::max(count,amrex::Long(entry.second.count));
            }
            for(amrex::MFIter mfi(p);mfi.isValid();++mfi) {
                auto const box=p[mfi].box();std::vector<Real> a(box.numPts()),b(a.size());
                amrex::Gpu::dtoh_memcpy(a.data(),p[mfi].dataPtr(c),a.size()*sizeof(Real));
                amrex::Gpu::dtoh_memcpy(b.data(),n[mfi].dataPtr(c),b.size()*sizeof(Real));
                for(amrex::BoxIterator it(mfi.validbox());it.ok();++it) {
                    auto const index=box.index(it());
                    auto const found=bounds.find({std::uint64_t(mfi.index()),std::uint64_t(it()[0]),std::uint64_t(it()[1])});
                    SumBound const sum=found==bounds.end() ? SumBound{} : found->second;
                    Real const delta=std::abs(a[index]-b[index]),bound=sum.bound;
                    bool const finite=std::isfinite(a[index]) && std::isfinite(b[index]) && std::isfinite(delta);
                    bool const ok=sum.valid && finite && (sum.absolute==0 ?
                        (a[index]==0. && b[index]==0.) : delta<=bound);
                    bad=std::max(bad,int(!ok));
                    measurements[0]=std::max(measurements[0],delta);
                    measurements[1]=std::max(measurements[1],bound);
                    measurements[2]=std::max(measurements[2],bound>0. ? delta/bound : (ok ? 0. : 2.));
                }
            }
            amrex::ParallelDescriptor::ReduceLongMax(count);amrex::ParallelDescriptor::ReduceIntMax(bad);
            amrex::ParallelDescriptor::ReduceRealMax(measurements,3);
            amrex::Print()<<std::setprecision(17)<<"ENDPOINT_MOMENT_ROUNDOFF phase="<<phase
                <<" species_index="<<c/7<<" component="<<c%7<<" count_max="<<count
                <<" depth_max="<<(count ? count+2 : 0)<<" difference_max="<<measurements[0]
                <<" bound_max="<<measurements[1]<<" max_node_fraction="<<measurements[2]
                <<" invalid="<<bad<<std::endl;
            AMREX_ALWAYS_ASSERT(bad==0);
        }
    }
    void InjectMomentCorruption(amrex::MultiFab& p,std::string const& kind) const {
        using Real=amrex::Real;
        if(kind=="empty") {
            int const component=p.nComp()-1;AMREX_ALWAYS_ASSERT(Bounds(component).empty());
            p.setVal(1.e-30,component,1,0);return;
        }
        AMREX_ALWAYS_ASSERT(kind=="nonzero");auto const bounds=Bounds(0);
        for(amrex::MFIter mfi(p);mfi.isValid();++mfi) {
            auto const box=p[mfi].box();std::vector<Real> values(box.numPts());
            amrex::Gpu::dtoh_memcpy(values.data(),p[mfi].dataPtr(0),values.size()*sizeof(Real));
            for(amrex::BoxIterator it(mfi.validbox());it.ok();++it) {
                auto const found=bounds.find({std::uint64_t(mfi.index()),std::uint64_t(it()[0]),std::uint64_t(it()[1])});
                if(found!=bounds.end() && found->second.bound>0.)values[box.index(it())]+=4*found->second.bound;
            }
            amrex::Gpu::htod_memcpy(p[mfi].dataPtr(0),values.data(),values.size()*sizeof(Real));
        }
    }

    amrex::Real PositiveScale(int component) const {
        // Independently sum absolute per-particle moment terms in host long
        // double, by exact stored source cell. Diagnostic only; no gate fitted.
        int const species=component/7,channel=component%7;
        std::map<std::array<std::uint64_t,3>,long double> sums;
        for(auto const& r:m_records) {
            if(r[0]!=std::uint64_t(species))continue;
            long double term=value(r[5]);
            if(channel) {
                long double const u=value(r[6+(channel-1)%3]);
                term*=std::abs(u);if(channel>=4)term*=std::abs(u);
            }
            sums[{r[1],r[2],r[3]}]+=term;
        }
        long double maximum=0;for(auto const& entry:sums)maximum=std::max(maximum,entry.second);
        amrex::Real result=maximum;amrex::ParallelDescriptor::ReduceRealMax(result);return result;
    }
};

inline void ReportMomentDifferences(amrex::MultiFab const& p,amrex::MultiFab const& n,
    std::vector<warpx::thermal::KineticSpeciesDescriptor> const& species,
    EndpointSourceTuples const& inputs,std::string const& phase) {
    using Real=amrex::Real;
    amrex::MultiFab delta(p.boxArray(),p.DistributionMap(),p.nComp(),0);
    amrex::MultiFab::Copy(delta,p,0,0,p.nComp(),0);amrex::MultiFab::Subtract(delta,n,0,0,p.nComp(),0);
    for(int c=0;c<p.nComp();++c) {
        Real const difference=delta.norminf(c),reference=n.norminf(c),positive=inputs.PositiveScale(c);
        amrex::Print()<<std::setprecision(17)<<"ENDPOINT_MOMENT_DIAGNOSTIC phase="<<phase
            <<" species="<<species[c/7].name<<" component="<<c%7<<" absolute="<<difference
            <<" reference="<<reference<<" private="<<p.norminf(c)<<" positive="<<positive
            <<" original_ratio="<<difference/std::max(reference,Real(1.e-30))<<std::endl;
        bool printed=false;
        if(difference>0.)for(amrex::MFIter mfi(delta);mfi.isValid();++mfi) {
            auto const box=delta[mfi].box();std::vector<Real> d(box.numPts()),a(d.size()),b(d.size());
            amrex::Gpu::dtoh_memcpy(d.data(),delta[mfi].dataPtr(c),d.size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(a.data(),p[mfi].dataPtr(c),a.size()*sizeof(Real));
            amrex::Gpu::dtoh_memcpy(b.data(),n[mfi].dataPtr(c),b.size()*sizeof(Real));
            for(amrex::BoxIterator it(mfi.validbox());it.ok();++it) {
                auto const index=box.index(it());
                if(!printed && std::abs(d[index])==difference) {
                    amrex::AllPrint()<<std::setprecision(17)<<"ENDPOINT_MOMENT_LOCATION phase="<<phase
                        <<" species="<<species[c/7].name<<" component="<<c%7<<" grid="<<mfi.index()
                        <<" cell="<<it()<<" private="<<a[index]<<" native="<<b[index]
                        <<" difference="<<d[index]<<std::endl;printed=true;
                }
            }
        }
    }
}
} // namespace radial_fixture
#endif
