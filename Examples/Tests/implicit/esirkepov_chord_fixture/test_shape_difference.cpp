/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#include "Particles/Deposition/EsirkepovShapeDifference.H"
#include "Particles/ShapeFactors.H"
#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <quadmath.h>

using Quad=__float128;
template<int Order> void Check(std::ofstream& out) {
    double worst_stable=0.,worst_original=0.,worst_sum=0.;int samples=0;
    std::mt19937_64 rng(710000+Order);std::uniform_real_distribution<double> dist(2.,12.);
    for(int n=0;n<1000;++n)for(int power:{1,3,5,7,9,11,13}) {
        double a=dist(rng),b=a+std::pow(10.,-power)*(n%2?-1.:1.);
        if(n%4==0) {double const knot=7.+(Order%2==0?.5:0.);a=knot-.25*std::pow(10.,-power);b=knot+.75*std::pow(10.,-power);}
        std::array<double,Order+1> old{},now{};std::array<Quad,Order+1> qo{},qn{};
        int const io=Compute_shape_factor<Order>{}(old.data(),a),in=Compute_shape_factor<Order>{}(now.data(),b);
        AMREX_ALWAYS_ASSERT(Compute_shape_factor<Order>{}(qo.data(),Quad(a))==io);
        AMREX_ALWAYS_ASSERT(Compute_shape_factor<Order>{}(qn.data(),Quad(b))==in);
        double const delta=std::abs(b-a);double sum=0.;
        for(int i=std::min(io,in);i<=std::max(io,in)+Order;++i) {
            double const vo=i>=io&&i<=io+Order?old[i-io]:0.;double const vn=i>=in&&i<=in+Order?now[i-in]:0.;
            Quad const qvo=i>=io&&i<=io+Order?qo[i-io]:Quad(0);Quad const qvn=i>=in&&i<=in+Order?qn[i-in]:Quad(0);
            Quad const exact=qvo-qvn;
            double const value=warpx::particles::EsirkepovShapeDifference<Order>(a,b,io,in,i,vo,vn);
            double const stable=double(fabsq(Quad(value)-exact))/delta;
            double const original=double(fabsq(Quad(vo-vn)-exact))/delta;
            worst_stable=std::max(worst_stable,stable);worst_original=std::max(worst_original,original);sum+=value;
            AMREX_ALWAYS_ASSERT(stable<128*std::numeric_limits<double>::epsilon());
            double const reverse=warpx::particles::EsirkepovShapeDifference<Order>(b,a,in,io,i,vn,vo);
            AMREX_ALWAYS_ASSERT(std::abs(value+reverse)<64*std::numeric_limits<double>::epsilon()*delta);
            double const repeated=warpx::particles::EsirkepovShapeDifference<Order>(a,b,io,in,i,vo,vn);
            AMREX_ALWAYS_ASSERT(value==repeated);++samples;
        }
        worst_sum=std::max(worst_sum,std::abs(sum)/delta);
        AMREX_ALWAYS_ASSERT(std::abs(sum)<128*std::numeric_limits<double>::epsilon()*delta);
    }
    out<<"{\"order\":"<<Order<<",\"samples\":"<<samples<<",\"max_stable_error_over_displacement\":"<<worst_stable<<",\"max_original_error_over_displacement\":"<<worst_original<<",\"max_partition_defect_over_displacement\":"<<worst_sum<<"}\n";
}
int main(int argc,char** argv) {
    amrex::Initialize(argc,argv);
    {
        std::ofstream out;if(amrex::ParallelDescriptor::IOProcessor())out.open("SHAPE_DIFFERENCE.jsonl");out<<std::setprecision(17);
        Check<1>(out);Check<2>(out);Check<3>(out);Check<4>(out);
        amrex::Print()<<"PASS SHAPE_DIFFERENCE order1-4 represented-endpoint quad oracle, support crossings, partition and A/B/A\n";
    }
    amrex::Finalize();
}
