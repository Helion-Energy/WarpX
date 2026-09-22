/* Copyright 2026 The WarpX Community
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_HYBRID_PIC_RZ_OPERATOR_H_
#define WARPX_HYBRID_PIC_RZ_OPERATOR_H_
#ifdef WARPX_DIM_RZ

#include "FlexibleGMRES.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <memory>
#include <vector>

/** Native staggered m=0 RZ realization of the scaled electric/curl-curl
 * primal pair. Cartesian implementations live in HybridPICPC.H.
 *
 * Each component retains its own IndexType and physical boundary DOFs.
 * The pair is (E, W_hat), W_hat = s curl curl E. We unscale W_hat BEFORE
 * the spatial operators, including the variable-scale hyper-resistive leg.
 * This approximate Jacobian leaves the physical nonlinear residual intact.
 */
class HybridPICRZOperator {
  public:
    using RT = amrex::Real;
    using MF = amrex::MultiFab;
    using Pair =
        amrex::Array<std::unique_ptr<MF>, 3>; // E,W_hat components in each MF
    using Field = amrex::Array<std::unique_ptr<MF>, 3>;
    using BC = amrex::Array<FieldBoundaryType, 2>;
    // Edge coefficients: Hall prefactor, resistivity, hyper-resistivity,
    // inverse Ohm density times vacuum gate, auxiliary scale, ion response,
    // active Ohm-row weight.
    static constexpr int Hall = 0, Eta = 1, Hyper = 2, InvRho = 3, Scale = 4,
                         Ion = 5, RowWeight = 6;
    static constexpr int NC = 7;
    // Nodal coefficients: B[3], theta*dt*(Ji-Jp)[3], inertia length squared.
    static constexpr int NN = 7;

    void Define (amrex::Geometry const&, amrex::BoxArray const&,
                 amrex::DistributionMapping const&, BC const&, BC const&,
                 int mg_floor = 4);
    void PrepareCoefficients ();
    void Solve (Field const& out, Field const& rhs, int inner_max,
                RT inner_rtol, int vcycles);
    void ApplyField (Field const& out, Field const& in);
    void ApplyPair (int lev, Pair& out, Pair const& in);
    void Vcycle (int lev);
    void Smooth (int lev, int sweeps);
    void Restrict (int lev);
    void Prolong (int lev);
    void Fill (MF&, int lev, int component, bool magnetic,
               bool coefficient = false) const;
    void FillNodes (MF&, int lev) const;
    void Curl (Field const& out, Field const& in, int lev, bool upward) const;
    void PrepareCurlSource (int lev);
    void RestoreRows (MF& out, MF const& rhs, int lev, int component) const;

    [[nodiscard]] Pair makeVecRHS () const;
    [[nodiscard]] Pair
    makeVecLHS () const {
        return makeVecRHS();
    }
    void
    apply (Pair& out, Pair const& in) {
        ApplyPair(0, out, in);
    }
    void precond (Pair& out, Pair const& in);
    void assign (Pair& out, Pair const& in) const;
    void setToZero (Pair& x) const;
    void scale (Pair& x, RT a) const;
    void increment (Pair& x, Pair const& y, RT a) const;
    void linComb (Pair& out, RT a, Pair const& x, RT b, Pair const& y) const;
    [[nodiscard]] RT dotProduct (Pair const& x, Pair const& y) const;
    [[nodiscard]] RT norm2 (Pair const& x) const;

    struct Level {
        amrex::Geometry geom;
        amrex::BoxArray cells;
        Pair x, b, residual, action, correction;
        Field coeff, physical_w, electric, curl_source, curl_e, curlcurl_e,
            ion_current;
        std::unique_ptr<MF> nodes, cross, smoother;
        amrex::Array<std::unique_ptr<amrex::iMultiFab>, 3> owner;
    };
    std::vector<std::unique_ptr<Level>> levels;
    BC lo, hi;
    RT sigma = RT(1.3), sigma_w = RT(1.2), whistler_defect = RT(1.);
    bool verbose = false;
    bool pin_vector_potential = false;
    bool ion_response = true;
    int last_iterations = 0;
    RT last_residual = RT(0);
};
#endif
#endif
