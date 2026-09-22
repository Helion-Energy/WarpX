/* Copyright 2026 Prabhat Kumar
 * This file is part of WarpX. License: BSD-3-Clause-LBNL
 * Implementation detail included by HybridPICPC.H. See its provenance.
 */
#ifndef WARPX_HYBRID_PIC_PC_SOLVE_H_
#define WARPX_HYBRID_PIC_PC_SOLVE_H_

template <class T, class Ops>
void HybridPICPC<T,Ops>::InnerGMRES ()
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::InnerGMRES()");

    const int m = m_inner_max;
    constexpr int nkb = kNKB;
    // the s_W-weighted block norm is E-space accurate: inner_rtol applies
    // directly
    const RT rtol = m_inner_rtol;

    // v_0 = b/||b|| (the RHS block was assembled by Apply() into m_bKB)
    amrex::MultiFab::Copy(*m_kryB, *m_bKB, 0, 0, nkb, 0);
    m_xW[0]->setVal(0.0);
    m_xF[0]->setVal(0.0);
    std::vector<RT> d1(m + 2, RT(0.0));
    std::vector<RT> d2(m + 2, RT(0.0));
    BatchDot(*m_kryB, 0, true, d1.data());
    const RT beta = std::sqrt(d1[0]);
    if (beta <= RT(0.0)) { return; }
    m_kryB->mult(RT(1.0)/beta, 0, nkb, 0);

    std::vector<std::vector<RT>> H(m + 1, std::vector<RT>(m, RT(0.0)));
    std::vector<RT> cs(m, RT(0.0));
    std::vector<RT> sn(m, RT(0.0));
    std::vector<RT> g(m + 1, RT(0.0));
    g[0] = beta;
    amrex::GpuArray<amrex::Real, kMaxKry> cb{};

    int its = 0;
    for (int j = 0; j < m; j++) {
        its = j + 1;

        // z = M v_j (one V-cycle; result merged into the ghost-filled block
        // m_zB), w = A z
        ApplyMToW(*m_kryB, j*nkb);
        MergeBlock(*m_zB);
        FillGhosts(*m_zB, 0);
        ApplyDomainClosure(*m_zB, 0, ClosureClass::JField, 0, 3);
        ApplyDomainClosure(*m_zB, 0, ClosureClass::EField, 3, 3);
        ApplyReducedOp(*m_zB, *m_wB);

        // Classical Gram-Schmidt with reorthogonalization (CGS2):
        //   pass 1: d1 = V^T w,        w1 = w  - V d1
        //   pass 2: d2 = V^T w1,       w2 = w1 - V d2,  H_j = d1 + d2,
        //   h_{j+1,j} = ||w2|| = sqrt(||w1||^2 - ||d2||^2)  (Pythagoras; an
        //   explicit norm is taken when the second projection is not tiny)
        // Two batched reductions (two host syncs) per iteration instead of
        // the j+2 sequential ones of modified Gram-Schmidt; identical in
        // exact arithmetic.
        BatchDot(*m_wB, j + 1, false, d1.data());
        for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d1[i]); }
        BlockAxpy(*m_wB, 0, m_wB.get(), j + 1, cb, 1.0_rt);
        BatchDot(*m_wB, j + 1, true, d2.data());
        RT d2sq = RT(0.0);
        for (int i = 0; i <= j; i++) {
            H[i][j] = d1[i] + d2[i];
            d2sq += d2[i]*d2[i];
        }
        RT h2 = d2[j+1] - d2sq;
        const bool explicit_norm = (d2sq > RT(1.0e-8) * d2[j+1]) || (h2 <= RT(0.0));
        if (explicit_norm) {
            // near-breakdown regime: form w2 and measure it directly
            for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d2[i]); }
            BlockAxpy(*m_wB, 0, m_wB.get(), j + 1, cb, 1.0_rt);
            RT ww[1] = {RT(0.0)};
            BatchDot(*m_wB, 0, true, ww);
            h2 = ww[0];
        }
        const RT hj1 = std::sqrt(amrex::max(h2, RT(0.0)));

        for (int i = 0; i < j; i++) {
            const RT t = cs[i]*H[i][j] + sn[i]*H[i+1][j];
            H[i+1][j] = -sn[i]*H[i][j] + cs[i]*H[i+1][j];
            H[i][j] = t;
        }
        const RT denom = std::sqrt(H[j][j]*H[j][j] + hj1*hj1);
        cs[j] = H[j][j] / denom;
        sn[j] = hj1 / denom;
        H[j][j] = denom;
        g[j+1] = -sn[j]*g[j];
        g[j] = cs[j]*g[j];

        const bool converged = (std::abs(g[j+1]) <= rtol * beta);
        const bool breakdown = (hj1 <= RT(1.0e-14) * beta);
        if (converged || breakdown || j + 1 == m) { break; }

        // v_{j+1} = w2 / h_{j+1,j}, fused with the second projection when
        // w2 was not materialized above
        if (explicit_norm) {
            BlockAxpy(*m_kryB, (j+1)*nkb, m_wB.get(), 0, cb, 1.0_rt/static_cast<amrex::Real>(hj1));
        } else {
            for (int i = 0; i <= j; i++) { cb[i] = static_cast<amrex::Real>(d2[i]); }
            BlockAxpy(*m_kryB, (j+1)*nkb, m_wB.get(), j + 1, cb,
                      1.0_rt/static_cast<amrex::Real>(hj1));
        }
    }

    std::vector<RT> y(its, RT(0.0));
    for (int i = its - 1; i >= 0; i--) {
        RT s = g[i];
        for (int k = i + 1; k < its; k++) { s -= H[i][k]*y[k]; }
        y[i] = s / H[i][i];
    }

    // x = M (V y): the V-cycle is linear, so apply it once to the summed
    // basis (classical right-preconditioned GMRES update); one fused
    // linear-combination kernel forms V y (BlockAxpy with no source term
    // computes -sum c_i v_i, hence the sign)
    for (int i = 0; i < its; i++) { cb[i] = -static_cast<amrex::Real>(y[i]); }
    BlockAxpy(*m_zB, 0, nullptr, its, cb, 1.0_rt);
    ApplyMToW(*m_zB, 0);   // final W and E in m_xW[0]/m_xF[0]

    if (m_verbose) {
        amrex::Print() << "  pc_hybrid_pic inner GMRES: " << its << " its, rel res "
                       << std::abs(g[its])/beta << "\n";
    }
}

template <class T, class Ops>
void HybridPICPC<T,Ops>::Apply ( T& a_x, const T& a_b )
{
    using namespace amrex;

    BL_PROFILE("HybridPICPC::Apply()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        IsDefined(), "HybridPICPC::Apply() called on undefined object");

    // pe-as-Newton-unknown triangular leg (see m_pe_row_active): solve the
    // pe row with the pointwise diagonal D_p, then fold its Ohm gradient
    // coupling into the E-row RHS below (block upper-triangular approximate
    // Jacobian; the reverse pe <- E coupling is dropped). With the leg
    // inactive, or an E-only vector, the pe row falls through to the
    // identity passthrough at the end of this function.
    const bool pe_row = m_pe_row_active
        && (a_x.getScalarVecType() != warpx::fields::FieldType::None);
    if (pe_row) {
        amrex::MultiFab& xp = *a_x.getScalarVec()[0];
        const amrex::MultiFab& bp = *a_b.getScalarVec()[0];
        // the scheme scales the pe Newton row; the E leg needs physical dpe
        const amrex::Real inv_kappa =
            1.0_rt / m_ops->PressureScaleForPC();
        m_pe_row_scr->setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box vbx = mfi.tilebox();
            auto const& x  = xp.array(mfi);
            auto const& b  = bp.const_array(mfi);
            auto const& cf = m_coef[0]->const_array(mfi);
            auto const& s  = m_pe_row_scr->array(mfi);
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real v = b(i,j,k) / (1.0_rt + cf(i,j,k,10));
                    x(i,j,k) = v;
                    s(i,j,k) = v * inv_kappa;   // physical dpe for the E leg
                });
        }
#if !defined(WARPX_DIM_RZ)
        if (m_pe_row_sweeps > 0) {
            // damped-Jacobi sweeps carrying the signed UPWIND advection
            // stencil theta*dt*gamma ue0.grad(.) (comps 11-13 hold
            // theta*dt ue0), whose diagonal is exactly the advective part
            // of D_p = 1 + cf10, PLUS the conduction diffusion stencil
            // -c_k del2(.) whose Nyquist diagonal is exactly the conduction
            // part of D_p (c_k = (cf10 - adiag)/l2, so the sweep operator's
            // diagonal stays 1 + cf10): the off-diagonal content the
            // pointwise leg drops for both legs. Domain ghosts zero-extend.
            const amrex::Real gam = m_ops->GetHybridPICModel()->m_gamma;
            const amrex::Real l2_pe = static_cast<amrex::Real>(m_l2[0]);
            const auto dxi = m_geom[0].InvCellSizeArray();
            // marker-CFL clamp in the units of comps 11-13 (theta*dt ue):
            // c = pe_ue_cap_fac * dx_min; <= 0 = no clamp
            amrex::Real dx_min_pe = m_geom[0].CellSize(0);
            for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                dx_min_pe = std::min(dx_min_pe, m_geom[0].CellSize(d));
            }
            const amrex::Real capl = m_pe_ue_cap_fac * dx_min_pe;
            amrex::MultiFab& x2 = *m_pe_row_scr2;
            amrex::MultiFab::Copy(x2, xp, 0, 0, 1, 0);
            for (int sw = 0; sw < m_pe_row_sweeps; sw++) {
                FillGhosts(x2, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    const Box vbx = mfi.tilebox();
                    auto const& xo = xp.array(mfi);
                    auto const& xi = x2.const_array(mfi);
                    auto const& b  = bp.const_array(mfi);
                    auto const& cf = m_coef[0]->const_array(mfi);
                    amrex::ParallelFor(vbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            auto clamp_u = [=] (Real u) {
                                return (capl > 0.0_rt) ? capl * std::tanh(u / capl) : u;
                            };
                            // gamma * upwind(theta*dt ue . grad x) and the
                            // advective diagonal of D_p it contains
                            Real up = 0.0_rt;
                            Real adiag = 0.0_rt;
                            Real nbr = 0.0_rt;   // Laplacian neighbour sum
#if defined(WARPX_DIM_1D_Z)
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i-1,j,k))
                                : uz*(xi(i+1,j,k) - xi(i,j,k)));
                            adiag = gam * std::abs(uz) * dxi[0];
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0];
#elif defined(WARPX_DIM_XZ)
                            const Real ux = clamp_u(cf(i,j,k,11));
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (ux > 0.0_rt
                                ? ux*(xi(i,j,k) - xi(i-1,j,k))
                                : ux*(xi(i+1,j,k) - xi(i,j,k)));
                            up += gam * dxi[1] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i,j-1,k))
                                : uz*(xi(i,j+1,k) - xi(i,j,k)));
                            adiag = gam * (std::abs(ux) * dxi[0] + std::abs(uz) * dxi[1]);
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0]
                                + (xi(i,j+1,k) + xi(i,j-1,k))*dxi[1]*dxi[1];
#elif defined(WARPX_DIM_3D)
                            const Real ux = clamp_u(cf(i,j,k,11));
                            const Real uy = clamp_u(cf(i,j,k,12));
                            const Real uz = clamp_u(cf(i,j,k,13));
                            up += gam * dxi[0] * (ux > 0.0_rt
                                ? ux*(xi(i,j,k) - xi(i-1,j,k))
                                : ux*(xi(i+1,j,k) - xi(i,j,k)));
                            up += gam * dxi[1] * (uy > 0.0_rt
                                ? uy*(xi(i,j,k) - xi(i,j-1,k))
                                : uy*(xi(i,j+1,k) - xi(i,j,k)));
                            up += gam * dxi[2] * (uz > 0.0_rt
                                ? uz*(xi(i,j,k) - xi(i,j,k-1))
                                : uz*(xi(i,j,k+1) - xi(i,j,k)));
                            adiag = gam * (std::abs(ux) * dxi[0] + std::abs(uy) * dxi[1]
                                         + std::abs(uz) * dxi[2]);
                            nbr = (xi(i+1,j,k) + xi(i-1,j,k))*dxi[0]*dxi[0]
                                + (xi(i,j+1,k) + xi(i,j-1,k))*dxi[1]*dxi[1]
                                + (xi(i,j,k+1) + xi(i,j,k-1))*dxi[2]*dxi[2];
#endif
                            // conduction: -c_k del2 x = c_k (l2 x - nbr) with
                            // c_k = cond/l2, cond = cf10 - adiag; its
                            // diagonal is inside the (1 + cond) x term, so
                            // only the neighbour sum enters the residual
                            const Real ck = (cf(i,j,k,10) - adiag) / l2_pe;
                            const Real r = b(i,j,k)
                                - (1.0_rt + cf(i,j,k,10) - adiag)*xi(i,j,k)
                                + ck*nbr
                                - up;
                            xo(i,j,k) = xi(i,j,k) + r / (1.0_rt + cf(i,j,k,10));
                        });
                }
                if (sw + 1 < m_pe_row_sweeps) {
                    amrex::MultiFab::Copy(x2, xp, 0, 0, 1, 0);
                }
            }
            // refresh the physical dpe carried to the E leg
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(xp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box vbx = mfi.tilebox();
                auto const& x = xp.const_array(mfi);
                auto const& s = m_pe_row_scr->array(mfi);
                amrex::ParallelFor(vbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        s(i,j,k) = x(i,j,k) * inv_kappa;
                    });
            }
        }
#endif
        FillGhosts(*m_pe_row_scr, 0);
    }

    // b~: the PC input b (rotated by (I+S)^-1 with the ion response), with
    // the pe-row gradient folded in and the E-field closure applied
    for (int n = 0; n < 3; n++) {
        if (m_staggered) {
            CopyCompIn(*a_b.getArrayVec()[0][n], *m_bE, n, false);
        } else {
            amrex::MultiFab::Copy(*m_bE, *a_b.getArrayVec()[0][n], 0, n, 1, 0);
        }
    }
    if (pe_row) {
        // b_E <- b_E - grad(dpe)/rho_e: the dF_E/dpe Ohm coupling the
        // identity pe leg left entirely to GMRES. Domain ghosts of the dpe
        // scratch are zero-extended; the affected wall-tangential E rows are
        // identity-overridden on output anyway.
        const auto dxi = m_geom[0].InvCellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*m_bE, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box vbx = mfi.tilebox();
            auto const& be = m_bE->array(mfi);
            auto const& s  = m_pe_row_scr->const_array(mfi);
            auto const& cf = m_coef[0]->const_array(mfi);
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    const Real ir = 1.0_rt / cf(i,j,k,9);
#if defined(WARPX_DIM_1D_Z)
                    be(i,j,k,2) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
#elif defined(WARPX_DIM_XZ)
                    be(i,j,k,0) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
                    be(i,j,k,2) -= ir * (s(i,j+1,k) - s(i,j-1,k)) * 0.5_rt * dxi[1];
#elif defined(WARPX_DIM_3D)
                    be(i,j,k,0) -= ir * (s(i+1,j,k) - s(i-1,j,k)) * 0.5_rt * dxi[0];
                    be(i,j,k,1) -= ir * (s(i,j+1,k) - s(i,j-1,k)) * 0.5_rt * dxi[1];
                    be(i,j,k,2) -= ir * (s(i,j,k+1) - s(i,j,k-1)) * 0.5_rt * dxi[2];
#else
                    amrex::ignore_unused(i, j, k, ir, be, s, dxi);
#endif
                });
        }
    }
    if (m_have_mm) {
        // the pair is posed on b~ = (I+S)^-1 b (see the class docs)
        RotateValid(*m_bE);
    }
    FillGhosts(*m_bE, 0);
    // PEC dead tangential-E columns (wall values zeroed; the identity rows
    // are restored on the output below) + parity ghosts; EB nodes zeroed
    ApplyDomainClosure(*m_bE, 0, ClosureClass::EField);

    // the E-row RHS is b~, the W-row RHS is zero
    if (m_inner_max > 0) {
        // block RHS (W-half comps 0..2, E-half comps 3..5) for the inner GMRES
        m_bKB->setVal(0.0, 0, 3, 0);
        amrex::MultiFab::Copy(*m_bKB, *m_bE, 0, 3, 3, 0);
        InnerGMRES();
    } else {
        // fixed linear operator: m_vcycles V-cycles from a zero initial guess
        amrex::MultiFab::Copy(*m_bF[0], *m_bE, 0, 0, 3, 0);
        m_bW[0]->setVal(0.0);
        if (!(m_zero_free && m_eb_flag == nullptr)) {
            m_xW[0]->setVal(0.0);
            m_xF[0]->setVal(0.0);
        }
        Vcycle(0, true);
        for (int c = 1; c < m_vcycles; c++) { Vcycle(0, false); }
    }

    // the E-half is the output
    if (m_staggered) {
        // WriteXOut reads the top nodal planes from the first ghost ring of
        // the cell container: fill it (periodic grids only)
        FillGhosts(*m_xF[0], 0);
        WriteXOut(*m_xF[0], a_x);
    } else {
        for (int n = 0; n < 3; n++) {
            amrex::MultiFab::Copy(*a_x.getArrayVec()[0][n], *m_xF[0], n, 0, 1, 0);
        }
    }
    // exact identity rows of the true Jacobian: x = b on wall-tangential E
    // at PEC faces and on EB-covered nodes
    OverrideIdentityRows(a_x, a_b);

    // pe-as-Newton-unknown fallback: with the triangular pe-row leg
    // inactive, pass the scalar (pe) row through as identity so the
    // right-preconditioned operator is nonsingular in those rows (the
    // precond entry zeroes a_x first, so skipping this would pin dpe = 0).
    // With the leg active, a_x's scalar already holds the D_p solve above.
    if (!pe_row && a_x.getScalarVecType() != warpx::fields::FieldType::None) {
        amrex::MultiFab::Copy(*a_x.getScalarVec()[0], *a_b.getScalarVec()[0],
                              0, 0, 1, 0);
    }
}

#endif
