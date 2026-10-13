/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_HYPER_RESISTIVITY_CARTESIAN_H_
#define WARPX_HYPER_RESISTIVITY_CARTESIAN_H_

#include "MotionalFieldCartesian.H"
#include "EmbeddedBoundary/Enabled.H"
#include "WarpX.H"
#include <ablastr/coarsen/sample.H>

#if defined(WARPX_DIM_3D)
/** Curl-curl stabilizer with a physical-volume adjoint at domain and ECT walls.
 * C is the Cartesian Yee edge-to-face curl, P removes conductor edges, and Q
 * removes covered faces. K = eta Q C P J, E_H = P V_e^-1 C^T V_f K.
 * Consequently <J,E_H>_e = <Q C P J,eta Q C P J>_f >= 0 for eta >= 0.
 * The curl pair reads valid faces/edges and one-cell internal ghosts only.
 * Physical-domain current/flux ghosts are excluded from the curl pair. The
 * face coefficient uses the caller's usual filled rho/B interpolation halos.
 */
inline void ComputeHyperResistivityCartesian (WarpX &warpx, int lev,
                                              ablastr::fields::VectorField const &current,
                                              ablastr::fields::VectorField const &B,
                                              amrex::MultiFab const &rho,
                                              MotionalFieldCartesian geometry,
                                              amrex::ParserExecutor<2> eta, bool has_B)
{
    using warpx::fields::FieldType;
    using namespace ablastr::coarsen::sample;
    auto K = warpx.m_fields.get_alldirs("hybrid_hyperres_curlJ_fp", lev);
    auto EH = warpx.m_fields.get_alldirs("hybrid_hyperres_E_fp", lev);
    auto lengths = EB::enabled() && geometry.conformal
                                 ? warpx.m_fields.get_alldirs(FieldType::edge_lengths, lev)
                                 : ablastr::fields::VectorField{nullptr, nullptr, nullptr};
    auto const &geom = warpx.Geom(lev);
    amrex::GpuArray<QdsmcVolumeElement, 3> vf;
    amrex::GpuArray<amrex::GpuArray<int, 3>, 3> bstag;
    for (int c = 0; c < 3; ++c)
    {
        vf[c] = MakeQdsmcVolumeElement(geom, B[c]->ixType());
        auto const iv = B[c]->ixType().toIntVect();
        bstag[c] = {iv[0], iv[1], iv[2]};
        K[c]->setVal(0.0);
        EH[c]->setVal(0.0);
    }
    for (int c = 0; c < 3; ++c)
    {
        for (amrex::MFIter mfi(*K[c]); mfi.isValid(); ++mfi)
        {
            auto mask = geometry;
            amrex::Array4<int const> face_mask;
            if (EB::enabled())
            {
                for (int d = 0; d < 3; ++d)
                {
                    mask.update[d] = warpx.GetEBUpdateEFlag()[lev][d]->const_array(mfi);
                    if (lengths[d]) { mask.length[d] = lengths[d]->const_array(mfi); }
                }
                face_mask = warpx.GetEBUpdateBFlag()[lev][c]->const_array(mfi);
            }
            amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> j, b;
            for (int d = 0; d < 3; ++d)
            {
                j[d] = current[d]->const_array(mfi);
                b[d] = B[d]->const_array(mfi);
            }
            auto const r = rho.const_array(mfi);
            auto const out = K[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int jj, int k)
                {
                    if (face_mask && face_mask(i, jj, k) == 0)
                    {
                        return;
                    }
                    int const a = (c + 1) % 3, d = (c + 2) % 3;
                    auto sample = [&] AMREX_GPU_DEVICE(int component, int axis, int offset)
                    {
                        int const ii = i + (axis == 0 ? offset : 0);
                        int const y = jj + (axis == 1 ? offset : 0);
                        int const z = k + (axis == 2 ? offset : 0);
                        return mask.active(ii, y, z, component) ? j[component](ii, y, z) : 0.0;
                    };
                    amrex::Real const curl = (sample(d, a, 1) - sample(d, a, 0)) / mask.dx[a] -
                                             (sample(a, d, 1) - sample(a, d, 0)) / mask.dx[d];
                    amrex::GpuArray<int, 3> const nodal{1, 1, 1}, coarsen{1, 1, 1};
                    amrex::Real const density = Interp(r, nodal, bstag[c], coarsen, i, jj, k, 0);
                    amrex::Real b2 = 0.0;
                    if (has_B)
                    {
                        for (int component = 0; component < 3; ++component)
                        {
                            auto const value = Interp(b[component], bstag[component], bstag[c],
                                                      coarsen, i, jj, k, 0);
                            b2 += value * value;
                        }
                    }
                    out(i, jj, k) = eta(density, std::sqrt(b2)) * curl;
                });
        }
        K[c]->FillBoundary(geom.periodicity());
    }
    for (int c = 0; c < 3; ++c)
    {
        for (amrex::MFIter mfi(*EH[c]); mfi.isValid(); ++mfi)
        {
            auto mask = geometry;
            if (EB::enabled())
            {
                for (int d = 0; d < 3; ++d)
                {
                    mask.update[d] = warpx.GetEBUpdateEFlag()[lev][d]->const_array(mfi);
                    if (lengths[d]) { mask.length[d] = lengths[d]->const_array(mfi); }
                }
            }
            amrex::GpuArray<amrex::Array4<amrex::Real const>, 3> values;
            for (int d = 0; d < 3; ++d)
            {
                values[d] = K[d]->const_array(mfi);
            }
            auto const out = EH[c]->array(mfi);
            amrex::ParallelFor(
                mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    if (!mask.active(i, j, k, c))
                    {
                        return;
                    }
                    auto weighted = [&] AMREX_GPU_DEVICE(int component, int axis, int offset)
                    {
                        int const idx[3] = {i + (axis == 0 ? offset : 0),
                                            j + (axis == 1 ? offset : 0),
                                            k + (axis == 2 ? offset : 0)};
                        for (int d = 0; d < 3; ++d)
                        {
                            if (!mask.periodic[d] &&
                                (idx[d] < mask.lo[d] || idx[d] > mask.hi[d] + (d == component)))
                            {
                                return 0.0;
                            }
                        }
                        return vf[component](idx[0], idx[1], idx[2]) *
                               values[component](idx[0], idx[1], idx[2]);
                    };
                    int const a = (c + 1) % 3, b = (c + 2) % 3;
                    out(i, j, k) = ((weighted(b, a, 0) - weighted(b, a, -1)) / mask.dx[a] -
                                    (weighted(a, b, 0) - weighted(a, b, -1)) / mask.dx[b]) /
                                   mask.ve[c](i, j, k);
                });
        }
        EH[c]->FillBoundary(geom.periodicity());
    }
}
#endif
#endif
