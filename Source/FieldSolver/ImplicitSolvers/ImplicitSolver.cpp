#include "ImplicitSolver.H"
#include "Fields.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace amrex;
using namespace amrex::literals;

void ImplicitSolver::CreateParticleAttributes () const
{
    // Set comm to false so that the attributes are not communicated
    // nor written to the checkpoint files
    int const comm = 0;

    // Add space to save the positions and velocities at the start of the time steps
    for (auto const& pc : m_WarpX->GetPartContainer()) {
#if !defined(WARPX_DIM_1D_Z)
        pc->AddRealComp("x_n", comm);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        pc->AddRealComp("y_n", comm);
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        pc->AddRealComp("z_n", comm);
#endif
        pc->AddRealComp("ux_n", comm);
        pc->AddRealComp("uy_n", comm);
        pc->AddRealComp("uz_n", comm);

        if (m_particle_suborbits) {
            pc->AddIntComp("nsuborbits", comm);
        }
    }
}

const Geometry& ImplicitSolver::GetGeometry (const int a_lvl) const
{
    AMREX_ASSERT((a_lvl >= 0) && (a_lvl < m_num_amr_levels));
    return m_WarpX->Geom(a_lvl);
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryLo () const
{
    return m_WarpX->GetFieldBoundaryLo();
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryHi () const
{
    return m_WarpX->GetFieldBoundaryHi();
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCLo () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryLo(),/*bdry_side=*/0);
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCHi () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryHi(),/*bdry_side=*/1);
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::convertFieldBCToLinOpBC (const Array<FieldBoundaryType,AMREX_SPACEDIM>& a_fbc, const int bdry_side) const
{
    Array<LinOpBCType, AMREX_SPACEDIM> lbc;
    for (auto& bc : lbc) { bc = LinOpBCType::interior; }
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
        if (a_fbc[i] == FieldBoundaryType::PML) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Periodic) {
            lbc[i] = LinOpBCType::Periodic;
        } else if (a_fbc[i] == FieldBoundaryType::PEC) {
            lbc[i] = LinOpBCType::Dirichlet;
        } else if (a_fbc[i] == FieldBoundaryType::Damped) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Absorbing_Silver_Mueller) {
            ablastr::warn_manager::WMRecordWarning("Implicit solver",
                "With SilverMueller, in the Curl-Curl preconditioner Symmetry boundary will be used since the full boundary is not yet implemented.",
                ablastr::warn_manager::WarnPriority::medium);
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::Neumann) {
            // Also for FieldBoundaryType::PMC
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::PEC_Insulator) {
            const int voltage_driven = m_WarpX->GetPECInsulator_IsESet(i,bdry_side);
            if (voltage_driven) { // Dirichlet for E
                lbc[i] = LinOpBCType::Dirichlet;
            } else { // Dirichlet for B
                ablastr::warn_manager::WMRecordWarning("Implicit solver with current-driven PECInsulator",
                    "in the Curl-Curl preconditioner. Symmetry boundary will be used since the full boundary is not yet implemented.",
                    ablastr::warn_manager::WarnPriority::medium);
                lbc[i] = LinOpBCType::symmetry;
            }
        } else if (a_fbc[i] == FieldBoundaryType::None) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Open) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else {
            WARPX_ABORT_WITH_MESSAGE("Invalid value for FieldBoundaryType");
        }
    }
    return lbc;
}

void ImplicitSolver::CumulateJ ()
{

    // Add J0, which contains J from particles included in the mass matrices (MM) to current_fp, which
    // is either zero or contains J from suborbit particles that are not included in the MM.
    // Do this BEFORE call to SyncCurrentAndRho().
    //
    // J during the linear stage of JFNK is computed as J(E=E0+dE) = J_suborbit + J0 + MM*(E - E0),
    // where MM are the mass matrices (i.e., dJ/dE), E0 is the electric field at the start of the Newton
    // step (see SaveE function), J0 is the current associated with particles that are included in the MM
    // using E0, and J_suborbit is the current associated with particles that have suborbits.

    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        const ablastr::fields::VectorField J0 = m_WarpX->m_fields.get_alldirs(FieldType::current_fp_non_suborbit, lev);
        amrex::MultiFab::Add(*J[0], *J0[0], 0, 0, J0[0]->nComp(), J0[0]->nGrowVect());
        amrex::MultiFab::Add(*J[1], *J0[1], 0, 0, J0[1]->nComp(), J0[1]->nGrowVect());
        amrex::MultiFab::Add(*J[2], *J0[2], 0, 0, J0[2]->nComp(), J0[2]->nGrowVect());
    }

}

void ImplicitSolver::SaveE ()
{

    // Copy Efield_fp to E0.

    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField E0 = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev);
        amrex::MultiFab::Copy(*E0[0], *E[0], 0, 0, E[0]->nComp(), E[0]->nGrowVect());
        amrex::MultiFab::Copy(*E0[1], *E[1], 0, 0, E[1]->nComp(), E[1]->nGrowVect());
        amrex::MultiFab::Copy(*E0[2], *E[2], 0, 0, E[2]->nComp(), E[2]->nGrowVect());
    }

}

void ImplicitSolver::ApplyMassMatrices (
    ablastr::fields::MultiLevelVectorField& a_out,
    const ablastr::fields::MultiLevelVectorField& a_in,
    const ablastr::fields::MultiLevelVectorField* a_in_ref,
    const ablastr::fields::MultiLevelVectorField* a_baseline,
    const amrex::Real a_scale,
    const bool a_zero_out_first )
{
    BL_PROFILE("ImplicitSolver::ApplyMassMatrices()");
    using namespace amrex::literals;

    using warpx::fields::FieldType;

    const int ncomps = 1;
    const int nlevs = static_cast<int>(a_out.size());
    const bool use_delta = (a_in_ref != nullptr);
    const bool use_baseline = (a_baseline != nullptr);

    AMREX_ALWAYS_ASSERT(a_in.size() == nlevs);
    if (use_delta) {
        AMREX_ALWAYS_ASSERT(a_in_ref->size() == nlevs);
    }
    if (use_baseline) {
        AMREX_ALWAYS_ASSERT(a_baseline->size() == nlevs);
    }

    for (int lev = 0; lev < nlevs; ++lev) {

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

        if (a_zero_out_first) {
            a_out[lev][0]->setVal(0.0);
            a_out[lev][1]->setVal(0.0);
            a_out[lev][2]->setVal(0.0);
        }

        const amrex::IntVect outx_nodal = a_out[lev][0]->ixType().toIntVect();
        const amrex::IntVect outy_nodal = a_out[lev][1]->ixType().toIntVect();
        const amrex::IntVect outz_nodal = a_out[lev][2]->ixType().toIntVect();

        // Compute the component offset in each direction (careful with staggering)
        amrex::IntVect offset_xx, offset_xy, offset_xz;
        amrex::IntVect offset_yx, offset_yy, offset_yz;
        amrex::IntVect offset_zx, offset_zy, offset_zz;
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            offset_xx[dir] = (m_ncomp_xx[dir]-1)/2;
            offset_xy[dir] = (outx_nodal[dir] > outy_nodal[dir]) ?  (m_ncomp_xy[dir]/2)
                                                                 : ((m_ncomp_xy[dir]-1)/2);
            offset_xz[dir] = (outx_nodal[dir] > outz_nodal[dir]) ?  (m_ncomp_xz[dir]/2)
                                                                 : ((m_ncomp_xz[dir]-1)/2);
            offset_yx[dir] = (outy_nodal[dir] > outx_nodal[dir]) ?  (m_ncomp_yx[dir]/2)
                                                                 : ((m_ncomp_yx[dir]-1)/2);
            offset_yy[dir] = (m_ncomp_yy[dir]-1)/2;
            offset_yz[dir] = (outy_nodal[dir] > outz_nodal[dir]) ?  (m_ncomp_yz[dir]/2)
                                                                 : ((m_ncomp_yz[dir]-1)/2);
            offset_zx[dir] = (outz_nodal[dir] > outx_nodal[dir]) ?  (m_ncomp_zx[dir]/2)
                                                                 : ((m_ncomp_zx[dir]-1)/2);
            offset_zy[dir] = (outz_nodal[dir] > outy_nodal[dir]) ?  (m_ncomp_zy[dir]/2)
                                                                 : ((m_ncomp_zy[dir]-1)/2);
            offset_zz[dir] = (m_ncomp_zz[dir]-1)/2;
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*a_out[lev][0], false); mfi.isValid(); ++mfi )
        {
            amrex::Array4<amrex::Real> const& out_arr_x = a_out[lev][0]->array(mfi);
            amrex::Array4<amrex::Real> const& out_arr_y = a_out[lev][1]->array(mfi);
            amrex::Array4<amrex::Real> const& out_arr_z = a_out[lev][2]->array(mfi);

            amrex::Array4<const amrex::Real> const& in_arr_x = a_in[lev][0]->array(mfi);
            amrex::Array4<const amrex::Real> const& in_arr_y = a_in[lev][1]->array(mfi);
            amrex::Array4<const amrex::Real> const& in_arr_z = a_in[lev][2]->array(mfi);

            // These are only read when use_delta/use_baseline is true; otherwise
            // they are left as empty (null) Array4 handles and never dereferenced.
            amrex::Array4<const amrex::Real> const& ref_arr_x = use_delta ? (*a_in_ref)[lev][0]->array(mfi) : amrex::Array4<const amrex::Real>{};
            amrex::Array4<const amrex::Real> const& ref_arr_y = use_delta ? (*a_in_ref)[lev][1]->array(mfi) : amrex::Array4<const amrex::Real>{};
            amrex::Array4<const amrex::Real> const& ref_arr_z = use_delta ? (*a_in_ref)[lev][2]->array(mfi) : amrex::Array4<const amrex::Real>{};

            amrex::Array4<const amrex::Real> const& baseline_arr_x = use_baseline ? (*a_baseline)[lev][0]->array(mfi) : amrex::Array4<const amrex::Real>{};
            amrex::Array4<const amrex::Real> const& baseline_arr_y = use_baseline ? (*a_baseline)[lev][1]->array(mfi) : amrex::Array4<const amrex::Real>{};
            amrex::Array4<const amrex::Real> const& baseline_arr_z = use_baseline ? (*a_baseline)[lev][2]->array(mfi) : amrex::Array4<const amrex::Real>{};

            amrex::Array4<const amrex::Real> const& Sxx = SX[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxy = SX[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxz = SX[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Syx = SY[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syy = SY[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syz = SY[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Szx = SZ[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szy = SZ[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szz = SZ[2]->array(mfi);

            // The outer loop below reads Sxx/Sxy/Sxz (etc.) directly at (i,j,k),
            // so it must stay within the mass matrices' own ghost region - grow
            // by the min of the input's and the mass matrices' ghost widths.
            amrex::Box outbx = amrex::convert(mfi.validbox(),a_out[lev][0]->ixType());
            amrex::Box outby = amrex::convert(mfi.validbox(),a_out[lev][1]->ixType());
            amrex::Box outbz = amrex::convert(mfi.validbox(),a_out[lev][2]->ixType());
            outbx.grow(amrex::elemwiseMin(a_out[lev][0]->nGrowVect(), SX[0]->nGrowVect()));
            outby.grow(amrex::elemwiseMin(a_out[lev][1]->nGrowVect(), SY[1]->nGrowVect()));
            outbz.grow(amrex::elemwiseMin(a_out[lev][2]->nGrowVect(), SZ[2]->nGrowVect()));

            // The inner stencil reads are bounded by the input field's own
            // (potentially wider) ghost region, which holds correct
            // periodic-wrapped data via FillBoundaryAndSync.
            amrex::Box in_fullbx = amrex::convert(mfi.validbox(),a_in[lev][0]->ixType());
            amrex::Box in_fullby = amrex::convert(mfi.validbox(),a_in[lev][1]->ixType());
            amrex::Box in_fullbz = amrex::convert(mfi.validbox(),a_in[lev][2]->ixType());
            in_fullbx.grow(a_in[lev][0]->nGrowVect());
            in_fullby.grow(a_in[lev][1]->nGrowVect());
            in_fullbz.grow(a_in[lev][2]->nGrowVect());

            const amrex::IntVect ncomp_xx = m_ncomp_xx;
            const amrex::IntVect ncomp_xy = m_ncomp_xy;
            const amrex::IntVect ncomp_xz = m_ncomp_xz;
            const amrex::IntVect ncomp_yx = m_ncomp_yx;
            const amrex::IntVect ncomp_yy = m_ncomp_yy;
            const amrex::IntVect ncomp_yz = m_ncomp_yz;
            const amrex::IntVect ncomp_zx = m_ncomp_zx;
            const amrex::IntVect ncomp_zy = m_ncomp_zy;
            const amrex::IntVect ncomp_zz = m_ncomp_zz;

            amrex::ParallelFor(
            outbx, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Sxx*d_in_x
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xx[dim],in_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xx[dim]-1-offset_xx[dim],in_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Sxx_d_in_x = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xx[0],
                                         + ncomp_xx[0]*( jj+offset_xx[1] ),
                                         + ncomp_xx[0]*ncomp_xx[1]*( kk+offset_xx[2] ) );
                            amrex::Real dval = in_arr_x(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_x(i+ii,j+jj,k+kk,n); }
                            Sxx_d_in_x += Sxx(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Sxy*d_in_y
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xy[dim],in_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xy[dim]-1-offset_xy[dim],in_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Sxy_d_in_y = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xy[0],
                                         + ncomp_xy[0]*( jj+offset_xy[1] ),
                                         + ncomp_xy[0]*ncomp_xy[1]*( kk+offset_xy[2] ) );
                            amrex::Real dval = in_arr_y(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_y(i+ii,j+jj,k+kk,n); }
                            Sxy_d_in_y += Sxy(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Sxz*d_in_z
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xz[dim],in_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xz[dim]-1-offset_xz[dim],in_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Sxz_d_in_z = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_xz[0],
                                         + ncomp_xz[0]*( jj+offset_xz[1] ),
                                         + ncomp_xz[0]*ncomp_xz[1]*( kk+offset_xz[2] ) );
                            amrex::Real dval = in_arr_z(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_z(i+ii,j+jj,k+kk,n); }
                            Sxz_d_in_z += Sxz(i,j,k,Nc)*dval;
                        }
                    }
                }

                if (use_baseline) { out_arr_x(i,j,k,n) += baseline_arr_x(i,j,k,n); }
                out_arr_x(i,j,k,n) += a_scale * (Sxx_d_in_x + Sxy_d_in_y + Sxz_d_in_z);
            });
            amrex::ParallelFor(
            outby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Syx*d_in_x
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yx[dim],in_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yx[dim]-1-offset_yx[dim],in_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Syx_d_in_x = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yx[0],
                                         + ncomp_yx[0]*( jj+offset_yx[1] ),
                                         + ncomp_yx[0]*ncomp_yx[1]*( kk+offset_yx[2] ) );
                            amrex::Real dval = in_arr_x(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_x(i+ii,j+jj,k+kk,n); }
                            Syx_d_in_x += Syx(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Syy*d_in_y
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yy[dim],in_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yy[dim]-1-offset_yy[dim],in_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Syy_d_in_y = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yy[0],
                                         + ncomp_yy[0]*( jj+offset_yy[1] ),
                                         + ncomp_yy[0]*ncomp_yy[1]*( kk+offset_yy[2] ) );
                            amrex::Real dval = in_arr_y(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_y(i+ii,j+jj,k+kk,n); }
                            Syy_d_in_y += Syy(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Syz*d_in_z
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yz[dim],in_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yz[dim]-1-offset_yz[dim],in_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Syz_d_in_z = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_yz[0],
                                         + ncomp_yz[0]*( jj+offset_yz[1] ),
                                         + ncomp_yz[0]*ncomp_yz[1]*( kk+offset_yz[2] ) );
                            amrex::Real dval = in_arr_z(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_z(i+ii,j+jj,k+kk,n); }
                            Syz_d_in_z += Syz(i,j,k,Nc)*dval;
                        }
                    }
                }

                if (use_baseline) { out_arr_y(i,j,k,n) += baseline_arr_y(i,j,k,n); }
                out_arr_y(i,j,k,n) += a_scale * (Syx_d_in_x + Syy_d_in_y + Syz_d_in_z);
            });
            amrex::ParallelFor(
            outbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Szx*d_in_x
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zx[dim],in_fullbx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zx[dim]-1-offset_zx[dim],in_fullbx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Szx_d_in_x = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zx[0],
                                         + ncomp_zx[0]*( jj+offset_zx[1] ),
                                         + ncomp_zx[0]*ncomp_zx[1]*( kk+offset_zx[2] ) );
                            amrex::Real dval = in_arr_x(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_x(i+ii,j+jj,k+kk,n); }
                            Szx_d_in_x += Szx(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Szy*d_in_y
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zy[dim],in_fullby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zy[dim]-1-offset_zy[dim],in_fullby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Szy_d_in_y = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zy[0],
                                         + ncomp_zy[0]*( jj+offset_zy[1] ),
                                         + ncomp_zy[0]*ncomp_zy[1]*( kk+offset_zy[2] ) );
                            amrex::Real dval = in_arr_y(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_y(i+ii,j+jj,k+kk,n); }
                            Szy_d_in_y += Szy(i,j,k,Nc)*dval;
                        }
                    }
                }

                // Compute Szz*d_in_z
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zz[dim],in_fullbz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zz[dim]-1-offset_zz[dim],in_fullbz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real Szz_d_in_z = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            const int Nc = AMREX_D_TERM( ii+offset_zz[0],
                                         + ncomp_zz[0]*( jj+offset_zz[1] ),
                                         + ncomp_zz[0]*ncomp_zz[1]*( kk+offset_zz[2] ) );
                            amrex::Real dval = in_arr_z(i+ii,j+jj,k+kk,n);
                            if (use_delta) { dval -= ref_arr_z(i+ii,j+jj,k+kk,n); }
                            Szz_d_in_z += Szz(i,j,k,Nc)*dval;
                        }
                    }
                }

                if (use_baseline) { out_arr_z(i,j,k,n) += baseline_arr_z(i,j,k,n); }
                out_arr_z(i,j,k,n) += a_scale * (Szx_d_in_x + Szy_d_in_y + Szz_d_in_z);
            });
        }
    }
}

void ImplicitSolver::ComputeJfromMassMatrices (const bool  a_J_from_MM_only)
{
    BL_PROFILE("ImplicitSolver::ComputeJfromMassMatrices()");
    using warpx::fields::FieldType;

    const int finest_level = m_num_amr_levels - 1;

    ablastr::fields::MultiLevelVectorField J_ml =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level);
    const ablastr::fields::MultiLevelVectorField E_ml =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    const ablastr::fields::MultiLevelVectorField E0_ml =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp_save, finest_level);
    const ablastr::fields::MultiLevelVectorField J0_ml =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp_non_suborbit, finest_level);

    ApplyMassMatrices(
        /* a_out           = */ J_ml,
        /* a_in            = */ E_ml,
        /* a_in_ref        = */ &E0_ml,
        /* a_baseline      = */ &J0_ml,
        /* a_scale         = */ 1.0_rt,
        /* a_zero_out_first = */ a_J_from_MM_only);
}


void ImplicitSolver::parseNonlinearSolverParams ( const amrex::ParmParse&  pp )
{

    pp.get("nonlinear_solver", m_nlsolver_type);

    if (m_nlsolver_type == NonlinearSolverType::picard) {

        // Picard
        m_nlsolver = std::make_unique<PicardSolver<WarpXSolverVec,ImplicitSolver>>();
        m_max_particle_iterations = 1;
        m_particle_tolerance = 0.0;

    }
    else if (      (m_nlsolver_type == NonlinearSolverType::newton)
                || (m_nlsolver_type == NonlinearSolverType::petsc_snes) ) {

        // JFNK solvers
        if (m_nlsolver_type == NonlinearSolverType::newton) {
            m_nlsolver = std::make_unique<NewtonSolver<WarpXSolverVec,ImplicitSolver>>();
        } else {
#ifdef AMREX_USE_PETSC
            m_nlsolver = std::make_unique<PETScSNES<WarpXSolverVec,ImplicitSolver>>();
#else
            WARPX_ABORT_WITH_MESSAGE("ImplicitSolver::parseNonlinearSolverParams(): must compile with PETSc to use petsc_snes (AMREX_USE_PETSC must be defined)");
#endif
        }
        pp.query("max_particle_iterations", m_max_particle_iterations);
        pp.query("particle_tolerance", m_particle_tolerance);
        pp.query("particle_suborbits", m_particle_suborbits);
        pp.query("reflect_particles_at_rmax", m_reflect_particles_at_rmax);
        pp.query("adjoint_gather_ghosts", m_adjoint_gather_ghosts);
        pp.query("print_unconverged_particle_details", m_print_unconverged_particle_details);
        pp.query("use_mass_matrices_jacobian", m_use_mass_matrices_jacobian);
        pp.query("use_mass_matrices_pc", m_use_mass_matrices_pc);
        if (m_use_mass_matrices_jacobian || m_use_mass_matrices_pc) {
            m_use_mass_matrices = true;
        }
        if (m_use_mass_matrices_jacobian) {
            // Default m_skip_particle_picard_init to true if using suborbits
            if (m_particle_suborbits) { m_skip_particle_picard_init = true; }
            pp.query("skip_particle_picard_init", m_skip_particle_picard_init);
            pp.query("mass_matrices_boundary_rows", m_mass_matrices_boundary_rows);
            pp.query("verify_mm_jvp_step", m_verify_mm_jvp_step);
            if (m_mass_matrices_boundary_rows && m_use_mass_matrices_pc) {
                std::stringstream warningMsg;
                warningMsg << "mass_matrices_boundary_rows folds the physical-boundary "
                    "response into the Jacobian mass-matrix bands; the preconditioner "
                    "containers copy diagonal band entries from those bands and then "
                    "apply their own guard-current fold, so combining both options "
                    "applies a boundary treatment twice on the preconditioner side.";
                ablastr::warn_manager::WMRecordWarning("ImplicitSolver", warningMsg.str());
            }
        }
        if (m_use_mass_matrices_pc) {
            m_mass_matrices_pc_width = 0;
#if AMREX_SPACEDIM != 3
            pp.query("mass_matrices_pc_width", m_mass_matrices_pc_width);
#endif
        }
#if defined(WARPX_DIM_RSPHERE)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices,
            "Using mass matrices is not setup for DIM = RSPHERE!");
#endif
#if defined(WARPX_DIM_3D)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices_jacobian,
            "Using mass matrices for jacobian can not be used for DIM = 3");
#endif
        if ( (WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor ||
              WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov) &&
             (WarpX::nox < 2) ) {
            std::stringstream warningMsg;
            warningMsg << "Particle-suppressed JFNK (e.g., theta-implicit evolve with newton nonlinear solver) ";
            warningMsg << "is being used with a charge-conserving deposition (esirkepov or villasenor) and particle_shape = 1.\n";
            warningMsg << "Some particle orbits may not converge!!!\n";
            warningMsg << "Consider using particle_shape > 1.\n";
            ablastr::warn_manager::WMRecordWarning("ImplicitSolver", warningMsg.str());
        }
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "invalid nonlinear_solver specified. Valid options are picard and newton.");
    }

}

void ImplicitSolver::SaveEoldMultifab ()
{
    using warpx::fields::FieldType;
    // E_old multifab is needed for diagnostics and saving at checkpoints
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Efp = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField E_old = m_WarpX->m_fields.get_alldirs(FieldType::E_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*E_old[n], *Efp[n], 0, 0, E_old[n]->nComp(), E_old[n]->nGrowVect());
        }
    }
}

void ImplicitSolver::InitializeMassMatrices ()
{

    // Initializes the MassMatrices and MassMatrices_PC containers
    // The latter has a reduced number of elements that is used for the preconditioner.
    //
    // dJx = MassMatrices_xx*dEx + MassMatrices_xy*dEy + MassMatrices_xz*dEz
    // dJy = MassMatrices_yx*dEx + MassMatrices_yy*dEy + MassMatrices_yz*dEz
    // dJz = MassMatrices_zx*dEx + MassMatrices_zy*dEy + MassMatrices_zz*dEz

    // check that PC is being used by nonlinear solver
    if (m_use_mass_matrices_pc) {
        const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
        if (pc_type == PreconditionerType::none) {
            m_use_mass_matrices_pc = false;
        }
        if (pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            // This PC does not yet support off-diagonal mass matrix terms
            if (m_use_mass_matrices_pc) { m_mass_matrices_pc_width = 0; }
            else { m_mass_matrices_pc_width = -1; }
        }
    }

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int shape = WarpX::nox;
    const amrex::IntVect ngJ = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, 0)->nGrowVect();
    const amrex::IntVect ngE = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();

    // Get nodal flags for each component of J
    const ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, 0);
    const amrex::IntVect Jx_nodal = J[0]->ixType().toIntVect();
    const amrex::IntVect Jy_nodal = J[1]->ixType().toIntVect();
    const amrex::IntVect Jz_nodal = J[2]->ixType().toIntVect();

    // Compute the total number of components for each mass matrices container.
    // This depends on the particle shape factor and the type of current deposition.
    int Nc_tot_xx = 1, Nc_tot_xy = 1, Nc_tot_xz = 1;
    int Nc_tot_yx = 1, Nc_tot_yy = 1, Nc_tot_yz = 1;
    int Nc_tot_zx = 1, Nc_tot_zy = 1, Nc_tot_zz = 1;
    if (m_use_mass_matrices_jacobian) {

        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( ngE[dir]>=ngJ[dir],
                "Mass Matrices for Jacobian requires guard cells for E "
                "to be at least as many as those for J.");
        }

        if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Direct) {
            for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                m_ncomp_xx[dir] = 1 + 2*shape;
                m_ncomp_xy[dir] = 1 + 2*shape + ( (Jx_nodal[dir] + Jy_nodal[dir]) % 2 );
                m_ncomp_xz[dir] = 1 + 2*shape + ( (Jx_nodal[dir] + Jz_nodal[dir]) % 2 );
                m_ncomp_yy[dir] = 1 + 2*shape;
                m_ncomp_yx[dir] = 1 + 2*shape + ( (Jy_nodal[dir] + Jx_nodal[dir]) % 2 );
                m_ncomp_yz[dir] = 1 + 2*shape + ( (Jy_nodal[dir] + Jz_nodal[dir]) % 2 );
                m_ncomp_zz[dir] = 1 + 2*shape;
                m_ncomp_zx[dir] = 1 + 2*shape + ( (Jz_nodal[dir] + Jx_nodal[dir]) % 2 );
                m_ncomp_zy[dir] = 1 + 2*shape + ( (Jz_nodal[dir] + Jy_nodal[dir]) % 2 );
                //
                Nc_tot_xx *= m_ncomp_xx[dir];
                Nc_tot_xy *= m_ncomp_xy[dir];
                Nc_tot_xz *= m_ncomp_xz[dir];
                Nc_tot_yx *= m_ncomp_yx[dir];
                Nc_tot_yy *= m_ncomp_yy[dir];
                Nc_tot_yz *= m_ncomp_yz[dir];
                Nc_tot_zx *= m_ncomp_zx[dir];
                Nc_tot_zy *= m_ncomp_zy[dir];
                Nc_tot_zz *= m_ncomp_zz[dir];
            }
        }
        else if (WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor) {
#ifndef WARPX_DIM_3D
            const int max_crossings = ngJ[0] - shape + 1;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings > 0,
                "Mass Matrices for Jacobian with Villasenor deposition requires particles.max_grid_crossings > 0.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings == m_WarpX->particle_max_grid_crossings,
                "Guard cells for J are not consistent with particle_max_grid_crossings.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings <= 2,
                "Mass Matrices for Jacobian with Villasenor deposition requires particles.max_grid_crossings <= 2.");
#endif
            // Comment on direction-dependent number of mass matrices components
            // set below for charge-conserving Villasenor deposition:
            // 1 + 2*(shape - 1) (both comps centered)
            // 0 + 2*shape       (mixed nodal/centered comps)
            // 1 + 2*shape       (both comps nodal)
#if defined(WARPX_DIM_1D_Z)
            // x and y are nodal, z is centered
            m_ncomp_xx[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*(shape-1) + 2*max_crossings;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // x is centered, y and z are nodal
            m_ncomp_xx[0] = 1 + 2*(shape-1) + 2*max_crossings;
            m_ncomp_xy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*shape + 2*max_crossings;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // dir = 0: x is centered, y and z are nodal
            m_ncomp_xx[0] = 1 + 2*(shape-1) + 2*max_crossings;
            m_ncomp_xy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*shape + 2*max_crossings;
            // dir = 1: x and y are nodal, z is centered
            m_ncomp_xx[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xy[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xz[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yy[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zx[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zz[1] = 1 + 2*(shape-1) + 2*max_crossings;
#endif
            for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                Nc_tot_xx *= m_ncomp_xx[dir];
                Nc_tot_xy *= m_ncomp_xy[dir];
                Nc_tot_xz *= m_ncomp_xz[dir];
                Nc_tot_yx *= m_ncomp_yx[dir];
                Nc_tot_yy *= m_ncomp_yy[dir];
                Nc_tot_yz *= m_ncomp_yz[dir];
                Nc_tot_zx *= m_ncomp_zx[dir];
                Nc_tot_zy *= m_ncomp_zy[dir];
                Nc_tot_zz *= m_ncomp_zz[dir];
            }
        }
        else {
            WARPX_ABORT_WITH_MESSAGE("Mass matrices can only be used with Direct and Villasenor depositions.");
        }
    }
    else { // Mass matrices used for PC only
        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
            m_ncomp_xx[dir] = 1;
            m_ncomp_xy[dir] = 0;
            m_ncomp_xz[dir] = 0;
            m_ncomp_yx[dir] = 0;
            m_ncomp_yy[dir] = 1;
            m_ncomp_yz[dir] = 0;
            m_ncomp_zx[dir] = 0;
            m_ncomp_zy[dir] = 0;
            m_ncomp_zz[dir] = 1;
            //
            Nc_tot_xx *= m_ncomp_xx[dir];
            Nc_tot_xy *= m_ncomp_xy[dir];
            Nc_tot_xz *= m_ncomp_xz[dir];
            Nc_tot_yx *= m_ncomp_yx[dir];
            Nc_tot_yy *= m_ncomp_yy[dir];
            Nc_tot_yz *= m_ncomp_yz[dir];
            Nc_tot_zx *= m_ncomp_zx[dir];
            Nc_tot_zy *= m_ncomp_zy[dir];
            Nc_tot_zz *= m_ncomp_zz[dir];
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Jx = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Jy = m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Jz = m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev)->boxArray();
        const auto& dm = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->DistributionMap();
        //
        if (m_use_mass_matrices_jacobian) {
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{0}, lev, ba_Jx, dm, 1, ngE, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{1}, lev, ba_Jy, dm, 1, ngE, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{2}, lev, ba_Jz, dm, 1, ngE, 0.0_rt);
        }
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{0}, lev, ba_Jx, dm, Nc_tot_xx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{1}, lev, ba_Jx, dm, Nc_tot_xy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{2}, lev, ba_Jx, dm, Nc_tot_xz, ngJ, 0.0_rt);
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{0}, lev, ba_Jy, dm, Nc_tot_yx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{1}, lev, ba_Jy, dm, Nc_tot_yy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{2}, lev, ba_Jy, dm, Nc_tot_yz, ngJ, 0.0_rt);
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{0}, lev, ba_Jz, dm, Nc_tot_zx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{1}, lev, ba_Jz, dm, Nc_tot_zy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{2}, lev, ba_Jz, dm, Nc_tot_zz, ngJ, 0.0_rt);
        //
        if (m_use_mass_matrices_pc) {
            int ncomp_tot_pc_xx = 1;
            int ncomp_tot_pc_yy = 1;
            int ncomp_tot_pc_zz = 1;

            // Additional MM components in PC not setup yet for when MM is only used for the PC
            const int ncomp_dir_pc = (m_use_mass_matrices_jacobian ? 1 + 2*m_mass_matrices_pc_width : 1);
            for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                m_ncomp_pc_xx[dir] = std::min(m_ncomp_xx[dir],ncomp_dir_pc);
                m_ncomp_pc_yy[dir] = std::min(m_ncomp_yy[dir],ncomp_dir_pc);
                m_ncomp_pc_zz[dir] = std::min(m_ncomp_zz[dir],ncomp_dir_pc);
                ncomp_tot_pc_xx *= m_ncomp_pc_xx[dir];
                ncomp_tot_pc_yy *= m_ncomp_pc_yy[dir];
                ncomp_tot_pc_zz *= m_ncomp_pc_zz[dir];
            }

            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{0}, lev, ba_Jx, dm, ncomp_tot_pc_xx, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{1}, lev, ba_Jy, dm, ncomp_tot_pc_yy, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{2}, lev, ba_Jz, dm, ncomp_tot_pc_zz, ngJ, 0.0_rt);
        }
    }

    // Set the pointer to mass matrix MultiFab
    if (m_use_mass_matrices_pc) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_mmpc_mfarrvec.push_back(m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev));
        }
    }

}

void ImplicitSolver::PreLinearSolve ()
{
    BL_PROFILE("ImplicitSolver::PreLinearSolve()");

    if (m_use_mass_matrices) {

        m_WarpX->DepositMassMatrices();

        if (m_use_mass_matrices_jacobian) {
            FinishMassMatrices();
            // NOTE: the E0 snapshot paired with J0 (SaveE) is taken in
            // PreRHSOp immediately before the deposit, NOT here: by the
            // time PreLinearSolve runs, solvers whose ComputeRHS overwrites
            // Efield_fp with the assembled RHS (the hybrid Ohm solve leaves
            // E_ohm there) no longer hold the gather field in Efield_fp,
            // and snapshotting it would inject a residual-proportional
            // M*F(U) offset into every linear-stage matvec.
        }

        if (m_use_mass_matrices_pc) {
            SyncMassMatricesPCAndApplyBCs();
            const amrex::Real theta_dt = m_theta*m_dt;
            SetMassMatricesForPC( theta_dt );
        }

    }

}

void ImplicitSolver::PreRHSOp ( const amrex::Real  a_cur_time,
                                const int          a_nl_iter,
                                const bool         a_from_jacobian )
{
    BL_PROFILE("ImplicitSolver::PreRHSOp()");

    using warpx::fields::FieldType;

    if (WarpX::use_filter) {
        const int finest_level = 0;
        m_WarpX->ApplyFilterMF(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), 0);
    }

    // Restore discrete energy adjointness between the guard-current fold and
    // the particle field gather at a reflecting r-max wall: overwrite the r-hi
    // guard cells of the gather-source Efield_fp with the parity mirror that is
    // the exact energy adjoint of the J fold (ApplyReflectiveBoundarytoJfield).
    // Ordering makes this a gather-only modification within each ComputeRHS
    // evaluation: the Faraday curl already consumed the PEC-imaged ghosts in
    // the UpdateWarpXFields step (SetElectricFieldAndApplyBCs fills ghosts,
    // then UpdateMagneticFieldAndApplyBCs takes curl(E); see e.g.
    // ThetaImplicitHybrid::ComputeRHS, which calls UpdateWarpXFields before
    // PreRHSOp), and the Ohm/field solve rewrites Efield_fp after PreRHSOp
    // returns, so nothing downstream reads the overridden ghosts. Applied
    // after the E filter above so the mirrored values are the filtered ones
    // the interior gather sees. Runs in Jacobian evaluations too: the residual
    // map must include it for the linearization to be consistent.
    if (m_adjoint_gather_ghosts) {
        ApplyAdjointGatherGhosts();
    }

    // Advance the particle positions by 1/2 dt,
    // particle velocities by dt, then take average of old and new v,
    // deposit currents, giving J at n+1/2
    // This uses Efield_fp and Bfield_fp, the field at n+1/2 from the previous iteration.
    const bool skip_deposition = false;

    // Set the implict solver options for particles and setting the current density
    ImplicitOptions options;
    options.linear_stage_of_jfnk = a_from_jacobian;
    options.use_mass_matrices_pc = m_use_mass_matrices_pc;
    options.use_mass_matrices_jacobian = m_use_mass_matrices_jacobian;
    options.evolve_suborbit_particles_only = false;
    options.reflect_particles_at_rmax = m_reflect_particles_at_rmax;

    if (a_nl_iter == 0 && !a_from_jacobian &&
        m_use_mass_matrices_jacobian && m_skip_particle_picard_init) {
        // Only do a single Picard iteration for particles on the initial Newton step
        options.max_particle_iterations = 1;
        options.particle_tolerance = 0.0;
    }
    else {
        options.max_particle_iterations = m_max_particle_iterations;
        options.particle_tolerance = m_particle_tolerance;
    }

    if (m_use_mass_matrices_jacobian && a_from_jacobian) { // Called from linear stage of JFNK and using mass matrices for Jacobian
        if (m_particle_suborbits) {
            options.evolve_suborbit_particles_only = true;
            m_WarpX->PushParticlesandDeposit(a_cur_time, skip_deposition, PositionPushType::Full, MomentumPushType::Full, &options);
        }
        const bool J_from_MM_only = !options.evolve_suborbit_particles_only;
        ComputeJfromMassMatrices( J_from_MM_only );
    }
    else { // Conventional particle-suppressed JFNK
        if (m_use_mass_matrices_jacobian) {
            // Snapshot the gather field paired with this deposit: the
            // linear stage composes J = J0 + MM*(E - E0), and E0 must be
            // the field the J0-depositing particles gathered in THIS
            // evaluation (Efield_fp right now, after the E-filter and any
            // scheme-specific push-field adjustments). Snapshotting later
            // breaks for residuals that overwrite Efield_fp (hybrid Ohm).
            SaveE();
        }
        m_WarpX->PushParticlesandDeposit(a_cur_time, skip_deposition, PositionPushType::Full, MomentumPushType::Full, &options);
        CumulateJ();
    }

    // During the mass-matrix linear stage nothing deposits rho or the
    // per-species moments: they stay frozen at the last nonlinear
    // evaluation's (already synced) state. Re-syncing rho here would
    // re-add guard-cell images per matvec (SumBoundary is not
    // idempotent) and, in radial geometry, re-apply the inverse-volume
    // scaling to the stale density.
    const bool mm_linear_stage =
        m_use_mass_matrices_jacobian && a_from_jacobian && !m_particle_suborbits;

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    // Apply the inverse volume scaling for radial geometries after the total
    // current has been accumulated from all containers above. The charge
    // density needs no such treatment here: rho is deposited directly and is
    // scaled inside WarpX::PushParticlesandDeposit(), on the implicit path too.
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        m_WarpX->ApplyInverseVolumeScalingToCurrentDensity(J[0], J[1], J[2], lev);
        // The particle evolve applies the radial-geometry volume scaling only on the
        // explicit path; charge density deposited during implicit residual evaluations
        // must be scaled here or solvers that divide by rho (e.g. the hybrid Ohm's law)
        // see an unscaled density.
        if (!mm_linear_stage && m_WarpX->m_fields.has(FieldType::rho_fp, lev)) {
            m_WarpX->ApplyInverseVolumeScalingToChargeDensity(m_WarpX->m_fields.get(FieldType::rho_fp, lev), lev);
        }
    }
#endif

    // Apply BCs to J and communicate (rho only when it was re-deposited)
    if (mm_linear_stage) {
        using ablastr::fields::Direction;
        m_WarpX->SyncCurrent("current_fp");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_WarpX->ApplyJfieldBoundary(lev,
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev),
                PatchType::fine);
        }
        // One-shot debug verification of the mass-matrix current response
        // against a full particle push/deposit at this exact field state
        // (aborts on completion by design).
        if (m_verify_mm_jvp_step >= 0 && !m_verify_mm_jvp_done &&
            m_WarpX->getistep(0) >= m_verify_mm_jvp_step) {
            m_verify_mm_jvp_done = true;
            VerifyMassMatricesJVP(a_cur_time);
        }
    } else {
        m_WarpX->SyncCurrentAndRho();
    }

    if (m_nlsolver_type == NonlinearSolverType::petsc_snes && !a_from_jacobian) {
        // The native Newton solver calls this routine immediately before the linear solve,
        // and only when a linear solve is required (i.e., the system is not converged).
        // PETSc's SNES solver does not provide this optimization, so we must call it here.
        PreLinearSolve();
    }

}

void ImplicitSolver::ApplyAdjointGatherGhosts ()
{
#if defined(WARPX_DIM_RZ)
    using warpx::fields::FieldType;

    // Gate: the adjoint fill below is derived for the r-hi boundary combination
    // field PEC + particle Reflecting, where the guard-J fold takes the
    // particle-Reflecting parity precedence (psign assignment in
    // PEC::ApplyReflectiveBoundarytoJfield): J_r (normal) folds odd,
    // J_theta/J_z (tangential) fold even -- while the gather ghosts hold the
    // PEC conductor image (SetEfieldOnPEC) with the opposite parity on all
    // three components.
    if (WarpX::field_boundary_hi[0] != FieldBoundaryType::PEC) { return; }
    if (WarpX::particle_boundary_hi[0] != ParticleBoundaryType::Reflecting) { return; }

    // Parity per component = the J-fold psign for this combination. The fold's
    // radius weight (rscale = r_guard/r_valid) transposes against the
    // deposition volumes (V ~ r after the inverse-volume scaling, at the
    // component-staggered radii) to exactly 1, so the energy-adjoint ghost
    // fill is a pure parity mirror across the wall (node index N):
    //   E_r(N+g)     = -E_r(N-1-g)   (cc-in-r: mirror about the wall plane)
    //   E_theta(N+g) = +E_theta(N-g) (nodal-in-r: mirror about the wall node)
    //   E_z(N+g)     = +E_z(N-g)
    // With this fill, Sum_p q v.E(x_p) equals the r-weighted Sum_i V_i J_i E_i
    // over the interior for every deposit (verified to roundoff standalone).
    // The wall node itself is untouched: the fold's self-mirror doubling there
    // is the wall node's half-volume normalization, not an energy term. B
    // ghosts are left unchanged (B does no work on the particles; revisit if
    // the wall momentum channel ever needs the matching treatment).
    const amrex::GpuArray<amrex::Real,3> parity = {-1.0_rt, 1.0_rt, 1.0_rt};

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::Box domain_box = m_WarpX->Geom(lev).Domain();
        domain_box.convert(amrex::IntVect::TheNodeVector());
        const int ir_wall_node = domain_box.bigEnd(0);

        const ablastr::fields::VectorField E =
            m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);

        for (int dir = 0; dir < 3; ++dir) {
            amrex::MultiFab& Emf = *E[dir];
            const int is_nodal_r = static_cast<int>(Emf.ixType().nodeCentered(0));
            // Mirror about the wall in this component's index space; same
            // arithmetic as the J-fold mirror (mirrorfac, hi side).
            const int mirrorfac = 2*ir_wall_node - (1 - is_nodal_r);
            const amrex::Real psign = parity[dir];

            // No tiling: ghost regions are written, and the r-hi guard band
            // must be handled exactly once per fab.
            for (amrex::MFIter mfi(Emf, false); mfi.isValid(); ++mfi) {
                const amrex::Box& vbx = mfi.validbox();
                // Only fabs whose valid region touches the r-hi wall own the
                // physical guard band there; other fabs' r-hi ghosts overlap
                // neighboring valid data and keep their FillBoundary values.
                if (vbx.bigEnd(0) != ir_wall_node - (1 - is_nodal_r)) { continue; }

                const amrex::Box& fabbox = mfi.fabbox();
                const int ir_lo = fabbox.smallEnd(0);
                // Guard band: first index beyond the wall through the fab
                // edge, over the full transverse extent (including transverse
                // ghosts, matching the transverse-grown fold boxes).
                amrex::Box gbx = fabbox;
                gbx.setSmall(0, ir_wall_node + is_nodal_r);

                auto const& Earr = Emf.array(mfi);
                // Writes go only to guard indices beyond the wall and reads
                // come only from indices at or below it, so iterations are
                // independent.
                amrex::ParallelFor(gbx, Emf.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                {
                    const int im = mirrorfac - i;
                    if (im >= ir_lo) {
                        Earr(i,j,k,n) = psign * Earr(im,j,k,n);
                    }
                });
            }
        }
    }
#endif
}

void ImplicitSolver::VerifyMassMatricesJVP ( const amrex::Real a_cur_time )
{
#if defined(WARPX_DIM_RZ)
    // One-shot ground-truth check of the mass-matrix Jacobian action: the
    // linear stage has just produced the mass-matrix current
    //   J_mm(Z) = fold(scale(J0 + S*(E(Z) - E0)))
    // at the perturbed field state Z currently held in Efield_fp. Here the
    // true current response at the identical state is evaluated with a full
    // particle push/deposit, J_true(Z) = fold(scale(deposit(Z))), and the
    // baseline J_base = fold(scale(J0)) is rebuilt from the stored raw J0.
    // Since J_mm is affine in E, J_mm - J_true measures the Jacobian-action
    // error of the mass matrices on the actual Krylov direction eps*v =
    // E(Z) - E0, and (J_true - J_base) is the true response that normalizes
    // it. Errors are reported per radial row (aggregated over z), which
    // resolves the wall band. The probe overwrites the linear-stage state
    // (particle iterates, rho, J0 pairing), so the run aborts on completion.
    using namespace amrex::literals;
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    amrex::Print() << "\n==== ImplicitSolver::VerifyMassMatricesJVP (one-shot) ====\n";
    amrex::Print() << "step index " << m_WarpX->getistep(0)
                   << ", time " << a_cur_time
                   << ", boundary rows " << (m_mass_matrices_boundary_rows ? "ON" : "OFF")
                   << ", adjoint gather ghosts " << (m_adjoint_gather_ghosts ? "ON" : "OFF")
                   << "\n";

    const int lev0 = 0;
    ablastr::fields::VectorField Jfp = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev0);
    const ablastr::fields::VectorField J0 =
        m_WarpX->m_fields.get_alldirs(FieldType::current_fp_non_suborbit, lev0);
    const ablastr::fields::VectorField E =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev0);
    const ablastr::fields::VectorField E0 =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev0);

    // Magnitude of the Krylov direction eps*v = E - E0 (valid cells)
    amrex::Real dE2 = 0.0;
    for (int dir = 0; dir < 3; ++dir) {
        amrex::MultiFab tmp(E[dir]->boxArray(), E[dir]->DistributionMap(), 1, 0);
        amrex::MultiFab::LinComb(tmp, 1.0_rt, *E[dir], 0, -1.0_rt, *E0[dir], 0, 0, 1, 0);
        const amrex::Real n2 = tmp.norm2(0);
        dE2 += n2*n2;
    }
    amrex::Print() << "|eps*v| = |E - E0| = " << std::sqrt(dE2) << "\n";

    // (1) Save J_mm(Z): the folded, volume-scaled linear-stage current.
    std::array<std::unique_ptr<amrex::MultiFab>,3> Jmm, Jbase;
    for (int dir = 0; dir < 3; ++dir) {
        Jmm[dir] = std::make_unique<amrex::MultiFab>(
            Jfp[dir]->boxArray(), Jfp[dir]->DistributionMap(),
            Jfp[dir]->nComp(), Jfp[dir]->nGrowVect());
        amrex::MultiFab::Copy(*Jmm[dir], *Jfp[dir], 0, 0, Jfp[dir]->nComp(), Jfp[dir]->nGrowVect());
    }

    // Shared post-deposit sequence: the same scaling/communication/boundary
    // operations the residual evaluation applies to a raw deposit.
    auto finish_raw_J = [&]() {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            ablastr::fields::VectorField Jl =
                m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
            m_WarpX->ApplyInverseVolumeScalingToCurrentDensity(Jl[0], Jl[1], Jl[2], lev);
        }
        m_WarpX->SyncCurrent("current_fp");
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_WarpX->ApplyJfieldBoundary(lev,
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
                m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev),
                PatchType::fine);
        }
    };

    // (2) Rebuild J_base = fold(scale(J0)) from the stored raw baseline.
    for (int dir = 0; dir < 3; ++dir) {
        amrex::MultiFab::Copy(*Jfp[dir], *J0[dir], 0, 0, J0[dir]->nComp(), J0[dir]->nGrowVect());
    }
    finish_raw_J();
    for (int dir = 0; dir < 3; ++dir) {
        Jbase[dir] = std::make_unique<amrex::MultiFab>(
            Jfp[dir]->boxArray(), Jfp[dir]->DistributionMap(),
            Jfp[dir]->nComp(), Jfp[dir]->nGrowVect());
        amrex::MultiFab::Copy(*Jbase[dir], *Jfp[dir], 0, 0,
                              Jfp[dir]->nComp(), Jfp[dir]->nGrowVect());
    }

    // (3) True response: full particle push/deposit at the identical field
    // state (a nonlinear-stage evaluation, no mass-matrix routing shortcuts).
    {
        ImplicitOptions options;
        options.linear_stage_of_jfnk = false;
        options.use_mass_matrices_pc = false;
        options.use_mass_matrices_jacobian = false;
        options.evolve_suborbit_particles_only = false;
        options.reflect_particles_at_rmax = m_reflect_particles_at_rmax;
        options.max_particle_iterations = m_max_particle_iterations;
        options.particle_tolerance = m_particle_tolerance;
        const bool skip_deposition = false;
        m_WarpX->PushParticlesandDeposit(a_cur_time, skip_deposition,
            PositionPushType::Full, MomentumPushType::Full, &options);
        CumulateJ();
        finish_raw_J();
    }
    // current_fp now holds J_true(Z)

    // (4) Per-radial-row aggregates over the level-0 valid region.
    amrex::Box dom_node = m_WarpX->Geom(lev0).Domain();
    dom_node.convert(amrex::IntVect::TheNodeVector());
    const int ir_lo = dom_node.smallEnd(0);
    const int nrow = dom_node.length(0) + 1;
    const char* comp_names[3] = {"Jr", "Jt", "Jz"};

    for (int dir = 0; dir < 3; ++dir) {
        std::vector<amrex::Real> num2(nrow, 0.0), den2(nrow, 0.0), mm2(nrow, 0.0);
        const int is_nodal_r = static_cast<int>(Jfp[dir]->ixType().nodeCentered(0));
        const int ir_wall = dom_node.bigEnd(0) - (1 - is_nodal_r);

        for (amrex::MFIter mfi(*Jfp[dir]); mfi.isValid(); ++mfi) {
            const amrex::Box& vbx = mfi.validbox();
            const int ncomp = Jfp[dir]->nComp();
            amrex::FArrayBox h_true(vbx, ncomp, amrex::The_Pinned_Arena());
            amrex::FArrayBox h_mm(vbx, ncomp, amrex::The_Pinned_Arena());
            amrex::FArrayBox h_base(vbx, ncomp, amrex::The_Pinned_Arena());
            h_true.copy<amrex::RunOn::Device>((*Jfp[dir])[mfi], vbx, 0, vbx, 0, ncomp);
            h_mm.copy<amrex::RunOn::Device>((*Jmm[dir])[mfi], vbx, 0, vbx, 0, ncomp);
            h_base.copy<amrex::RunOn::Device>((*Jbase[dir])[mfi], vbx, 0, vbx, 0, ncomp);
            amrex::Gpu::streamSynchronize();
            auto const& at = h_true.const_array();
            auto const& am = h_mm.const_array();
            auto const& ab = h_base.const_array();
            const auto lo = amrex::lbound(vbx);
            const auto hi = amrex::ubound(vbx);
            for (int n = 0; n < ncomp; ++n) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        const int irow = i - ir_lo;
                        if (irow < 0 || irow >= nrow) { continue; }
                        const amrex::Real dnum = am(i,j,0,n) - at(i,j,0,n);
                        const amrex::Real dden = at(i,j,0,n) - ab(i,j,0,n);
                        num2[irow] += dnum*dnum;
                        den2[irow] += dden*dden;
                        mm2[irow]  += (am(i,j,0,n) - ab(i,j,0,n))*(am(i,j,0,n) - ab(i,j,0,n));
                    }
                }
            }
        }
        amrex::ParallelDescriptor::ReduceRealSum(num2.data(), nrow);
        amrex::ParallelDescriptor::ReduceRealSum(den2.data(), nrow);
        amrex::ParallelDescriptor::ReduceRealSum(mm2.data(), nrow);

        amrex::Print() << "\n-- " << comp_names[dir]
            << " (wall row ir = " << ir_wall << ", rows printed wall -> axis) --\n";
        amrex::Print() << "  dist  ir    |dJ_true|      |dJ_mm|        |dJ_mm-dJ_true|  rel\n";
        amrex::Real num2_wall = 0.0, den2_wall = 0.0, num2_int = 0.0, den2_int = 0.0;
        for (int irow = ir_wall - ir_lo; irow >= 0; --irow) {
            const int dist = ir_wall - (irow + ir_lo);
            const amrex::Real den = std::sqrt(den2[irow]);
            const amrex::Real num = std::sqrt(num2[irow]);
            const amrex::Real mmv = std::sqrt(mm2[irow]);
            const amrex::Real rel = num / std::max(den, std::numeric_limits<amrex::Real>::min());
            if (dist <= 3) { num2_wall += num2[irow]; den2_wall += den2[irow]; }
            else           { num2_int  += num2[irow]; den2_int  += den2[irow]; }
            amrex::Print() << "  " << std::setw(4) << dist
                << "  " << std::setw(4) << (irow + ir_lo)
                << "  " << std::scientific << std::setprecision(6) << den
                << "  " << mmv << "  " << num
                << "  " << std::setprecision(3) << rel << "\n";
        }
        constexpr amrex::Real tiny = std::numeric_limits<amrex::Real>::min();
        amrex::Print() << "  summary " << comp_names[dir]
            << ": wall band (dist<=3) rel = "
            << std::scientific << std::setprecision(3)
            << std::sqrt(num2_wall)/std::max(std::sqrt(den2_wall), tiny)
            << ", interior rel = "
            << std::sqrt(num2_int)/std::max(std::sqrt(den2_int), tiny)
            << "\n";
    }

    amrex::Print() << "\n==== VerifyMassMatricesJVP complete ====\n\n";
    WARPX_ABORT_WITH_MESSAGE(
        "ImplicitSolver: verify_mm_jvp complete (one-shot diagnostic; the probe "
        "overwrites solver state, so the run aborts by design).");
#else
    amrex::ignore_unused(a_cur_time);
    WARPX_ABORT_WITH_MESSAGE(
        "ImplicitSolver: verify_mm_jvp_step is only implemented for RZ geometry.");
#endif
}

void ImplicitSolver::SyncMassMatricesPCAndApplyBCs ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Add select mass matrices elements to the preconditioner containers,
    // which may alread include contributions from suborbit particles that
    // are not included in the mass matrices.

    const int diag_comp_xx = (AMREX_D_TERM(m_ncomp_xx[0],*m_ncomp_xx[1],*m_ncomp_xx[2])-1)/2;
    const int diag_comp_yy = (AMREX_D_TERM(m_ncomp_yy[0],*m_ncomp_yy[1],*m_ncomp_yy[2])-1)/2;
    const int diag_comp_zz = (AMREX_D_TERM(m_ncomp_zz[0],*m_ncomp_zz[1],*m_ncomp_zz[2])-1)/2;
    int MM_PC_ncomp_xx[3] = {1, 1, 1};
    int MM_PC_ncomp_yy[3] = {1, 1, 1};
    int MM_PC_ncomp_zz[3] = {1, 1, 1};
    int MM_PC_width_xx[3] = {0, 0, 0};
    int MM_PC_width_yy[3] = {0, 0, 0};
    int MM_PC_width_zz[3] = {0, 0, 0};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        MM_PC_ncomp_xx[dir]  = m_ncomp_pc_xx[dir];
        MM_PC_ncomp_yy[dir]  = m_ncomp_pc_yy[dir];
        MM_PC_ncomp_zz[dir]  = m_ncomp_pc_zz[dir];
        MM_PC_width_xx[dir]  = (m_ncomp_pc_xx[dir] - 1)/2;
        MM_PC_width_yy[dir]  = (m_ncomp_pc_yy[dir] - 1)/2;
        MM_PC_width_zz[dir]  = (m_ncomp_pc_zz[dir] - 1)/2;
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        const amrex::MultiFab* MM_xx = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
        const amrex::MultiFab* MM_yy = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
        const amrex::MultiFab* MM_zz = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);
        ablastr::fields::VectorField MM_PC = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev);

        // Below is general for 1D and 2D. It works for 3D because for now we limit width = 0 in 3D.

        const int diag_comp_pc_xx = (MM_PC[0]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_xx[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_xx[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_xx    - MM_PC_width_xx[0] + m_ncomp_xx[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_xx - MM_PC_width_xx[0] + m_ncomp_pc_xx[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[0], *MM_xx, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_xx[0], MM_xx->nGrowVect());
        }
        const int diag_comp_pc_yy = (MM_PC[1]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_yy[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_yy[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_yy    - MM_PC_width_yy[0] + m_ncomp_yy[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_yy - MM_PC_width_yy[0] + m_ncomp_pc_yy[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[1], *MM_yy, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_yy[0], MM_yy->nGrowVect());
        }
        const int diag_comp_pc_zz = (MM_PC[2]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_zz[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_zz[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_zz    - MM_PC_width_zz[0] + m_ncomp_zz[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_zz - MM_PC_width_zz[0] + m_ncomp_pc_zz[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[2], *MM_zz, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_zz[0], MM_zz->nGrowVect());
        }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        m_WarpX->ApplyInverseVolumeScalingToMassMatricesPC(MM_PC[0], MM_PC[1], MM_PC[2], lev);
#endif
    }

    // Do addOp Exchange on MassMatrices_PC
    m_WarpX->SyncMassMatricesPC();

    // Apply BCs to MassMatrices_PC
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        m_WarpX->ApplyJfieldBoundary(lev,
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{0}, lev),
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{1}, lev),
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{2}, lev),
            PatchType::fine);
    }
}

void ImplicitSolver::SetMassMatricesForPC ( const amrex::Real a_theta_dt )
{

    using namespace amrex::literals;
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Scale mass matrices used by preconditioner by c^2*mu0*theta*dt.
    // Add one to diagonal terms when using the curl_curl_mlmg pc_type.
    // The pc_type petsc already has the one from the curl curl operator
    // Note: This should be done after Sync/communication has been called

    const amrex::Real pc_factor = PhysConst::c2 * PhysConst::mu0 * a_theta_dt;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::MultiFab* MMxx_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{0}, lev);
        amrex::MultiFab* MMyy_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{1}, lev);
        amrex::MultiFab* MMzz_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{2}, lev);
        MMxx_PC->mult(pc_factor, 0, MMxx_PC->nComp());
        MMyy_PC->mult(pc_factor, 0, MMyy_PC->nComp());
        MMzz_PC->mult(pc_factor, 0, MMzz_PC->nComp());
        const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
        if (pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            // Need to add 1 to the diagonal terms for the curl_curl pc
            const int diag_comp_Mxx = (MMxx_PC->nComp()-1)/2;
            const int diag_comp_Myy = (MMyy_PC->nComp()-1)/2;
            const int diag_comp_Mzz = (MMzz_PC->nComp()-1)/2;
            MMxx_PC->plus(1.0_rt, diag_comp_Mxx, 1, 0);
            MMyy_PC->plus(1.0_rt, diag_comp_Myy, 1, 0);
            MMzz_PC->plus(1.0_rt, diag_comp_Mzz, 1, 0);
        }
    }

}

void ImplicitSolver::FinishMassMatrices ()
{
    BL_PROFILE("ImplicitSolver::FinishMassMatrices()");

    // The MM deposit routine takes advantage of symmetry for the diagonal mass
    // matrices to only deposit roughly half of the values. The remainder are
    // computed via copy here in this routine.

#if AMREX_SPACEDIM < 3
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

#if AMREX_SPACEDIM > 1
    const int ncomp_tot_xx = AMREX_D_TERM(m_ncomp_xx[0],*m_ncomp_xx[1],*m_ncomp_xx[2]);
    const int ncomp_tot_yy = AMREX_D_TERM(m_ncomp_yy[0],*m_ncomp_yy[1],*m_ncomp_yy[2]);
    const int ncomp_tot_zz = AMREX_D_TERM(m_ncomp_zz[0],*m_ncomp_zz[1],*m_ncomp_zz[2]);
#endif

    amrex::GpuArray<int,3> ncomp_xx = {1,1,1};
    amrex::GpuArray<int,3> ncomp_yy = {1,1,1};
    amrex::GpuArray<int,3> ncomp_zz = {1,1,1};
    amrex::GpuArray<int,3> Sxx_width = {0,0,0};
    amrex::GpuArray<int,3> Syy_width = {0,0,0};
    amrex::GpuArray<int,3> Szz_width = {0,0,0};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        ncomp_xx[dir] = m_ncomp_xx[dir];
        ncomp_yy[dir] = m_ncomp_yy[dir];
        ncomp_zz[dir] = m_ncomp_zz[dir];
        Sxx_width[dir] = (ncomp_xx[dir] - 1)/2;
        Syy_width[dir] = (ncomp_yy[dir] - 1)/2;
        Szz_width[dir] = (ncomp_zz[dir] - 1)/2;
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*SX[0], false); mfi.isValid(); ++mfi )
        {

            amrex::Array4<amrex::Real> const& Sxx = SX[0]->array(mfi);
            amrex::Array4<amrex::Real> const& Syy = SY[1]->array(mfi);
            amrex::Array4<amrex::Real> const& Szz = SZ[2]->array(mfi);

            // Use grown boxes here with all S guard cells
            amrex::Box Sbx = amrex::convert(mfi.validbox(),SX[0]->ixType());
            amrex::Box Sby = amrex::convert(mfi.validbox(),SY[1]->ixType());
            amrex::Box Sbz = amrex::convert(mfi.validbox(),SZ[2]->ixType());
            Sbx.grow(SX[0]->nGrowVect());
            Sbz.grow(SZ[2]->nGrowVect());
            Sby.grow(SY[1]->nGrowVect());

#if AMREX_SPACEDIM == 1
            amrex::ParallelFor( Sbx, Sby, Sbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Sxx(i,d + n) = Sxx(i + n,d - n), where d = Sxx_width[0]
                const int width = amrex::min(Sxx_width[0],Sbx.bigEnd(0)-i);
                for (int n = 1; n <= width; ++n) {
                    const int dst_comp = Sxx_width[0] + n;
                    const int src_comp = Sxx_width[0] - n;
                    Sxx(i,j,k,dst_comp) = Sxx(i + n,j,k,src_comp);
                }
            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Syy(i,d + n) = Syy(i + n,d - n), where d = Syy_width[0]
                const int width = std::min(Syy_width[0], Sby.bigEnd(0) - i);
                for (int n = 1; n <= width; n++) {
                    const int dst_comp = Syy_width[0] + n;
                    const int src_comp = Syy_width[0] - n;
                    Syy(i,j,k,dst_comp) = Syy(i + n,j,k,src_comp);
                }
            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Szz(i,d + n) = Szz(i + n,d - n), where d = Szz_width[0]
                const int width_zz = std::min(Szz_width[0],Sbz.bigEnd(0) - i);
                for (int n = 1; n <= width_zz; n++) {
                    const int dst_comp = Szz_width[0] + n;
                    const int src_comp = Szz_width[0] - n;
                    Szz(i,j,k,dst_comp) = Szz(i + n,j,k,src_comp);
                }
            });

#elif AMREX_SPACEDIM == 2
            // In-place fold of the mass matrices: for every (ncomp_x, ncomp_y)
            // combination, the components written at iv_dst are disjoint from
            // the components read at any i-offset source, so iterations of the
            // vectorized i loop are independent, as required by ParallelFor
            // (see issue #7097). Reads across j rely on the serial ascending j
            // loop on CPU and must not be reordered.
            amrex::ParallelFor( Sbx, Sby, Sbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = amrex::max(0,ncomp_xx[1] - ncomp_xx[0]);

                for (int m = row_start; m < ncomp_xx[1]; ++m) {
                    const int jj = m - Sxx_width[1];

                    const int above_diag = (m > Sxx_width[1]) ? 1 : 0;
                    const int width0 = amrex::min(m + above_diag - row_start + 1, ncomp_xx[0]);

                    for (int n = 0; n < width0; ++n) {
                        const int ii = Sxx_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sbx.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_xx[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_xx - 1 - dst_comp;

                        Sxx(iv_dst,dst_comp) = Sxx(iv_src,src_comp);
                    }

                }

            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = 1;

                for (int m = row_start; m < ncomp_yy[1]; m++) {
                    const int jj = m - Syy_width[1];

                    const int above_diag = (m > Syy_width[1]) ? 1 : 0;
                    const int width0 = std::min(m + above_diag - row_start + 1, ncomp_yy[0]);

                    for (int n = 0; n < width0; n++) {
                        const int ii = Syy_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sby.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_yy[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_yy - 1 - dst_comp;

                        Syy(iv_dst,dst_comp) = Syy(iv_src,src_comp);
                    }
                }

            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = std::max(0,ncomp_zz[1] - ncomp_zz[0]);

                for (int m = row_start; m < ncomp_zz[1]; m++) {
                    const int jj = m - Szz_width[1];

                    const int above_diag = (m > Szz_width[1]) ? 1 : 0;
                    const int width0 = std::min(m - row_start + above_diag + 1, ncomp_zz[0]);

                    for (int n = 0; n < width0; n++) {
                        const int ii = Szz_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sbz.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_zz[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_zz - 1 - dst_comp;

                        Szz(iv_dst,dst_comp) = Szz(iv_src,src_comp);
                    }
                }

            });
#endif
        }
    }
#endif

    // Fold the physical-boundary response into the completed bands. Must run
    // after the symmetry completion above, which reads the raw guard rows.
    if (m_mass_matrices_boundary_rows) {
        ApplyMassMatricesBoundaryRows();
    }
}

void ImplicitSolver::ApplyMassMatricesBoundaryRows ()
{
#if defined(WARPX_DIM_RZ)
    BL_PROFILE("ImplicitSolver::ApplyMassMatricesBoundaryRows()");
    using namespace amrex::literals;

    // Physical-boundary rows for the particle mass matrices at the r-hi wall,
    // for the boundary combination field PEC + particle Reflecting (same gate
    // as the guard-current fold parity precedence and the adjoint ghost fill).
    //
    // The deposit writes raw band entries S_ab(i; D) coupling J_a(i) to
    // E_b(i+D), including guard rows (i beyond the wall) and guard columns
    // (i+D beyond the wall), exactly as it writes raw guard J. The true
    // residual composes that band with two boundary maps (see e.g. Chen &
    // Chacon, Comput. Phys. Commun. 197 (2015) 73, and Chacon & Chen,
    // J. Comput. Phys. 316 (2016) 578, for the mass-matrix formulation of the
    // linearized current response):
    //  - a gather-side ghost image G: guard E_b equals q_b times its interior
    //    mirror (PEC conductor image by default; the fold-adjoint image when
    //    adjoint_gather_ghosts is on), with the wall-node tangential E zeroed
    //    on the conductor before every gather;
    //  - a deposit-side row fold F: guard J_a rows fold onto their interior
    //    mirrors with the particle-Reflecting parity precedence (J_r odd,
    //    J_theta/J_z even) and the radius weight rscale = r_guard/r_interior,
    //    including the wall-node self-fold doubling for nodal-in-r rows
    //    (the wall node's half-volume normalization; cf. Verboncoeur,
    //    J. Comput. Phys. 174 (2001) 421).
    // On the raw (pre-inverse-volume) bands rscale cancels exactly against
    // the guard/interior ring-volume ratio (V ~ r at the staggered radius),
    // so both legs reduce to pure parity maps here.
    //
    // Folded guard rows and guard columns are zeroed afterwards: their
    // content now enters through interior entries, and the vector-side guard
    // fold and ghost-image fill applied around the mass-matrix matvec then
    // contribute nothing extra, so the matvec is unchanged while the stored
    // bands become the true composed interior operator (required by any
    // consumer that reads band entries directly rather than acting through
    // the folded matvec).
    if (WarpX::field_boundary_hi[0] != FieldBoundaryType::PEC) { return; }
    if (WarpX::particle_boundary_hi[0] != ParticleBoundaryType::Reflecting) { return; }

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Row parity p_a: the guard-current fold parity for this boundary
    // combination (psign of ApplyReflectiveBoundarytoJfield, Reflecting
    // precedence). Column parity q_b: the ghost-image parity the gather
    // actually uses, which must follow the adjoint_gather_ghosts knob.
    const amrex::GpuArray<amrex::Real,3> p_row = {-1.0_rt, 1.0_rt, 1.0_rt};
    const amrex::GpuArray<amrex::Real,3> q_col = m_adjoint_gather_ghosts ?
        amrex::GpuArray<amrex::Real,3>{-1.0_rt,  1.0_rt,  1.0_rt} :
        amrex::GpuArray<amrex::Real,3>{ 1.0_rt, -1.0_rt, -1.0_rt};

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        amrex::Box domain_box = m_WarpX->Geom(lev).Domain();
        domain_box.convert(amrex::IntVect::TheNodeVector());
        const int ir_wall_node = domain_box.bigEnd(0);

        const ablastr::fields::VectorField J =
            m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        const std::array<ablastr::fields::VectorField,3> S = {
            m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev),
            m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev),
            m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev) };

        const std::array<std::array<amrex::IntVect,3>,3> ncomp_ab =
            {{ {m_ncomp_xx, m_ncomp_xy, m_ncomp_xz},
               {m_ncomp_yx, m_ncomp_yy, m_ncomp_yz},
               {m_ncomp_zx, m_ncomp_zy, m_ncomp_zz} }};

        for (int a = 0; a < 3; ++a) {
            // row staggering in r (J_a); E_b and J_b share per-component staggering
            const int na = static_cast<int>(J[a]->ixType().nodeCentered(0));
            for (int b = 0; b < 3; ++b) {
                amrex::MultiFab& Smf = *S[a][b];
                const int nb = static_cast<int>(J[b]->ixType().nodeCentered(0));
                const int ncomp0 = ncomp_ab[a][b][0];
                const int ncomp1 = ncomp_ab[a][b][1];
                // Band-center offset in r; same staggering rule as ApplyMassMatrices
                const int off0 = (a == b) ? (ncomp0 - 1)/2
                               : ((na > nb) ? ncomp0/2 : (ncomp0 - 1)/2);
                const amrex::Real pa = p_row[a];
                const amrex::Real qb = q_col[b];
                // Mirror maps about the wall (node index N): nodal m(i) = 2N - i,
                // cell-centered m(i) = 2N - 1 - i; same arithmetic as the
                // guard-current fold (mirrorfac, hi side).
                const int mfac_row = 2*ir_wall_node - (1 - na);
                const int mfac_col = 2*ir_wall_node - (1 - nb);
                // First index beyond the wall per staggering
                const int first_guard_row = ir_wall_node + na;
                const int first_guard_col = ir_wall_node + nb;
                const int wall_row = ir_wall_node - (1 - na);

                // No tiling: guard regions are read/written and the wall band
                // must be handled exactly once per fab.
                for (amrex::MFIter mfi(Smf, false); mfi.isValid(); ++mfi) {
                    const amrex::Box& vbx = mfi.validbox();
                    // Only fabs whose valid region touches the r-hi wall own
                    // the physical guard band there.
                    if (vbx.bigEnd(0) != wall_row) { continue; }
                    const amrex::Box& fbx = mfi.fabbox();
                    auto const& Sarr = Smf.array(mfi);

                    // ---- Column remap (rows at or inside the wall) ----
                    // Entries whose column lies beyond the wall move to the
                    // imaged interior column with parity q_b; wall-node
                    // tangential columns are dead (that E is zeroed on the
                    // conductor before every gather). Each iteration reads
                    // and writes only its own row's components, and source
                    // components (columns beyond the wall) are disjoint from
                    // destinations (at or inside it), so grid iterations are
                    // independent as required by ParallelFor (see the note
                    // for issue #7097 at the symmetry-completion fold).
                    // Guard rows are excluded: their column remap composes
                    // with the row fold below, where the re-mapped offset is
                    // measured from the destination row so the folded entry
                    // stays inside the allocated band.
                    // First column index needing treatment: the dead wall
                    // node for nodal-in-r columns, the first guard column
                    // otherwise.
                    const int first_bc_col = (nb == 1) ? ir_wall_node : first_guard_col;
                    amrex::Box cbx = fbx;
                    cbx.setSmall(0, std::max(fbx.smallEnd(0),
                                             first_bc_col - (ncomp0 - 1 - off0)));
                    cbx.setBig(0, wall_row);
                    if (cbx.ok()) {
                        amrex::ParallelFor(cbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            for (int m = 0; m < ncomp1; ++m) {
                                for (int n = ncomp0 - 1; n >= 0; --n) {
                                    const int c = i + (n - off0);
                                    if (c < first_guard_col) {
                                        if (nb == 1 && c == ir_wall_node) {
                                            Sarr(i,j,k, n + ncomp0*m) = 0.0_rt;
                                        }
                                        continue;
                                    }
                                    const int comp_src = n + ncomp0*m;
                                    const amrex::Real v = Sarr(i,j,k,comp_src);
                                    Sarr(i,j,k,comp_src) = 0.0_rt;
                                    if (v == 0.0_rt) { continue; }
                                    const int n_dst = (mfac_col - c) - i + off0;
                                    if (n_dst >= 0 && n_dst < ncomp0) {
                                        Sarr(i,j,k, n_dst + ncomp0*m) += qb*v;
                                    }
                                }
                            }
                        });
                    }

                    // ---- Row fold (guard rows onto interior mirrors) ----
                    // For each interior destination row iv the single mirror
                    // source is the guard row ig = mfac_row - iv, whose
                    // entries are still raw (column remap skipped them): the
                    // composed map applies q_b for columns beyond the wall,
                    // drops dead wall-node tangential columns, and lands at
                    // offset D' = c_final - iv. The wall-node SELF-fold
                    // (iv == ig at the wall node for nodal-in-r rows) is
                    // deliberately NOT applied here: the raw-deposit fold
                    // machinery still acts on the mass-matrix matvec result
                    // (it must, for the raw baseline current), and it doubles
                    // the wall-node row there - folding the doubling into the
                    // band as well double-counts it (measured as an exact
                    // factor-2 wall-row error when tried). The stored bands
                    // therefore carry the wall-node row in the raw-deposit
                    // convention: half-normalized until the standard guard
                    // fold doubles it. Each iteration writes only its own row
                    // and reads only its guard mirror, so iterations are
                    // independent.
                    amrex::Box rbx = fbx;
                    rbx.setSmall(0, std::max(fbx.smallEnd(0),
                                             mfac_row - fbx.bigEnd(0)));
                    rbx.setBig(0, wall_row - na); // strictly below the self-fold row
                    if (rbx.ok()) {
                        amrex::ParallelFor(rbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                            const int ig = mfac_row - i;
                            for (int m = 0; m < ncomp1; ++m) {
                                for (int n = 0; n < ncomp0; ++n) {
                                    const amrex::Real v = Sarr(ig,j,k, n + ncomp0*m);
                                    if (v == 0.0_rt) { continue; }
                                    const int c = ig + (n - off0);
                                    if (nb == 1 && c == ir_wall_node) {
                                        continue; // dead wall-node column
                                    }
                                    const bool ghost_col = (c >= first_guard_col);
                                    const int c_final = ghost_col ? (mfac_col - c) : c;
                                    const amrex::Real w = ghost_col ? qb : 1.0_rt;
                                    const int n_dst = c_final - i + off0;
                                    if (n_dst >= 0 && n_dst < ncomp0) {
                                        Sarr(i,j,k, n_dst + ncomp0*m) += pa*w*v;
                                    }
                                    // out-of-band entries cannot occur for
                                    // this fold (the reflected offset stays
                                    // inside the band; see the design note)
                                }
                            }
                        });
                    }

                    // ---- Zero the folded guard rows ----
                    amrex::Box gbx = fbx;
                    gbx.setSmall(0, first_guard_row);
                    if (gbx.ok()) {
                        amrex::ParallelFor(gbx, ncomp0*ncomp1,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
                        {
                            Sarr(i,j,k,n) = 0.0_rt;
                        });
                    }
                }
            }
        }
    }
#endif
}

void ImplicitSolver::PrintBaseImplicitSolverParameters () const
{
    amrex::Print() << "max particle iterations:             " << m_max_particle_iterations << "\n";
    amrex::Print() << "particle relative tolerance:         " << m_particle_tolerance << "\n";
    amrex::Print() << "use particle suborbits:              " << (m_particle_suborbits ? "true":"false") << "\n";
    amrex::Print() << "reflect particles at r-max:          "
                   << (m_reflect_particles_at_rmax ? "true":"false") << "\n";
    amrex::Print() << "adjoint gather ghosts at r-max:      "
                   << (m_adjoint_gather_ghosts ? "true":"false") << "\n";
    amrex::Print() << "print unconverged particle details:  " << (m_print_unconverged_particle_details ? "true":"false") << "\n";
    amrex::Print() << "Nonlinear solver type:               " << amrex::getEnumNameString(m_nlsolver_type) << "\n";
    if ( (m_nlsolver_type == NonlinearSolverType::newton)
      || (m_nlsolver_type == NonlinearSolverType::petsc_snes) ) {
        amrex::Print() << "use mass matrices:                   " << (m_use_mass_matrices ? "true":"false") << "\n";
        if (m_use_mass_matrices) {
            amrex::Print() << "    for jacobian calc:   " << (m_use_mass_matrices_jacobian ? "true":"false") << "\n";
            if (m_use_mass_matrices_jacobian) {
                amrex::Print() << "        skip particle picard init:  " << (m_skip_particle_picard_init ? "true":"false") << "\n";
                amrex::Print() << "        boundary rows:              "
                               << (m_mass_matrices_boundary_rows ? "true":"false") << "\n";
            }
            amrex::Print() << "    for preconditioner:  " << (m_use_mass_matrices_pc ? "true":"false") << "\n";
            if (m_use_mass_matrices_pc) {
                amrex::Print() << "    mass matrices pc width:  " << m_mass_matrices_pc_width << "\n";
            }
            amrex::Print() << "    ncomp_xx:  " << m_ncomp_xx << ";  ncomp_pc_xx:  " << m_ncomp_pc_xx << "\n";
            amrex::Print() << "    ncomp_xy:  " << m_ncomp_xy << ";  ncomp_pc_xy:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_xz:  " << m_ncomp_xz << ";  ncomp_pc_xz:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_yx:  " << m_ncomp_yx << ";  ncomp_pc_yx:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_yy:  " << m_ncomp_yy << ";  ncomp_pc_yy:  " << m_ncomp_pc_yy << "\n";
            amrex::Print() << "    ncomp_yz:  " << m_ncomp_yz << ";  ncomp_pc_yz:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zx:  " << m_ncomp_zx << ";  ncomp_pc_zx:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zy:  " << m_ncomp_zy << ";  ncomp_pc_zy:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zz:  " << m_ncomp_zz << ";  ncomp_pc_zz:  " << m_ncomp_pc_zz << "\n";
        }
    }
}
