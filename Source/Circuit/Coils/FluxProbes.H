/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CIRCUIT_FLUXPROBES_H_
#define WARPX_CIRCUIT_FLUXPROBES_H_

#include "CoilSet.H"

#include <AMReX_Box.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <map>
#include <vector>

namespace warpx::circuit
{
    /** Which linkage measurement a coil's back-EMF port uses. */
    enum class ProbeKind
    {
        none,        //!< coil is not measured (drive-only)
        disk,        //!< plasma-frame disk flux through the coil circle
        reciprocity  //!< free-space reciprocity dot with the plasma current
    };

    /** Plasma flux linkage of a coil by the DISK rule: the plasma-frame
     * B_z integrated over the coil's quarter-offset circle,
     *     lambda_phys = 2 pi Int_0^{r_off} B_z(r, z_off) r dr,
     * with the identical staircase conventions as the discrete
     * self-inductance (STRICT r_cell < r_off cell test, unclamped linear z
     * interpolation between the two bracketing node planes) -- probe and
     * port self-inductance must share the rule or the coupled system's
     * passivity is violated. Returns lambda_phys * I_ref * n_turns (the
     * linkage of the coil's reference drive, whose time derivative divided
     * by I_ref * n_turns is the port EMF in volts). Collective
     * (all-reduces the partial sums); RZ (m = 0), level 0.
     *
     * Valid with conducting walls (it measures the model's own flux
     * through the disk); use it whenever the reciprocity probe is not
     * valid.
     *
     * \param[in] coil     the measured coil
     * \param[in] Bz       plasma-frame B_z (cell r, node z), level 0
     */
    amrex::Real DiskFluxLinkage (const Coil& coil, const amrex::MultiFab& Bz);

    /** Plasma flux linkage of a coil by FREE-SPACE RECIPROCITY: the dot
     * product of the coil's unit vector potential with the plasma current,
     *     lambda = Int A_theta^(I_ref n) J_theta,plasma dV
     *            = 2 pi Sum w_r r A_theta J_theta dr dz,
     * over the nodal (node r, node z) J_theta, with trapezoid end-weights
     * in r (w = 1/2 at i = 0 and i = nr) and each shared nodal point
     * counted once (owner-masked). A carries the coil's reference
     * amp-turns, so the result is lambda_phys * I_ref * n_turns directly.
     * Exact linkage in free space; INVALID with a conducting wall between
     * plasma and coil (the image response is not in A^unit) -- pair it
     * with the Green's-function open boundary, or use the disk probe.
     * Collective; RZ (m = 0), level 0.
     *
     * \param[in] A_theta  the coil's <field_name>_Aext theta component
     *                     (node r, node z), holding the field at
     *                     I_ref * n_turns, level 0
     * \param[in] J_theta  plasma-frame J_theta (node r, node z), level 0
     */
    amrex::Real ReciprocityLinkage (const amrex::MultiFab& A_theta,
                                    const amrex::MultiFab& J_theta);

    /** The coupling power Int J . E dV over the RZ (m = 0) domain, summed
     * component-wise on matched (E/J) staggerings with owner-masked nodal
     * points and volume weights 2 pi r dr dz (r at each sample position).
     * Used for the field side of the coupling-power double entry against
     * -sum_k (ds_k/dt) lambda_k. Collective; level 0.
     *
     * \param[in] J  the three plasma-current components (E/J staggering)
     * \param[in] E  the three field components on the same staggerings
     */
    amrex::Real CouplingPowerIntegral (
        const std::array<const amrex::MultiFab*, 3>& J,
        const std::array<const amrex::MultiFab*, 3>& E);

    /** Geometry-keyed owner-mask cache: the owner mask of a MultiFab's
     * (BoxArray, DistributionMapping, index type) triple under the given
     * periodicity, computed once and reused until the layout changes (a
     * regrid allocates a new BoxArray/DistributionMapping ref, which
     * misses the identity check and rebuilds the entry). Replaces the
     * per-call iMultiFab allocation of the probe integrals. The cached
     * BoxArray/DistributionMapping copies keep the identity refs alive,
     * so a pointer-identity hit can never alias a rebuilt layout. */
    const amrex::iMultiFab& CachedOwnerMask (const amrex::MultiFab& mf,
                                             const amrex::Periodicity& period);

    /** Batched multi-coil linkage measurement.
     *
     * Layered ABOVE the single-coil probe functions (which remain the
     * reference implementation and keep their explicit-MultiFab
     * signatures): all static factors of every coil's probe integrand --
     * owner masks, radius/trapezoid/plane weights, 2 pi dr (dz) volume
     * factors, the disk probe's I_ref * n_turns -- are folded once into
     * per-(coil, box) weight tables on the run mesh, so one measurement
     * is a batch of weight-dot-field products: every device kernel is
     * issued asynchronously, block partials land in fixed slots, one
     * fixed-order combine kernel forms the coil vector, and the host
     * pays exactly ONE stream synchronization, ONE device-to-host copy
     * of n_coils doubles and ONE MPI all-reduce over the vector --
     * against one sync + one all-reduce PER COIL for the serial loop.
     * The fixed reduction tree makes the result deterministic
     * run-to-run (no atomics); it matches the single-coil functions to
     * summation-order roundoff (1e-15-class relative), not bitwise.
     *
     * The weight tables are geometry-keyed (BoxArray +
     * DistributionMapping + Geometry identity) and rebuilt on a regrid,
     * like CachedOwnerMask. RZ (m = 0), level 0, collective.
     */
    class LinkageBatch
    {
    public:
        /** Measure every measured coil's plasma flux linkage.
         *
         * \param[in]  coils    the coil set (rows of the batch)
         * \param[in]  probes   per-coil probe kind (ProbeKind::none rows
         *                      report lambda = 0)
         * \param[in]  a_theta  per-coil unit vector potential
         *                      <field_name>_Aext theta component (node r,
         *                      node z) at I_ref * n_turns; may be null
         *                      for non-reciprocity rows
         * \param[in]  bz       plasma-frame B_z (cell r, node z); may be
         *                      null when no row uses the disk probe
         * \param[in]  j_theta  plasma-frame J_theta (node r, node z); may
         *                      be null when no row uses reciprocity
         * \param[out] lambda   per-coil lambda_phys * I_ref * n_turns,
         *                      coil order, zeros for unmeasured rows
         */
        void Measure (const CoilSet& coils,
                      const std::vector<ProbeKind>& probes,
                      const std::vector<const amrex::MultiFab*>& a_theta,
                      const amrex::MultiFab* bz,
                      const amrex::MultiFab* j_theta,
                      std::vector<amrex::Real>& lambda);

        /** One weight-dot-field job: the cells of one box measured for
         * one coil row. Public for the CUDA extended-lambda rule (device
         * lambdas must live in public member functions); not API. */
        struct Job
        {
            int row = 0;             //!< coil index
            amrex::Box box;          //!< cell enumeration box
            long weight_offset = 0;  //!< into m_weights
            long partial_offset = 0; //!< into m_partials (GPU)
            int nblocks = 0;         //!< reduction blocks (GPU)
        };

        /** (Re)build the geometry-keyed weight pack; called by Measure on
         * a key miss. Public for the CUDA extended-lambda rule; not API. */
        void BuildPack (const CoilSet& coils,
                        const std::vector<ProbeKind>& probes,
                        const std::vector<const amrex::MultiFab*>& a_theta,
                        const amrex::MultiFab* bz,
                        const amrex::MultiFab* j_theta);

    private:
        bool m_built = false;
        int m_ncoils = 0;
        //! layout identity of the pack (regrid invalidation)
        amrex::BoxArray m_key_ba_j, m_key_ba_bz;
        amrex::DistributionMapping m_key_dm_j, m_key_dm_bz;
        amrex::Box m_key_domain;

        //! per-source jobs, keyed by the source's MFIter box index
        std::map<int, std::vector<Job>> m_jobs_j;
        std::map<int, std::vector<Job>> m_jobs_bz;

        amrex::Gpu::DeviceVector<double> m_weights;
        amrex::Gpu::DeviceVector<double> m_partials;
        amrex::Gpu::DeviceVector<double> m_lambda_device;
        //! per-row fixed-order combine segments: for row r, the partial
        //! slabs [seg_partial_offset[k], + seg_nblocks[k]) for k in
        //! [seg_begin[r], seg_begin[r+1])
        amrex::Gpu::DeviceVector<long> m_seg_partial_offset;
        amrex::Gpu::DeviceVector<int> m_seg_nblocks;
        amrex::Gpu::DeviceVector<int> m_seg_begin;
        amrex::Gpu::PinnedVector<double> m_lambda_host;
    };

} // namespace warpx::circuit

#endif // WARPX_CIRCUIT_FLUXPROBES_H_
