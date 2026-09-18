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
        reciprocity, //!< free-space reciprocity dot with the plasma current
        loop         //!< reciprocity against the analytic loop A of the declared filament (LoopLinkage)
    };

    /** The three components of a level-0 vector field (E/J or B
     * staggering), by direction. Components a probe does not read may be
     * null (documented per probe). */
    using VectorFieldPtrs = std::array<const amrex::MultiFab*, 3>;

    /** Plasma flux linkage of a coil by the DISK rule: the plasma-frame
     * B_z integrated over the coil's circle,
     *     lambda_phys = Int_disk B_z dA,
     * with the identical staircase conventions as the discrete
     * self-inductance (LoopInductance.H) -- probe and port
     * self-inductance must share the rule or the coupled system's
     * passivity is violated. Returns lambda_phys * I_ref * n_turns (the
     * linkage of the coil's reference drive, whose time derivative divided
     * by I_ref * n_turns is the port EMF in volts). Collective (all-reduces
     * the partial sums); level 0.
     *
     * RZ (m = 0): the coil's QUARTER-OFFSET circle (r_off, z_off),
     *     lambda_phys = 2 pi Int_0^{r_off} B_z(r, z_off) r dr,
     * STRICT r_cell < r_off cell test, unclamped linear z interpolation
     * between the two bracketing node planes.
     *
     * 3D: the coil's NOMINAL circle (the filament position the 3D unit
     * field is painted at; the quarter offset is an RZ-mesh policy),
     *     lambda_phys = Sum_{x_c^2 + y_c^2 < r^2} B_z(x_c, y_c, z_coil) dx dy,
     * over the (cell x, cell y, node z) B_z with the STRICT squared
     * cell-center radius test and the same unclamped linear z
     * interpolation between the bracketing node planes; shared node
     * planes are counted once (owner mask).
     *
     * Valid with conducting walls (it measures the model's own flux
     * through the disk); use it whenever the reciprocity probe is not
     * valid. A circle reaching outside the domain is truncated to the
     * domain (the same convention as the discrete inductance table).
     *
     * \param[in] coil     the measured coil
     * \param[in] Bz       plasma-frame B_z on its own staggering, level 0
     */
    amrex::Real DiskFluxLinkage (const Coil& coil, const amrex::MultiFab& Bz);

    /** Plasma flux linkage of a coil by FREE-SPACE RECIPROCITY: the dot
     * product of the coil's unit vector potential with the plasma current,
     *     lambda = Int A^(I_ref n) . J_plasma dV,
     * component-wise on the matched (E/J) staggerings with each shared
     * point counted once (owner-masked). A carries the coil's reference
     * amp-turns, so the result is lambda_phys * I_ref * n_turns directly.
     * Exact linkage in free space; INVALID with a conducting wall between
     * plasma and coil (the image response is not in A^unit) -- pair it
     * with the Green's-function open boundary, or use the disk probe.
     * Collective; level 0.
     *
     * RZ (m = 0): the theta components only (A[1], J[1]; the others may
     * be null): 2 pi Sum w_r r A_theta J_theta dr dz over the nodal (node
     * r, node z) J_theta with trapezoid end-weights in r (w = 1/2 at i = 0
     * and i = nr).
     *
     * 3D: all three components, Sum_c Sum w_c A_c J_c dx dy dz, with the
     * trapezoid end-weight w_c = 1/2 on every non-periodic domain end
     * plane of each direction in which component c is node-centered (the
     * generalization of the RZ w_r rule; CouplingPowerIntegral uses the
     * same weights so the coupling-power double entry closes).
     *
     * \param[in] A  the coil's <field_name>_Aext components (E staggering)
     *               holding the field at I_ref * n_turns, level 0
     * \param[in] J  plasma-frame J components (same staggerings), level 0
     */
    amrex::Real ReciprocityLinkage (const VectorFieldPtrs& A,
                                    const VectorFieldPtrs& J);

    /** Plasma flux linkage of a coil by free-space reciprocity against the
     * ANALYTIC loop vector potential of the coil's declared filament,
     *     A_theta = I_ref n_turns * YeeLoopATheta(rho, z, coil.r, coil.z),
     * i.e. the discrete-convention loop kernel (Legendre-parameter clip
     * [1e-15, 1 - 1e-12], f(m)/m form, exactly 0 on the axis) of a
     * filament at the coil's (r, z) AS GIVEN -- no quarter-cell offset is
     * added here: a caller placing the filament in the discrete
     * convention declares the offset position -- with no image and no
     * softening, EXCLUDING the points within exclusion_radius of the
     * filament (their unresolved near-field response would register as an
     * in-phase spurious linkage with maximal weight). Free-space
     * functional: pair it with the Green's-function open boundary.
     * Returns lambda_phys * I_ref * n_turns. Collective; level 0.
     *
     * RZ (m = 0): 2 pi Sum_{j < nz} w_r r A_theta J_theta dr dz over the
     * nodal J_theta (J[1]) with trapezoid end-weights in r and EXCLUDING
     * the upper axial node plane j = nz. This is, term for term, the
     * integrand of the python coupling reference's loop probe (its
     * analytic A with the scipy elliptic integrals, its exclusion mask,
     * and its node range that drops the last axial plane), so a native
     * driver measuring with it sees the reference's linkage to
     * elliptic-integral roundoff.
     *
     * 3D: A_x = -A_theta y/rho, A_y = A_theta x/rho on the J_x / J_y
     * staggerings (A_z = 0), the ReciprocityLinkage end-weight rule, the
     * exclusion measured from the filament ring, no plane exclusion.
     *
     * \param[in] coil              the measured coil
     * \param[in] J                 plasma-frame J components (E/J staggering)
     * \param[in] exclusion_radius  mask radius [m] around the filament
     *                              (0 = no mask)
     */
    amrex::Real LoopLinkage (const Coil& coil, const VectorFieldPtrs& J,
                             double exclusion_radius);

    /** The coupling power Int J . E dV over the level-0 domain, summed
     * component-wise on matched (E/J) staggerings with owner-masked
     * shared points and the SAME end-weight rule as ReciprocityLinkage
     * (RZ: volume weights 2 pi r dr dz, r at each sample position, with
     * the trapezoid w_r on r-nodal components; 3D: dx dy dz with the
     * nodal-direction end weights), so the coupling-power double entry
     * against -sum_k (ds_k/dt) lambda_k closes exactly. Collective;
     * level 0.
     *
     * \param[in] J  the three plasma-current components (E/J staggering)
     * \param[in] E  the three field components on the same staggerings
     */
    amrex::Real CouplingPowerIntegral (const VectorFieldPtrs& J,
                                       const VectorFieldPtrs& E);

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
     * owner masks, radius/trapezoid/plane weights, volume factors, the
     * disk probe's I_ref * n_turns -- are folded once into per-(coil,
     * source component, box) weight tables on the run mesh, so one
     * measurement is a batch of weight-dot-field products: every device
     * kernel is issued asynchronously, block partials land in fixed
     * slots, one fixed-order combine kernel forms the coil vector, and
     * the host pays exactly ONE stream synchronization, ONE
     * device-to-host copy of n_coils doubles and ONE MPI all-reduce over
     * the vector -- against one sync + one all-reduce PER COIL for the
     * serial loop. The fixed reduction tree makes the result
     * deterministic run-to-run (no atomics); it matches the single-coil
     * functions to summation-order roundoff (1e-15-class relative), not
     * bitwise.
     *
     * The weight tables are geometry-keyed (BoxArray +
     * DistributionMapping + Geometry identity) and rebuilt on a regrid,
     * like CachedOwnerMask. Level 0, collective. RZ (m = 0) reads the
     * theta components; 3D reads all three.
     */
    class LinkageBatch
    {
    public:
        /** Measure every measured coil's plasma flux linkage.
         *
         * \param[in]  coils    the coil set (rows of the batch)
         * \param[in]  probes   per-coil probe kind (ProbeKind::none rows
         *                      report lambda = 0)
         * \param[in]  exclusion_radius  per-coil mask radius [m] of the
         *                      loop probe (ignored by the other kinds)
         * \param[in]  a_ext    per-coil unit vector potential
         *                      <field_name>_Aext components (E staggering)
         *                      at I_ref * n_turns; entries may be null for
         *                      non-reciprocity rows (RZ reads [1] only)
         * \param[in]  bz       plasma-frame B_z; may be null when no row
         *                      uses the disk probe
         * \param[in]  j        plasma-frame J components (E/J
         *                      staggering); may be null when no row uses
         *                      reciprocity or loop (RZ reads [1] only)
         * \param[out] lambda   per-coil lambda_phys * I_ref * n_turns,
         *                      coil order, zeros for unmeasured rows
         */
        void Measure (const CoilSet& coils,
                      const std::vector<ProbeKind>& probes,
                      const std::vector<double>& exclusion_radius,
                      const std::vector<VectorFieldPtrs>& a_ext,
                      const amrex::MultiFab* bz,
                      const VectorFieldPtrs& j,
                      std::vector<amrex::Real>& lambda);

        /** Device-resident global linkage vector, CUDA/HIP only. Its storage
         * remains valid until the next measurement or pack rebuild. Single-rank
         * final combination is asynchronous on the current stream, after the
         * MFIter fences joining the partial reductions. Multiple ranks
         * require GPU-aware MPI and a producer-stream fence; no host staging is
         * allowed. Prepare the weight pack before residual evaluation. */
        const amrex::Gpu::DeviceVector<double>& MeasureDevice (
            const CoilSet& coils, const std::vector<ProbeKind>& probes,
            const std::vector<double>& exclusion_radius,
            const std::vector<VectorFieldPtrs>& a_ext,
            const amrex::MultiFab* bz, const VectorFieldPtrs& j);

        /** Shared measurement implementation, public for CUDA extended lambdas. */
        void MeasureImpl (const CoilSet& coils, const std::vector<ProbeKind>& probes,
            const std::vector<double>& exclusion_radius,
            const std::vector<VectorFieldPtrs>& a_ext,
            const amrex::MultiFab* bz, const VectorFieldPtrs& j,
            std::vector<amrex::Real>& lambda, bool device_only);

        /** One weight-dot-field job: the cells of one box of one source
         * component measured for one coil row. Public for the CUDA
         * extended-lambda rule (device lambdas must live in public member
         * functions); not API. */
        struct Job
        {
            int row = 0;             //!< coil index
            int comp = 0;            //!< source J component (unused for B_z jobs)
            amrex::Box box;          //!< cell enumeration box
            long weight_offset = 0;  //!< into m_weights
            long partial_offset = 0; //!< into m_partials (GPU)
            int nblocks = 0;         //!< reduction blocks (GPU)
        };

        /** (Re)build the geometry-keyed weight pack; called by Measure on
         * a key miss. Public for the CUDA extended-lambda rule; not API. */
        void BuildPack (const CoilSet& coils,
                        const std::vector<ProbeKind>& probes,
                        const std::vector<double>& exclusion_radius,
                        const std::vector<VectorFieldPtrs>& a_ext,
                        const amrex::MultiFab* bz,
                        const VectorFieldPtrs& j);

    private:
        bool m_built = false;
        int m_ncoils = 0;
        //! layout identity of the pack (regrid invalidation)
        std::array<amrex::BoxArray, 3> m_key_ba_j;
        std::array<amrex::DistributionMapping, 3> m_key_dm_j;
        amrex::BoxArray m_key_ba_bz;
        amrex::DistributionMapping m_key_dm_bz;
        amrex::Box m_key_domain;

        //! per-source-component jobs, keyed by the source's MFIter box index
        std::array<std::map<int, std::vector<Job>>, 3> m_jobs_j;
        std::map<int, std::vector<Job>> m_jobs_bz;

        amrex::Gpu::DeviceVector<double> m_weights;
        amrex::Gpu::DeviceVector<double> m_partials;
        amrex::Gpu::DeviceVector<double> m_lambda_device;
        amrex::Gpu::DeviceVector<double> m_exchange_device;
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
