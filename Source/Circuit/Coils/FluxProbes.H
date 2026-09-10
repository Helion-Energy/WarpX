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

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <array>
#include <map>
#include <string>
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

    /** Wall-polyline region classification of the nodal (node r, node z)
     * J_theta mesh for the J-based linkage probes (circuit.probe_region,
     * circuit.probe_region_report).
     *
     * THE RULE (binary, so the classification is the whole contract; the
     * python coupling reference applies the identical rule with numpy):
     * node (r_i, z_j), r_i = r_lo + i dr, z_j = z_lo + j dz, is INSIDE the
     * wall iff
     *     r_i < r_wall(z_j),
     * where r_wall(z) is the piecewise-LINEAR interpolation in z of the
     * wall polyline ("z, r" rows, z non-decreasing), CLAMPED at the ends
     * (constant extrapolation), evaluated with numpy.interp semantics:
     * for z <= z_0 -> r_0, for z >= z_last -> r_last, otherwise on the
     * segment [z_k, z_{k+1}) with the LARGEST k such that z_k <= z (so a
     * duplicate-z vertex pair resolves to the segment that follows it)
     *     r_wall(z) = (r_{k+1} - r_k) / (z_{k+1} - z_k) * (z - z_k) + r_k,
     * in exactly that floating-point order (bit-identical to
     * numpy.interp(z, z_poly, r_poly)). Python:
     *     r_node < np.interp(z_node, z_poly, r_poly)
     * with r_node = r_lo + i * dr and z_node = z_lo + j * dz.
     *
     * The report's radial split refines the inside into the INTERIOR
     * (r_i < r_wall(z_j) - 2 dr) and the two-cell wall BAND
     * (r_wall(z_j) - 2 dr <= r_i < r_wall(z_j)); the EXTERIOR is
     * r_i >= r_wall(z_j). The three regions partition the nodes exactly
     * (both thresholds are tabulated once per z node and every kernel
     * compares against the same tabulated values).
     *
     * This is a NODAL classification of the J_theta mesh and is NOT the
     * theta-implicit MHD wall mask's cell-centered stair classification
     * (that one marks a CELL conductor when its center radius is on or
     * outside the polyline, and reduces zero-length vertical segments to
     * the smaller radius); the two agree on which side of the wall a
     * point lies except within half a cell of the contour, by
     * construction of the different sample points.
     *
     * The disk probe (a B_z surface integral through the coil circle) is
     * unaffected by the region mask: it is not a current integral.
     * RZ (m = 0), level 0. */
    class ProbeRegionTable
    {
    public:
        /** Tabulate r_wall and r_wall - 2 dr at every z node of the
         * level-0 domain (nodal j in [j_lo, j_hi]). z_poly must be
         * non-decreasing with at least two points. */
        void Define (const amrex::Geometry& geom,
                     const std::vector<double>& z_poly,
                     const std::vector<double>& r_poly);

        [[nodiscard]] bool IsDefined () const { return m_defined; }

        /** THE RULE's r_wall(z) (host; numpy.interp semantics, see the
         * class comment). */
        [[nodiscard]] static double WallRadiusAt (
            const std::vector<double>& z_poly,
            const std::vector<double>& r_poly, double z);

        /** Device tables indexed by (j - JLo()). */
        [[nodiscard]] const double* WallRadiusTable () const { return m_r_wall.data(); }
        [[nodiscard]] const double* BandRadiusTable () const { return m_r_band.data(); }
        [[nodiscard]] int JLo () const { return m_j_lo; }
        [[nodiscard]] int NumNodes () const { return static_cast<int>(m_r_wall_host.size()); }
        /** Host copy of r_wall(z_j), j - JLo() indexed. */
        [[nodiscard]] const std::vector<double>& WallRadiusHost () const { return m_r_wall_host; }
        [[nodiscard]] double BandWidth () const { return m_band_width; }

    private:
        bool m_defined = false;
        int m_j_lo = 0;
        double m_band_width = 0.0;   //!< 2 dr
        std::vector<double> m_r_wall_host;
        amrex::Gpu::DeviceVector<double> m_r_wall;
        amrex::Gpu::DeviceVector<double> m_r_band;
    };

    /** Radial region of a J_theta node relative to the wall polyline
     * (see ProbeRegionTable). */
    enum class RadialRegion : int
    {
        any = 0,      //!< no radial restriction (the whole domain)
        inside_wall,  //!< r < r_wall(z): the circuit.probe_region = wall_interior mask
        interior,     //!< r < r_wall(z) - 2 dr
        band,         //!< r_wall(z) - 2 dr <= r < r_wall(z)
        exterior      //!< r >= r_wall(z)
    };

    /** Physical-share class of a J_theta node by the nodal linkage weight
     * w = eta_phys / eta_field (the theta-implicit MHD solver's
     * "circuit_linkage_weight" register; see CircuitCoupler::Params::
     * probe_weight). */
    enum class WeightClass : int
    {
        any = 0,  //!< no weight restriction
        plasma,   //!< w > 0.9: the physical eta carries the current
        mixed,    //!< 0.1 <= w <= 0.9
        boost     //!< w < 0.1: the vacuum-boosted (mock displacement) current
    };

    /** Host-side node filter of the J-based probes: an optional radial
     * region (needs a defined ProbeRegionTable) and an optional weight
     * class (needs the nodal weight MultiFab on the J_theta layout).
     * Default-constructed = no filter = the unrestricted integral,
     * bit-identical to the probes without the filter argument. */
    struct ProbeNodeFilter
    {
        const ProbeRegionTable* region = nullptr;
        RadialRegion radial = RadialRegion::any;
        const amrex::MultiFab* weight = nullptr;
        WeightClass wclass = WeightClass::any;

        [[nodiscard]] bool Active () const
        {
            return radial != RadialRegion::any || wclass != WeightClass::any;
        }
    };

    /** Device-side (POD, captured by value) twin of ProbeNodeFilter for
     * one box; built by MakeNodeSelector. */
    struct NodeSelector
    {
        const double* r_wall = nullptr;
        const double* r_band = nullptr;
        int j_lo = 0;
        int radial = 0;
        amrex::Array4<const amrex::Real> w;
        int wclass = 0;

        /** Whether node (i, j) at radius r passes the filter. Pure
         * comparisons: the probe arithmetic is untouched, so an inactive
         * selector (radial = wclass = 0) is bit-identical to no filter. */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        bool Contains (const int i, const int j, const double r) const noexcept
        {
            if (radial != 0) {
                const double rw = r_wall[j - j_lo];
                const double rb = r_band[j - j_lo];
                const bool in_wall = (r < rw);
                const bool in_interior = (r < rb);
                if (radial == 1 && !in_wall) { return false; }
                if (radial == 2 && !in_interior) { return false; }
                if (radial == 3 && (in_interior || !in_wall)) { return false; }
                if (radial == 4 && in_wall) { return false; }
            }
            if (wclass != 0) {
                const double wv = static_cast<double>(w(i, j, 0));
                const bool plasma = (wv > 0.9);
                const bool boost = (wv < 0.1);
                if (wclass == 1 && !plasma) { return false; }
                if (wclass == 2 && (plasma || boost)) { return false; }
                if (wclass == 3 && !boost) { return false; }
            }
            return true;
        }
    };

    /** The device selector of a filter on one box (mfi must iterate the
     * J_theta layout the weight MultiFab lives on). Aborts when a radial
     * region is requested without a defined table, or a weight class
     * without a weight MultiFab. */
    NodeSelector MakeNodeSelector (const ProbeNodeFilter& filter,
                                   const amrex::MFIter& mfi);

    /** Root-mean-square of the nodal J_theta over the owner-masked nodes
     * of each radial region -- interior, band, exterior (in that order;
     * see RadialRegion) -- of a defined ProbeRegionTable; 0 for an empty
     * region. Collective; RZ (m = 0), level 0. */
    std::array<double, 3> NodalRmsByRegion (const amrex::MultiFab& J_theta,
                                            const ProbeRegionTable& region);

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
     * \param[in] filter   optional node filter (default: none = the whole
     *                     domain, bit-identical to the unfiltered probe);
     *                     see ProbeNodeFilter
     */
    amrex::Real ReciprocityLinkage (const amrex::MultiFab& A_theta,
                                    const amrex::MultiFab& J_theta,
                                    const ProbeNodeFilter& filter = ProbeNodeFilter{});

    /** Plasma flux linkage of a coil by free-space reciprocity against the
     * ANALYTIC loop vector potential of the coil's declared filament,
     *     lambda = 2 pi Sum_{j < nz} w_r r A_theta(r, z) J_theta dr dz,
     *     A_theta = I_ref n_turns * YeeLoopATheta(r, z, coil.r, coil.z),
     * i.e. the discrete-convention loop kernel (Legendre-parameter clip
     * [1e-15, 1 - 1e-12], f(m)/m form, exactly 0 on the axis) of a
     * filament at the coil's (r, z) AS GIVEN -- no quarter-cell offset is
     * added here: a caller placing the filament in the discrete
     * convention declares the offset position -- with no image and no
     * softening, over the nodal (node r, node z) J_theta with trapezoid
     * end-weights in r, EXCLUDING the nodes within exclusion_radius of
     * the filament (their unresolved near-field response would register
     * as an in-phase spurious linkage with maximal weight) and EXCLUDING
     * the upper axial node plane j = nz. This is, term for term, the
     * integrand of the python coupling reference's loop probe (its
     * analytic A with the scipy elliptic integrals, its exclusion mask,
     * and its node range that drops the last axial plane), so a native
     * driver measuring with it sees the reference's linkage to
     * elliptic-integral roundoff. Free-space functional: pair it with the
     * Green's-function open boundary. Returns lambda_phys * I_ref *
     * n_turns. Collective; RZ (m = 0), level 0.
     *
     * \param[in] coil              the measured coil
     * \param[in] J_theta           plasma-frame J_theta (node r, node z)
     * \param[in] exclusion_radius  mask radius [m] around the filament
     *                              (0 = no mask)
     * \param[in] filter            optional node filter (default: none,
     *                              bit-identical to the unfiltered probe)
     */
    amrex::Real LoopLinkage (const Coil& coil, const amrex::MultiFab& J_theta,
                             double exclusion_radius,
                             const ProbeNodeFilter& filter = ProbeNodeFilter{});

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
         * \param[in]  exclusion_radius  per-coil mask radius [m] of the
         *                      loop probe (ignored by the other kinds)
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
         * \param[in]  filter   node filter of the J-based rows (the
         *                      circuit.probe_region = wall_interior mask,
         *                      folded into the weight tables; default
         *                      none, bit-identical). Static like the
         *                      other weight factors: a filter change
         *                      requires a new LinkageBatch.
         */
        void Measure (const CoilSet& coils,
                      const std::vector<ProbeKind>& probes,
                      const std::vector<double>& exclusion_radius,
                      const std::vector<const amrex::MultiFab*>& a_theta,
                      const amrex::MultiFab* bz,
                      const amrex::MultiFab* j_theta,
                      std::vector<amrex::Real>& lambda,
                      const ProbeNodeFilter& filter = ProbeNodeFilter{});

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
                        const std::vector<double>& exclusion_radius,
                        const std::vector<const amrex::MultiFab*>& a_theta,
                        const amrex::MultiFab* bz,
                        const amrex::MultiFab* j_theta,
                        const ProbeNodeFilter& filter);

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
