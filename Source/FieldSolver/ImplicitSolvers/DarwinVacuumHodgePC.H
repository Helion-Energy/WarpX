/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_DARWIN_VACUUM_HODGE_PC_H_
#define WARPX_DARWIN_VACUUM_HODGE_PC_H_
#include <AMReX_iMultiFab.H>
#include <ablastr/fields/MultiFabRegister.H>
#include <cstddef>
#include <memory>
#include <string>
class WarpX;
namespace warpx::darwin {
/** Private distributed vacuum primitives and augmented PC. P/V are raw
 * charged/empty support, never a density threshold. No physical state is
 * modified. W retains each floating charged-component constant. The first
 * capability requires a real scalar PEC anchor and opens no runtime route. */
class DarwinVacuumHodgePC {
public:
    using View = ablastr::fields::VectorField;
    using Mask = amrex::Array<amrex::iMultiFab const*, 3>;
    struct Options {
        amrex::Real b = 0.; // (theta*dt)^2/(epsilon0*mu0), m^2.
        amrex::Real length = 0.; // fixed domain scale, m.
        amrex::Real relative_tolerance = 1.e-12;
        amrex::Real absolute_tolerance = 0.;
        int max_iterations = 2000, restart_length = 100;
        bool use_auxiliary_preconditioner = false; // R04 diagonal control remains selectable.
        bool use_yee_green_interface_inverse = false; // Optional exact masked RZ/PMC inverse.
        int max_interface_dofs = 4096; // Explicit dense-cache memory/setup bound.
        bool use_green_auxiliary_inverse = false; // Optional exact nodal inverse; RZ/no-EB/FFT only.
    };
    struct Statistics {
        int label_iterations = 0, floating_components = 0;
        amrex::Long vacuum_edges = 0, potential_dofs = 0;
        std::size_t local_owned_bytes = 0; // Real work and Krylov storage, excludes Core/masks/AMReX buffers.
        long curl_actions = 0, gradient_actions = 0, adjoint_actions = 0;
        long augmented_actions = 0, linear_iterations = 0;
        long auxiliary_cycles = 0, hodge_completion_actions = 0;
        long green_auxiliary_solves = 0;
        int interface_dofs = 0;
        long interface_setup_columns = 0, interface_applications = 0;
        long yee_green_component_solves = 0;
    };
    struct Result {
        bool converged = false, finite = false;
        int status = -1, iterations = 0;
        amrex::Real initial = 0., residual = 0., target = 0.;
    };
    struct Impl; // Public opaque scope permits CUDA extended lambdas.
    DarwinVacuumHodgePC(WarpX&, View const& electric, View const& magnetic, Options);
    ~DarwinVacuumHodgePC();
    DarwinVacuumHodgePC(DarwinVacuumHodgePC const&) = delete;
    DarwinVacuumHodgePC& operator=(DarwinVacuumHodgePC const&) = delete;
    /** Complete preflight before replacing private state. P/V partition free
     * rows and vanish on physical fixed traces. At most one floating component. */
    bool Prepare(Mask const& charged, Mask const& vacuum, std::string& error);
    bool Ready() const noexcept;
    void Invalidate() noexcept;
    Statistics const& Stats() const;
    amrex::iMultiFab const& Labels() const;
    amrex::MultiFab const& NodeMetric() const;
    /** Distributed nodal W values are identical on every P component. Scalar
     * weighted reductions retain its physical floating constant. */
    void CanonicalPotential(amrex::MultiFab& output, amrex::MultiFab const& input);
    void Gradient(View const& output, amrex::MultiFab const& potential);
    /** G_W^dagger with native weights and authoritative valid input only. */
    void Adjoint(amrex::MultiFab& output, View const& field);
    /** Physical compensated I_V curl curl, field/m^2. Charged inputs retain
     * K_VP. Caller input/output allocations must be distinct. */
    void Curl(View const& output, View const& field);
    amrex::Real EdgeDot(View const&, View const&) const;
    amrex::Real PotentialDot(amrex::MultiFab const&, amrex::MultiFab const&) const;
    /** Exact metric-adjoint nodal-vector transfer used only in the PC. */
    void ProlongAuxiliary(View const& output_edge, View const& input_nodes);
    void RestrictAuxiliary(View const& output_nodes, View const& input_edge);
    amrex::MultiFab const& AuxiliaryMetric(int component) const;
    /** Apply the opt-in inverse of length^2 (K + G_W G_W^dagger).
     * Rebuilt with Prepare; physical field/current state is never modified. */
    void ApplyHodgeInverse(View const& output, View const& input);
    /** Full PC-only saddle in u=b*e/length^2, psi=phi/length:
     * [length^2 K, length G; length G^dagger,0]. No dropped RHS or penalty. */
    void ApplyAugmented(View const& output_edge, amrex::MultiFab& output_node,
                        View const& input_edge, amrex::MultiFab const& input_node);
    /** b K e+G phi=-d, G^dagger e=G^dagger h; finite true defect/status explicit. */
    Result Solve(View const& electric, amrex::MultiFab& potential,
                 View const& d, View const& h);
private:
    std::unique_ptr<Impl> m_impl;
};
} // namespace warpx::darwin
#endif
