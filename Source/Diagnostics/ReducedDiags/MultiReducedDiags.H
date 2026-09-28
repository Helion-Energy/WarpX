/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_MULTIREDUCEDDIAGS_H_

#include "MultiReducedDiags_fwd.H"

#include "ReducedDiags.H"

#include <memory>
#include <string>
#include <vector>

/**
 * This class holds multiple instances of ReducedDiagnostics, and contains
 * general functions to initialize, compute, and write these diagnostics
 * to file.
 */
class MultiReducedDiags
{
public:

    /// Bool: whether or not reduced diagnostics are activated
    int m_plot_rd = 0;

    /// names of reduced diagnostics
    std::vector<std::string> m_rd_names;

    /// m_multi_rd stores a pointer to each reduced diagnostics
    std::vector<std::unique_ptr<ReducedDiags>> m_multi_rd;

    /// constructor
    MultiReducedDiags ();

    /** Owned state of the supported midpoint diagnostics. The handle is local
     *  to one MultiReducedDiags instance and must not outlive that instance.
     *  Destruction only discards the saved data; it never restores implicitly.
     */
    class MidStepSnapshot
    {
    public:
        MidStepSnapshot ();
        ~MidStepSnapshot ();
        MidStepSnapshot (MidStepSnapshot&&) noexcept;
        MidStepSnapshot& operator= (MidStepSnapshot&&) = delete;
        MidStepSnapshot (MidStepSnapshot const&) = delete;
        MidStepSnapshot& operator= (MidStepSnapshot const&) = delete;

    private:
        friend class MultiReducedDiags;
        struct State;
        std::unique_ptr<State> m_state;
    };

    /** Collective, observation-only capture. Rejects an active snapshot or an
     *  unsupported diagnostic without replacing an existing saved state.
     *  The schema covers all registered data vectors and the FieldPoyntingFlux
     *  midpoint flag. Other supported native types have no midpoint override.
     */
    bool CaptureMidStepState (MidStepSnapshot& snapshot) const;

    /** Collective, pure owner/schema/order/size preflight. */
    bool CanRestoreMidStepState (MidStepSnapshot const& snapshot) const;

    /** Collective restore after the complete all-rank preflight succeeds.
     *  Failure changes neither diagnostic data nor the snapshot. Success
     *  restores exact stored data and consumes the snapshot.
     */
    bool RestoreMidStepState (MidStepSnapshot& snapshot);

    /** Discard this instance's snapshot after the caller's final collective
     *  acceptance preflight. No sampling, I/O, or collective is performed.
     */
    void CommitMidStepState (MidStepSnapshot& snapshot) noexcept;


    /** Loop over all ReducedDiags and call their InitData
     */
    void InitData ();

    /** Redistribute parallel data structures during load balance
     */
    void LoadBalance ();

    /** Loop over all ReducedDiags and call their ComputeDiags
     *  @param[in] step current iteration time */
    void ComputeDiags (int step);

    /** Loop over all ReducedDiags and call their ComputeDiagsMidStep
     *  @param[in] step current iteration time */
    void ComputeDiagsMidStep (int step);

    /** Loop over all ReducedDiags and call their WriteToFile
     *  @param[in] step current iteration time */
    void WriteToFile (int step);

    /** Check if any diagnostics will be done */
    bool DoDiags(int step);

    /** \brief Loop over all ReducedDiags and call their WriteCheckpointData
     *  @param[in] dir checkpoint directory */
    void WriteCheckpointData (std::string const & dir);

    /** \brief Loop over all ReducedDiags and call their ReadCheckpointData
     *  @param[in] dir checkpoint directory */
    void ReadCheckpointData (std::string const & dir);
};

#endif
