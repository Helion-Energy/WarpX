/* Copyright 2019-2020 Maxence Thevenet, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MultiReducedDiags.H"

#include "BeamRelevant.H"
#include "ChargeOnEB.H"
#include "ColliderRelevant.H"
#include "DifferentialLuminosity.H"
#include "DifferentialLuminosity2D.H"
#include "FieldEnergy.H"
#include "FieldMaximum.H"
#include "FieldMomentum.H"
#include "FieldPoyntingFlux.H"
#include "FieldProbe.H"
#include "FieldReduction.H"
#include "HybridDissipation.H"
#include "LoadBalanceCosts.H"
#include "LoadBalanceEfficiency.H"
#include "ParticleBoundaryFlux.H"
#include "ParticleEnergy.H"
#include "ParticleExtrema.H"
#include "ParticleHistogram.H"
#include "ParticleHistogram2D.H"
#include "ParticleMomentum.H"
#include "ParticleNumber.H"
#include "RhoMaximum.H"
#include "ScrapedParticleEnergy.H"
#include "Timestep.H"
#include "Utils/TextMsg.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cstring>
#include <typeinfo>
#include <utility>
#include <functional>
#include <iterator>
#include <map>

using namespace amrex;

// constructor
MultiReducedDiags::MultiReducedDiags ()
{
    // read reduced diags names
    const ParmParse pp_warpx("warpx");
    m_plot_rd = pp_warpx.queryarr("reduced_diags_names", m_rd_names);

    // if names are not given, reduced diags will not be done
    if ( m_plot_rd == 0 ) { return; }

    using CS = const std::string& ;
    const auto reduced_diags_dictionary =
        std::map<std::string, std::function<std::unique_ptr<ReducedDiags>(CS)>>{
            {"BeamRelevant",          [](CS s){return std::make_unique<BeamRelevant>(s);}},
            {"ChargeOnEB",            [](CS s){return std::make_unique<ChargeOnEB>(s);}},
            {"ColliderRelevant",      [](CS s){return std::make_unique<ColliderRelevant>(s);}},
            {"DifferentialLuminosity",[](CS s){return std::make_unique<DifferentialLuminosity>(s);}},
            {"DifferentialLuminosity2D",[](CS s){return std::make_unique<DifferentialLuminosity2D>(s);}},
            {"HybridDissipation",     [](CS s){return std::make_unique<HybridDissipation>(s);}},
            {"ParticleBoundaryFlux",  [](CS s){return std::make_unique<ParticleBoundaryFlux>(s);}},
            {"ParticleEnergy",        [](CS s){return std::make_unique<ParticleEnergy>(s);}},
            {"ParticleExtrema",       [](CS s){return std::make_unique<ParticleExtrema>(s);}},
            {"ParticleHistogram",     [](CS s){return std::make_unique<ParticleHistogram>(s);}},
            {"ParticleHistogram2D",   [](CS s){return std::make_unique<ParticleHistogram2D>(s);}},
            {"ParticleMomentum",      [](CS s){return std::make_unique<ParticleMomentum>(s);}},
            {"ParticleNumber",        [](CS s){return std::make_unique<ParticleNumber>(s);}},
            {"FieldEnergy",           [](CS s){return std::make_unique<FieldEnergy>(s);}},
            {"FieldMaximum",          [](CS s){return std::make_unique<FieldMaximum>(s);}},
            {"FieldMomentum",         [](CS s){return std::make_unique<FieldMomentum>(s);}},
            {"FieldPoyntingFlux",     [](CS s){return std::make_unique<FieldPoyntingFlux>(s);}},
            {"FieldProbe",            [](CS s){return std::make_unique<FieldProbe>(s);}},
            {"FieldReduction",        [](CS s){return std::make_unique<FieldReduction>(s);}},
            {"LoadBalanceCosts",      [](CS s){return std::make_unique<LoadBalanceCosts>(s);}},
            {"LoadBalanceEfficiency", [](CS s){return std::make_unique<LoadBalanceEfficiency>(s);}},
            {"RhoMaximum",            [](CS s){return std::make_unique<RhoMaximum>(s);}},
            {"ScrapedParticleEnergy", [](CS s){return std::make_unique<ScrapedParticleEnergy>(s);}},
            {"Timestep",              [](CS s){return std::make_unique<Timestep>(s);}}
    };
    // loop over all reduced diags and fill m_multi_rd with requested reduced diags
    std::transform(m_rd_names.begin(), m_rd_names.end(), std::back_inserter(m_multi_rd),
        [&](const auto& rd_name){
            const ParmParse pp_rd_name(rd_name);

            // read reduced diags type
            std::string rd_type;
            pp_rd_name.get("type", rd_type);

            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                reduced_diags_dictionary.contains(rd_type),
                rd_type + " is not a valid type for reduced diagnostic " + rd_name
            );

            return reduced_diags_dictionary.at(rd_type)(rd_name);
        });
    // end loop over all reduced diags
}
// end constructor

void MultiReducedDiags::InitData ()
{
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> InitData();
    }
}

void MultiReducedDiags::LoadBalance () {
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> LoadBalance();
    }
}

// call functions to compute diags
void MultiReducedDiags::ComputeDiags (int step)
{
    ABLASTR_PROFILE("MultiReducedDiags::ComputeDiags()");

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> ComputeDiags(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::ComputeDiags

// call functions to compute diags at the mid step time level
void MultiReducedDiags::ComputeDiagsMidStep (int step)
{
    ABLASTR_PROFILE("MultiReducedDiags::ComputeDiagsMidStep()");

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd] -> ComputeDiagsMidStep(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::ComputeDiagsMidStep

// function to write data
void MultiReducedDiags::WriteToFile (int step)
{
    // Only the I/O rank does
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        // Judge if the diags should be done
        if (!m_multi_rd[i_rd]->m_intervals.contains(step+1)) { continue; }

        // call the write to file function
        m_multi_rd[i_rd]->WriteToFile(step);
    }
    // end loop over all reduced diags
}
// end void MultiReducedDiags::WriteToFile

// Check if any diagnostics will be done
bool MultiReducedDiags::DoDiags(int step)
{
    bool result = false;
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        result = result || m_multi_rd[i_rd] -> DoDiags(step);
    }
    return result;
}
// end bool MultiReducedDiags::DoDiags

void MultiReducedDiags::WriteCheckpointData (std::string const & dir)
{
    // Only the I/O rank does
    if ( !ParallelDescriptor::IOProcessor() ) { return; }

    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd]->WriteCheckpointData(dir);
    }
    // end loop over all reduced diags
}

void MultiReducedDiags::ReadCheckpointData (std::string const & dir)
{
    // loop over all reduced diags
    for (int i_rd = 0; i_rd < static_cast<int>(m_rd_names.size()); ++i_rd)
    {
        m_multi_rd[i_rd]->ReadCheckpointData(dir);
    }
    // end loop over all reduced diags
}

namespace
{
    // Explicit schema: adding a midpoint override requires extending the saved
    // state here. Unknown types fail closed instead of silently losing state.
    int MidStepStateSchema (ReducedDiags const& diagnostic)
    {
        auto const& type = typeid(diagnostic);
        if (type == typeid(FieldPoyntingFlux)) { return 2; }
        std::type_info const* const stateless_midpoint_types[] = {
            &typeid(BeamRelevant), &typeid(ChargeOnEB), &typeid(ColliderRelevant),
            &typeid(DifferentialLuminosity), &typeid(DifferentialLuminosity2D),
            &typeid(HybridDissipation), &typeid(ParticleBoundaryFlux),
            &typeid(ParticleEnergy), &typeid(ParticleExtrema),
            &typeid(ParticleHistogram), &typeid(ParticleHistogram2D),
            &typeid(ParticleMomentum), &typeid(ParticleNumber),
            &typeid(FieldEnergy), &typeid(FieldMaximum), &typeid(FieldMomentum),
            &typeid(FieldProbe), &typeid(FieldReduction), &typeid(LoadBalanceCosts),
            &typeid(LoadBalanceEfficiency), &typeid(RhoMaximum),
            &typeid(ScrapedParticleEnergy), &typeid(Timestep)
        };
        for (auto const* known : stateless_midpoint_types) {
            if (type == *known) { return 1; }
        }
        return 0;
    }

    bool AllMidStepRanks (bool const local)
    {
        int valid = local ? 1 : 0;
        amrex::ParallelDescriptor::ReduceIntMin(valid);
        return valid != 0;
    }
}

struct MultiReducedDiags::MidStepSnapshot::State
{
    struct Entry
    {
        ReducedDiags const* object = nullptr;
        int schema = 0;
        std::type_info const* type = nullptr;
        std::string name;
        std::vector<amrex::Real> data;
        bool poynting_midpoint = false;
    };
    MultiReducedDiags const* owner = nullptr;
    int plot = 0;
    std::vector<std::string> names;
    std::vector<Entry> entries;
};

MultiReducedDiags::MidStepSnapshot::MidStepSnapshot () = default;
MultiReducedDiags::MidStepSnapshot::~MidStepSnapshot () = default;
MultiReducedDiags::MidStepSnapshot::MidStepSnapshot (MidStepSnapshot&&) noexcept = default;

bool MultiReducedDiags::CaptureMidStepState (MidStepSnapshot& snapshot) const
{
    bool valid = !snapshot.m_state && m_rd_names.size() == m_multi_rd.size();
    for (auto const& diagnostic : m_multi_rd) {
        valid = valid && diagnostic && MidStepStateSchema(*diagnostic) != 0;
    }
    if (!AllMidStepRanks(valid)) { return false; }

    auto state = std::make_unique<MidStepSnapshot::State>();
    state->owner = this;
    state->plot = m_plot_rd;
    state->names = m_rd_names;
    state->entries.reserve(m_multi_rd.size());
    for (auto const& diagnostic : m_multi_rd) {
        auto& entry = state->entries.emplace_back();
        entry.object = diagnostic.get();
        entry.schema = MidStepStateSchema(*diagnostic);
        entry.type = &typeid(*diagnostic);
        entry.name = diagnostic->m_rd_name;
        entry.data.resize(diagnostic->m_data.size());
        if (!entry.data.empty()) {
            std::memcpy(entry.data.data(), diagnostic->m_data.data(),
                        entry.data.size()*sizeof(amrex::Real));
        }
        if (entry.schema == 2) {
            entry.poynting_midpoint =
                static_cast<FieldPoyntingFlux const&>(*diagnostic).use_mid_step_value;
        }
    }
    snapshot.m_state = std::move(state);
    return true;
}

bool MultiReducedDiags::CanRestoreMidStepState (MidStepSnapshot const& snapshot) const
{
    auto const* state = snapshot.m_state.get();
    bool valid = state && state->owner == this && state->plot == m_plot_rd
        && state->names == m_rd_names && state->entries.size() == m_multi_rd.size();
    if (valid) {
        for (std::size_t i = 0; i < m_multi_rd.size(); ++i) {
            auto const& diagnostic = m_multi_rd[i];
            auto const& entry = state->entries[i];
            if (!diagnostic || entry.object != diagnostic.get()
                || entry.schema != MidStepStateSchema(*diagnostic)
                || !entry.type || *entry.type != typeid(*diagnostic)
                || entry.name != diagnostic->m_rd_name
                || entry.data.size() != diagnostic->m_data.size()) {
                valid = false;
                break;
            }
        }
    }
    return AllMidStepRanks(valid);
}

bool MultiReducedDiags::RestoreMidStepState (MidStepSnapshot& snapshot)
{
    // No diagnostic or saved-state write precedes this collective preflight.
    if (!CanRestoreMidStepState(snapshot)) { return false; }
    auto const& entries = snapshot.m_state->entries;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        auto const& entry = entries[i];
        auto& diagnostic = *m_multi_rd[i];
        if (!entry.data.empty()) {
            std::memcpy(diagnostic.m_data.data(), entry.data.data(),
                        entry.data.size()*sizeof(amrex::Real));
        }
        if (entry.schema == 2) {
            static_cast<FieldPoyntingFlux&>(diagnostic).use_mid_step_value =
                entry.poynting_midpoint;
        }
    }
    snapshot.m_state.reset();
    return true;
}

void MultiReducedDiags::CommitMidStepState (MidStepSnapshot& snapshot) noexcept
{
    if (snapshot.m_state && snapshot.m_state->owner == this) {
        snapshot.m_state.reset();
    }
}
