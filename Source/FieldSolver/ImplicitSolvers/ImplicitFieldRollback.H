/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_IMPLICIT_FIELD_ROLLBACK_H_
#define WARPX_IMPLICIT_FIELD_ROLLBACK_H_
#include "ablastr/fields/MultiFabRegister.H"
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace warpx::implicit {
// Device-resident snapshot at the physical step boundary. Includes all existing
// registered field components and guards, including aliases. Newly allocated
// derived scratch may remain allocated after rejection. The caller may also
// explicitly omit rebuildable scratch that can resize during a trial; its owner must
// invalidate its caches separately. This does not snapshot particles, circuit
// engines, RNG, model flags or unregistered solver vectors.
class FieldRollback {
  public:
    void Capture (ablastr::fields::MultiFabRegister& fields,
                  std::vector<amrex::MultiFab const*> const& rebuildable_scratch = {}) {
        AMREX_ALWAYS_ASSERT(!m_open);
        auto names=fields.list();
        std::erase_if(names, [&] (std::string const& name) {
            return std::find(rebuildable_scratch.begin(), rebuildable_scratch.end(),
                             fields.internal_get(name)) != rebuildable_scratch.end();
        });
        bool reuse=names.size()==m_entries.size();
        if (reuse) {
            for (std::size_t i=0;i<names.size();++i) {
                reuse=reuse && names[i]==m_entries[i].name &&
                    Matches(fields,m_entries[i]);
            }
        }
        if (!reuse) {
            m_entries.clear();
            for (auto const& name:names) {
                auto* field=fields.internal_get(name);
                Entry entry;
                entry.name=name; entry.original=field;
                entry.saved=std::make_unique<amrex::MultiFab>(field->boxArray(),
                    field->DistributionMap(),field->nComp(),field->nGrowVect());
                for (amrex::MFIter mfi(*field);mfi.isValid();++mfi) {
                    entry.storage.emplace_back(mfi.index(),(*field)[mfi].dataPtr());
                }
                m_entries.push_back(std::move(entry));
            }
        }
        for (auto& entry:m_entries) {
            auto const& field=*entry.original;
            amrex::MultiFab::Copy(*entry.saved,field,0,0,field.nComp(),field.nGrowVect());
        }
        amrex::Gpu::streamSynchronize();
        m_open=true;
    }
    [[nodiscard]] bool CanRestore (ablastr::fields::MultiFabRegister& fields) const {
        bool valid=m_open;
        for (auto const& entry:m_entries) { valid=Matches(fields,entry) && valid; }
        amrex::ParallelDescriptor::ReduceBoolAnd(valid);
        return valid;
    }
    // Validate the entire transaction across all ranks before writing any field.
    [[nodiscard]] bool Restore (ablastr::fields::MultiFabRegister& fields) const {
        if (!CanRestore(fields)) { return false; }
        for (auto const& entry:m_entries) {
            auto& field=*entry.original;
            amrex::MultiFab::Copy(field,*entry.saved,0,0,field.nComp(),field.nGrowVect());
        }
        amrex::Gpu::streamSynchronize();
        return true;
    }
    // A const, owner-lifetime view used only while the original transaction is
    // open. It exposes no restore/commit permission or mutable snapshot bytes.
    [[nodiscard]] amrex::MultiFab const* SavedField(amrex::MultiFab const* original) const {
        if (!m_open) { return nullptr; }
        for (auto const& entry:m_entries) {
            if (entry.original==original) { return entry.saved.get(); }
        }
        return nullptr;
    }
    void Discard () { m_open=false; }
    [[nodiscard]] bool Open () const { return m_open; }
    [[nodiscard]] std::size_t Size () const { return m_entries.size(); }
  private:
    struct Entry {
        std::string name;
        amrex::MultiFab* original=nullptr;
        std::unique_ptr<amrex::MultiFab> saved;
        std::vector<std::pair<int,const amrex::Real*>> storage;
    };
    static bool Matches (ablastr::fields::MultiFabRegister& fields,const Entry& e) {
        if (!fields.internal_has(e.name)) { return false; }
        auto const* f=fields.internal_get(e.name);
        if (f!=e.original || f->boxArray()!=e.saved->boxArray() ||
            f->DistributionMap()!=e.saved->DistributionMap() ||
            f->nComp()!=e.saved->nComp() || f->nGrowVect()!=e.saved->nGrowVect()) {
            return false;
        }
        std::size_t n=0;
        for (amrex::MFIter mfi(*f);mfi.isValid();++mfi,++n) {
            if (n>=e.storage.size() || mfi.index()!=e.storage[n].first ||
                (*f)[mfi].dataPtr()!=e.storage[n].second) { return false; }
        }
        return n==e.storage.size();
    }
    std::vector<Entry> m_entries;
    bool m_open=false;
};
} // namespace warpx::implicit
#endif
