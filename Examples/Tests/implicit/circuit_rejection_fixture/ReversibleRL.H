/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef REJECTION_FIXTURE_RL_H_
#define REJECTION_FIXTURE_RL_H_
#include "Circuit/ExternalCircuit.H"
#include "Circuit/ExternalCircuitRejection.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
// Native C++ backward-Euler RL oracle, matching the maintained RL fixture law.
// A synthetic BeginStep phase/clock mutation exercises the same *kind* of
// pre-acceptance state ownership as C3 ArmLocks; it is not a C3 network model.
class ReversibleRL final : public ExternalCircuit {
  public:
    struct State {
        std::vector<double> accepted, live, eps;
        bool open = false, accepted_this_step = false;
        int phase = 0, begins = 0, finishes = 0, accepts = 0, trials = 0;
        double t0 = 0., dt = 0.;
        bool operator==(const State&) const = default;
    } state;
    int snapshots = 0, cancellations = 0, releases = 0, checkpoint_calls = 0;
    bool fail_snapshot = false, fail_cancel = false, throw_accept = false;
    bool mutate_phase = false;
    void
    Define (const std::vector<std::string>& names,
            const std::vector<amrex::Real>& reference,
            const std::string&) override {
        if (names.size() != 2 || reference.size() != 2)
            throw std::runtime_error("RL ports");
        iref.assign(reference.begin(), reference.end());
        state.accepted = {.5, -.25};
        state.live = state.accepted;
    }
    void
    BeginStep (amrex::Real t0, amrex::Real dt) override {
        if (state.open)
            throw std::runtime_error("RL step already open");
        state.open = true;
        state.accepted_this_step = false;
        state.t0 = t0;
        state.dt = dt;
        ++state.begins;
        if (mutate_phase && state.phase == 0 && t0 > 0 && t0 - 1. >= -.5 * dt) {
            state.phase = 1;
            for (auto& value : state.accepted)
                value *= 1.25;
        }
        state.live = state.accepted;
    }
    void
    AdvanceInterval (amrex::Real t0, amrex::Real t1,
                     const std::vector<amrex::Real>& eps, bool accept,
                     std::vector<amrex::Real>& scales) override {
        if (!state.open || state.accepted_this_step ||
            eps.size() != iref.size())
            throw std::runtime_error("RL bad advance lifecycle");
        if (accept && throw_accept)
            throw std::runtime_error("injected accepting failure");
        scales.resize(iref.size());
        state.eps.assign(eps.begin(), eps.end());
        auto dt = t1 - t0;
        double resistance = state.phase ? 2. : 1.;
        for (std::size_t p = 0; p < iref.size(); ++p) {
            state.live[p] = (state.accepted[p] + dt * (3. - eps[p])) /
                            (1. + dt * resistance);
            scales[p] = state.live[p] / iref[p];
        }
        if (accept) {
            state.accepted = state.live;
            state.accepted_this_step = true;
            ++state.accepts;
        } else
            ++state.trials;
    }
    void
    FinishStep () override {
        if (!state.open || !state.accepted_this_step)
            throw std::runtime_error("RL finish requires acceptance");
        state.open = false;
        ++state.finishes;
    }
    void
    WriteCheckpoint (const std::string&) const override {
        throw std::runtime_error("disk checkpoint forbidden in transaction");
    }
    void
    ReadCheckpoint (const std::string&) override {
        ++checkpoint_calls;
        throw std::runtime_error("disk checkpoint forbidden in transaction");
    }
    static const WarpxCircuitRejectionApiV1*
    API () {
        static const WarpxCircuitRejectionApiV1 api{
            sizeof(api),
            WARPX_CIRCUIT_REJECTION_API_V1,
            WARPX_CIRCUIT_REJECTION_PRE_ACCEPT_V1,
            Snapshot,
            Cancel,
            Release};
        return &api;
    }
    uint64_t
    Token () const {
        return token;
    }

  private:
    std::vector<double> iref;
    std::unique_ptr<State> saved;
    uint64_t epoch = 0, token = 0;
    static ReversibleRL*
    Cast (void* ptr) {
        return static_cast<ReversibleRL*>(static_cast<ExternalCircuit*>(ptr));
    }
    static int32_t
    Snapshot (void* ptr, uint64_t* out, char* error, uint64_t capacity) {
        if (out)
            *out = 0;
        try {
            auto* self = Cast(ptr);
            if (!self || !out || self->saved || self->state.open)
                return WARPX_REJECTION_BAD_REQUEST;
            if (self->fail_snapshot)
                return WARPX_REJECTION_INTERNAL_ERROR;
            auto copy = std::make_unique<State>(self->state);
            self->saved = std::move(copy);
            self->token = ++self->epoch;
            *out = self->token;
            ++self->snapshots;
            return WARPX_REJECTION_OK;
        } catch (...) {
            if (error && capacity)
                std::snprintf(error, capacity, "RL snapshot failed");
            return WARPX_REJECTION_INTERNAL_ERROR;
        }
    }
    static int32_t
    Cancel (void* ptr, uint64_t token, char* error, uint64_t capacity) {
        try {
            auto* self = Cast(ptr);
            if (!self || !token || token != self->token || !self->saved)
                return WARPX_REJECTION_BAD_REQUEST;
            if (self->fail_cancel)
                return WARPX_REJECTION_INTERNAL_ERROR;
            self->state = std::move(*self->saved);
            self->saved.reset();
            self->token = 0;
            ++self->cancellations;
            return WARPX_REJECTION_OK;
        } catch (...) {
            if (error && capacity)
                std::snprintf(error, capacity, "RL cancel failed");
            return WARPX_REJECTION_INTERNAL_ERROR;
        }
    }
    static void
    Release (void* ptr, uint64_t token) {
        auto* self = Cast(ptr);
        if (self && token && token == self->token) {
            self->saved.reset();
            self->token = 0;
            ++self->releases;
        }
    }
};
#endif
