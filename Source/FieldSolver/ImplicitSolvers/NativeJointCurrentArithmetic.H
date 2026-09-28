/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
#ifndef WARPX_NATIVE_JOINT_CURRENT_ARITHMETIC_H_
#define WARPX_NATIVE_JOINT_CURRENT_ARITHMETIC_H_
#include "WarpXSolverVec.H"
#include <AMReX_iMultiFab.H>
#include <array>
#include <cstdint>
#include <memory>
class WarpX;
namespace warpx::thermal {
/** Private attempt-owned arithmetic for the existing RZ PMC joint equations.
 * No field/particle scope is granted here. The joint owner supplies its actual
 * map generation; all physical high publications and endpoint gates remain.
 */
class NativeJointCurrentArithmetic {
public:
    using Real=amrex::Real;
    using MF=amrex::MultiFab;
    using Vec=WarpXSolverVec;
    using View=ablastr::fields::VectorField;
    explicit NativeJointCurrentArithmetic(WarpX&,Vec const& initial);
    ~NativeJointCurrentArithmetic();
    static bool SupportedLocal(WarpX&);
    void Invalidate() noexcept;
    bool BeginSolve(Vec const&);
    bool FormInput(Vec&,Vec const&,Real,Vec const&,bool probe,std::uint64_t next);
    bool AcceptInput(Vec const&,std::uint64_t);
    void EndSolve(bool) noexcept;
    bool RestoreInput(Vec const&);
    bool BindInput(Vec const&,bool probe,std::uint64_t generation);
    bool FreezeInput(Vec const&,std::uint64_t);
    void CompleteInput(bool probe,std::uint64_t);
    bool Ready(std::uint64_t) const noexcept;
    void BuildFields(Real interval,amrex::IntVect const& physical_grow,std::uint64_t);
    void PublishCurrent(std::uint64_t);
    void SubtractDisplacement(Real interval,View const* thermal_low,std::uint64_t);
    void AssembleElectric(View const&,std::uint64_t) const;
    void CaptureOhm(View const&,std::uint64_t);
    std::array<MF const*,3> CurrentRemainder(std::uint64_t) const;
    void FormResidual(Vec&,Vec const&,View const& ion,
        amrex::Array<amrex::iMultiFab,3> const& vacuum,View const& current,
        Real alpha,std::uint64_t) const;
    void AddTransverseLow(MF& total,MF const& longitudinal,MF const* longitudinal_low,
        int component,std::uint64_t) const;
    void AppendReceipt(std::vector<MF*>&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
}
#endif
