/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ImplicitSolvers/WarpXSolverVec.H"
#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {
using ablastr::fields::Direction;
using amrex::Real;
using warpx::fields::FieldType;
constexpr auto energy_name = "hybrid_electron_energy_fp";

void
near (Real actual, Real expected, const char* message)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(actual) &&
                                         std::abs(actual - expected) <=
                                             2.0e-12 * std::max(Real(1.0), std::abs(expected)),
                                     message);
}

void
setup (WarpX& sim, int size, int max_grid, bool periodic)
{
    const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(size - 1));
    const amrex::RealBox physical({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
    amrex::Array<int, AMREX_SPACEDIM> periods{};
    periods[AMREX_SPACEDIM - 1] = int(periodic);
#ifdef WARPX_DIM_RZ
    constexpr int coord = 1;
#else
    constexpr int coord = 0;
#endif
    sim.geometry = amrex::Geometry(domain, &physical, coord, periods.data());
    amrex::BoxArray cells(domain);
    cells.maxSize(max_grid);
    const amrex::DistributionMapping dm(cells);
    for (int component = 0; component < 3; ++component) {
        amrex::IntVect centering(1);
#ifdef WARPX_DIM_RZ
        if (component == 0) {
            centering[0] = 0;
        }
        if (component == 2) {
            centering[1] = 0;
        }
#else
        centering[component] = 0;
#endif
        sim.m_fields.alloc_init(FieldType::Efield_fp, Direction{component}, 0,
                                amrex::convert(cells, centering), dm, 1, amrex::IntVect(1), 0.0);
    }
    sim.m_fields.alloc_init(FieldType::phi_fp, 0, amrex::convert(cells, amrex::IntVect(1)), dm, 1,
                            amrex::IntVect(1), 0.0);
    sim.m_fields.alloc_init(FieldType::hybrid_electron_energy_fp, 0, cells, dm, 1,
                            amrex::IntVect(1), 0.0, true, true, true);
    sim.m_fields.alloc_init("layout_aux", 0, amrex::convert(cells, amrex::IntVect(1)), dm, 2,
                            amrex::IntVect(1), 0.0);
}

amrex::Long
count (const WarpX& sim, const amrex::IntVect& centering)
{
    amrex::Long result = 1;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        result *=
            sim.Geom(0).Domain().length(d) + (centering[d] && !sim.Geom(0).isPeriodic(d) ? 1 : 0);
    }
    return result;
}

std::vector<Real>
packed (const WarpXSolverVec& vec)
{
    amrex::Gpu::DeviceVector<Real> device(vec.nDOF_local());
    vec.copyTo(device.data());
    std::vector<Real> host(device.size());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, device.begin(), device.end(), host.begin());
    return host;
}

void
check_ids (const WarpXSolverVec& vec)
{
    std::vector<int> local;
    std::vector<int> global;
    auto collect = [&] (const amrex::iMultiFab& dofs) {
        for (amrex::MFIter mfi(dofs); mfi.isValid(); ++mfi) {
            const auto& fab = dofs[mfi];
            amrex::IArrayBox host(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(), fab.dataPtr() + fab.size(),
                             host.dataPtr());
            const auto a = host.const_array();
            amrex::LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
                for (int n = 0; n < fab.nComp(); n += 2) {
                    if (a(i, j, k, n) >= 0) {
                        local.push_back(a(i, j, k, n));
                        global.push_back(a(i, j, k, n + 1));
                    }
                }
            });
        }
    };
    const auto& dofs = *vec.getDOFsObject();
    for (const auto& level : dofs.m_array) {
        for (const auto& field : level) {
            if (field) {
                collect(*field);
            }
        }
    }
    for (const auto& field : dofs.m_scalar) {
        if (field) {
            collect(*field);
        }
    }
    for (const auto& block : dofs.m_multifab_blocks) {
        for (const auto& field : block.dofs) {
            collect(*field);
        }
    }
    AMREX_ALWAYS_ASSERT(local.size() == std::size_t(vec.nDOF_local()));
    std::sort(local.begin(), local.end());
    for (std::size_t i = 0; i < local.size(); ++i) {
        AMREX_ALWAYS_ASSERT(local[i] == int(i));
    }
    amrex::Vector<amrex::Long> counts(amrex::ParallelDescriptor::NProcs());
    const auto nlocal = vec.nDOF_local();
#ifdef AMREX_USE_MPI
    MPI_Allgather(&nlocal, 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
                  counts.data(), 1, amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
                  amrex::ParallelDescriptor::Communicator());
#else
    counts[0] = nlocal;
#endif
    const auto offset = std::accumulate(
        counts.begin(), counts.begin() + amrex::ParallelDescriptor::MyProc(), amrex::Long(0));
    std::sort(global.begin(), global.end());
    for (std::size_t i = 0; i < global.size(); ++i) {
        AMREX_ALWAYS_ASSERT(global[i] == int(i + offset));
    }
}

std::vector<Real>
all_values (const WarpXSolverVec& vec)
{
    std::vector<Real> result;
    auto collect = [&] (const amrex::MultiFab& field) {
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            const auto& fab = field[mfi];
            const auto offset = result.size();
            result.resize(offset + fab.size());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, fab.dataPtr(), fab.dataPtr() + fab.size(),
                             result.begin() + offset);
        }
    };
    if (vec.getArrayVecType() != FieldType::None) {
        for (auto* field : vec.getArrayVec()[0]) {
            collect(*field);
        }
    }
    if (vec.getScalarVecType() != FieldType::None) {
        collect(*vec.getScalarVec()[0]);
    }
    for (const auto& spec : vec.getMultiFabBlockSpecs()) {
        collect(vec.getMultiFabBlock(spec.name, 0));
    }
    return result;
}

void
seed (WarpXSolverVec& vec)
{
    const auto periods = vec.getWarpX()->Geom(0).isPeriodicArray();
    const auto lengths = vec.getWarpX()->Geom(0).Domain().length3d();
    auto fill = [&] (amrex::MultiFab& field) {
        for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
            const auto arr = field.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), field.nComp(), [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    const int ii = periods[0] ? i % lengths[0] : i;
                    const int jj = periods[1] ? j % lengths[1] : j;
#if (AMREX_SPACEDIM == 3)
                    const int kk = periods[2] ? k % lengths[2] : k;
#else
                    const int kk = k;
#endif
                    arr(i, j, k, n) = 1.0 + ii * 0.25 + jj * 0.0625 + kk * 0.015625 + n * 0.0078125;
                });
        }
    };
    if (vec.getArrayVecType() != FieldType::None) {
        for (auto* field : vec.getArrayVec()[0]) {
            fill(*field);
        }
    }
    if (vec.getScalarVecType() != FieldType::None) {
        fill(*vec.getScalarVec()[0]);
    }
    for (const auto& spec : vec.getMultiFabBlockSpecs()) {
        fill(vec.getMultiFabBlock(spec.name, 0));
    }
}

void
roundtrip (WarpXSolverVec& vec)
{
    const auto original = packed(vec);
    const auto valid_original = all_values(vec);
    amrex::Gpu::DeviceVector<Real> device(original.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, original.begin(), original.end(), device.begin());
    vec.setVal(0.0);
    vec.copyFrom(device.data());
    AMREX_ALWAYS_ASSERT(packed(vec) == original);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(all_values(vec) == valid_original,
                                     "copyFrom failed to restore duplicate/periodic valid points");
    Real norm = std::inner_product(original.begin(), original.end(), original.begin(), Real(0.0));
    amrex::ParallelAllReduce::Sum(norm, amrex::ParallelContext::CommunicatorSub());
    near(vec.dotProduct(vec), norm, "packed norm differs from vector norm");
    const auto blocks = vec.blockNorms();
    near(std::inner_product(blocks.begin(), blocks.end(), blocks.begin(), Real(0.0)), norm,
         "per-block norms do not sum to solver norm");
    check_ids(vec);
}

void
check_ops (const WarpXSolverVec& source)
{
    WarpXSolverVec x, y, z;
    x.Copy(source);
    y.Define(source);
    y.Copy(source);
    z.Define(source);
    z.linComb(2.0, x, -1.0, y);
    const auto initial = packed(source);
    AMREX_ALWAYS_ASSERT(packed(z) == initial);
    z += y;
    z -= x;
    z.increment(y, 1.0);
    z.scale(0.5);
    AMREX_ALWAYS_ASSERT(packed(z) == initial);
    z.setVal(8.0);
    roundtrip(z);
    seed(z);
    roundtrip(z);
}

void
run (int max_grid, bool periodic)
{
    WarpX sim;
    setup(sim, 8, max_grid, periodic);
    auto e = std::make_unique<WarpXSolverVec>();
    e->Define(&sim, "Efield_fp");
    WarpXSolverVec phi, energy, combined, aux;
    phi.Define(&sim, "none", "phi_fp");
    energy.Define(&sim, "none", "none", {{energy_name, 4.0}});
    combined.Define(&sim, "Efield_fp", "none", {{energy_name, 4.0}}, 2.0);
    aux.Define(&sim, "Efield_fp", "phi_fp", {{energy_name, 4.0}, {"layout_aux", 8.0}}, 2.0, 16.0);
    AMREX_ALWAYS_ASSERT(e->getDOFsObject() != energy.getDOFsObject());
    AMREX_ALWAYS_ASSERT(phi.getDOFsObject() != combined.getDOFsObject());
    amrex::Long ne = 0;
    for (int n = 0; n < 3; ++n) {
        ne += count(sim,
                    sim.m_fields.get(FieldType::Efield_fp, Direction{n}, 0)->ixType().toIntVect());
    }
    const auto nu = count(sim, amrex::IntVect(0));
    const auto np = count(sim, amrex::IntVect(1));
    AMREX_ALWAYS_ASSERT(e->nDOF_global() == ne);
    AMREX_ALWAYS_ASSERT(phi.nDOF_global() == np);
    AMREX_ALWAYS_ASSERT(energy.nDOF_global() == nu);
    AMREX_ALWAYS_ASSERT(combined.nDOF_global() == ne + nu);
    AMREX_ALWAYS_ASSERT(aux.nDOF_global() == ne + nu + 3 * np);
    AMREX_ALWAYS_ASSERT(energy.getMultiFabBlock(energy_name, 0).ixType().cellCentered());
    AMREX_ALWAYS_ASSERT(energy.hasMultiFabBlock(energy_name));
    AMREX_ALWAYS_ASSERT(!energy.hasMultiFabBlock("phi_fp"));
    AMREX_ALWAYS_ASSERT(
        (combined.blockNames() == std::vector<std::string>{"Efield_fp", energy_name}));
    AMREX_ALWAYS_ASSERT((combined.blockScales() == std::vector<Real>{2.0, 4.0}));
    e->setVal(2.0);
    phi.setVal(3.0);
    energy.setVal(8.0);
    combined.setVal(2.0);
    combined.getMultiFabBlock(energy_name, 0).setVal(8.0);
    aux.setVal(16.0);
    aux.getMultiFabBlock("layout_aux", 0).setVal(32.0, 1, 1, 0);
    near(e->dotProduct(*e), 4 * ne, "E compatibility norm");
    near(phi.dotProduct(phi), 9 * np, "phi compatibility norm");
    near(energy.dotProduct(energy), 4 * nu, "energy scaled norm");
    near(combined.dotProduct(combined), ne + 4 * nu, "independent combined scales");
    near(combined.dotProduct(combined, false), 4 * ne + 64 * nu, "unscaled norm");
    near(combined.blockNorms()[0], std::sqrt(Real(ne)), "E block norm");
    near(combined.blockNorms()[1], 2 * std::sqrt(Real(nu)), "energy block norm");
    for (auto* v : {e.get(), &phi, &energy, &combined, &aux}) {
        roundtrip(*v);
        check_ops(*v);
    }

    // Read/write named field data explicitly; legacy Copy remains field-only.
    sim.m_fields.get(FieldType::hybrid_electron_energy_fp, 0)->setVal(24.0);
    combined.CopyMultiFabBlocksFromFields();
    near(combined.blockNorms()[1], 6 * std::sqrt(Real(nu)), "registry energy import");
    combined.Copy(FieldType::Efield_fp);
    near(combined.blockNorms()[1], 6 * std::sqrt(Real(nu)), "legacy Copy overwrote energy");
    combined.getMultiFabBlock(energy_name, 0).setVal(28.0);
    combined.CopyMultiFabBlocksToFields();
    near(sim.m_fields.get(energy_name, 0)->min(0), 28.0, "registry energy export");

    // Actual register persistence route, opted-in allocation only. Rebuild
    // layouts after restart.
    const std::string prefix = "layout_checkpoint_" + std::to_string(max_grid) + "_" +
                               std::to_string(periodic) + "_" +
                               std::to_string(amrex::ParallelDescriptor::NProcs());
    amrex::UtilCreateCleanDirectory(prefix, true);
    sim.m_fields.write_checkpoints(0, prefix + "/");
    sim.m_fields.get(energy_name, 0)->setVal(-1.0);
    const auto loaded = sim.m_fields.read_restarts(0, prefix + "/");
    AMREX_ALWAYS_ASSERT(loaded.size() == 1);
    near(sim.m_fields.get(energy_name, 0)->min(0), 28.0, "checkpoint energy roundtrip");
    WarpXSolverVec restarted;
    restarted.Define(&sim, "none", "none", {{energy_name, 4.0}});
    restarted.CopyMultiFabBlocksFromFields();
    near(restarted.blockNorms()[0], 7 * std::sqrt(Real(nu)), "restart layout reconstruction");
    roundtrip(restarted);

    // Per-layout lifetime: an E work vector survives its original, unrelated
    // U/phi destruction.
    WarpXSolverVec survivor;
    survivor.Define(*e);
    survivor.Copy(*e);
    auto shared = survivor.getDOFsObject();
    AMREX_ALWAYS_ASSERT(shared == e->getDOFsObject());
    e.reset();
    const auto saved_norm = survivor.norm2();
    for (auto& [name, mask] : sim.masks) {
        mask->setVal(0);
    }
    near(survivor.norm2(), saved_norm, "DOF layout borrowed a mutable registry mask");
    sim.masks.clear();
    roundtrip(survivor);
    WarpXSolverVec moved(std::move(energy));
    AMREX_ALWAYS_ASSERT(!energy.IsDefined());
    roundtrip(moved);
    std::weak_ptr<const WarpXSolverDOF> released = phi.getDOFsObject();
    phi = std::move(moved);
    AMREX_ALWAYS_ASSERT(!moved.IsDefined() && released.expired());
    roundtrip(phi);

    // Interleave a different simulation and grid; original metadata must be
    // unchanged.
    {
        WarpX second;
        setup(second, 6, 3, !periodic);
        WarpXSolverVec second_u;
        second_u.Define(&second, "none", "none", {{energy_name, 2.0}});
        second_u.setVal(6.0);
        roundtrip(second_u);
        AMREX_ALWAYS_ASSERT(second_u.nDOF_global() != phi.nDOF_global());
    }
    roundtrip(survivor);
    roundtrip(phi);
    roundtrip(combined);

    // A field-register remake cannot silently change an existing work-vector
    // layout or masks.
    sim.m_fields.erase(energy_name, 0);
    const amrex::BoxArray different(amrex::Box(amrex::IntVect(0), amrex::IntVect(3)));
    sim.m_fields.alloc_init(energy_name, 0, different, amrex::DistributionMapping(different), 1,
                            amrex::IntVect(0), 0.0);
    WarpXSolverVec frozen_clone;
    frozen_clone.Define(phi);
    frozen_clone.Copy(phi);
    AMREX_ALWAYS_ASSERT(frozen_clone.nDOF_global() == nu);
    roundtrip(frozen_clone);
    amrex::Print() << "PASS solver layouts: E=" << ne << " phi=" << np << " U=" << nu
                   << " boxes=" << max_grid << " periodic=" << periodic << "\n";
}

void
reject (const std::string& name)
{
    WarpX sim;
    setup(sim, 8, 4, false);
    WarpXSolverVec vec;
    if (name == "empty") {
        vec.Define(&sim, "none", "none", {});
    } else if (name == "duplicate") {
        vec.Define(&sim, "none", "none", {{energy_name, 1.0}, {energy_name, 2.0}});
    } else if (name == "scalar_alias") {
        vec.Define(&sim, "none", "phi_fp", {{"phi_fp", 1.0}});
    } else if (name == "nan_scale") {
        vec.Define(&sim, "none", "none", {{energy_name, std::numeric_limits<Real>::quiet_NaN()}});
    } else if (name == "inf_scale") {
        vec.Define(&sim, "Efield_fp", "none", {}, std::numeric_limits<Real>::infinity());
    } else if (name == "zero_scale") {
        vec.Define(&sim, "none", "phi_fp", {}, 1.0, 0.0);
    } else if (name == "bad_layout") {
        vec.Define(&sim, "none", "none", {{energy_name, 1.0}});
        sim.m_fields.erase(energy_name, 0);
        const amrex::BoxArray different(amrex::Box(amrex::IntVect(0), amrex::IntVect(3)));
        sim.m_fields.alloc_init(energy_name, 0, different, amrex::DistributionMapping(different), 1,
                                amrex::IntVect(0), 0.0);
        WarpXSolverVec other;
        other.Define(&sim, "none", "none", {{energy_name, 1.0}});
        vec.Copy(other);
    }
}
} // namespace

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("test");
        int boxes = 4;
        bool periodic = false;
        std::string invalid;
        pp.query("max_grid_size", boxes);
        pp.query("periodic", periodic);
        pp.query("reject", invalid);
        if (invalid.empty()) {
            run(boxes, periodic);
        } else {
            reject(invalid);
        }
    }
    amrex::Finalize();
}
