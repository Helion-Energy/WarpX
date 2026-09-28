/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Private DarwinVacuumHodgePC::Impl methods. All full-grid actions stay on device.
// Only topology metadata and the small dense interface factor visit the host at setup.
    bool BuildGreenInterface (std::string& error)
    {
        m_yee_green = std::make_unique<DarwinRZYeeGreenSolver>(geometry, true, true);
        Define(pc_action, 0);
        Define(pc_residual, 0);
        Define(pc_correction, 0);
        amrex::Long local_count = 0;
        for (int c = 0; c < 3; ++c) {
            m_interface_index[c].define(layout[c]->boxArray(),
                                        layout[c]->DistributionMap(), 1, 0);
            m_interface_index[c].setVal(-1);
            int const direction = Dimension(c);
            amrex::ReduceOps<amrex::ReduceOpSum> op;
            amrex::ReduceData<amrex::Long> data(op);
            using Tuple = decltype(data)::Type;
            for (amrex::MFIter it(m_interface_index[c]); it.isValid(); ++it) {
                auto const index = m_interface_index[c].array(it);
                auto const v = vacuum[c].const_array(it);
                auto const p = charged[c].const_array(it);
                auto const label = graph->labels.const_array(it);
                auto const measure = node_measure.const_array(it);
                auto const owner = edge_owner[c]->const_array(it);
                op.eval(it.validbox(), data,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple {
                    amrex::IntVect const q(AMREX_D_DECL(i, j, k));
                    bool interface = false;
                    if (v(q)) {
                        // The Schur correction can reach only same-component
                        // full-L neighbors of a charged row.
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            auto const off = amrex::IntVect::TheDimensionVector(d);
                            interface = interface || p(q - off) || p(q + off);
                        }
                        // Collapsing a charged potential component changes G G^dagger
                        // only on edges incident on that component. Grounded interior
                        // nodes count too; physical PEC scalar traces have zero measure.
                        if (direction >= 0) {
                            auto const off = amrex::IntVect::TheDimensionVector(direction);
                            interface = interface || (label(q) >= 0 && measure(q) > 0.) ||
                                (label(q + off) >= 0 && measure(q + off) > 0.);
                        }
                    }
                    index(q) = interface ? 0 : -1;
                    return {amrex::Long(interface && owner(q))};
                });
            }
            local_count += amrex::get<0>(data.value());
        }
        amrex::Long count = local_count;
        amrex::ParallelDescriptor::ReduceLongSum(count);
        if (count > options.max_interface_dofs) {
            error = "Yee Green interface exceeds the explicit dense-interface dof cap";
            return false;
        }
        int const n = static_cast<int>(count);
        int const local = static_cast<int>(local_count);
        stats.interface_dofs = n;
        amrex::Gpu::DeviceVector<amrex::Long> local_ids(local);
        amrex::Gpu::DeviceVector<int> cursor(1);
        auto* position = cursor.data();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) { *position = 0; });
        auto* packed = local_ids.data();
        amrex::Long offset = 0;
        for (int c = 0; c < 3; ++c) {
            auto const domain = amrex::convert(geometry.Domain(), layout[c]->ixType());
            int const nx = domain.length(0);
            for (amrex::MFIter it(m_interface_index[c]); it.isValid(); ++it) {
                auto const index = m_interface_index[c].const_array(it);
                auto const owner = edge_owner[c]->const_array(it);
                // Shared counter/scatter: For deliberately has no CPU SIMD promise.
                amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (index(i, j, k) >= 0 && owner(i, j, k)) {
                        int const slot = amrex::Gpu::Atomic::Add(position, 1);
                        packed[slot] = offset + amrex::Long(i) + amrex::Long(nx) * j;
                    }
                });
            }
            offset += domain.numPts();
        }
        std::vector<amrex::Long> host_local(local), ids(n);
        int packed_count = -1;
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, cursor.begin(), cursor.end(), &packed_count);
        AMREX_ALWAYS_ASSERT(packed_count == local);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         local_ids.begin(), local_ids.end(), host_local.begin());
#ifdef AMREX_USE_MPI
        int const ranks = amrex::ParallelDescriptor::NProcs();
        std::vector<int> counts(ranks), starts(ranks, 0);
        AMREX_ALWAYS_ASSERT(MPI_Allgather(
            &local, 1, MPI_INT, counts.data(), 1, MPI_INT,
            amrex::ParallelDescriptor::Communicator()) == MPI_SUCCESS);
        for (int rank = 1; rank < ranks; ++rank) {
            starts[rank] = starts[rank - 1] + counts[rank - 1];
        }
        AMREX_ALWAYS_ASSERT(starts.back() + counts.back() == n);
        auto const type = amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type();
        AMREX_ALWAYS_ASSERT(MPI_Allgatherv(
            host_local.data(), local, type, ids.data(), counts.data(), starts.data(), type,
            amrex::ParallelDescriptor::Communicator()) == MPI_SUCCESS);
#else
        ids = std::move(host_local);
#endif
        std::sort(ids.begin(), ids.end());
        AMREX_ALWAYS_ASSERT(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
        m_interface_ids.resize(n);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, ids.begin(), ids.end(), m_interface_ids.begin());
        auto const* global_ids = m_interface_ids.data();
        offset = 0;
        for (int c = 0; c < 3; ++c) {
            auto const domain = amrex::convert(geometry.Domain(), layout[c]->ixType());
            int const nx = domain.length(0);
            for (amrex::MFIter it(m_interface_index[c]); it.isValid(); ++it) {
                auto const index = m_interface_index[c].array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (index(i, j, k) >= 0) {
                        amrex::Long const code = offset + amrex::Long(i) + amrex::Long(nx) * j;
                        int low = 0, high = n;
                        while (low < high) {
                            int const mid = low + (high - low) / 2;
                            if (global_ids[mid] < code) {
                                low = mid + 1;
                            } else {
                                high = mid;
                            }
                        }
                        AMREX_ALWAYS_ASSERT(low < n && global_ids[low] == code);
                        index(i, j, k) = low;
                    }
                });
            }
            offset += domain.numPts();
        }
        m_interface_rhs.resize(n);
#if defined(AMREX_USE_MPI) && defined(AMREX_USE_GPU)
        if (amrex::ParallelDescriptor::NProcs() > 1) {
            m_interface_reduce_scratch.resize(n);
            stats.local_owned_bytes += std::size_t(n) * sizeof(Real);
        }
#endif
        m_interface_lu.resize(std::size_t(n) * n);
        m_interface_pivots.resize(n);
        stats.local_owned_bytes += (std::size_t(n) * n + n) * sizeof(Real);
        for (auto const* fields : {&pc_action, &pc_residual, &pc_correction}) {
            for (auto const& field : *fields) {
                for (amrex::MFIter it(field); it.isValid(); ++it) {
                    stats.local_owned_bytes += field[it].size() * sizeof(Real);
                }
            }
        }
        if (n == 0) {
            amrex::Gpu::streamSynchronize();
            return true;
        }
        // C = I^T H M0 I. Length factors cancel between H and M0.
        // C is generally nonsymmetric, so use pivoted LU, never Cholesky.
        std::vector<Real> matrix(std::size_t(n) * n), column(n);
        auto* rhs = m_interface_rhs.data();
        for (int col = 0; col < n; ++col) {
            amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) { rhs[i] = i == col ? 1. : 0.; });
            InjectInterface(V(pc_residual));
            YeeBaseInverse(V(pc_correction), V(pc_residual));
            HodgeCompletion(V(pc_action), V(pc_correction));
            ExtractInterface(V(pc_action));
            amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                             m_interface_rhs.begin(), m_interface_rhs.end(), column.begin());
            for (int row = 0; row < n; ++row) {
                matrix[std::size_t(row) * n + col] = column[row];
            }
            ++stats.interface_setup_columns;
        }
        std::vector<int> pivots(n);
        for (int col = 0; col < n; ++col) {
            int pivot = col;
            for (int row = col + 1; row < n; ++row) {
                if (std::abs(matrix[std::size_t(row) * n + col]) >
                    std::abs(matrix[std::size_t(pivot) * n + col])) {
                    pivot = row;
                }
            }
            pivots[col] = pivot;
            Real const value = matrix[std::size_t(pivot) * n + col];
            if (!std::isfinite(value) || value == 0.) {
                error = "Yee Green interface matrix has a singular or nonfinite LU pivot";
                return false;
            }
            if (pivot != col) {
                for (int j = 0; j < n; ++j) {
                    std::swap(matrix[std::size_t(col) * n + j],
                              matrix[std::size_t(pivot) * n + j]);
                }
            }
            for (int row = col + 1; row < n; ++row) {
                Real& multiplier = matrix[std::size_t(row) * n + col];
                multiplier /= matrix[std::size_t(col) * n + col];
                for (int j = col + 1; j < n; ++j) {
                    matrix[std::size_t(row) * n + j] -=
                        multiplier * matrix[std::size_t(col) * n + j];
                }
            }
        }
        if (!std::all_of(matrix.begin(), matrix.end(), [](Real x) { return std::isfinite(x); })) {
            error = "Yee Green interface factorization is nonfinite";
            return false;
        }
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         matrix.begin(), matrix.end(), m_interface_lu.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         pivots.begin(), pivots.end(), m_interface_pivots.begin());
        amrex::Gpu::streamSynchronize();
        return true;
    }

    void YeeBaseInverse (View const& out, View const& rhs)
    {
        for (int c = 0; c < 3; ++c) {
            m_yee_green->SolveComponent(*out[c], *rhs[c], c);
            out[c]->mult(1. / (options.length * options.length), 0, 1, 0);
            ++stats.yee_green_component_solves;
        }
        Restrict(out);
    }

    void InjectInterface (View const& out)
    {
        auto const* values = m_interface_rhs.data();
        for (int c = 0; c < 3; ++c) {
            out[c]->setVal(0.);
            for (amrex::MFIter it(*out[c]); it.isValid(); ++it) {
                auto const a = out[c]->array(it);
                auto const index = m_interface_index[c].const_array(it);
                amrex::ParallelFor(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    int const row = index(i, j, k);
                    if (row >= 0) {
                        a(i, j, k) = values[row];
                    }
                });
            }
        }
    }

    void ExtractInterface (View const& in)
    {
        int const n = stats.interface_dofs;
        auto* values = m_interface_rhs.data();
        amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) { values[i] = 0.; });
        for (int c = 0; c < 3; ++c) {
            for (amrex::MFIter it(*in[c]); it.isValid(); ++it) {
                auto const a = in[c]->const_array(it);
                auto const index = m_interface_index[c].const_array(it);
                auto const owner = edge_owner[c]->const_array(it);
                amrex::For(it.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    int const row = index(i, j, k);
                    if (row >= 0 && owner(i, j, k)) {
                        values[row] = a(i, j, k);
                    }
                });
            }
        }
        amrex::Gpu::streamSynchronize();
#ifdef AMREX_USE_MPI
        if (amrex::ParallelDescriptor::NProcs() > 1) {
#ifdef AMREX_USE_GPU
            // Reuse the circuit's device arithmetic and GPU-aware point-to-point
            // reduction. MPI's own local reduction need not support device buffers.
            warpx::circuit::SumDeviceVector(m_interface_rhs, m_interface_reduce_scratch);
#else
            auto const type = amrex::ParallelDescriptor::Mpi_typemap<Real>::type();
            AMREX_ALWAYS_ASSERT(MPI_Allreduce(MPI_IN_PLACE, values, n, type, MPI_SUM,
                amrex::ParallelDescriptor::Communicator()) == MPI_SUCCESS);
#endif
        }
#endif
    }

    void SolveInterface ()
    {
        int const n = stats.interface_dofs;
        auto* x = m_interface_rhs.data();
        auto const* lu = m_interface_lu.data();
        auto const* pivots = m_interface_pivots.data();
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) {
            for (int col = 0; col < n; ++col) {
                int const pivot = pivots[col];
                Real const value = x[col];
                x[col] = x[pivot];
                x[pivot] = value;
            }
            for (int row = 0; row < n; ++row) {
                for (int col = 0; col < row; ++col) {
                    x[row] -= lu[std::size_t(row) * n + col] * x[col];
                }
            }
            for (int row = n - 1; row >= 0; --row) {
                for (int col = row + 1; col < n; ++col) {
                    x[row] -= lu[std::size_t(row) * n + col] * x[col];
                }
                x[row] /= lu[std::size_t(row) * n + row];
            }
        });
    }

    void GreenInterfacePrecond (View const& out, View const& rhs)
    {
        Distinct(out, rhs);
        for (int c = 0; c < 3; ++c) {
            MF::Copy(pc_residual[c], *rhs[c], 0, 0, 1, 0);
            pc_residual[c].OverrideSync(geometry.periodicity());
        }
        Restrict(V(pc_residual));
        YeeBaseInverse(out, V(pc_residual));
        if (stats.interface_dofs > 0) {
            HodgeCompletion(V(pc_action), out);
            for (int c = 0; c < 3; ++c) {
                MF::Subtract(pc_action[c], pc_residual[c], 0, 0, 1, 0);
            }
            ExtractInterface(V(pc_action));
            SolveInterface();
            InjectInterface(V(pc_residual));
            YeeBaseInverse(V(pc_correction), V(pc_residual));
            for (int c = 0; c < 3; ++c) {
                MF::Subtract(*out[c], pc_correction[c], 0, 0, 1, 0);
            }
        }
        ++stats.interface_applications;
    }
