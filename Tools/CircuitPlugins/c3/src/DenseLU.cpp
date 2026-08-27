/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "DenseLU.H"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace c3 {

void
DenseLU::Factor (long n,
                 std::vector<long> const& rows,
                 std::vector<long> const& cols,
                 std::vector<double> const& vals)
{
    m_n = n;
    m_lu.assign(static_cast<std::size_t>(n) * n, 0.0);
    m_piv.assign(n, 0);
    for (std::size_t t = 0; t < vals.size(); ++t) {
        m_lu[rows[t] * n + cols[t]] += vals[t];   // duplicates sum
    }

    double* a = m_lu.data();
    for (long k = 0; k < n; ++k) {
        long p = k;
        double amax = std::fabs(a[k * n + k]);
        for (long i = k + 1; i < n; ++i) {
            double const v = std::fabs(a[i * n + k]);
            if (v > amax) { amax = v; p = i; }
        }
        if (amax == 0.0) {
            m_n = 0;
            throw std::runtime_error("DenseLU: zero pivot at column "
                                     + std::to_string(k));
        }
        m_piv[k] = p;
        if (p != k) {
            std::swap_ranges(a + k * n, a + (k + 1) * n, a + p * n);
        }
        double const inv_piv = 1.0 / a[k * n + k];
        for (long i = k + 1; i < n; ++i) {
            double const lik = a[i * n + k] * inv_piv;
            a[i * n + k] = lik;
            if (lik == 0.0) { continue; }
            double const* rk = a + k * n;
            double* ri = a + i * n;
            for (long j = k + 1; j < n; ++j) {
                ri[j] -= lik * rk[j];
            }
        }
    }
}

void
DenseLU::Solve (std::vector<double> const& rhs, std::vector<double>& x) const
{
    long const n = m_n;
    x = rhs;
    for (long k = 0; k < n; ++k) {
        if (m_piv[k] != k) { std::swap(x[k], x[m_piv[k]]); }
    }
    double const* a = m_lu.data();
    for (long i = 1; i < n; ++i) {          // L y = P rhs (unit diagonal)
        double s = x[i];
        double const* ri = a + i * n;
        for (long j = 0; j < i; ++j) {
            s -= ri[j] * x[j];
        }
        x[i] = s;
    }
    for (long i = n - 1; i >= 0; --i) {     // U x = y
        double s = x[i];
        double const* ri = a + i * n;
        for (long j = i + 1; j < n; ++j) {
            s -= ri[j] * x[j];
        }
        x[i] = s / ri[i];
    }
}

} // namespace c3
