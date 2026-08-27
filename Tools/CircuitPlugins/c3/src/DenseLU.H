/* Copyright 2026 The WarpX Community
 *
 * This file is part of the c3 external-circuit plugin (Tools/CircuitPlugins/c3).
 *
 * Authors: S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef C3_DENSE_LU_H_
#define C3_DENSE_LU_H_

#include <vector>

namespace c3 {

/** Dense column-pivoted LU factorization (partial pivoting: at elimination
 * column k the pivot is the largest |A(i,k)|, i >= k).
 *
 * The banked circuit systems are small (a few thousand unknowns at most),
 * so a self-contained dense factorization replaces the Python reference's
 * scipy.sparse splu: mathematically the same solve, with different pivot
 * ordering and therefore different (tiny) roundoff -- the validation
 * harness bounds the accumulated trajectory difference.
 *
 * PA = LU with L unit-lower and U upper, both stored in place; Solve
 * applies the row swaps then forward/back substitution.
 */
class DenseLU
{
public:
    /** Factorize the n x n matrix given as COO triplets (duplicate
     * entries sum, matching scipy's coo -> csc construction). Throws
     * std::runtime_error on a zero pivot. */
    void Factor (long n,
                 std::vector<long> const& rows,
                 std::vector<long> const& cols,
                 std::vector<double> const& vals);

    /** Solve A x = rhs (Factor must have succeeded). */
    void Solve (std::vector<double> const& rhs, std::vector<double>& x) const;

    bool Valid () const { return m_n > 0; }

private:
    long m_n = 0;
    std::vector<double> m_lu;   //!< row-major; unit-L strictly below, U on/above
    std::vector<long> m_piv;    //!< row swapped with k at elimination step k
};

} // namespace c3

#endif // C3_DENSE_LU_H_
