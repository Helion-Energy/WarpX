/* Copyright 2026 The WarpX Community. License: BSD-3-Clause-LBNL */
// Private native device utilities; no registry writes.
View V(Field &f) { return {&f[0], &f[1], &f[2]}; }
Field Clone(View const &f)
{
    Field a;
    for (int c = 0; c < 3; ++c)
    {
        a[c].define(f[c]->boxArray(), f[c]->DistributionMap(), 1, f[c]->nGrowVect());
        MF::Copy(a[c], *f[c], 0, 0, 1, a[c].nGrowVect());
    }
    return a;
}
void Copy(View const &a, View const &b)
{
    for (int c = 0; c < 3; ++c)
        MF::Copy(*a[c], *b[c], 0, 0, 1, amrex::min(a[c]->nGrowVect(), b[c]->nGrowVect()));
}
void Zero(View const &a)
{
    for (auto *f : a)
        f->setVal(0.);
}
double Error(View const &a, View const &b, int ng = 0)
{
    auto x = Clone(a);
    double e = 0.;
    for (int c = 0; c < 3; ++c)
    {
        MF::Subtract(x[c], *b[c], 0, 0, 1, ng);
        e = std::max(e, x[c].norminf(0, ng));
    }
    return e;
}
void Gradient(WarpX &w, MF const &phi, View const &field)
{
    auto dx = w.Geom(0).CellSizeArray();
    for (int c = 0; c < 3; ++c)
    {
        int direction = c;
#if defined(WARPX_DIM_RZ)
        direction = c == 1 ? -1 : (c == 2 ? 1 : 0);
#endif
        if (direction < 0)
        {
            field[c]->setVal(0.);
            continue;
        }
        auto off = amrex::IntVect::TheDimensionVector(direction);
        double inverse = 1. / dx[direction];
        for (amrex::MFIter mfi(*field[c]); mfi.isValid(); ++mfi)
        {
            auto a = field[c]->array(mfi);
            auto f = phi.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   amrex::IntVect q(AMREX_D_DECL(i, j, k));
                                   a(q) = (f(q + off) - f(q)) * inverse;
                               });
        }
    }
}

void VectorImages(WarpX &w, View const &f, bool gather, View const *reference = nullptr)
{
    for (auto *x : f)
    {
        x->setBndry(0.);
        x->OverrideSync(w.Geom(0).periodicity());
        x->FillBoundary(w.Geom(0).periodicity());
    }
    amrex::GpuArray<int, AMREX_SPACEDIM> lo{}, hi{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        lo[d] = WarpX::field_boundary_lo[d] == FieldBoundaryType::PMC;
        hi[d] = WarpX::field_boundary_hi[d] == FieldBoundaryType::PMC;
    }
    if (gather)
    {
        amrex::Vector<amrex::IntVect> ratios;
        PEC::ApplyPECtoEfield(f, WarpX::field_boundary_lo, WarpX::field_boundary_hi,
                              FieldBoundaryType::PEC, w.get_ng_fieldgather(), w.Geom(0), 0,
                              PatchType::fine, ratios);
        PEC::ApplyPECtoBfield(f, WarpX::field_boundary_lo, WarpX::field_boundary_hi,
                              FieldBoundaryType::PMC, w.get_ng_fieldgather(), w.Geom(0), 0,
                              PatchType::fine, ratios);
    }
#if defined(WARPX_DIM_RZ)
    if (gather)
        w.ApplyFieldBoundaryOnAxis(f[0], f[1], f[2], 0);
#endif
    for (int c = 0; c < 3; ++c)
        ApplyDarwinPMCVectorBoundary(*f[c], w.Geom(0), lo, hi,
                                     reference ? (*reference)[c] : nullptr);
}
