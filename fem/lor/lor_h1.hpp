// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LOR_H1
#define MFEM_LOR_H1

#include "lor_batched.hpp"

namespace mfem
{

// BatchedLORKernel specialization for H1 spaces. Not user facing. See the
// classes BatchedLORAssembly and BatchedLORKernel .
class BatchedLOR_H1 : BatchedLORKernel
{
protected:
   // TODO: for now only supporting constant coefficients
   double mass_coeff, diffusion_coeff;
public:
   template <int ORDER> void Assemble2D();
   template <int ORDER> void Assemble3D();
   BatchedLOR_H1(BilinearForm &a,
                 FiniteElementSpace &fes_ho_,
                 Vector &X_vert_,
                 Vector &sparse_ij_,
                 Array<int> &sparse_mapping_);
};

}

#endif
