// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ELECTROMAGNETICS_HPP
#define MFEM_ELECTROMAGNETICS_HPP

namespace mfem
{

namespace electromagnetics
{

// Physical Constants

// Permittivity of Free Space (units F/m)
// 真空介电常数, 单位 F/m
static const double epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
// 真空磁导率, 单位 henry per metre
static const double mu0_ = 4.0e-7*M_PI;

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_ELECTROMAGNETICS_HPP
