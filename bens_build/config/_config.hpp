// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_CONFIG_HEADER
#define MFEM_CONFIG_HEADER

// MFEM version: integer of the form: (major*100 + minor)*100 + patch.
#define MFEM_VERSION 40001

// MFEM version string of the form "3.3" or "3.3.1".
#define MFEM_VERSION_STRING "4.0.1"

// MFEM version type, see the MFEM_VERSION_TYPE_* constants below.
#define MFEM_VERSION_TYPE ((MFEM_VERSION)%2)

// MFEM version type constants.
#define MFEM_VERSION_TYPE_RELEASE 0
#define MFEM_VERSION_TYPE_DEVELOPMENT 1

// Separate MFEM version numbers for major, minor, and patch.
#define MFEM_VERSION_MAJOR ((MFEM_VERSION)/10000)
#define MFEM_VERSION_MINOR (((MFEM_VERSION)/100)%100)
#define MFEM_VERSION_PATCH ((MFEM_VERSION)%100)

// MFEM source directory.
#define MFEM_SOURCE_DIR "/Users/ben/Documents/SoftwareLibraries/mfem"

// MFEM install directory.
#define MFEM_INSTALL_DIR "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build"

// Description of the git commit used to build MFEM.
#define MFEM_GIT_STRING "heads/AIR_pr-0-g57c5c412a05c1370b7cfc85e34cb73af78a187c5"

// Build the parallel MFEM library.
// Requires an MPI compiler, and the libraries HYPRE and METIS.
#define MFEM_USE_MPI

// Enable debug checks in MFEM.
/* #undef MFEM_DEBUG */

// Throw an exception on errors.
/* #undef MFEM_USE_EXCEPTIONS */

// Enable gzstream in MFEM.
/* #undef MFEM_USE_GZSTREAM */

// Enable backtraces for mfem_error through libunwind.
/* #undef MFEM_USE_LIBUNWIND */

// Enable MFEM features that use the METIS library (parallel MFEM).
#define MFEM_USE_METIS

// Enable this option if linking with METIS version 5 (parallel MFEM).
#define MFEM_USE_METIS_5

// Use LAPACK routines for various dense linear algebra operations.
#define MFEM_USE_LAPACK

// Use thread-safe implementation. This comes at the cost of extra memory
// allocation and de-allocation.
/* #undef MFEM_THREAD_SAFE */

// Enable the OpenMP backend.
/* #undef MFEM_USE_OPENMP */

// [Deprecated] Enable experimental OpenMP support. Requires MFEM_THREAD_SAFE.
/* #undef MFEM_USE_LEGACY_OPENMP */

// Enable MFEM functionality based on the Mesquite library.
/* #undef MFEM_USE_MESQUITE */

// Enable MFEM functionality based on the SuiteSparse library.
/* #undef MFEM_USE_SUITESPARSE */

// Enable MFEM functionality based on the SuperLU_DIST library.
/* #undef MFEM_USE_SUPERLU */

// Enable MFEM functionality based on the STRUMPACK library.
/* #undef MFEM_USE_STRUMPACK */

// Internal MFEM option: enable group/batch allocation for some small objects.
#define MFEM_USE_MEMALLOC

// Enable functionality based on the Gecko library
/* #undef MFEM_USE_GECKO */

// Enable MFEM functionality based on the GnuTLS library
/* #undef MFEM_USE_GNUTLS */

// Enable MFEM functionality based on the NetCDF library
/* #undef MFEM_USE_NETCDF */

// Enable MFEM functionality based on the PETSc library
/* #undef MFEM_USE_PETSC */

// Enable MFEM functionality based on the Sidre library
/* #undef MFEM_USE_SIDRE */

// Enable MFEM functionality based on Conduit
/* #undef MFEM_USE_CONDUIT */

// Enable MFEM functionality based on the PUMI library
/* #undef MFEM_USE_PUMI */

// Build the GPU/CUDA-enabled version of the MFEM library.
// Requires a CUDA compiler (nvcc).
/* #undef MFEM_USE_CUDA */

// Enable MFEM functionality based on the RAJA library
/* #undef MFEM_USE_RAJA */

// Enable MFEM functionality based on the OCCA library
/* #undef MFEM_USE_OCCA */

// Which library functions to use in class StopWatch for measuring time.
// For a list of the available options, see INSTALL.
// If not defined, an option is selected automatically.
#define MFEM_TIMER_TYPE 4

// Enable MFEM functionality based on the SUNDIALS libraries.
/* #undef MFEM_USE_SUNDIALS */

// Version of HYPRE used for building MFEM.
#define MFEM_HYPRE_VERSION 21700

// Macro defined when PUMI is built with support for the Simmetrix SimModSuite
// library.
/* #undef MFEM_USE_SIMMETRIX */

#endif // MFEM_CONFIG_HEADER
