// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PLASMA_HPP
#define MFEM_PLASMA_HPP

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../mfem.hpp"
#include "../../general/text.hpp"

namespace mfem
{

namespace plasma
{

// Physical Constants

// Permittivity of Free Space (units F/m)
static const double epsilon0_ = 8.8541878176e-12;

// Permeability of Free Space (units H/m)
static const double mu0_ = 4.0e-7 * M_PI;

// Speed of light in Free Space (units m/s)
static const double c0_ = 1.0 / sqrt(epsilon0_ * mu0_);

static const double q_     = 1.602176634e-19; // Elementary charge in coulombs
static const double J_per_eV_   = 1.602176634e-19; // 1 eV in Joules
static const double kg_per_amu_ =
   1.660539040e-27; // Atomic mass unit in kilograms
static const double me_kg_ = 9.10938356e-31;  // Mass of electron in kilograms
static const double me_u_  = 5.4857990907e-4; // Mass of electron in a.m.u

/**
   Returns the cyclotron frequency in radians/second
   m is the mass in kg
   q is the charge in Coulombs
   B is the magnetic field magnitude in tesla
 */
inline double cyclotronFrequency(double B, double m, double q)
{
   return fabs(q * B / m);
}

/** Coulomg Logarithm for electron-ion collisions from 2019 NRL Plasma Formulary

   Page 34 of the NRL Formulary gives three formulae for the case of
   electron-ion collisions. Each case is suitable for a different
   range of electron and ion temperatures.

   For simplicity we will assume Te > Ti me/mi which will allow us to
   remove the dependence on the masses and the ion temperature.

   Te is the electron temperature in eV
   ne is the density of electrons in particles per meter^3
   zi is the charge number of the ion species

   Note that the NRL formulary uses cgs units in this section so their
   electron density is in particles / centimeter^3. This leads to an
   additional factor of 1e-3 in our version of these formulae.

   Also, in order to match results at the transition where Te = zi^2 10eV
   we shift the high temperature formula by 0.5 ln(10) - 1 ~ 0.15.
*/
double lambda_ei(double Te, double ne, double zi);

/// Derivative of lambda_ei wrt Te
double dlambda_ei_dTe(double Te, double ne, double zi);

/// Derivative of lambda_ei wrt ne
inline double dlambda_ei_dne(double Te, double ne, double zi)
{
   return -0.5 / ne;
}

/**
   Returns the mean Electron-Ion mean collision time in seconds (see
   equation 2.5e)
   Te is the electron temperature in eV
   ni is the density of ions (assuming ni=ne) in particles per meter^3
   zi is the charge number of the ion species
   lnLambda is the Coulomb Logarithm
*/
inline double tau_e(double Te, double zi, double ni, double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Te from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(Te * J_per_eV_, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}
/// Derivative of tau_e wrt ni
inline double dtau_e_dni(double Te, double zi, double ni, double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Te from eV to Joules
   return -0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(Te * J_per_eV_, 3) / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni * ni);
}
/// Derivative of tau_e wrt Te
inline double dtau_e_dTe(double Te, double zi, double ni, double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Te from eV to Joules
   return 1.125 * J_per_eV_ * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * Te * J_per_eV_ / M_PI) /
          (lnLambda * pow(q_, 4) * zi * zi * ni);
}
/// Derivative of tau_e wrt lnLambda
inline double dtau_e_dlambda(double Te, double zi, double ni, double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Te from eV to Joules
   return -0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(0.5 * me_kg_ * pow(Te * J_per_eV_, 3) / M_PI) /
          (lnLambda * lnLambda * pow(q_, 4) * zi * zi * ni);
}

/**
   Returns the mean Ion-Ion mean collision time in seconds (see equation 2.5i)
   mi is the ion mass in kg.
   zi is the charge number of the ion species
   ni is the density of ions in particles per meter^3
   Ti is the ion temperature in eV
   lnLambda is the Coulomb Logarithm
*/
inline double tau_i(double mi, double zi, double ni, double Ti,
                    double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Ti from eV to Joules
   return 0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * pow(Ti * J_per_eV_, 3) / M_PI) /
          (lnLambda * pow(q_ * zi, 4) * ni);
}
/// Derivative of tau_i wrt ni
inline double dtau_i_dni(double mi, double zi, double ni, double Ti,
                         double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Ti from eV to Joules
   return -0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * pow(Ti * J_per_eV_, 3) / M_PI) /
          (lnLambda * pow(q_ * zi, 4) * ni * ni);
}
/// Derivative of tau_i wrt Ti
inline double dtau_i_dTi(double mi, double zi, double ni, double Ti,
                         double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Ti from eV to Joules
   return 1.125 * J_per_eV_ * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * Ti * J_per_eV_ / M_PI) /
          (lnLambda * pow(q_ * zi, 4) * ni);
}
/// Derivative of tau_i wrt lnLambda
inline double dtau_i_dlambda(double mi, double zi, double ni, double Ti,
                             double lnLambda)
{
   // The factor of J_per_eV_ is included to convert Ti from eV to Joules
   return -0.75 * pow(4.0 * M_PI * epsilon0_, 2) *
          sqrt(mi * pow(Ti * J_per_eV_, 3) / M_PI) /
          (lnLambda * lnLambda * pow(q_ * zi, 4) * ni);
}

class G_EQDSK_Data
{
public:
   G_EQDSK_Data(std::istream &is);

   void PrintInfo(std::ostream & out = std::cout) const;
   void DumpGnuPlotData(const std::string &file) const;

   double InterpPsi(const Vector &x);
   void InterpNxGradPsi(const Vector &x, Vector &b);

private:
   void initInterpolation();

   std::vector<std::string> CASE_; // Identification character string

   int NW_; // Number of horizontal R grid points
   int NH_; // Number of vertical Z grid points

   double RDIM_;    // Horizontal dimension in meter of computational box
   double ZDIM_;    // Vertical dimension in meter of computational box
   double RLEFT_;   // Minimum R in meter of rectangular computational box
   double ZMID_;    // Z of center of computational box in meter
   double RMAXIS_;  // R of magnetic axis in meter
   double ZMAXIS_;  // Z of magnetic axis in meter
   double SIMAG_;   // poloidal flux at magnetic axis in Weber /rad
   double SIBRY_;   // poloidal flux at the plasma boundary in Weber /rad
   double RCENTR_;  // R in meter of vacuum toroidal magnetic field BCENTR
   double BCENTR_;  // Vacuum toroidal magnetic field in Tesla at RCENTR
   double CURRENT_; // Plasma current in Ampere

   // Poloidal current function in m-T, F = RBT on flux grid
   std::vector<double> FPOL_;

   // Plasma pressure in nt / m^2 on uniform flux grid
   std::vector<double> PRES_;

   // FF’(ψ) in (mT)2 / (Weber /rad) on uniform flux grid
   std::vector<double> FFPRIM_;

   // P’(ψ) in (nt /m2) / (Weber /rad) on uniform flux grid
   std::vector<double> PPRIME_;

   // Poloidal flux in Weber / rad on the rectangular grid points
   std::vector<double> PSIRZ_;

   // q values on uniform flux grid from axis to boundary
   std::vector<double> QPSI_;

   int                 NBBBS_;  // Number of boundary points
   std::vector<double> RBBBS_;  // R of boundary points in meter
   std::vector<double> ZBBBS_;  // Z of boundary points in meter

   int                 LIMITR_; // Number of limiter points
   std::vector<double> RLIM_;   // R of surrounding limiter contour in meter
   std::vector<double> ZLIM_;   // Z of surrounding limiter contour in meter

   // Divided differences for Akima's interpolation method
   double dr_, dz_;
   DenseMatrix c_;
   DenseMatrix d_;
   DenseMatrix e_;
};

class G_EQDSK_Psi_Coefficient : public Coefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_Psi_Coefficient(G_EQDSK_Data &g_eqdsk) : eqdsk(g_eqdsk) {}

   double Eval(ElementTransformation & T,
               const IntegrationPoint & ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return eqdsk.InterpPsi(transip);
   }

};

class G_EQDSK_NxGradPsi_Coefficient : public VectorCoefficient
{
private:
   G_EQDSK_Data &eqdsk;

public:

   G_EQDSK_NxGradPsi_Coefficient(G_EQDSK_Data &g_eqdsk)
      : VectorCoefficient(2), eqdsk(g_eqdsk) {}

   void Eval(Vector &b, ElementTransformation & T,
             const IntegrationPoint & ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      eqdsk.InterpNxGradPsi(transip, b);
   }

};

} // namespace plasma

} // namespace mfem

#endif // MFEM_PLASMA_HPP
