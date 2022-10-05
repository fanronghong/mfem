#ifndef _PNP_PROTEIN_HPP_
#define _PNP_PROTEIN_HPP_

#include <cassert>
#include "mfem.hpp"
#include "../utils/PQR_GreenFunc_PhysicalParameters.hpp"
#include "../utils/StdFunctionCoefficient.hpp"
#include "../utils/mfem_utils.hpp"
using namespace std;
using namespace mfem;


int p_order             = 1; //有限元基函数的多项式次数
const char* Linearize   = "newton"; // newton, gummel
const char* Discretize  = "cg"; // cg, dg
const char* AdvecStable = "eafe"; // none, supg, eafe
const char* prec_type   = "uzawa"; // preconditioner for Newton discretization: block, uzawa, simple
const char* options_src = "./newton_amg_lu";
int refine_times        = 0;
const char* output      = NULL;
bool self_debug         = false;
bool visualize          = false;
bool local_conservation = false;
bool show_peclet        = false;
double relax            = 0.2; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.
double schur_alpha1      = 1.0; // schur = A - alpha1 B1 A1^-1 C1 - alpha2 B2 A2^-1 C2, 这个alpha1就是该参数
double schur_alpha2      = 1.0;

double sigma            = -1.0; // symmetric parameter for DG
double kappa            = 10.0; // penalty parameter for DG
bool verbose            = false;

/* 只能定义如下集中参数
 * _1MAG_2:
 * _1bl8_tu:
 * */
#define _1MAG_2


#if defined(_1MAG_2_test_case)
#define SELF_DEBUG // only do tests for below parameters
const char* options_src = "./petsc_opts";
const int p_order = 1; //有限元基函数的多项式次数
const char* mesh_file = "../data/1MAG_2.msh"; // 带有蛋白的网格,与PQR文件必须匹配
const int refine_times = 0;
const char* pqr_file = "../data/1MAG.pqr"; // PQR文件,与网格文件必须匹配
const char* phi1_txt = "./1MAG_phi1.txt";
const int protein_marker = 1; // 这些marker信息可以从Gmsh中可视化得到
const int water_marker = 2;
const int interface_marker = 9;
const int top_marker = 8;
const int bottom_marker = 7;
const int Gamma_m_marker = 5;

double phi_top     = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 上边界是 Dirichlet, 乘以alpha1进行无量纲化
double phi_bottom  = 1.0 * alpha1; // 国际单位V, 电势在计算区域的 下边界是 Dirichlet
double phi_other   = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 其他边界是 Neumann

double c1_top      = 0.2 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 上边界是 Dirichlet,乘以alpha2是把mol/L换成Angstrom,单位统一
double c1_bottom   = 0.2 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 下边界是 Dirichlet
double c1_other    = 0.0 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 其他边界是 Neumann

double c2_top      = 0.2 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 上边界是 Dirichlet
double c2_bottom   = 0.2 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 下边界是 Dirichlet
double c2_other    = 0.0 * alpha3; // 国际单位mol/L, Cl-阳离子在计算区域的 其他边界是 Neumann
#elif defined(_1MAG_2)
const char* mesh_file   = "./1MAG_2.msh"; // 带有蛋白的网格,与PQR文件必须匹配
const char* pqr_file    = "./1MAG.pqr"; // PQR文件,与网格文件必须匹配

const int protein_marker   = 1; // 这些marker信息可以从Gmsh中可视化得到
const int water_marker     = 2;
const int interface_marker = 9;
const int top_marker       = 8;
const int bottom_marker    = 7;
const int Gamma_m_marker   = 5;

double phi_top     = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 上边界是 Dirichlet, 乘以alpha1进行无量纲化
double phi_bottom  = 2.5 * alpha1; // 国际单位V, 电势在计算区域的 下边界是 Dirichlet
double phi_other   = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 其他边界是 Neumann

double c1_top      = 0.9 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 上边界是 Dirichlet,乘以alpha2是把mol/L换成Angstrom,单位统一
double c1_bottom   = 0.1 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 下边界是 Dirichlet
double c1_other    = 0.0 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 其他边界是 Neumann

double c2_top      = 0.9 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 上边界是 Dirichlet
double c2_bottom   = 0.1 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 下边界是 Dirichlet
double c2_other    = 0.0 * alpha3; // 国际单位mol/L, Cl-阳离子在计算区域的 其他边界是 Neumann

double phi_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return phi_top;
    else if (abs(x[2] + 60.0) < 1E-10) return phi_bottom;
    else return 0.0;
}
double c1_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return c1_top;
    else if (abs(x[2] + 60.0) < 1E-10) return c1_bottom;
    else return 0.0;
}
double c2_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return c2_top;
    else if (abs(x[2] + 60.0) < 1E-10) return c2_bottom;
    else return 0.0;
}
double phi_D_top(const Vector& x)
{
    return phi_top;
}
double phi_D_bottom(const Vector& x)
{
    return phi_bottom;
}
double c1_D_top(const Vector& x)
{
    return c1_top;
}
double c1_D_bottom(const Vector& x)
{
    return c1_bottom;
}
double c2_D_top(const Vector& x)
{
    return c2_top;
}
double c2_D_bottom(const Vector& x)
{
    return c2_bottom;
}

FunctionCoefficient phi_D_coeff(phi_D_func);
FunctionCoefficient c1_D_coeff (c1_D_func);
FunctionCoefficient c2_D_coeff (c2_D_func);

ConstantCoefficient phi_D_top_coeff(phi_top);
ConstantCoefficient phi_D_bottom_coeff(phi_bottom);
ConstantCoefficient c1_D_top_coeff(c1_top);
ConstantCoefficient c1_D_bottom_coeff(c1_bottom);
ConstantCoefficient c2_D_top_coeff(c2_top);
ConstantCoefficient c2_D_bottom_coeff(c2_bottom);
#elif defined(_1bl8_tu)
const char* mesh_file = "./1bl8_tu.msh"; // 带有蛋白的网格,与PQR文件必须匹配
const char* pqr_file = "./1bl8.pqr"; // PQR文件,与网格文件必须匹配

const int protein_marker = 1;
const int water_marker = 2;
const int interface_marker = 9;
const int top_marker = 8;
const int bottom_marker = 7;
const int Gamma_m_marker   = 5;

double phi_top     = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 上边界是 Dirichlet, 乘以alpha1进行无量纲化
double phi_bottom  = 1.0 * alpha1; // 国际单位V, 电势在计算区域的 下边界是 Dirichlet
double phi_other   = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 其他边界是 Neumann

double c1_top      = 0.1 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 上边界是 Dirichlet,乘以alpha2是把mol/L换成Angstrom,单位统一
double c1_bottom   = 0.9 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 下边界是 Dirichlet
double c1_other    = 0.0 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 其他边界是 Neumann

double c2_top      = 0.1 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 上边界是 Dirichlet
double c2_bottom   = 0.9 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 下边界是 Dirichlet
double c2_other    = 0.0 * alpha3; // 国际单位mol/L, Cl-阳离子在计算区域的 其他边界是 Neumann

double phi_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return phi_top;
    else if (abs(x[2] + 60.0) < 1E-10) return phi_bottom;
    else return 0.0;
}
double c1_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return c1_top;
    else if (abs(x[2] + 60.0) < 1E-10) return c1_bottom;
    else return 0.0;
}
double c2_D_func(const Vector& x)
{
    if (abs(x[2] - 50.0) < 1E-10) return c2_top;
    else if (abs(x[2] + 60.0) < 1E-10) return c2_bottom;
    else return 0.0;
}
double phi_D_top(const Vector& x)
{
    return phi_top;
}
double phi_D_bottom(const Vector& x)
{
    return phi_bottom;
}
double c1_D_top(const Vector& x)
{
    return c1_top;
}
double c1_D_bottom(const Vector& x)
{
    return c1_bottom;
}
double c2_D_top(const Vector& x)
{
    return c2_top;
}
double c2_D_bottom(const Vector& x)
{
    return c2_bottom;
}

FunctionCoefficient phi_D_coeff(phi_D_func);
FunctionCoefficient c1_D_coeff (c1_D_func);
FunctionCoefficient c2_D_coeff (c2_D_func);

ConstantCoefficient phi_D_top_coeff(phi_top);
ConstantCoefficient phi_D_bottom_coeff(phi_bottom);
ConstantCoefficient c1_D_top_coeff(c1_top);
ConstantCoefficient c1_D_bottom_coeff(c1_bottom);
ConstantCoefficient c2_D_top_coeff(c2_top);
ConstantCoefficient c2_D_bottom_coeff(c2_bottom);
#endif


// ------------------------------- other parameters independent on mesh file and pqr file -------------------------------
const int Gummel_max_iters  = 20;
const double Gummel_rel_tol = 1e-8;
const double TOL            = 1e-10;


// ------------------------- 一些辅助变量(避免在main函数里面定义) ------------------------
ConstantCoefficient neg(-1.0);
ConstantCoefficient zero(0.0);
ConstantCoefficient one(1.0);
ConstantCoefficient two(2.0);
ConstantCoefficient sigma_coeff(sigma);
ConstantCoefficient kappa_coeff(kappa);

MarkProteinCoefficient mark_protein_coeff(protein_marker, water_marker); //在蛋白单元取值为1.0,在水中单元取值为0.0
MarkWaterCoefficient   mark_water_coeff(protein_marker, water_marker);   //在水中单元取值为1.0,在蛋白单元取值为0.0
ProductCoefficient     epsilon_water_mark(epsilon_water, mark_water_coeff);
ProductCoefficient     epsilon_protein_mark(epsilon_protein, mark_protein_coeff);
EpsilonCoefficient     Epsilon(protein_marker, water_marker, protein_rel_permittivity, water_rel_permittivity);
ProductCoefficient     neg_Epsilon(neg, Epsilon);
ProductCoefficient     sigma_Epsilon(sigma_coeff, Epsilon);
ProductCoefficient     kappa_Epsilon(kappa_coeff, Epsilon);
ProductCoefficient     sigma_water(sigma_coeff, mark_water_coeff);
ProductCoefficient     kappa_water(kappa_coeff, mark_water_coeff);

Green_func                     G_func(pqr_file);     // i.e., phi1
gradGreen_func                 gradG_func(pqr_file); // also can be obtained from grad(phi1)
StdFunctionCoefficient         G_coeff(G_func);
VectorStdFunctionCoefficient   gradG_coeff(3, gradG_func);
ScalarVectorProductCoefficient neg_gradG_coeff(neg, gradG_coeff);

ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
ProductCoefficient D1_water(D_K_, mark_water_coeff);
ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
ProductCoefficient neg_epsilon_protein(neg, epsilon_protein);
ProductCoefficient neg_D1(neg, D_K_);
ProductCoefficient neg_D2(neg, D_Cl_);
ProductCoefficient neg_D1_z1(neg_D1, v_K_coeff);
ProductCoefficient sigma_D1_z1(sigma_coeff, D1_prod_z1_water);
ProductCoefficient neg_D2_z2(neg_D2, v_Cl_coeff);
ProductCoefficient sigma_D2_z2(sigma_coeff, D2_prod_z2_water);

ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K(neg, alpha2_prod_alpha3_prod_v_K);
ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl(neg, alpha2_prod_alpha3_prod_v_Cl);
ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K_water(neg_alpha2_prod_alpha3_prod_v_K, mark_water_coeff);
ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl_water(neg_alpha2_prod_alpha3_prod_v_Cl, mark_water_coeff);

ProductCoefficient sigma_D1(sigma_coeff, D_K_);
ProductCoefficient kappa_D1(kappa_coeff, D_K_);
ProductCoefficient sigma_D2(sigma_coeff, D_Cl_);
ProductCoefficient kappa_D2(kappa_coeff, D_Cl_);
ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);

ConstantCoefficient scale_coeff(1000000000);
ConstantCoefficient scale2_coeff(1000);
#endif