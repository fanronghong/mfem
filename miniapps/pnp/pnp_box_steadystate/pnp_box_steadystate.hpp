#ifndef _PNP_BOX_HPP_
#define _PNP_BOX_HPP_

#include "mfem.hpp"
#include "../utils/PQR_GreenFunc_PhysicalParameters.hpp"
using namespace std;
using namespace mfem;

//#define SELF_VERBOSE

const char* mesh_file       = "./4_4_4_translate.msh";
int p_order                 = 1; //有限元基函数的多项式次数
int refine_times            = 0;
const char* Linearize       = "gummel"; // newton, gummel
bool zero_initial           = true; // 非线性迭代的初值是否为0
double initTol              = 1e-3; // 为得到非线性迭代的初值所需Gummel迭代
const char* Discretize      = "dg"; // cg, dg
const char* AdvecStable     = "none"; // none, eafe, supg
const char* options_src     = "./pnp_box_petsc_opts";
bool ComputeConvergenceRate = false; // 利用解析解计算误差阶
bool local_conservation     = false;
bool visualize              = false;
const char* output          = NULL;
int max_newton              = 20;
double relax                = 0.02; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.

const int bottom_attr       = 1;
const int top_attr          = 6;
const int left_attr         = 5;
const int front_attr        = 2;
const int back_attr         = 4;
const int right_attr        = 3;

const int Gummel_max_iters  = 50;
const double Gummel_rel_tol = 1e-10;
const double TOL            = 1e-20;
const double sigma = -1.0; // symmetric parameter for DG
const double kappa = 20; // penalty parameter for DG

/* 可以定义如下模型参数: 前三个宏定义参数在其他头文件定义
 * Angstrom_SCALE: 埃米尺度
 * Nano_SCALE: 纳米尺度
 * Micro_SCALE: 微米尺度
 * */
#define Angstrom_SCALE

#if defined(Angstrom_SCALE)
double phi_exact_(const Vector& x)
{
    return 1.9460874289826929e-6*x[2] + 4.3132524201962733e-6*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 1.9460874289826929e-6;
}
double c1_exact_(const Vector& x)
{
    return 0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003;
}
double c2_exact_(const Vector& x)
{
    return -0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003;
}
double f1_analytic_(const Vector& x)
{
    return -0.00047297787088862518*(1.3279472474040542e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) - 3.8143313608060783e-7)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 3.1415926535900001*(3.9985429747669003e-10*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 7.9970859495338007e-10)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 2.0859346583998194e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 6.2808966172957995e-10*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 6.2808966172957995e-10*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.00043685571852133725*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
double f2_analytic_(const Vector& x)
{
    return -0.00047297787088862518*(1.3753739348113418e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) - 3.9505574808348666e-7)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 3.1415926535900001*(4.1413480810085754e-10*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 8.2826961620171509e-10)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 2.1604323247712418e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.00060221412900000003)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 6.5052143536277921e-10*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 6.5052143536277921e-10*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.00045245770846852795*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
void J_(const Vector& x, Vector& y)
{
    y[0] = 0.00054201928465471595*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 0.00054201928465471595*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = 0.00054201928465471595*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) - 0.00015568699431861544;
}
void J1_(const Vector& x, Vector& y)
{
    y[0] = 1.3279472474040542e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003)*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 9.2703662694170545e-5*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 1.3279472474040542e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003)*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]) + 9.2703662694170545e-5*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = -0.19600000000000001*(-6.7752410581839497e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 1.9460874289826929e-6)*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.00060221412900000003) + 9.2703662694170545e-5*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}
void J2_(const Vector& x, Vector& y)
{
    y[0] = 1.3753739348113418e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.00060221412900000003)*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 9.6014507790390918e-5*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 1.3753739348113418e-6*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.00060221412900000003)*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]) - 9.6014507790390918e-5*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = -0.20300000000000001*(-6.7752410581839497e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 1.9460874289826929e-6)*(0.00030110706450000002*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.00060221412900000003) - 9.6014507790390918e-5*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

void adv1(const Vector& x, Vector& y)
{
    y[0] = -1.3279472474040542e-6*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = -1.3279472474040542e-6*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = -1.3279472474040542e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 3.8143313608060783e-7;
}
void adv2(const Vector& x, Vector& y)
{
    y[0] = 1.3753739348113418e-6*sin(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 1.3753739348113418e-6*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = 1.3753739348113418e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) - 3.9505574808348666e-7;
}
double div_adv1(const Vector& x)
{
    return -6.2578039751994586e-6*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
double div_adv2(const Vector& x)
{
    return 6.4812969743137253e-6*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

FunctionCoefficient phi_exact(phi_exact_);
FunctionCoefficient c1_exact(c1_exact_);
FunctionCoefficient c2_exact(c2_exact_);
FunctionCoefficient f1_analytic(f1_analytic_);
FunctionCoefficient f2_analytic(f2_analytic_);
VectorFunctionCoefficient J (3, J_);
VectorFunctionCoefficient J1(3, J1_);
VectorFunctionCoefficient J2(3, J2_);
FunctionCoefficient div_Adv1(div_adv1);
FunctionCoefficient div_Adv2(div_adv2);
#elif defined(Nano_SCALE)
double phi_exact_(Vector& x)
{
    return 19.4608742898269*x[2] + 0.431325242019627*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 19.4608742898269;
}
double c1_exact_(Vector& x)
{
    return 0.3011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.602214129;
}
double c2_exact_(Vector& x)
{
    return -0.3011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.602214129;
}
double f1_analytic_(Vector& x)
{
    return 0.000927036626941705*(-0.677524105818395*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 0.00625780397519946*(0.3011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.602214129)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.00062808966172958*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.00062808966172958*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.00436855718521337*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
double f2_analytic_(Vector& x)
{
    return 0.000960145077903909*(-0.677524105818395*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 0.00648129697431372*(0.3011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.602214129)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.000650521435362779*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.000650521435362779*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.00452457708468528*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
FunctionCoefficient phi_exact(phi_exact_);
VectorFunctionCoefficient grad_phi_exact(3, grad_phi_exact_);
FunctionCoefficient c1_exact(c1_exact_);
FunctionCoefficient c2_exact(c2_exact_);
FunctionCoefficient f1_analytic(f1_analytic_);
FunctionCoefficient f2_analytic(f2_analytic_);
#elif defined(Micro_SCALE) and !defined(PhysicalModel)
double phi_exact_(Vector& x)
{
    return 19.4608742898269*x[2] + 431325242019627.0*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 19.4608742898269;
}
double c1_exact_(Vector& x)
{
    return 301107064.5*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 602214129.0;
}
double c2_exact_(Vector& x)
{
    return -301107064.5*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 602214129.0;
}
double f1_analytic_(Vector& x)
{
    return 0.927036626941705*(-677524105818395.0*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 6257803.97519946*(301107064.5*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 602214129.0)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 628089661729580.0*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 628089661729580.0*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 4.36855718521337*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
double f2_analytic_(Vector& x)
{
    return 0.960145077903909*(-677524105818395.0*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 6481296.97431372*(301107064.5*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 602214129.0)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 650521435362779.0*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 650521435362779.0*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 4.52457708468528*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
FunctionCoefficient phi_exact(phi_exact_);
VectorFunctionCoefficient grad_phi_exact(3, grad_phi_exact_);
FunctionCoefficient c1_exact(c1_exact_);
FunctionCoefficient c2_exact(c2_exact_);
FunctionCoefficient f1_analytic(f1_analytic_);
FunctionCoefficient f2_analytic(f2_analytic_);
#endif


// ------------------------- 一些辅助变量(避免在main函数里面定义) ------------------------
bool use_np1spd             = false;
bool use_np2spd             = false;

// if not use PETSc, use below parameters for solvers
double phi_solver_atol = 1E-20;
double phi_solver_rtol = 1E-14;
int phi_solver_maxiter = 10000;
int phi_solver_printlv = -1;

double np1_solver_atol = 1E-20;
double np1_solver_rtol = 1E-14;
int np1_solver_maxiter = 10000;
int np1_solver_printlv = -1;

double np2_solver_atol = 1E-20;
double np2_solver_rtol = 1E-14;
int np2_solver_maxiter = 10000;
int np2_solver_printlv = -1;

const double newton_rtol   = 1.0e-8;
const double newton_atol   = 1.0e-20;
const double newton_maxitr = 50;
const int newton_printlvl  = 0;

const double jacobi_rtol = 1.0e-8;
const double jacobi_atol = 1.0e-20;
const int jacobi_maxiter = 10000;
const int jacobi_printlv = -1;

ConstantCoefficient sigma_coeff(sigma);
ConstantCoefficient kappa_coeff(kappa);
ConstantCoefficient zero(0.0);
ConstantCoefficient one(1.0);
ConstantCoefficient two(2.0);
ConstantCoefficient neg(-1.0);
ProductCoefficient epsilon_water_prod_kappa(epsilon_water, kappa_coeff);
ProductCoefficient neg_epsilon_water(neg, epsilon_water);
ProductCoefficient sigma_epsilon_water(sigma_coeff, epsilon_water);
ProductCoefficient neg_D1(neg, D_K_);
ProductCoefficient sigma_D1(sigma_coeff, D_K_);
ProductCoefficient neg_D2(neg, D_Cl_);
ProductCoefficient sigma_D2(sigma_coeff, D_Cl_);
ProductCoefficient neg_D1_z1(neg, D_K_prod_v_K);
ProductCoefficient sigma_D1_z1(sigma_coeff, D_K_prod_v_K);
ProductCoefficient neg_D2_z2(neg, D_Cl_prod_v_Cl);
ProductCoefficient sigma_D2_z2(sigma_coeff, D_Cl_prod_v_Cl);
ProductCoefficient kappa_D1(kappa_coeff, D_K_);
ProductCoefficient kappa_D2(kappa_coeff, D_Cl_);
ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl(neg, alpha2_prod_alpha3_prod_v_Cl);
ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K(neg, alpha2_prod_alpha3_prod_v_K);

ProductCoefficient neg_div_Adv1(neg, div_Adv1);
ProductCoefficient neg_div_Adv2(neg, div_Adv2);
#endif