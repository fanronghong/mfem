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
const char* Discretize      = "cg"; // cg, dg
const char* AdvecStable     = "none"; // none, eafe, supg
const char* options_src     = "./pnp_box_petsc_opts";
bool ComputeConvergenceRate = false; // 利用解析解计算误差阶
bool local_conservation     = false;
bool paraview               = false;
const char* output          = NULL;
int max_newton              = 20;
double relax                = 0.2; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.
int ode_type                = 11; // 1: backward Euler; 11: forward Euler
double init_t               = 0.0; // 初始时间
double dt                   = 0.03;// 时间步长
double t_final              = 0.1; // 最后时间
const int skip_zero_entries = 0; // 为了保证某些矩阵的sparsity pattern一致
int mpi_debug               = 0;

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
double phi_exact_time(const Vector& x, double t)
{
    return cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double c1_exact_time(const Vector& x, double t)
{
    return cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double c2_exact_time(const Vector& x, double t)
{
    return cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f_analytic_time(const Vector& x, double t)
{
    return 592.17626406543945*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f1_analytic_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.48361061565344232*pow(sin(1.570796326795*x[0]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.48361061565344232*pow(sin(1.570796326795*x[1]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.48361061565344232*pow(sin(1.570796326795*x[2]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2) - 1.450831846960327*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 1.450831846960327*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f2_analytic_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.50088242335535094*pow(sin(1.570796326795*x[0]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.50088242335535094*pow(sin(1.570796326795*x[1]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.50088242335535094*pow(sin(1.570796326795*x[2]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2) + 1.5026472700660527*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 1.5026472700660527*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

void J_time(const Vector& x, double t, Vector& y)
{
    y[0] = 125.6637061436*sin(1.570796326795*x[0])*cos(3.1415926535900001*t)*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 125.6637061436*sin(1.570796326795*x[1])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = 125.6637061436*sin(1.570796326795*x[2])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

void J1_time(const Vector& x, double t, Vector& y)
{
    y[0] = 0.30787608005182004*sin(1.570796326795*x[0])*pow(cos(3.1415926535900001*t), 2)*cos(1.570796326795*x[0])*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.30787608005182004*sin(1.570796326795*x[0])*cos(3.1415926535900001*t)*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 0.30787608005182004*sin(1.570796326795*x[1])*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*cos(1.570796326795*x[1])*pow(cos(1.570796326795*x[2]), 2) + 0.30787608005182004*sin(1.570796326795*x[1])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = 0.30787608005182004*sin(1.570796326795*x[2])*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*cos(1.570796326795*x[2]) + 0.30787608005182004*sin(1.570796326795*x[2])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

void J2_time(const Vector& x, double t, Vector& y)
{
    y[0] = -0.31887165433938502*sin(1.570796326795*x[0])*pow(cos(3.1415926535900001*t), 2)*cos(1.570796326795*x[0])*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.31887165433938502*sin(1.570796326795*x[0])*cos(3.1415926535900001*t)*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = -0.31887165433938502*sin(1.570796326795*x[1])*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*cos(1.570796326795*x[1])*pow(cos(1.570796326795*x[2]), 2) + 0.31887165433938502*sin(1.570796326795*x[1])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = -0.31887165433938502*sin(1.570796326795*x[2])*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*cos(1.570796326795*x[2]) + 0.31887165433938502*sin(1.570796326795*x[2])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

void adv1_time(const Vector& x, double t, Vector& y)
{
    y[0] = -0.30787608005182004*sin(1.570796326795*x[0])*cos(3.1415926535900001*t)*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = -0.30787608005182004*sin(1.570796326795*x[1])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = -0.30787608005182004*sin(1.570796326795*x[2])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

void adv2_time(const Vector& x, double t, Vector& y)
{
    y[0] = 0.31887165433938502*sin(1.570796326795*x[0])*cos(3.1415926535900001*t)*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
    y[1] = 0.31887165433938502*sin(1.570796326795*x[1])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[2]);
    y[2] = 0.31887165433938502*sin(1.570796326795*x[2])*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]);
}

double div_adv1_time(const Vector& x, double t)
{
    return -1.450831846960327*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double div_adv2_time(const Vector& x, double t)
{
    return 1.5026472700660527*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}


FunctionCoefficient phi_exact(phi_exact_time);
FunctionCoefficient c1_exact(c1_exact_time);
FunctionCoefficient c2_exact(c2_exact_time);
FunctionCoefficient f1_analytic(f1_analytic_time);
FunctionCoefficient f2_analytic(f2_analytic_time);
VectorFunctionCoefficient J (3, J_time);
VectorFunctionCoefficient J1(3, J1_time);
VectorFunctionCoefficient J2(3, J2_time);
FunctionCoefficient div_Adv1(div_adv1_time);
FunctionCoefficient div_Adv2(div_adv2_time);
#elif defined(Nano_SCALE)
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