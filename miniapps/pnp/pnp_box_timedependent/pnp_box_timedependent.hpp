#ifndef _PNP_BOX_HPP_
#define _PNP_BOX_HPP_

#include "mfem.hpp"
#include "../utils/PQR_GreenFunc_PhysicalParameters.hpp"
using namespace std;
using namespace mfem;

//#define SELF_VERBOSE

int p_order                     = 1; //有限元基函数的多项式次数
const char* mesh_file           = "../pnp_data/4_4_4_translate.msh";
const char* Linearize           = "newton"; // newton, gummel
const char* Discretize          = "dg"; // cg, dg
const char* prec_type           = "block"; // preconditioner for Newton discretization: block, uzawa, simple
const char* AdvecStable         = "none"; // none, eafe, supg
const char* options_src         = "../pnp_data/newton_amg";
bool zero_initial               = true; // 非线性迭代的初值是否为0
double initTol                  = 1e-3; // 为得到非线性迭代的初值所需Gummel迭代
bool local_conservation         = false;
bool visualization              = false;
bool paraview                   = false;
const char* output              = ""; // dummy参数, 为了显示bsub运行的程序名字
int max_newton                  = 20;
double relax                    = 0.2; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.
int ode_type                    = 1; // 1: backward Euler; 11: forward Euler
double t_init                   = 0.0; // 初始时间
double t_final                  = 0.003; // 最后时间
double t_stepsize               = 0.001; // 时间步长
int refine_mesh                 = 0; // 初始网格加密次数
int refine_time                 = 0;   // "加密时间次数"
double time_scale               = 1.0; // 类似网格加密(h -> 0.5 * h): dt -> time_scale * dt
bool TimeConvergRate            = false;
bool SpaceConvergRate           = false; // 利用解析解计算误差阶
bool SpaceConvergRate_Change_dt = false; // 为了计算误差: error = c1 dt + c2 h^2, 是否把dt设置为h^2的倍数?
double Change_dt_factor         = 1; // dt = factor * h^2
const int skip_zero_entries     = 0; // 为了保证某些矩阵的sparsity pattern一致
int mpi_debug                   = 1;
int verbose                     = 2; // 数字越大输出越多: 0,1,2
double sigma                    = -1.0; // symmetric parameter for DG
bool symmetry_with_boundary     = true; // 为true就可以不单独考虑weak Dirichlet.
double kappa                    = 0; // penalty parameter for DG
bool penalty_with_boundary      = true;

const int bottom_attr           = 1;
const int top_attr              = 6;
const int left_attr             = 5;
const int front_attr            = 2;
const int back_attr             = 4;
const int right_attr            = 3;

const int Gummel_max_iters      = 50;
double Gummel_rel_tol           = 1e-8;
const double TOL                = 1e-20;

/* 可以定义如下模型参数: 前三个宏定义参数在其他头文件定义
 * Angstrom_SCALE: 埃米尺度
 * Nano_SCALE: 纳米尺度
 * Micro_SCALE: 微米尺度
 * */
#define Angstrom_SCALE

#if defined(Angstrom_SCALE)
//======> Time Dependent Analytic Solutions:
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

double dphidt_exact_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double dc1dt_exact_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double dc2dt_exact_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f0_analytic_time(const Vector& x, double t)
{
    return 592.17626406543945*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f1_analytic_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.48361061565344232*pow(sin(1.570796326795*x[0]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.48361061565344232*pow(sin(1.570796326795*x[1]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.48361061565344232*pow(sin(1.570796326795*x[2]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2) + 1.450831846960327*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 1.450831846960327*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}

double f2_analytic_time(const Vector& x, double t)
{
    return -3.1415926535900001*sin(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.50088242335535094*pow(sin(1.570796326795*x[0]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.50088242335535094*pow(sin(1.570796326795*x[1]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.50088242335535094*pow(sin(1.570796326795*x[2]), 2)*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2) - 1.5026472700660527*pow(cos(3.1415926535900001*t), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) + 1.5026472700660527*cos(3.1415926535900001*t)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
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
FunctionCoefficient dc1dt_exact(dc1dt_exact_time);
FunctionCoefficient dc2dt_exact(dc2dt_exact_time);
FunctionCoefficient f0_analytic(f0_analytic_time);
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

ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
ProductCoefficient neg_sigma_D_K_v_K(neg, sigma_D_K_v_K);
ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);
ProductCoefficient neg_sigma_D_Cl_v_Cl(neg, sigma_D_Cl_v_Cl);
ProductCoefficient neg_D_K_v_K(neg, D_K_prod_v_K);
ProductCoefficient neg_D_Cl_v_Cl(neg, D_Cl_prod_v_Cl);

ProductCoefficient neg_div_Adv1(neg, div_Adv1);
ProductCoefficient neg_div_Adv2(neg, div_Adv2);
#endif