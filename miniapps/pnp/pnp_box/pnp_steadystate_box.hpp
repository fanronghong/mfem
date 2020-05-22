//
// Created by fan on 2020/4/17.
//
#ifndef _PNP_BOX_HPP_
#define _PNP_BOX_HPP_

#include "mfem.hpp"
#include "../utils/PQR_GreenFunc_PhysicalParameters.hpp"
using namespace std;
using namespace mfem;

//#define SELF_VERBOSE

const char* mesh_file       = "./translate_translate.msh";
int refine_times            = 3;
const int bottom_attr       = 1;
const int top_attr          = 6;
const int left_attr         = 5;
const int front_attr        = 2;
const int back_attr         = 4;
const int right_attr        = 3;

const int p_order           = 1; //有限元基函数的多项式次数
const int Gummel_max_iters  = 20;
const double Gummel_rel_tol = 1e-8;
const double TOL            = 1e-10;
const char* options_src     = "./pnp_steadystate_box_petsc.opts";

/* 可以定义如下模型参数: 前三个宏定义参数在其他头文件定义
 * Angstrom_SCALE: 埃米尺度
 * Nano_SCALE: 纳米尺度
 * Micro_SCALE: 微米尺度
 * PhysicalModel: 真实的模型(没有蛋白区域), 上下边界是Dirichlet, 四周是Neumann边界
 * */
#define PhysicalModel

#if defined(PhysicalModel)
// use Dirichlet bdc on top and bottom, zero Neumann for other boundaries
double phi_top     = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 上边界是 Dirichlet, 乘以alpha1进行无量纲化
double phi_bottom  = 0.5 * alpha1; // 国际单位V, 电势在计算区域的 下边界是 Dirichlet

double c1_top      = 0.1 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 上边界是 Dirichlet,乘以alpha2是把mol/L换成Angstrom,单位统一
double c1_bottom   = 2.0 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 下边界是 Dirichlet

double c2_top      = 0.1 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 上边界是 Dirichlet
double c2_bottom   = 2.0 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 下边界是 Dirichlet

double phi_D_func(const Vector& x)
{
    if (abs(x[2] - 1.0) < 1E-10) return phi_top;
    else if (abs(x[2] + 1.0) < 1E-10) return phi_bottom;
}
double c1_D_func(const Vector& x)
{
    if (abs(x[2] - 1.0) < 1E-10) return c1_top;
    else if (abs(x[2] + 1.0) < 1E-10) return c1_bottom;
}
double c2_D_func(const Vector& x)
{
    if (abs(x[2] - 1.0) < 1E-10) return c2_top;
    else if (abs(x[2] + 1.0) < 1E-10) return c2_bottom;
}
FunctionCoefficient phi_D_coeff(phi_D_func);
FunctionCoefficient c1_D_coeff (c1_D_func);
FunctionCoefficient c2_D_coeff (c2_D_func);

#elif defined(Angstrom_SCALE) and !defined(PhysicalModel)
#define COMPUTE_CONVERGENCE_RATE   //运行所有代码内部自己添加的assert检查. Note: 不要修改下面的输入参数, 否则会造成程序中的很多assert不能通过!
double phi_exact_(Vector& x)
{
    return 19.4608742898269*x[2] + 4.31325242019627e-6*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 19.4608742898269;
}
double c1_exact_(Vector& x)
{
    return 0.0003011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.000602214129;
}
double c2_exact_(Vector& x)
{
    return -0.0003011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.000602214129;
}
double f1_analytic_(Vector& x)
{
    return 9.27036626941705e-5*(-6.77524105818395e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 6.25780397519946e-6*(0.0003011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) + 0.000602214129)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 6.2808966172958e-10*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 6.2808966172958e-10*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) + 0.000436855718521337*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
double f2_analytic_(Vector& x)
{
    return 9.60145077903909e-5*(-6.77524105818395e-6*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 19.4608742898269)*sin(1.570796326795*x[2])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) + 6.48129697431372e-6*(0.0003011070645*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 0.000602214129)*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]) - 6.50521435362779e-10*pow(sin(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[2]), 2) - 6.50521435362779e-10*pow(sin(1.570796326795*x[1]), 2)*pow(cos(1.570796326795*x[0]), 2)*pow(cos(1.570796326795*x[2]), 2) - 0.000452457708468528*cos(1.570796326795*x[0])*cos(1.570796326795*x[1])*cos(1.570796326795*x[2]);
}
FunctionCoefficient phi_exact(phi_exact_);
FunctionCoefficient c1_exact(c1_exact_);
FunctionCoefficient c2_exact(c2_exact_);
FunctionCoefficient f1_analytic(f1_analytic_);
FunctionCoefficient f2_analytic(f2_analytic_);
#elif defined(Nano_SCALE) and !defined(PhysicalModel)
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
const double relax_phi = 0.2; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.
const double relax_c1  = 0.2;
const double relax_c2  = 0.2;

double sigma = -1.0;
double kappa = 200;

// 必须足够精确
double phi_solver_atol = 1E-20;
double phi_solver_rtol = 1E-14;
int phi_solver_maxiter = 1000;
int phi_solver_printlv = -1;

double np1_solver_atol = 1E-20;
double np1_solver_rtol = 1E-14;
int np1_solver_maxiter = 1000;
int np1_solver_printlv = -1;

double np2_solver_atol = 1E-20;
double np2_solver_rtol = 1E-14;
int np2_solver_maxiter = 1000;
int np2_solver_printlv = -1;

const double newton_rtol   = 1.0e-8;
const double newton_atol   = 1.0e-20;
const double newton_maxitr = 20;
const int newton_printlvl  = 1;

const double jacobi_rtol = 1.0e-8;
const double jacobi_atol = 1.0e-20;
const int jacobi_maxiter = 1000;
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

#endif