#ifndef MFEM_PNP_PROTEIN_TIMEDEPENDENT_HPP
#define MFEM_PNP_PROTEIN_TIMEDEPENDENT_HPP

#include <cassert>
#include "mfem.hpp"
#include "../utils/PQR_GreenFunc_PhysicalParameters.hpp"
#include "../utils/StdFunctionCoefficient.hpp"
#include "../utils/mfem_utils.hpp"
using namespace std;
using namespace mfem;


int p_order                     = 1; //有限元基函数的多项式次数
bool nonzero_NewtonInitial      = false;
int nonzero_maxGummel           = 3;
const char* Linearize           = "newton"; // newton, gummel
const char* Discretize          = "cg"; // cg, dg
const char* AdvecStable         = "eafe"; // none, supg, eafe
const char* prec_type           = "block"; // preconditioner for Newton discretization: block, uzawa, simple
const char* options_src         = "../pnp_data/protein_newton_amg";
const char* output              = "";
bool self_debug                 = false; // 针对特定的条件进行一些检测
bool visualize                  = false;
bool local_conservation         = false;
bool show_peclet                = false;
double relax                    = 0.2; //松弛因子: relax * phi^{k-1} + (1 - relax) * phi^k -> phi^k, 浓度 c_2^k 做同样处理. 取0表示不用松弛方法.
double schur_alpha1             = 1.0; // schur = A - alpha1 B1 A1^-1 C1 - alpha2 B2 A2^-1 C2, 这个alpha1就是该参数
double schur_alpha2             = 1.0;
int refine_mesh                 = 0; // 初始网格加密次数
int refine_time                 = 0;   // "加密时间次数"
double time_scale               = 1.0; // 类似网格加密(h -> 0.5 * h): dt -> time_scale * dt
bool TimeConvergRate            = false;
bool SpaceConvergRate           = false; // 利用解析解计算误差阶
bool SpaceConvergRate_Change_dt = false; // 为了计算误差: error = c1 dt + c2 h^2, 是否把dt设置为h^2的倍数?
double Change_dt_factor         = 1.0; // dt = factor * h^2
int ode_type                    = 1; // 1: backward Euler; 11: forward Euler
double t_init                   = 0.0; // 初始时间, 单位 µs
double t_final                  = 300; // 最后时间
double t_stepsize               = 100; // 时间步长

const int Gummel_max_iters      = 20;
const double Gummel_rel_tol     = 1e-10;
const double TOL                = 1e-10;

bool paraview                   = false;
const char* paraview_dir        = "";
bool skip_zero_entries          = false;
int verbose                     = 1;
double sigma                    = -1.0; // symmetric parameter for DG
double kappa                    = 10.0; // penalty parameter for DG

//const char* mesh_file      = "../pnp_data/1bl8_tu.msh"; // 带有蛋白的网格,与PQR文件必须匹配
//const char* pqr_file       = "../pnp_data/1bl8.pqr"; // PQR文件,与网格文件必须匹配
const char* mesh_file      = "../pnp_data/1MAG_2.msh"; // 带有蛋白的网格,与PQR文件必须匹配
const char* pqr_file       = "../pnp_data/1MAG.pqr"; // PQR文件,与网格文件必须匹配
const int protein_marker   = 1; // 蛋白区域   单元编号
const int water_marker     = 2; // 水溶液区域 单元编号
const int interface_marker = 9; // 蛋白区域 和 水溶液区域 的交界面编号
const int top_marker       = 8; // 计算区域 顶部边界 的编号
const int bottom_marker    = 7; // 计算区域 底部边界 的编号
const int Gamma_m_marker   = 5; // 蛋白区域的 外边界 的编号

double phi_top     = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 上边界是 Dirichlet, 乘以alpha1进行无量纲化
double phi_bottom  = 0.0 * alpha1; // 国际单位V, 电势在计算区域的 下边界是 Dirichlet
double c1_top      = 0.1 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 上边界是 Dirichlet,乘以alpha2是把mol/L换成Angstrom,单位统一
double c1_bottom   = 0.0 * alpha3; // 国际单位mol/L, K+阳离子在计算区域的 下边界是 Dirichlet
double c2_top      = 0.1 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 上边界是 Dirichlet
double c2_bottom   = 0.0 * alpha3; // 国际单位mol/L, Cl-阴离子在计算区域的 下边界是 Dirichlet

ConstantCoefficient phi_D_top_coeff(phi_top);
ConstantCoefficient phi_D_bottom_coeff(phi_bottom);
ConstantCoefficient c1_D_top_coeff(c1_top);
ConstantCoefficient c1_D_bottom_coeff(c1_bottom);
ConstantCoefficient c2_D_top_coeff(c2_top);
ConstantCoefficient c2_D_bottom_coeff(c2_bottom);


// ---------------------------------------------------------------------------------------
//                           一些辅助变量(避免在main函数里面定义)
// ---------------------------------------------------------------------------------------
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
ProductCoefficient             protein_G(mark_protein_coeff, G_coeff); // G 其实只在蛋白区域取值：第一种奇异分解式
VectorStdFunctionCoefficient   gradG_coeff(3, gradG_func);
ScalarVectorProductCoefficient protein_gradG(mark_protein_coeff, gradG_coeff); // gradG 其实只在蛋白区域取值：第一种奇异分解式
ScalarVectorProductCoefficient neg_protein_gradG(neg, protein_gradG);
ScalarVectorProductCoefficient neg_gradG(neg, gradG_coeff);
ProductCoefficient             neg_G(neg, G_coeff);
ProductCoefficient             neg_protein_G(neg, protein_G);

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

ProductCoefficient water_alpha2_prod_alpha3_prod_v_K(mark_water_coeff, alpha2_prod_alpha3_prod_v_K);
ProductCoefficient water_alpha2_prod_alpha3_prod_v_Cl(mark_water_coeff, alpha2_prod_alpha3_prod_v_Cl);

ProductCoefficient sigma_D1(sigma_coeff, D_K_);
ProductCoefficient kappa_D1(kappa_coeff, D_K_);
ProductCoefficient sigma_D2(sigma_coeff, D_Cl_);
ProductCoefficient kappa_D2(kappa_coeff, D_Cl_);
ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);

#endif //MFEM_PNP_PROTEIN_TIMEDEPENDENT_HPP
