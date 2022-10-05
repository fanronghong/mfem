#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>
#include "mfem.hpp"
using namespace mfem;

bool Run_FEM = true;
bool Run_EAFE = true;
bool Run_SUPG = true;
bool EAFE_Only_Dump_data = false;
bool SUPG_Only_Dump_data = false;

const char* mesh_file = "../../../data/inline-tet.mesh";
int refine_times = 3;
int p_order = 1; //Lagrange有限元次数

// 程序暂时只能处理各项同性的扩散系数
double alpha = 1E-1;
double beta = 1.0E+0;
void DiffusionTensor(const Vector &x, DenseMatrix &K) {
    K(0,0) = alpha;
    K(0,1) = 0;
    K(0,2) = 0;
    K(1,0) = 0;
    K(1,1) = alpha;
    K(1, 2) = 0;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = alpha;
}
void AdvectionVector(const Vector &x, Vector &advcoeff) {
    advcoeff[0] = beta;
    advcoeff[1] = beta;
    advcoeff[2] = beta;
}

double gmres_atol = 1e-20;
double gmres_rtol = 1e-8;
int gmres_maxiter = 200000;
int gmres_printlevel = 0;


// ---------------- 下面两个函数的返回值为0是因为要用python字符串匹配 -------------------
double analytic_solution(const Vector& x){return cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]);}
double analytic_rhs(const Vector& x)
{return 3.14159265358979*sin(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]) + 3.14159265358979*sin(3.14159265358979*x[1])*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[2]) + 3.14159265358979*sin(3.14159265358979*x[2])*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1]) + 2.96088132032681*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]);}
// 扩散速度的 (-divergence), 对于advection velocity不是divergence free的时候需要
double neg_div_advection(const Vector& x){return 0;}


// 为了方便, 把某些函数定义在这里
ConstantCoefficient one(1.0);
ConstantCoefficient neg(-1.0);

FunctionCoefficient analytic_rhs_(analytic_rhs);

FunctionCoefficient analytic_solution_(analytic_solution);

MatrixFunctionCoefficient diffusion_tensor(3, DiffusionTensor);
VectorFunctionCoefficient advection_vector(3, AdvectionVector);
FunctionCoefficient neg_div_advection_(neg_div_advection);
ProductCoefficient div_adv(neg_div_advection_, neg);

#endif