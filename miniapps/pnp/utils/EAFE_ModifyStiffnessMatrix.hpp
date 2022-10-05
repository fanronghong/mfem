#ifndef LEARN_MFEM_EAFE_MODIFYSTIFFNESSMATRIX_HPP
#define LEARN_MFEM_EAFE_MODIFYSTIFFNESSMATRIX_HPP

#include <cassert>
#include <iostream>

#include "mfem.hpp"
#include "mfem_utils.hpp"
using namespace mfem;
using namespace std;

double bernoulli_EAFE_ModifyStiffnessMatrix(const double z)
{
    // returns B(z) = z/(exp(z)-1)
    double tolb = 1e-12;
    double zlarge = 1e+10;
    if (fabs(z) < tolb)
        return (1.0 - z * 0.5); // around 0 this is the asymptotic;
    else if (z < zlarge)
        return (z / (exp(z) - 1.0));
    else //z>tlarge this is zero pretty much
        return 0.0;
}


/* Assemble -div (a(x) grad(u) + (b . u)).
 * Note cannot be -div(a(x) grad(u)) + b.grad(u) !!!
 * A is the stiffness matrix of (grad(u), grad(v)), diffusion coefficient is 1.0 !!!
 * A must be from discretizing (grad(u), grad(v)) with triangle/tetrahedron Lagrange 1 order (P1).
 * Diff is used to compute diffusion coefficient at a point
 * Adv is used to compute advection coefficient at a point
 */
void EAFE_Modify(Mesh& mesh, SparseMatrix& A,
        void (*Diff)(const Vector&, DenseMatrix&),
        void (*Adv)(const Vector&, Vector&))
{
    int* I = A.GetI(); //CSR矩阵的row offsets
    int* J = A.GetJ(); //CSR矩阵的column indices
    double* data = A.GetData(); //CSR矩阵的non-zero values

    int nrow = A.Size(); //矩阵的Height. A就是要进行EAFE修改的刚度矩阵
    int nv = mesh.GetNV();
    int dim = mesh.Dimension();
    assert(dim == 2 || dim == 3);

    Vector diag0(nrow);
    diag0 = 0.0;

    Vector adv(dim), mid(dim), tangential(dim);
    double xi, yi, zi, xj, yj, zj, alpha, beta;

    Vector coors(nv*dim);
    mesh.GetVertices(coors);

    for (int i=0; i<nrow; ++i)
    {
        xi = coors[i]; // 第i个vertex的坐标向量: 矩阵有nrow行, 那么就有nrow个vertices.
        yi = coors[nv+i];
        if (dim == 3) zi = coors[2*nv+i];

        //对CSR矩阵的第i行非0元素循环, jk是第i行非0元素在non-zero vals的索引,同时这个索引对应到J中就是非0元素的column index
        for (int jk=I[i]; jk<I[i+1]; jk++) //jk就是稀疏矩阵第i行的所有非零元素在向量J或者data中的索引
        {
            int j = J[jk]; //对应非0元素的column index
            if(i != j) // 非对角元素
            {
                xj = coors[j]; // 第j个vertex的坐标向量
                yj = coors[nv+j];
                if (dim == 3) zj = coors[2*nv+j];

                // compute the advection field at the middle of the edge and then the bernoulli function
                //在vertices的所有坐标值组成的向量是xy, xy中索引为i,j的两个顶点是相连的?因为只有相连才会在总刚度矩阵里面形成非零的单刚
                tangential[0] = xi - xj; //切向量
                tangential[1] = yi - yj;
                if (dim == 3) tangential[2] = zi - zj;

                mid[0] = (xi + xj) * 0.5; // 中点
                mid[1] = (yi + yj) * 0.5;
                if (dim == 3) mid[2] = (zi + zj) * 0.5;

                Adv(mid, adv); //计算中点处的对流速度(记为adv)
                //对流速度与切向量的内积.参考[1]:(3.24)式中的 \beta \cdot \tau_E
                beta = adv[0]*tangential[0] + adv[1]*tangential[1];
                if (dim == 3) beta += adv[2]*tangential[2];

                // diffusion coefficient for the flux J = a(x)\nabla u + \beta u;
                DenseMatrix diff_mat(dim);
                Diff(mid, diff_mat);
                // 这条边中点处的扩散系数, 其约等于真正的扩散系数在这条边上的调和平均. 参考[1]:(3.24)式中的 \tilde{\alpha_E}
                // 2 ways to choose alpha: both pass test
//                alpha = diff_mat(0, 0); //各项同性的diffusion tensor.
                alpha = diff_mat.MaxMaxNorm(); // 这里的范数肯定有其他选择

                /* alpha = a(xmid) \approx harmonic_average = |e|/(int_e 1/a); should be computed by quadrature in general for 1/a(x).
                 * a_{ij} = B(beta.t_e / harmonic_average) * harmonic_average * omega_e;
                 *	  for (i,j):    B(beta.t_e / harmonic_average);
                 *	  for (j,i):    B(-beta.t_e / harmonic_average), the minus comes from is because t_e=i-j;
                 */
                double temp = data[jk]; // for the 2nd way
                data[jk] *= alpha * bernoulli_EAFE_ModifyStiffnessMatrix(beta/alpha); //用EAFE修改当前这个非0元素, data是CSR矩阵的non-zero values.bte里面包含了正负号

                // First way: 主对角元素为非对角元素和的相反数
//                diag0[i] -= data[jk]; // the diagonal is equal to the negative column sum;

                // Second way: 和第一种方式的唯一区别就是主对角元素不是等于非对角元素的和的相反数. 严格来讲第二种方式才是对的(按照Xu的论文)
                diag0[i] += -1.0 * temp * alpha * bernoulli_EAFE_ModifyStiffnessMatrix(-1.0 * beta/alpha); //
            }
        }
    }

    // 主对角线上的元素: 两种修改方式对应的处理主对角元素的代码完全相同
    for (int i=0; i<nrow; i++)
    {
        for (int jk = I[i]; jk<I[i+1]; jk++)
        {
            //I[i]就是第CSR矩阵的第i行(从0开始)的第一个非0元素在non-zero vals这个向量的索引, 下面的J[jk]就是这个非0元素的列指标
            if(i == J[jk]) //行列指标相同,则A[jk]就是对角线上的元素
                data[jk] = diag0[i];
        }
    }
}

void EAFE_Modify(Mesh& mesh, SparseMatrix& A,
        Coefficient& Diff,
        VectorCoefficient& Adv)
{
    int* I = A.GetI(); //CSR矩阵的row offsets
    int* J = A.GetJ(); //CSR矩阵的column indices
    double* data = A.GetData(); //CSR矩阵的non-zero values

    int nrow = A.Size(); //矩阵的Height. A就是要进行EAFE修改的刚度矩阵
    int nv = mesh.GetNV();
    int dim = mesh.Dimension();
    assert(dim == 2 || dim == 3);

    Vector diag0(nrow);
    diag0 = 0.0;

    Vector adv(dim), mid(dim), tangential(dim);
    double xi, yi, zi, xj, yj, zj, alpha, beta;

    Vector coors(nv*dim);
    mesh.GetVertices(coors);

    for (int i=0; i<nrow; ++i)
    {
        xi = coors[i]; // 第i个vertex的坐标向量: 矩阵有nrow行, 那么就有nrow个vertices.
        yi = coors[nv+i];
        if (dim == 3) zi = coors[2*nv+i];

        //对CSR矩阵的第i行非0元素循环, jk是第i行非0元素在non-zero vals的索引,同时这个索引对应到J中就是非0元素的column index
        for (int jk=I[i]; jk<I[i+1]; jk++) //jk就是稀疏矩阵第i行的所有非零元素在向量J或者data中的索引
        {
            int j = J[jk]; //对应非0元素的column index
            if(i != j) // 非对角元素
            {
                xj = coors[j]; // 第j个vertex的坐标向量
                yj = coors[nv+j];
                if (dim == 3) zj = coors[2*nv+j];

                // compute the advection field at the middle of the edge and then the bernoulli function
                //在vertices的所有坐标值组成的向量是xy, xy中索引为i,j的两个顶点是相连的?因为只有相连才会在总刚度矩阵里面形成非零的单刚
                tangential[0] = xi - xj; //切向量
                tangential[1] = yi - yj;
                if (dim == 3) tangential[2] = zi - zj;

                mid[0] = (xi + xj) * 0.5; // 中点
                mid[1] = (yi + yj) * 0.5;
                if (dim == 3) mid[2] = (zi + zj) * 0.5;

                ElementTransformation* tran;
                Array<IntegrationPoint> ips;
                Array<int> elem_ids;
                {
                    if (0) {
                        DenseMatrix physical_point(dim, 1); // only 1 integrate point
                        for (int l = 0; l < dim; ++l) physical_point(l, 0) = mid[l];

                        mesh.FindPoints(physical_point, elem_ids, ips);
                        elem_ids.Print(cout << "elem_ids: ");
                        ips.Print(cout << "ips: ");

                        tran = mesh.GetElementTransformation(elem_ids[0]); // only 1 integration point
                        tran->SetIntPoint(&(ips[0]));
                    }
                    else {
                        for (int m=0; m<mesh.GetNE(); ++m) {
                            tran = mesh.GetElementTransformation(m);
                            ips.SetSize(1);
                            elem_ids.SetSize(1);

                            InverseElementTransformation invtran(tran);
                            int ret = invtran.Transform(mid, ips[0]);
                            if (ret == 0) {
                                elem_ids[0] = m;
                                break;
                            }
                        }
                    }
                }

                Adv.Eval(adv, *tran, ips[0]); //计算中点处的对流速度(记为adv)
                //对流速度与切向量的内积.参考[1]:(3.24)式中的 \beta \cdot \tau_E
                beta = adv[0]*tangential[0] + adv[1]*tangential[1];
                if (dim == 3) beta += adv[2]*tangential[2];

                // diffusion coefficient for the flux J = a(x)\nabla u + \beta u;
                alpha = Diff.Eval(*tran, ips[0]);

                /* alpha = a(xmid) \approx harmonic_average = |e|/(int_e 1/a); should be computed by quadrature in general for 1/a(x).
                 * a_{ij} = B(beta.t_e / harmonic_average) * harmonic_average * omega_e;
                 *	  for (i,j):    B(beta.t_e / harmonic_average);
                 *	  for (j,i):    B(-beta.t_e / harmonic_average), the minus comes from is because t_e=i-j;
                 */
                double temp = data[jk]; // for the 2nd way
                data[jk] *= alpha * bernoulli_EAFE_ModifyStiffnessMatrix(beta/alpha); //用EAFE修改当前这个非0元素, data是CSR矩阵的non-zero values.bte里面包含了正负号

                // First way: 主对角元素为非对角元素和的相反数
//                diag0[i] -= data[jk]; // the diagonal is equal to the negative column sum;

                // Second way: 和第一种方式的唯一区别就是主对角元素不是等于非对角元素的和的相反数. 严格来讲第二种方式才是对的(按照Xu的论文)
                diag0[i] += -1.0 * temp * alpha * bernoulli_EAFE_ModifyStiffnessMatrix(-1.0 * beta/alpha); //
            }
        }
    }

    // 主对角线上的元素: 两种修改方式对应的处理主对角元素的代码完全相同
    for (int i=0; i<nrow; i++)
    {
        for (int jk = I[i]; jk<I[i+1]; jk++)
        {
            //I[i]就是第CSR矩阵的第i行(从0开始)的第一个非0元素在non-zero vals这个向量的索引, 下面的J[jk]就是这个非0元素的列指标
            if(i == J[jk]) //行列指标相同,则A[jk]就是对角线上的元素
                data[jk] = diag0[i];
        }
    }
}


namespace _EAFE_ModifyStiffnessMatrix
{
// 2D test
double analytic_solution_2D(const Vector& x){return sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]);}
double analytic_rhs_2D(const Vector& x){return -6.28318530717959*x[0]*(-x[0] + 1)*(2*x[1] - 1)*cos(6.28318530717959*x[0])*cos(6.28318530717959*x[1]) + x[0]*(2*x[1] - 1)*sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]) + 6.28318530717959*x[1]*(-2*x[0] + 1)*(-x[1] + 1)*sin(6.28318530717959*x[0])*sin(6.28318530717959*x[1]) + x[1]*(-2*x[0] + 1)*sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]) - (-2*x[0] + 1)*(-x[1] + 1)*sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]) - (-x[0] + 1)*(2*x[1] - 1)*sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]) + 0.789568352087149*sin(6.28318530717959*x[0])*cos(6.28318530717959*x[1]);}
double div_advection_2D(const Vector& x){return x[0]*(2*x[1] - 1) + x[1]*(-2*x[0] + 1) - (-2*x[0] + 1)*(-x[1] + 1) - (-x[0] + 1)*(2*x[1] - 1);} // 扩散速度的 (-divergence)
void DiffusionTensor_2D(const Vector &x, DenseMatrix &K) {
    K(0,0) = 1e-2;
    K(0,1) = 0;
    K(1,0) = 0;
    K(1,1) = 1e-2;
}
void AdvectionVector_2D(const Vector &x, Vector &advcoeff) {
    advcoeff[0] = x[0] * (1 - x[0]) * (2 * x[1] - 1);
    advcoeff[1] = -(2 * x[0] - 1) * x[1] * (1 - x[1]);
}
void EAFE_advec_diffu_2D(Mesh& mesh, Array<double>& L2norms, Array<double>& meshsizes)
{
    int dim = mesh.Dimension();

    H1_FECollection h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    if (mesh.bdr_attributes.Size()) {
        ess_bdr = 1;
        h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm lf(&h1_space);
    FunctionCoefficient analytic_rhs_(analytic_rhs_2D);
    lf.AddDomainIntegrator(new DomainLFIntegrator(analytic_rhs_));
    lf.Assemble();

    BilinearForm blf(&h1_space);
    //EAFE基本的刚度矩阵只需要Poisson方程,后面修改的就是这个Poisson方程的刚度矩阵,参考[1],(3.24)式
    blf.AddDomainIntegrator(new DiffusionIntegrator);
    blf.Assemble();
    blf.Finalize();

    GridFunction uh(&h1_space);
    FunctionCoefficient analytic_solution_(analytic_solution_2D);
    uh.ProjectCoefficient(analytic_solution_);//使得uh满足边界条件,必须

    SparseMatrix& A = blf.SpMat();
    Vector &b=lf;

    EAFE_Modify(mesh, A, DiffusionTensor_2D, AdvectionVector_2D); // use EAFE scheme, not change b for Ax=b

    blf.EliminateVDofs(ess_tdof_list, uh, lf);

    GMRESSolver solver;
    solver.SetOperator(A);
    solver.SetAbsTol(1e-20);
    solver.SetRelTol(1e-10);
    solver.SetPrintLevel(0);
    solver.SetMaxIter(20000);
    Vector x(lf.Size());
    solver.Mult(b, x);
    if (!solver.GetConverged()) throw "GMRES solver not converged!";

    uh = x;
    double l2norm = uh.ComputeL2Error(analytic_solution_);
    L2norms.Append(l2norm);

    double totle_size = 0.0;
    for (int i=0; i<mesh.GetNE(); i++) {
        totle_size += mesh.GetElementSize(0, 1);
    }
    meshsizes.Append(totle_size / mesh.GetNE());
}
void Test1_EAFE_Modify()
{
    const char* mesh_file = "../../../data/inline-tri.mesh";
    Mesh mesh(mesh_file, 1, 1);

    Array<double> L2norms;
    Array<double> meshsizes;
    for (int i=0; i<5; i++)
    {
        mesh.UniformRefinement();
        EAFE_advec_diffu_2D(mesh, L2norms, meshsizes);
    }

    Array<double> rates = compute_convergence(L2norms, meshsizes);
    int size = rates.Size() - 1;
    assert(abs(rates[size - 0] - 2.0) < 0.1); // the last 3 must be close 2.0 (L2 errornorm convergence rate)
    assert(abs(rates[size - 1] - 2.0) < 0.1);
    assert(abs(rates[size - 2] - 2.0) < 0.1);
}


// 3D test
double analytic_solution_3D(const Vector& x){return cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]);}
double analytic_rhs_3D(const Vector& x)
{return 3.14159265358979*sin(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]) + 3.14159265358979*sin(3.14159265358979*x[1])*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[2]) + 3.14159265358979*sin(3.14159265358979*x[2])*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1]) + 2.96088132032681*cos(3.14159265358979*x[0])*cos(3.14159265358979*x[1])*cos(3.14159265358979*x[2]);}
double div_advection_3D(const Vector& x){return 0;} // 扩散速度的 (-divergence), 对于advection velocity不是divergence free的时候需要
void DiffusionTensor_3D(const Vector &x, DenseMatrix &K) {
    K(0,0) = 0.1;
    K(0,1) = 0;
    K(0,2) = 0;
    K(1,0) = 0;
    K(1,1) = 0.1;
    K(1, 2) = 0;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = 0.1;
}
void AdvectionVector_3D(const Vector &x, Vector &advcoeff) {
    advcoeff[0] = 1.0;
    advcoeff[1] = 1.0;
    advcoeff[2] = 1.0;
}
void EAFE_advec_diffu_3D(Mesh& mesh, Array<double>& L2norms, Array<double>& meshsizes)
{
    int dim = mesh.Dimension();
    H1_FECollection h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    if (mesh.bdr_attributes.Size()) {
        ess_bdr = 1;
        h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm lf(&h1_space);
    FunctionCoefficient analytic_rhs_(analytic_rhs_3D);
    lf.AddDomainIntegrator(new DomainLFIntegrator(analytic_rhs_)); // (f, v)
    lf.Assemble();

    BilinearForm blf(&h1_space);
    //EAFE基本的刚度矩阵只需要Poisson方程,后面修改的就是这个Poisson方程的刚度矩阵,参考[1],(3.24)式
    blf.AddDomainIntegrator(new DiffusionIntegrator); // (grad(u), grad(v))
    blf.Assemble();
    blf.Finalize();

    GridFunction uh(&h1_space);
    FunctionCoefficient analytic_solution_(analytic_solution_3D);
    uh.ProjectCoefficient(analytic_solution_);//使得uh满足边界条件,必须

    SparseMatrix& A = blf.SpMat();
    Vector &b=lf;

    EAFE_Modify(mesh, A, DiffusionTensor_3D, AdvectionVector_3D);

    blf.EliminateVDofs(ess_tdof_list, uh, lf);

    GMRESSolver solver;
    solver.SetOperator(A);
    solver.SetAbsTol(1e-20);
    solver.SetRelTol(1e-8);
    solver.SetPrintLevel(0);
    solver.SetMaxIter(200000);
    Vector x(lf.Size());
    solver.Mult(b, x);
    if (!solver.GetConverged()) throw "GMRES solver not converged!";

    uh = x;
    double l2norm = uh.ComputeL2Error(analytic_solution_);
    L2norms.Append(l2norm);

    double totle_size = 0.0;
    for (int i=0; i<mesh.GetNE(); i++) {
        totle_size += mesh.GetElementSize(0, 1);
    }
    meshsizes.Append(totle_size / mesh.GetNE());
}
void Test2_EAFE_Modify()
{
    const char* mesh_file = "../../../data/inline-tet.mesh";
    Mesh mesh(mesh_file, 1, 1);

    Array<double> L2norms;
    Array<double> meshsizes;
    for (int i=0; i<3; i++)
    {
        mesh.UniformRefinement();
        EAFE_advec_diffu_3D(mesh, L2norms, meshsizes);
    }

    Array<double> rates = compute_convergence(L2norms, meshsizes);
//    rates.Print(std::cout);
    int size = rates.Size() - 1;
    assert(abs(rates[size - 0] - 2.0) < 0.1); // the last 2 must be close 2.0 (L2 errornorm convergence rate)
//    assert(abs(rates[size - 1] - 2.0) < 0.1);
}

}


void Test_EAFE_ModifyStiffnessMatrix()
{
    using namespace _EAFE_ModifyStiffnessMatrix;
    Test1_EAFE_Modify();
    Test2_EAFE_Modify();

    cout << "===> Test Pass: EAFE_ModifyStiffnessMatrix.hpp" << endl;
}
#endif //LEARN_MFEM_EAFE_MODIFYSTIFFNESSMATRIX_HPP
