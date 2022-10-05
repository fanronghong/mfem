//
// Created by plan on 2019/12/20.
//

#ifndef LEARN_MFEM_SUPG_INTEGRATOR_HPP
#define LEARN_MFEM_SUPG_INTEGRATOR_HPP

#include "mfem.hpp"
using namespace mfem;


// supg方法的线性型项: Q * \tau_K * (f, adv * \nabla v)_{K}, adv就是对流速度
// advection-diffusion equation: - \nabla\cdot(Diff \nabla u + Adv u) = f.
// Note: advection-diffusion equation cannot be -\nabla\cdot(Diff \nabla u) + Adv \cdot\nabla u = f !!!
class SUPG_LinearFormIntegrator : public LinearFormIntegrator
{
private:
    DenseMatrix dshape, adjJ, adv_ir;
    Vector vec2, BdFidxT;
    MatrixCoefficient& diff; // diffusion coefficient
    VectorCoefficient& adv;  //对流扩散方程的对流速度
    Coefficient& f;          //对流扩散方程的右端项
    Coefficient& Q;          //
    Mesh &mesh;

public:
    // Default constructor:
    SUPG_LinearFormIntegrator(MatrixCoefficient& diff_, VectorCoefficient& adv_, Coefficient& Q_, Coefficient& f_, Mesh& mesh_)
            : diff(diff_), adv(adv_), Q(Q_), f(f_), mesh(mesh_) {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,  ElementTransformation &Tr, Vector &elvect)
    {
        //Given a particular Finite Element(el) and a transformation (Tr) computes the element vector(elvect)
        double h_K = mesh.GetElementSize(Tr.ElementNo); //Streamline diffusion params.
        cout.precision(14);
        int ndofs = el.GetDof(); //el单元的自由度个数
        int dim = el.GetDim();

        //形函数梯度在积分点处的取值,假设shape function为v1,v2,v3,网格单元是二维三角形,则dshape就是3*2的矩阵.
        // 从这里也可以看出, 每个shape function的各个偏导数放在同一行fffff
        dshape.SetSize(ndofs, dim);
        adjJ.SetSize(dim);
        vec2.SetSize(dim);
        BdFidxT.SetSize(ndofs);
        elvect.SetSize(ndofs); //为单元荷载向量分配内存空间
        elvect = 0.0;

        const IntegrationRule *ir = IntRule; //IntRule保存了所有的积分点
        if (ir == NULL) {
            int order = Tr.OrderGrad(&el) + Tr.Order() + el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
        }

        //ir 指向一个数组(元素是IntegrationPoint), 计算对流速度在所有积分点处的取值, 组成一个DenseMatrix(adv_ir)
        //adv_ir 的每一列对应对流速度在一个积分点处的取值形成的向量
        adv.Eval(adv_ir, Tr, *ir);
        Vector vec1;

        elvect = 0.0;
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            el.CalcDShape(ip, dshape); //计算shape function的梯度在ip处的取值

            Tr.SetIntPoint(&ip);
            CalcAdjugate(Tr.Jacobian(), adjJ); //计算Tr.Jacobian()的伴随矩阵adjJ

            DenseMatrix diff_mat(dim);
            diff.Eval(diff_mat, Tr, ip); // diffustion matrix coefficient at integration point
            double alpha = diff_mat.MaxMaxNorm(); // 这里的范数肯定有其他选择ffff

            adv_ir.GetColumnReference(i, vec1);  //得到矩阵的第i列,即对流速度在第i个积分点处的取值
            double beta = vec1.Normlinf(); // 这里的范数肯定有其他选择ffff

            double Peclet = beta * h_K / (2 * alpha);  // Peclet number

            Vector transip(dim);
            Tr.Transform(ip, transip); //把reference element上的积分点变换到physical element可以删除ffffffffffffffffffffff
            double tau_K = 0.0; // stability parameter fff. several kinds of tau_K
            if (Peclet > 1) {
                tau_K = h_K / (2 * beta);
            } else {
                tau_K = h_K*h_K / (12 * alpha);
            }

            double vals = Q.Eval(Tr, ip) * tau_K * ip.weight * f.Eval(Tr, ip);

            adjJ.Mult(vec1, vec2);        //adjJ * adv => vec2
            dshape.Mult(vec2, BdFidxT); //dshape * vec2 => BdFidxT
            BdFidxT *= vals;
            elvect += BdFidxT;
        }
    }
};



// supg方法的双线性型项: \tau_K * (q1 * adv * \nabla u + q2 * divergence(adv)u, adv * \nabla v), adv就是对流速度
// advection-diffusion equation: - \nabla\cdot(Diff \nabla u + Adv u) = f
// Note: advection-diffusion equation cannot be -\nabla\cdot(Diff \nabla u) + Adv \cdot\nabla u = f !!!
class SUPG_BilinearFormIntegrator: public BilinearFormIntegrator
{
private:
    Mesh& mesh;
    MatrixCoefficient* diff;
    Coefficient* Diff;
    bool mark;

    VectorCoefficient& adv;
    Coefficient& div_adv;
    Coefficient &q1, &q2;

    Vector shape, adv_vec, vec1, vec2;
    DenseMatrix dshape, dshapedxt, adv_ir;

public:
    SUPG_BilinearFormIntegrator(MatrixCoefficient* diff_, Coefficient& q1_, VectorCoefficient& adv_,
                                Coefficient& q2_, Coefficient& div_adv_, Mesh& mesh_)
                    : diff(diff_), q1(q1_), adv(adv_), q2(q2_), div_adv(div_adv_), mesh(mesh_) { mark = false; }
    SUPG_BilinearFormIntegrator(Coefficient* diff_, Coefficient& q1_, VectorCoefficient& adv_,
                                Coefficient& q2_, Coefficient& div_adv_, Mesh& mesh_)
            : Diff(diff_), q1(q1_), adv(adv_), q2(q2_), div_adv(div_adv_), mesh(mesh_) { mark=true; }

    virtual void AssembleElementMatrix(const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat)
    {
        int dim = mesh.Dimension();
        int spaceDim = Trans.GetSpaceDim(); //dimension of the target (physical) space
        int nd = el.GetDof();

        dshape.SetSize(nd, dim);
        dshapedxt.SetSize(nd, spaceDim);
        shape.SetSize(nd);
        adv_vec.SetSize(dim);
        vec1.SetSize(nd);
        vec2.SetSize(nd);
        elmat.SetSize(nd); //给elmat分配内存空间
        elmat = 0.0; //应该就是单元刚度矩阵

        const IntegrationRule* ir = IntRule;
        if (ir == NULL) {
            int order = 2 * el.GetOrder() - 2;
            ir = &IntRules.Get(el.GetGeomType(), order);
        }

        //ir 指向一个数组(元素是IntegrationPoint), 计算对流速度在所有积分点处的取值, 组成一个DenseMatrix(adv_ir)
        //adv_ir 的每一列对应对流速度在一个积分点处的取值形成的向量
        adv.Eval(adv_ir, Trans, *ir);

        double h_K = mesh.GetElementSize(Trans.ElementNo);
        for (int i = 0; i < ir->GetNPoints(); i++) //对所有积分点循环
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Trans.SetIntPoint(&ip);

            double w1 = q1.Eval(Trans, ip) / Trans.Weight(); // q1 / |J|
            double w2 = q2.Eval(Trans, ip) * div_adv.Eval(Trans, ip); // q1 * divergence(Adv)

            adv_ir.GetColumnReference(i, adv_vec);  //得到矩阵的第i列,即对流速度在第i个积分点处的取值

            double tau_K; // stability parameter fff. several kinds of tau_K
            {
                double alpha = 0;
                if (!mark) {
                    DenseMatrix diff_mat(dim);
                    diff->Eval(diff_mat, Trans, ip);
                    alpha = diff_mat.MaxMaxNorm(); // 这里的范数肯定有其他选择ffff
                } else {
                    alpha = Diff->Eval(Trans, ip);
                }

                double beta = adv_vec.Normlinf(); // 这里的范数肯定有其他选择ffff
                double Peclet = beta * h_K / (2 * alpha);  // Peclet number

                if (Peclet > 1) {
                    tau_K = h_K / (2 * beta);
                } else {
                    tau_K = h_K * h_K / (12 * alpha);
                }
            }

            el.CalcShape(ip, shape);
            el.CalcDShape(ip, dshape); //每个shape function的各个偏导数放在同一行fffff

            Mult(dshape, Trans.AdjugateJacobian(), dshapedxt); // (adj J) \cdot \hat{\nabla} \hat{v}. dshape * Trans.AdjugateJacobian() -> dshapedxt

            dshapedxt.Mult(adv_vec, vec1); // first term
            add(w1, vec1, w2, shape, vec2);

            AddMult_a_VWt(ip.weight * tau_K, vec1, vec2, elmat);
        }
    }
};



// alpha = 1.0E-8
void smallDiffusionTensor_(const Vector& x, DenseMatrix& K)
{
    K(0,0) = 1.0E-8;
    K(0,1) = 0;
    K(1,0) = 0;
    K(1,1) = 1.0E-8;
}
// alpha = 10.0
void bigDiffusionTensor_(const Vector& x, DenseMatrix& K)
{
    K(0,0) = 10.0;
    K(0,1) = 0;
    K(1,0) = 0;
    K(1,1) = 10.0;
}
// beta = 1.0
void AdvectionVector_(const Vector& x, Vector& adv)
{
    adv[0] = 1.0;
    adv[1] = 1.0;
}
// u = x + 2*y
double u_(const Vector& x)
{
    return x[0] + 2 * x[1];
}

void Test1_big_diff()
{
    Mesh mesh(10, 10, Element::TRIANGLE, 1, 1.0, 1.0);
    int dim = mesh.Dimension();

    H1_FECollection    h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    double alpha = 10.0;
    MatrixFunctionCoefficient diff(dim, bigDiffusionTensor_); // alpha
    double beta = 1.0;
    VectorFunctionCoefficient adv(dim, AdvectionVector_); // beta

    const double h = 0.10745699318235; // 从网格中得出. mesh.PrintInfo()
    double Peclet = beta * h / (2 * alpha); // all element Peclet is same
    double tau_K = 0.0; // stability parameter fff. several kinds of tau_K
    if (Peclet > 1) {
        tau_K = h / (2 * beta);
    } else {
        tau_K = h*h / (12 * alpha); // 这些取法从SUPG_LinearFormIntegrator类的定义中得到
    }

    BilinearForm blf(&h1_space);
    blf.AddDomainIntegrator(new ConvectionIntegrator(adv, tau_K));
    blf.Assemble();

    SparseMatrix blf_mat(blf.SpMat());

    FunctionCoefficient u(u_);
    GridFunction uh(&h1_space);
    uh.ProjectCoefficient(u);

    Vector result(h1_space.GetNDofs());
    blf_mat.MultTranspose(uh, result);

    LinearForm lf(&h1_space);
    ConstantCoefficient one(1.0);
    GridFunctionCoefficient uh_coeff(&uh);
    lf.AddDomainIntegrator(new SUPG_LinearFormIntegrator(diff, adv, one, uh_coeff, mesh));
    lf.Assemble();

    result -= lf;
    for (int i=0; i<result.Size(); ++i)
    {
        if (abs(result[i]) > 1E-10) throw "Something wrong in SUPG_LinearFormIntegrator!";
    }
}

void Test1_small_diff()
{
    Mesh mesh(10, 10, Element::TRIANGLE, 1, 1.0, 1.0);
    int dim = mesh.Dimension();

    H1_FECollection    h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    double alpha = 1.0E-8;
    MatrixFunctionCoefficient diff(dim, smallDiffusionTensor_); // alpha
    double beta = 1.0;
    VectorFunctionCoefficient adv(dim, AdvectionVector_); // beta

    const double h = 0.10745699318235; // 从网格中得出. mesh.PrintInfo()
    double Peclet = beta * h / (2 * alpha); // all element Peclet are same
    double tau_K = 0.0; // all element stability parameters are same
    if (Peclet > 1) {
        tau_K = h / (2 * beta);
    } else {
        tau_K = h*h / (12 * alpha);
    }

    BilinearForm blf(&h1_space);
    // tau_K * (adv * grad(u), v)
    blf.AddDomainIntegrator(new ConvectionIntegrator(adv, tau_K));
    blf.Assemble();

    SparseMatrix blf_mat(blf.SpMat());

    FunctionCoefficient u(u_);
    GridFunction uh(&h1_space);
    uh.ProjectCoefficient(u);

    Vector result(h1_space.GetNDofs());
    blf_mat.MultTranspose(uh, result); // 矩阵转置在乘以向量

    LinearForm lf(&h1_space);
    ConstantCoefficient one(1.0);
    GridFunctionCoefficient uh_coeff(&uh);
    // one * \tau_K * (u, adv * grad(v))
    lf.AddDomainIntegrator(new SUPG_LinearFormIntegrator(diff, adv, one, uh_coeff, mesh));
    lf.Assemble();

    result -= lf; // 二者的每个元素应该相同
    for (int i=0; i<result.Size(); ++i)
    {
        if (abs(result[i]) > 1E-10) throw "Something wrong in SUPG_LinearFormIntegrator!";
    }
}

void Matrix_One(const Vector& x, DenseMatrix& K)
{
    K(0,0) = 1.0;
    K(0,1) = 1.0;
    K(1,0) = 1.0;
    K(1,1) = 1.0;
}
void Test2_bilinear()
{
    Mesh mesh(10, 10, Element::TRIANGLE, 1, 1.0, 1.0);
    int dim = mesh.Dimension();

    H1_FECollection    h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    double alpha = 10.0;
    MatrixFunctionCoefficient diff(dim, bigDiffusionTensor_); // alpha
    double beta = 1.0;
    VectorFunctionCoefficient adv(dim, AdvectionVector_); // beta

    const double h = 0.10745699318235; // 从网格中得出. mesh.PrintInfo()
    double Peclet = beta * h / (2 * alpha); // all element Peclet is same
    double tau_K = 0.0; // stability parameter fff. several kinds of tau_K
    if (Peclet > 1) {
        tau_K = h / (2 * beta);
    } else {
        tau_K = h*h / (12 * alpha); // 这些取法从SUPG_LinearFormIntegrator类的定义中得到
    }

    ConstantCoefficient one(1.0);
    MatrixFunctionCoefficient matrix_one(2, Matrix_One); // Matrix_One由对流速度与自己做叉积得到

    BilinearForm blf(&h1_space);
    blf.AddDomainIntegrator(new DiffusionIntegrator(matrix_one));
    blf.Assemble();
    blf.Finalize();

    SparseMatrix blf_mat(blf.SpMat());
    blf_mat *= tau_K;

    BilinearForm supg(&h1_space);
    ConstantCoefficient div_adv(0.0);
    ConstantCoefficient zero(0.0);
    supg.AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&diff, one, adv, zero, div_adv, mesh));
    supg.Assemble();
    supg.Finalize();

    SparseMatrix supg_mat(supg.SpMat());

    DenseMatrix res(supg_mat.Size());
    supg_mat.ToDenseMatrix(res);
    Add(blf_mat, -1.0, res);

    for (int i=0; i<h1_space.GetNDofs(); ++i) {
        for (int j=0; j<h1_space.GetNDofs(); ++j) {
            if (abs(res(i, j)) > 1E-10)
            {
                MFEM_ABORT("Something wrong in SUPG_BilinearFormIntegrator")
            }
        }
    }
}




void Test_SUPG_Integrator()
{
    Test1_small_diff();
    Test1_big_diff();
    Test2_bilinear();

    cout << "===> (Need more tests!!!) Test Pass: SUPG_Integrator.hpp" << endl;
}

#endif //LEARN_MFEM_SUPG_INTEGRATOR_HPP
