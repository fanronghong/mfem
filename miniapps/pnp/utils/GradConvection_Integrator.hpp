#ifndef GRAD_CONVECTION_INTEGRATOR
#define GRAD_CONVECTION_INTEGRATOR

#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include "mfem.hpp"
using namespace std;
using namespace mfem;

/* 计算
 *
 *    Q (u grad(w), grad(v))  or
 *    Q (u adv, grad(v)),
 *
 * Given w(类型为GridFunction), or given adv is VectorCoefficient (对流速度),
 * u is Trial function, v is Test function.
 *
 * The stiffness matrix of GradConvection_BLFIntegrator is just the transpose
 * of ConvectionIntegrator (ref: Test_gradConvectionIntegrator2)
 * */
class GradConvection_BLFIntegrator: public BilinearFormIntegrator
{
protected:
    GradientGridFunctionCoefficient* grad_w;
    VectorCoefficient* adv;
    bool param = false;
    Coefficient *Q, *diff;

    DenseMatrix dshape, dshapedxt;
    Vector gradw, shape, vec1;

public:
    Array<double> local_peclet; // for more info

public:
    GradConvection_BLFIntegrator(const GridFunction& w_, Coefficient* Q_, Coefficient* diff_=NULL): Q(Q_), diff(diff_)
    { grad_w = new GradientGridFunctionCoefficient(&w_); }
    GradConvection_BLFIntegrator(VectorCoefficient* adv_, Coefficient* Q_, Coefficient* diff_=NULL)
        : Q(Q_), adv(adv_), diff(diff_) { param = true; }
    ~GradConvection_BLFIntegrator() { delete grad_w; }

    virtual void AssembleElementMatrix(const FiniteElement& el,
            ElementTransformation& eltran, DenseMatrix& elmat)
    {
        int ndofs = el.GetDof();
        int dim = el.GetDim();
        vec1.SetSize(ndofs);
        shape.SetSize(ndofs);
        dshape.SetSize(ndofs, dim);
        dshapedxt.SetSize(ndofs, dim);
        elmat.SetSize(ndofs);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL) {
            int order = eltran.OrderGrad(&el) + eltran.Order() + el.GetOrder();
            ir = &IntRules.Get(el.GetGeomType(), order);
        }

        vector<double> elem_peclet;
        for (int i=0; i<ir->GetNPoints(); i++)
        {
            const IntegrationPoint& ip = ir->IntPoint(i);
            eltran.SetIntPoint(&ip);

            el.CalcDShape(ip, dshape);
            el.CalcShape(ip, shape);

            double w = ip.weight;
            if (Q) {
                w *= Q->Eval(eltran, ip);
            }

            Mult(dshape, eltran.AdjugateJacobian(), dshapedxt);

            if (param) {
                adv->Eval(gradw, eltran, ip);
            } else {
                grad_w->Eval(gradw, eltran, ip); // low precision for gradw
            }

            elem_peclet.push_back(gradw.Norml1());

            dshapedxt.Mult(gradw, vec1);
            AddMult_a_VWt(w, vec1, shape, elmat); // elmat += w * (vec1 shape^t)
        }

        double mean_peclet = accumulate(elem_peclet.begin(), elem_peclet.end(), 0.0) / elem_peclet.size();
        local_peclet.Append(mean_peclet);
    }
};

/* 计算
 *     Q*(grad(w), grad(v)),
 *
 * Given Q is Coefficient, w is GridFunction
 * */
class GradConvection_LFIntegrator: public LinearFormIntegrator
{
protected:
    Coefficient *q;
    GradientGridFunctionCoefficient *gradw;

    DenseMatrix adjJ, dshape, tmp;
    Vector gradw_val, tmp_vec;

public:
    GradConvection_LFIntegrator(Coefficient* q_, GridFunction* w): q(q_)
    { gradw = new GradientGridFunctionCoefficient(w); }
    ~GradConvection_LFIntegrator() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
        int nd = el.GetDof();
        int dim = el.GetDim();

        adjJ.SetSize(dim);
        gradw_val.SetSize(dim);

        dshape.SetSize(nd, dim);
        tmp.SetSize(nd, dim);
        tmp_vec.SetSize(nd);
        elvect.SetSize(nd);
        elvect = 0.0;

        int order = Tr.OrderGrad(&el) + Tr.Order() + el.GetOrder();
        const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

        for (int i=0; i<ir->GetNPoints(); ++i)
        {
            const IntegrationPoint& ip = ir->IntPoint(i);

            Tr.SetIntPoint(&ip);
            CalcAdjugate(Tr.Jacobian(), adjJ);

            el.CalcDShape(ip, dshape);

            gradw->Eval(gradw_val, Tr, ip);
            double wi = ip.weight * q->Eval(Tr, ip);

            Mult(dshape, adjJ, tmp);
            tmp.Mult(gradw_val, tmp_vec);
            elvect.Add(wi, tmp_vec);
        }
    }

    using LinearFormIntegrator::AssembleRHSElementVect;
};


namespace _GradConvection_Integrator
{
    double sin_cfun(const Vector& x)
    {
        return sin(x[0])*sin(x[0]) + cos(x[1])*cos(x[1]);
    }
    void grad_sin_cfun(const Vector& x, Vector& y)
    {
        y[0] = 2 * sin(x[0]) * cos(x[0]);
        y[1] = -2 * cos(x[1]) * sin(x[1]);
    }

    double func(const Vector& x)
    {
        return 4*x[0] + 10*x[1];
    }
    void grad_func(const Vector& x, Vector& y)
    {
        y[0] = 4;
        y[1] = 10;
    }


    void Test_gradConvectionIntegrator1()
    {
        Mesh mesh(50, 50, Element::TRIANGLE, 1, 1.0, 1.0);
        int dim = mesh.Dimension();

        H1_FECollection    h1_fec(1, dim);
        FiniteElementSpace h1_space(&mesh, &h1_fec);

        FunctionCoefficient       sin_coeff(sin_cfun);
        VectorFunctionCoefficient grad_sin_coeff(dim, grad_sin_cfun);
        ConstantCoefficient       one(1.2345678);

        GridFunction w(&h1_space);
        w.ProjectCoefficient(sin_coeff);

        // 用自己写的积分算子生成刚度矩阵
        BilinearForm blf1(&h1_space);
        blf1.AddDomainIntegrator(new GradConvection_BLFIntegrator(w, &one)); // (u grad(w), grad(v)): w取GridFunction
        blf1.Assemble();

        SparseMatrix blf1_mat(blf1.SpMat());
        GridFunction One(&h1_space), res1(&h1_space), res2(&h1_space);
        One = 1.0;
        blf1_mat.Mult(One, res1);

        // 下面用另外一种方式生成刚度矩阵做对比
        BilinearForm blf2(&h1_space);
        blf2.AddDomainIntegrator(new DiffusionIntegrator(one)); // (u grad(w), grad(v)): u取1
        blf2.Assemble();

        SparseMatrix blf2_mat(blf2.SpMat());
        blf2_mat.Mult(w, res2);

        res1 -= res2;
        for (int i=0; i<res1.Size(); i++)
        {
            if (abs(res1[i]) > 1e-8) mfem_error("Wrong: GradConvection_Integrator.hpp");
        }
    }

    void Test_gradConvectionIntegrator2()
    {
        Mesh mesh(100, 100, Element::TRIANGLE, 1, 1.0, 1.0);

        H1_FECollection    h1_fec(1, 2);
        FiniteElementSpace h1_space(&mesh, &h1_fec);

        FunctionCoefficient       func_coeff(func);
        VectorFunctionCoefficient grad_func_coeff(2, grad_func);
        double                    alpha=1.23456789;
        ConstantCoefficient       one(alpha);

        GridFunction func_gf(&h1_space);
        func_gf.ProjectCoefficient(func_coeff);

        // 用自己写的积分算子生成刚度矩阵
        BilinearForm blf1(&h1_space);
        blf1.AddDomainIntegrator(new GradConvection_BLFIntegrator(func_gf, &one));
        blf1.Assemble();

        GridFunction rand_gf(&h1_space), res1(&h1_space), res2(&h1_space);
        for (int i=0; i<h1_space.GetNDofs(); ++i) rand_gf[i] = rand() % 10;

        SparseMatrix blf1_mat(blf1.SpMat());
        blf1_mat.Mult(rand_gf, res1);

        // 下面用另外一种方式生成刚度矩阵做对比
        BilinearForm blf2(&h1_space);
        blf2.AddDomainIntegrator(new ConvectionIntegrator(grad_func_coeff, alpha));
        blf2.Assemble();

        SparseMatrix blf2_mat(blf2.SpMat());
    //    blf2_mat.Mult(rand_gf, res2); // wrong!!!
        blf2_mat.MultTranspose(rand_gf, res2); //

        res1 -= res2;
        for (int i=0; i<res1.Size(); i++)
        {
            if (abs(res1[i]) > 1e-10) mfem_error("Wrong: GradConvection_Integrator.hpp");
        }
    }

    void Test_GradConvectionIntegrator2_1()
    {
        Mesh mesh(100, 100, Element::TRIANGLE, 1, 1.0, 1.0);

        H1_FECollection    h1_fec(1, 2);
        FiniteElementSpace h1_space(&mesh, &h1_fec);
        int size = h1_space.GetTrueVSize();

        ConstantCoefficient one(1.23456);

        GridFunction rand_gf(&h1_space), res1(&h1_space), res2(&h1_space);
        for (int i=0; i<h1_space.GetNDofs(); ++i) rand_gf[i] = rand() % 10;

        BilinearForm blf1(&h1_space);
        blf1.AddDomainIntegrator(new DiffusionIntegrator(one));
        blf1.Assemble();

        SparseMatrix blf1_mat(blf1.SpMat());
        blf1_mat.Mult(rand_gf, res1);

        // 下面用另外一种方式生成刚度矩阵做对比
        LinearForm lf(&h1_space);
        lf.AddDomainIntegrator(new GradConvection_LFIntegrator(&one, &rand_gf));
        lf.Assemble();

        res1 -= lf;
        for (int i=0; i<res1.Size(); i++)
        {
            if (abs(res1[i]) > 1e-10) mfem_error("Wrong: class GradConvection_LFIntegrator");
        }
    }
}


void Test_GradConvection_Integrator()
{
    using namespace _GradConvection_Integrator;

    Test_gradConvectionIntegrator1();
    Test_gradConvectionIntegrator2();
    Test_GradConvectionIntegrator2_1();

    /* ---------- 完成本文件中所有类的全部测试 ------------ */
}

#endif
