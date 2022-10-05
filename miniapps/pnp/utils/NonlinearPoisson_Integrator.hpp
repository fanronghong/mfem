//
// Created by fan on 2019/9/24.
//

#ifndef LEARN_MFEM_NONLINEARPOISSON_INTEGRATOR_HPP
#define LEARN_MFEM_NONLINEARPOISSON_INTEGRATOR_HPP

#include <iostream>
#include "mfem.hpp"
using namespace std;
using namespace mfem;


// 假定非线性问题为 F(u)=0, F(u)=-Laplace u + u^2 - f
// Residual: (F(u), v) = (grad(u), grad(v)) + (u^2 - f, v)
// Jacobian: J(du, v; u) = (-Laplace du + 2*u*du, v) = (grad(du), grad(v)) + 2*u*(du, v)
// 这个非线性积分子结合后面的NonlinearForm(), 就是用来计算Residual和Jacobian的
class NLFIntegrator: public NonlinearFormIntegrator
{
private:
    Vector shape;
    Coefficient* f; // 非线性算子F(u)=-Laplace u + u^2 - f
    DenseMatrix dshape, dshapedxt, invdfdx;
    Vector vec, pointflux;

public:
    NLFIntegrator(Coefficient* f_): f(f_) {}

    // 计算当前解的残量
    virtual void AssembleElementVector(const FiniteElement& el,
                                       ElementTransformation& Tr,
                                       const Vector& elfun,
                                       Vector& elvect)
    {
        int dof = el.GetDof();
        int dim = el.GetDim();
        shape.SetSize(dof);
        dshape.SetSize(dof, dim);
        invdfdx.SetSize(dim); //Jacobi变换矩阵的伴随矩阵
        vec.SetSize(dim);
        pointflux.SetSize(dim);

        elvect.SetSize(dof);
        elvect = 0.0;
        vec = 0.0;

        const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder()+Tr.OrderW());

        for (int i=0; i<ir->GetNPoints(); i++) {
            const IntegrationPoint& ip = ir->IntPoint(i);
            el.CalcShape(ip, shape);
            el.CalcDShape(ip, dshape);
            Tr.SetIntPoint(&ip);

            //给定u, 计算 (u^2-f, v), v是shape function. https://github.com/mfem/mfem/issues/160
            double fun_val = (elfun*shape)*(elfun*shape) - (*f).Eval(Tr, ip);
            double w = ip.weight * Tr.Weight() * fun_val;
            add(elvect, w, shape, elvect);//elvect + w*shape => elvect

            // 给定u, 计算 (grad(u), grad(v)), v是shape function. ref: DiffusionIntegrator::AssembleElementVector()
            CalcAdjugate(Tr.Jacobian(), invdfdx); // invdfdx = adj(J). J^{-1} = adj(J)^T / |J|
            dshape.MultTranspose(elfun, vec); //这个时候vec就是 grad(u), u= \sum u_i phi_i, i=1,...,n; elfun=(u_1,...,u_n)
            invdfdx.MultTranspose(vec, pointflux);
            double ww = ip.weight / Tr.Weight();
            pointflux *= ww;
            invdfdx.Mult(pointflux, vec);
            dshape.AddMult(vec, elvect);
        }
    }

    // 计算当前解的Jacobian
    virtual void AssembleElementGrad(const FiniteElement& el,
                                     ElementTransformation& Tr,
                                     const Vector& elfun,
                                     DenseMatrix& elmat)
    {
        int dof = el.GetDof();
        int dim = el.GetDim();
        dshapedxt.SetSize(dof, dim);
        dshape.SetSize(dof, dim);
        shape.SetSize(dof);
        elmat.SetSize(dof);
        elmat = 0.0;

        const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder()+Tr.OrderW());

        for (int i=0; i<ir->GetNPoints(); i++) {
            const IntegrationPoint& ip = ir->IntPoint(i);
            el.CalcShape(ip, shape);
            el.CalcDShape(ip, dshape);
            Tr.SetIntPoint(&ip);

            // 计算单刚矩阵 (grad(du), grad(v)). 参考DiffusionIntegrator::AssembleElementMatrix()
            double w = ip.weight / Tr.Weight();
            Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
            AddMult_a_AAt(w, dshapedxt, elmat);

            // 计算单刚矩阵 2*u*(du,v), v是shape function
            double fun_val = 2 * (elfun*shape) * ip.weight * Tr.Weight(); //计算系数2*u, u是当前解
            AddMult_a_VVt(fun_val, shape, elmat); // 2*u*(du, v)
        }
    }
};



// u = sin( 2 *  pi * x)
double u_exact_func_NonlinearPoisson_Integrator(const Vector& x)
{
    return sin(2*M_PI*x[0]);
}
// -Laplace u + u^2 = f
double f_exact_func_NonlinearPoisson_Integrator(const Vector& x)
{
    return 4*M_PI*M_PI*sin(2*M_PI*x[0]) + sin(2*M_PI*x[0])*sin(2*M_PI*x[0]);
}

void Test_NLFIntegrator()
{
    Mesh mesh(50, 50, Element::TRIANGLE, 1, 1.0, 1.0);
    int dim = mesh.Dimension();

    H1_FECollection h1_fec(1, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int size = h1_space.GetVSize();

    FunctionCoefficient f_coeff(f_exact_func_NonlinearPoisson_Integrator);
    ConstantCoefficient negative(-1.0);
    ProductCoefficient neg_f(negative, f_coeff);

    GridFunction u(&h1_space);
    GridFunctionCoefficient u_coeff(&u);
    ProductCoefficient u2(u_coeff, u_coeff);
    ConstantCoefficient two(2.0);
    ProductCoefficient u_prod_2(u_coeff, two);

    {
        // 测试 NLFIntegrator::AssembleElementVector()
        LinearForm lf(&h1_space);
        lf.AddDomainIntegrator(new DomainLFIntegrator(u2));
        lf.AddDomainIntegrator(new DomainLFIntegrator(neg_f));
        lf.Assemble();

        NonlinearForm nlf(&h1_space);
        nlf.AddDomainIntegrator(new NLFIntegrator(&f_coeff));
        Vector res1(size);
        nlf.Mult(u, res1);

        for (int i=0; i<size; i++) {
            if (abs(res1[i] - lf[i]) > 1E-6)
                throw "Wrong element vector 1.";
        }
    }

    {
        // 再次测试 NLFIntegrator::AssembleElementVector()
        Array<int> ess_tdof_list;
        if (mesh.bdr_attributes.Size()) {
            Array<int> ess_bdr(mesh.bdr_attributes.Max());
            ess_bdr = 1;
            h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
        }

        GridFunction u_0(&h1_space);
        FunctionCoefficient u_coeff(u_exact_func_NonlinearPoisson_Integrator);
        u_0.ProjectCoefficient(u_coeff);

        NonlinearForm nlf(&h1_space);
        nlf.AddDomainIntegrator(new NLFIntegrator(&f_coeff));
        nlf.SetEssentialTrueDofs(ess_tdof_list);

        Vector y(size);
        nlf.Mult(u_0, y); // 计算在真解处的残量,y应该为0
        for (int i=0; i<size; i++) {
            if (abs(y[i]) > 1E-5)
                throw "Wrong element vector 2.";
        }

    }

    {
        // 测试 NLFIntegrator::AssembleElementGrad()
        BilinearForm blf(&h1_space);
        blf.AddDomainIntegrator(new MassIntegrator(u_prod_2));
        ConstantCoefficient one(1.0);
        blf.AddDomainIntegrator(new DiffusionIntegrator(one));
        blf.Assemble();

        NonlinearForm nlf(&h1_space);
        nlf.AddDomainIntegrator(new NLFIntegrator(&f_coeff));
        SparseMatrix Jacobian;
        Jacobian = dynamic_cast<SparseMatrix&> (nlf.GetGradient(u)); // 得到在当前 u 处的Jacobian

        Jacobian *= -1.0;
        Jacobian += blf.SpMat();
        DenseMatrix* Jacobian_(Jacobian.ToDenseMatrix());
        for (int i=0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (abs((*Jacobian_)(i, j)) > 1E-6)
                    throw "Wrong Jacobian!";
            }
        }
    }
}

void Test_NonlinearPoisson_Integrator()
{
    Test_NLFIntegrator();

    cout << "===> Test Pass: NonlinearPoisson_Integrator.hpp" << endl;
}

#endif //LEARN_MFEM_NONLINEARPOISSON_INTEGRATOR_HPP
