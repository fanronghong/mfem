//
// Created by fan on 2020/4/12.
//

#ifndef LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP
#define LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP

#include <iostream>
#include "mfem.hpp"
#include <cassert>

using namespace std;
using namespace mfem;


/* 计算(区域边界积分)
 *   q<u_D, v grad(w).n>_E
 * q, u_D are Coefficient
 * w is GridFunction
 * */
class DGSelfBdrFaceIntegrator: public LinearFormIntegrator
{
protected:
    Coefficient *q, *u_D;
    GradientGridFunctionCoefficient *gradw;

    Vector nor, shape, grad_w;

public:
    DGSelfBdrFaceIntegrator(Coefficient *q_, Coefficient *u_D_, GridFunction* w)
            : q(q_), u_D(u_D_) { gradw = new GradientGridFunctionCoefficient(w); }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("Not Support");
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el,
                                        FaceElementTransformations& Tr,
                                        Vector& elvect)
    {
        int dim  = el.GetDim();
        int ndof = el.GetDof();

        nor.SetSize(dim);
        grad_w.SetSize(dim);

        shape.SetSize(ndof);
        elvect.SetSize(ndof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order = 2*el.GetOrder();
            ir = &IntRules.Get(Tr.FaceGeom, order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            IntegrationPoint eip;

            Tr.Face->SetIntPoint(&ip);
            if (dim == 1)
                nor(0) = 2*eip.x - 1.0; //1维参考单元的法向量(不是单位的)
            else
                CalcOrtho(Tr.Face->Jacobian(), nor);

            Tr.Loc1.Transform(ip, eip);
            el.CalcShape(eip, shape);

            gradw->Eval(grad_w, *Tr.Elem1, eip);

            double val = ip.weight * q->Eval(*Tr.Face, ip) * u_D->Eval(*Tr.Face, ip) * (nor * grad_w);

            shape *= val;
            elvect += shape;
        }
    }
};


/* 计算(边界或者内部Face都可以):
 *    q <{u grad(w).n}, [v]>_E,
 *
 * u is trial function, v is test function
 * q are Coefficient. q在边E的两边连续
 * w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
 * */
class DGSelfTraceIntegrator_1 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;
    Vector nor, shape1, shape2, grad_w;
    double val1, val2;

public:
    DGSelfTraceIntegrator_1(Coefficient &q, GridFunction &w)
            : Q(&q)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    ~DGSelfTraceIntegrator_1() { delete gradw; }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1, ndof2;

        dim = el1.GetDim();
        grad_w.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        if (Trans.Elem2No >= 0)
            ndof2 = el2.GetDof();
        else ndof2 = 0;

        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);
        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (Trans.Elem2No >= 0)
                order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                         2 * max(el1.GetOrder(), el2.GetOrder()));
            else {
                order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();
            }
            if (el1.Space() == FunctionSpace::Pk) {
                order++;
            }
            ir = &IntRules.Get(Trans.FaceGeom, order); //得到face上的积分规则(里面包含积分点)
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            IntegrationPoint eip1, eip2;

            Trans.Loc1.Transform(ip, eip1);
            el1.CalcShape(eip1, shape1);
            if (ndof2)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
            }

            Trans.Face->SetIntPoint(&ip);
            if (dim == 1)
                nor(0) = 2*eip1.x - 1.0;
            else
                CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量

            Trans.Elem1->SetIntPoint(&eip1);
            gradw->Eval(grad_w, *Trans.Elem1, eip1);

            val1 = ip.weight * Q->Eval(*Trans.Elem1, eip1) * (grad_w * nor);
            if (!ndof2)
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);
            }
            else
            {
                val1 *= 0.5;
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);

                Trans.Elem2->SetIntPoint(&eip2);
                gradw->Eval(grad_w, *Trans.Elem2, eip2);
                val2 = ip.weight * Q->Eval(*Trans.Elem2, eip2) * (grad_w * nor) * 0.5;

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(ndof1 + i, j) -= shape2(i) * val1 * shape1(j);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, ndof1+j) += shape1(i) * val2 * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(ndof1 + i, ndof1 + j) -= val2 * shape2(i) * shape2(j);
            }
        }
    }
};


/* 计算(边界或者内部Face都可以):
 *   q <[u], {v grad(w).n}>_E,
 *
 * u is trial function, v is test function
 * q are Coefficient. q在边E的两边连续
 * w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
 * */
class DGSelfTraceIntegrator_2 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;
    Vector nor, shape1, shape2, grad_w;

public:
    DGSelfTraceIntegrator_2(Coefficient &q, GridFunction &w)
            : Q(&q)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    ~DGSelfTraceIntegrator_2() { delete gradw; }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1, ndof2;

        dim = el1.GetDim();
        grad_w.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        if (Trans.Elem2No >= 0)
            ndof2 = el2.GetDof();
        else ndof2 = 0;

        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);
        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (Trans.Elem2No >= 0)
                order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                         2 * max(el1.GetOrder(), el2.GetOrder()));
            else {
                order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();
            }
            if (el1.Space() == FunctionSpace::Pk) {
                order++;
            }
            ir = &IntRules.Get(Trans.FaceGeom, order); //得到face上的积分规则(里面包含积分点)
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            IntegrationPoint eip1, eip2;

            Trans.Loc1.Transform(ip, eip1);
            el1.CalcShape(eip1, shape1);
            if (ndof2)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
            }

            Trans.Face->SetIntPoint(&ip);
            if (dim == 1)
                nor(0) = 2*eip1.x - 1.0;
            else
                CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量

            Trans.Elem1->SetIntPoint(&eip1);
            gradw->Eval(grad_w, *Trans.Elem1, eip1);

            double val = ip.weight * Q->Eval(*Trans.Elem1, eip1) * (grad_w * nor);
            if (!ndof2)
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val * shape1(i) * shape1(j);
            }
            else
            {
                Trans.Elem2->SetIntPoint(&eip2);
                gradw->Eval(grad_w, *Trans.Elem2, eip2);
                val += ip.weight * Q->Eval(*Trans.Elem2, eip2) * (grad_w * nor);
                val *= 0.5;

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val * shape1(i) * shape1(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(ndof1 + i, j) -= val * shape2(i) * shape1(j);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, ndof1+j) -= shape1(i) * val * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(ndof1 + i, ndof1 + j) += val * shape2(i) * shape2(j);
            }
        }
    }
};


/* 计算(边界或者内部Face都可以):
 *   q <[u], [v]>_E,
 *
 * u is trial function, v is test function
 * q are Coefficient. q在边E的两边连续
 * */
class DGSelfTraceIntegrator_3: public BilinearFormIntegrator
{

};


/* 计算(边界或者内部Face都可以):
 *   q <[u], [v]>_E,
 *
 * u is GridFunction, v is test function
 * q are Coefficient. q在边E的两边连续
 * */
class DGSelfTraceIntegrator_4: public LinearFormIntegrator
{

};


/* 计算(边界或者内部Face都可以):
 *    q <{grad(u).n}, [v]>_E,
 *
 * u is given GridFunction, v is test function
 * q are Coefficient. q在边E的两边连续
 * */
class DGSelfTraceIntegrator_5 : public LinearFormIntegrator
{

};


namespace _DGSelfTraceIntegrator
{
    double sin_cfun(const Vector& x)
    {
        return sin(x[0]) * sin(x[1]); // sin(x) * sin(y)
    }

    double cos_cfun(const Vector& x)
    {
        return cos(x[0]) * cos(x[1]);
    }


    void Test_DGSelfTraceIntegrator_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

//    H1_FECollection h1_fec(1, mesh->Dimension());
        DG_FECollection h1_fec(1, mesh->Dimension());
        FiniteElementSpace h1_space(mesh, &h1_fec);

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&h1_space), one_gf(&h1_space);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        { // 只在内部边积分, test DGSelfTraceIntegrator_1
            Vector out1(h1_space.GetVSize()), out2(h1_space.GetVSize());

            BilinearForm blf1(&h1_space);
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&h1_space);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<h1_space.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }

        { // 只在边界积分, test DGSelfTraceIntegrator_1
            Vector out1(h1_space.GetVSize()), out2(h1_space.GetVSize());

            BilinearForm blf1(&h1_space);
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&h1_space);
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<h1_space.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }

        delete mesh;
    }

    void Test_DGSelfTraceIntegrator_2()
    {
        Mesh* mesh = new Mesh(20, 20, Element::TRIANGLE, true, 1.0, 1.0);

//    H1_FECollection h1_fec(1, mesh->Dimension());
        DG_FECollection h1_fec(1, mesh->Dimension());
        FiniteElementSpace h1_space(mesh, &h1_fec);

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&h1_space), one_gf(&h1_space);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        { // test: DGSelfTraceIntegrator_2
            Vector out1(h1_space.GetVSize()), out2(h1_space.GetVSize()), out3(h1_space.GetVSize());

            BilinearForm blf1(&h1_space);
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(neg, sin_gf));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(sin_gf, out1);

            BilinearForm blf2(&h1_space);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            BilinearForm blf3(&h1_space);
            blf3.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.Assemble();
            blf3.Finalize();
            blf3.Mult(one_gf, out3);

            out3 += out2;
            for (int i=0; i<h1_space.GetVSize(); ++i)
            {
//            assert(abs(out1[i] - out2[i]) < 1E-10);
//            cout << "hh" << endl;
            }
        }
        delete mesh;
        cout << "     Needs more tests here!" << endl;
    }

    void Test_DGSelfBdrFaceIntegrator_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection h1_fec(1, mesh->Dimension());
        FiniteElementSpace h1_space(mesh, &h1_fec);

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        FunctionCoefficient cos_coeff(cos_cfun);
        GridFunction sin_gf(&h1_space), one_gf(&h1_space);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);
        GradientGridFunctionCoefficient grad_sin(&sin_gf);

        {
            LinearForm lf1(&h1_space);
            lf1.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(cos_coeff, grad_sin, 2.0, 0.0));
            lf1.Assemble();

            LinearForm lf2(&h1_space);
            lf2.AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&one, &cos_coeff, &sin_gf));
            lf2.Assemble();

//        lf1.Print(cout << "lf1: " , h1_space.GetVSize());
//        lf2.Print(cout << "lf2: ", h1_space.GetVSize());
            for (int i=0; i<h1_space.GetVSize(); ++i)
            {
                assert(abs(lf1[i] - lf2[i]) < 1E-10);
            }
        }

        delete mesh;
    }
}

void Test_DGSelfTraceIntegrator()
{
    using namespace _DGSelfTraceIntegrator;
    Test_DGSelfTraceIntegrator_1();
    Test_DGSelfTraceIntegrator_2();
    Test_DGSelfBdrFaceIntegrator_1();

    cout << "===> Test Pass: DGSelfTraceIntegrator.hpp" << endl;
}

#endif //LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP
