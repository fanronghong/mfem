//
// Created by fan on 2020/4/12.
//

#ifndef LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP
#define LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP

#include <iostream>
#include "mfem.hpp"
#include <cassert>
#include <random>
#include <numeric>

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
 *   q <h^{-1} [u], [v]>_E
 *
 * u is trial function, v is test function
 * q are Coefficient. q在边E的两边连续
 * */
class DGSelfTraceIntegrator_3: public BilinearFormIntegrator
{
protected:
    Coefficient* Q;

    Vector shape1, shape2, nor;

public:
    DGSelfTraceIntegrator_3(Coefficient& q) : Q(&q) {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement& el1,
                                    const FiniteElement& el2,
                                    FaceElementTransformations& Trans,
                                    DenseMatrix& elmat)
    {
        int dim, ndof1, ndof2, ndofs;
        dim = el1.GetDim();
        ndof1 = el1.GetDof();
        ndof2 = 0;

        nor.SetSize(dim);
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }

        ndofs = ndof1 + ndof2;
        elmat.SetSize(ndofs);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order;
            if (ndof2)
            {
                order = 2*max(el1.GetOrder(), el2.GetOrder());
            }
            else
            {
                order = 2*el1.GetOrder();
            }
            ir = &IntRules.Get(Trans.GetGeometryType(), order);
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            IntegrationPoint eip1, eip2;

            Trans.Loc1.Transform(ip, eip1);

            Trans.Face->SetIntPoint(&ip);
            if (dim == 1)
            {
                nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
                CalcOrtho(Trans.Face->Jacobian(), nor);
            }

            Trans.Elem1->SetIntPoint(&eip1);
            double h_E = Trans.Elem1->Weight() / nor.Norml2();

            double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * nor.Norml2() / h_E;

            el1.CalcShape(eip1, shape1);
            for (int i=0; i<ndof1; ++i)
                for (int j=0; j<ndof1; ++j)
                    elmat(i, j) += w * shape1(i) * shape1(j);

            // interior facet
            if (ndof2 > 0)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, j+ndof1) -= w * shape1(i) * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i+ndof1, j) -= w * shape2(i) * shape1(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i+ndof1, j+ndof2) += w * shape2(i) * shape2(j);
            }
        }
    }
};


/* 计算(边界或者内部Face都可以):
 *   q <h^{-1} [u], [v]>_E,
 *
 * u is given GridFunction, v is test function
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


/* 计算(边界或者内部Face都可以):
 *    q <{Q grad(u).n}, [v]>_E,
 *
 * u is trial function, v is test function
 * q are Coefficient. q在边E的两边连续
 * */
class DGSelfTraceIntegrator_6 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q, *q;

    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    DGSelfTraceIntegrator_6(Coefficient* q_, Coefficient* Q_): q(q_), Q(Q_) {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1, ndof2, ndofs;
        double w, wq = 0.0;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);

        shape1.SetSize(ndof1);
        dshape1.SetSize(ndof1, dim);
        dshape1dn.SetSize(ndof1);
        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
            dshape2.SetSize(ndof2, dim);
            dshape2dn.SetSize(ndof2);
        }
        else
        {
            ndof2 = 0;
        }

        ndofs = ndof1 + ndof2;
        elmat.SetSize(ndofs);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order;
            if (ndof2)
            {
                order = 2*max(el1.GetOrder(), el2.GetOrder());
            }
            else
            {
                order = 2*el1.GetOrder();
            }
            ir = &IntRules.Get(Trans.GetGeometryType(), order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);
            IntegrationPoint eip1, eip2;

            Trans.Loc1.Transform(ip, eip1);
            Trans.SetIntPoint(&ip);
            if (dim == 1)
            {
                nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
                CalcOrtho(Trans.Jacobian(), nor);
            }

            el1.CalcShape(eip1, shape1);
            el1.CalcDShape(eip1, dshape1);
            Trans.Elem1->SetIntPoint(&eip1);
            w = ip.weight/Trans.Elem1->Weight();
            if (ndof2)
            {
                w /= 2;
            }

            w *= Q->Eval(*Trans.Elem1, eip1);
            ni.Set(w, nor);

            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);

            // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
            // independent of Loc1 and always gives the size of element 1 in
            // direction perpendicular to the face. Indeed, for linear transformation
            //     |nor|=measure(face)/measure(ref. face),
            //   det(J1)=measure(element)/measure(ref. element),
            // and the ratios measure(ref. element)/measure(ref. face) are
            // compatible for all element/face pairs.
            // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
            // for any tetrahedron vol(tet)=(1/3)*height*area(base).
            // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

            dshape1.Mult(nh, dshape1dn);
            for (int i = 0; i < ndof1; i++)
                for (int j = 0; j < ndof1; j++)
                {
                    elmat(i, j) += shape1(i) * dshape1dn(j);
                }

            if (ndof2)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                Trans.Elem2->SetIntPoint(&eip2);
                w = ip.weight/2/Trans.Elem2->Weight();
                w *= Q->Eval(*Trans.Elem2, eip2);
                ni.Set(w, nor);

                CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
                adjJ.Mult(ni, nh);

                dshape2.Mult(nh, dshape2dn);

                for (int i = 0; i < ndof1; i++)
                    for (int j = 0; j < ndof2; j++)
                    {
                        elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
                    }

                for (int i = 0; i < ndof2; i++)
                    for (int j = 0; j < ndof1; j++)
                    {
                        elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
                    }

                for (int i = 0; i < ndof2; i++)
                    for (int j = 0; j < ndof2; j++)
                    {
                        elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
                    }
            }
        }

        for (int i = 0; i < ndofs; i++)
        {
            for (int j = 0; j < i; j++)
            {
                double aij = elmat(i,j), aji = elmat(j,i);
                elmat(i,j) =  - aij;
                elmat(j,i) =  - aji;
            }
            elmat(i,i) *= ( - 1.);
        }
    }
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

    void Test_DGSelfTraceIntegrator_3()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;

        {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(sin_coeff));
            blf1.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0, 0));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(sin_coeff));
            blf1.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0, 0));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1);

            BilinearForm blf2(&fsp);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 1.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 1.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }

        {
            ConstantCoefficient neg(-1.0);

            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(sin_coeff));
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&neg, &sin_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(sin_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&neg, &sin_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1);

            BilinearForm blf2(&fsp);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 1.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 1.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
    }

    void Test_DGSelfTraceIntegrator_6()
    {
        Mesh* mesh = new Mesh(60, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);

        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;

        Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

        BilinearForm blf1(&fsp);
        blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&neg, &sin_coeff));
        blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&neg, &sin_coeff));
        blf1.Assemble();
        blf1.Finalize();
        blf1.Mult(rand_gf, out1);

        BilinearForm blf2(&fsp);
        blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 0.0));
        blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 0.0));
        blf2.Assemble();
        blf2.Finalize();
        blf2.Mult(rand_gf, out2);

//        out1.Print(cout << "out1: ", fsp.GetTrueVSize());
//        out2.Print(cout << "out2: ", fsp.GetTrueVSize());
        for (int i=0; i<fsp.GetVSize(); ++i)
            assert(abs(out1[i] - out2[i]) < 1E-10);

    }
}

void Test_DGSelfTraceIntegrator()
{
    using namespace _DGSelfTraceIntegrator;
    Test_DGSelfTraceIntegrator_1();
    Test_DGSelfTraceIntegrator_2();
    Test_DGSelfBdrFaceIntegrator_1();

    Test_DGSelfTraceIntegrator_3();

    Test_DGSelfTraceIntegrator_6();

    cout << "===> Test Pass: DGSelfTraceIntegrator.hpp" << endl;
}

#endif //LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP
