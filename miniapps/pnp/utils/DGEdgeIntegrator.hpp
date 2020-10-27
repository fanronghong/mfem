#ifndef MFEM_DGEDGEINTEGRATOR1_HPP
#define MFEM_DGEDGEINTEGRATOR1_HPP

#include <iostream>
#include "mfem.hpp"
#include "DGDiffusion_Edge_Symmetry_Penalty.hpp"
using namespace std;
using namespace mfem;


/* 单元边界和计算区域边界 的Facet积分:
 *
 *     q <{u grad(w).n}, [v]>,
 *
 * u is Trial function, v is Test function
 * q is given Coefficient, q在边E的两边连续
 * w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
 * */
class DGEdgeBLFIntegrator1: public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GridFunction& w;
    GradientGridFunctionCoefficient* gradw;

    Vector nor, shape1, shape2, grad_w;
    double val1, val2;

public:
    DGEdgeBLFIntegrator1(GridFunction &w_) : Q(NULL), w(w_)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    DGEdgeBLFIntegrator1(Coefficient &q, GridFunction &w_) : Q(&q), w(w_)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    ~DGEdgeBLFIntegrator1() { delete gradw; }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1(0), ndof2(0);

        dim = el1.GetDim();

        grad_w.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        if (Trans.Elem2No >= 0) // 内部边界
        {
            ndof2 = el2.GetDof(); // 后面判断是否是内部边界可以通过判断 ndof2 是否为0判断
        }

        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);

        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (ndof2)
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

            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            el1.CalcShape(eip1, shape1);
            el2.CalcShape(eip2, shape2);

            if (dim == 1) {
                nor(0) = 2 * eip1.x - 1.0;
            }
            else {
                CalcOrtho(Trans.Jacobian(), nor); // 计算Face的法向量
            }

            gradw->Eval(grad_w, *Trans.Elem1, eip1); // fff 并行不会出错, 对比下面
            val1 = ip.weight * Q->Eval(*Trans.Elem1, eip1) * (grad_w * nor);

            if (Trans.Elem2No >= 0) // 内部边界
            {
                val1 *= 0.5;
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);

                gradw->Eval(grad_w, *Trans.Elem2, eip2); // fff 并行会出错, 对比上面
//                w.GetGradient(*Trans.Elem2, grad_w); // 同上

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
            else
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);
            }
        }
    }
};


/* 单元边界和计算区域边界 的Facet积分:
 *
 *     <[u], {q v grad(w).n}>,
 *
 * u is Trial function, v is Test function
 * q is given Coefficient, q在边E的两边连续
 * w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
 * */
class DGEdgeBLFIntegrator2 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;
    const Mesh* mesh;
    int marker;

    Vector nor, shape1, shape2, grad_w_T1, grad_w_T2;
    double Q_T1, Q_T2;

public:
    DGEdgeBLFIntegrator2(Coefficient &q, GridFunction &w)
            : Q(&q), mesh(NULL)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    DGEdgeBLFIntegrator2(Coefficient &q, GridFunction &w, const Mesh* mesh_, int marker_)
            : Q(&q), mesh(mesh_), marker(marker_)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    ~DGEdgeBLFIntegrator2() { delete gradw; }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1(0), ndof2(0);

        dim = el1.GetDim();
        grad_w_T1.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);

        if (Trans.Elem2No >= 0)
        {
            grad_w_T2.SetSize(dim);
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }

        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        if (mesh) {
            const Element *elm1, *elm2;
            int attr1, attr2;
            elm1 = mesh->GetElement(Trans.Elem1No);
            attr1 = elm1->GetAttribute();
            if (Trans.Elem2No >= 0) {
                elm2 = mesh->GetElement(Trans.Elem2No);
                attr2 = elm2->GetAttribute();
                if (attr1 != marker || attr2 != marker) return;
            } else {
                if (attr1 != marker) return;
            }
        }

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
            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            el1.CalcShape(eip1, shape1);
            el2.CalcShape(eip2, shape2);

            if (dim == 1)
                nor(0) = 2*eip1.x - 1.0;
            else
                CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量

            gradw->Eval(grad_w_T1, *Trans.Elem1, eip1);
            Q_T1 = Q->Eval(*Trans.Elem1, eip1);

            double val = ip.weight;
            if (ndof2) val *= 0.5;

            for (int i=0; i<ndof1; ++i)
                for (int j=0; j<ndof1; ++j)
                    elmat(i, j) += val * Q_T1 * (grad_w_T1 * nor) * shape1(i) * shape1(j);

            if (ndof2)
            {
                gradw->Eval(grad_w_T2, *Trans.Elem2, eip2);
                Q_T2 = Q->Eval(*Trans.Elem2, eip2);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, ndof1+j) -= val * Q_T1 * (grad_w_T1 * nor) * shape1(i) * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(ndof1+i, j) += val * Q_T2 * (grad_w_T2 * nor) * shape2(i) * shape1(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(ndof1+i, ndof1+j) -= val * Q_T2 * (grad_w_T2 * nor) * shape2(i) * shape2(j);
            }
        }
    }
};

/* 计算
 *
 *    <[q1 u], {v q2.n}>,
 *
 * u is Trial function, v is Test function.
 * q1 is Coefficient, q2 is VectorCoefficient
 * */
class DGEdgeBLFIntegrator3 : public BilinearFormIntegrator
{
protected:
    Coefficient &q1;
    VectorCoefficient& q2;
    Vector shape1, shape2, nor, q2_val_T1, q2_val_T2;

public:
    DGEdgeBLFIntegrator3(Coefficient& q1_, VectorCoefficient& q2_): q1(q1_), q2(q2_) {}
    ~DGEdgeBLFIntegrator3() {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement& el1,
                                    const FiniteElement& el2,
                                    FaceElementTransformations& Trans,
                                    DenseMatrix& elmat)
    {
        int dim(0), ndof1(0), ndof2(0);

        dim = el1.GetDim();
        nor.SetSize(dim);
        q2_val_T1.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0) {
            q2_val_T2.SetSize(dim);
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }

        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (ndof2)
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
            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            el1.CalcShape(eip1, shape1);
            el2.CalcShape(eip2, shape2);

            if (dim == 1) {
                nor(0) = 2 * eip1.x - 1.0;
            }
            else {
                CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量
            }

            double val = ip.weight;
            if (ndof2) val *= 0.5;

            double q1_val_T1 = q1.Eval(*Trans.Elem1, eip1);
            q2.Eval(q2_val_T1, *Trans.Elem1, eip1);

            for (int i=0; i<ndof1; ++i)
                for (int j=0; j<ndof1; ++j)
                    elmat(i, j) += (q2_val_T1*nor) * shape1(i) * q1_val_T1 * shape1(j) * val;

            if (ndof2)
            {
                double q1_val_T2 = q1.Eval(*Trans.Elem2, eip2);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, j+ndof1) -= (q2_val_T1*nor) * shape1(i) * q1_val_T2 * shape2(j) * val;

                q2.Eval(q2_val_T2, *Trans.Elem2, eip2);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i+ndof1, j) += (q2_val_T2*nor) * shape2(i) * q1_val_T1 * shape1(j) * val;

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i+ndof1, j+ndof1) -= (q2_val_T2*nor) * shape2(i) * q1_val_T2 * shape2(j) * val;
            }
        }
    }
};




/* 单元边界和计算区域边界 的Facet积分:
 *
 *     <[u], {q grad(v).n}>,
 *
 * v is Test function
 * u and q are Coefficient,
 * */
class DGEdgeLFIntegrator1 : public LinearFormIntegrator
{
protected:
    Coefficient *Q, *u;
    Vector nor, shape1, shape2, tmp1, tmp2;
    DenseMatrix adjJ1, adjJ2, dshape1, dshape2;

public:
    DGEdgeLFIntegrator1(Coefficient &q, Coefficient &u_) : Q(&q), u(&u_) {}
    ~DGEdgeLFIntegrator1() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect) {}

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();
        nor.SetSize(dim);
        tmp1.SetSize(dim);
        tmp2.SetSize(dim);
        adjJ1.SetSize(dim);
        dshape1.SetSize(ndof1, dim);

        {
            ndof2 = 0;
        }

        ndofs = ndof1 + ndof2;
        elvect.SetSize(ndofs);
        elvect = 0.0;
        if (ndof2 != 0) return;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order;
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
            Trans.SetIntPoint(&ip);
            if (dim == 1)
            {
                nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
                CalcOrtho(Trans.Face->Jacobian(), nor);
            }

            Trans.Elem1->SetIntPoint(&eip1);
            el1.CalcDShape(eip1, dshape1);
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ1);
            double j1 = Trans.Elem1->Weight();

            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1)
                           * u->Eval(*Trans.Elem1, eip1) / j1;

                adjJ1.Mult(nor, tmp1);
                Vector dummy(ndof1);
                dshape1.Mult(tmp1, dummy);
                elvect.Add(w, dummy);
            }
        }

    }

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();
        nor.SetSize(dim);
        tmp1.SetSize(dim);
        tmp2.SetSize(dim);
        adjJ1.SetSize(dim);
        dshape1.SetSize(ndof1, dim);

        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            adjJ2.SetSize(dim);
            dshape2.SetSize(ndof2, dim);
        } else
        {
            ndof2 = 0;
        }

        ndofs = ndof1 + ndof2;
        elvect.SetSize(ndofs);
        elvect = 0.0;
        if (ndof2 == 0) return;

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
            Trans.SetIntPoint(&ip);
            if (dim == 1)
            {
                nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
                CalcOrtho(Trans.Face->Jacobian(), nor);
            }

            Trans.Elem1->SetIntPoint(&eip1);
            el1.CalcDShape(eip1, dshape1);
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ1);
            double j1 = Trans.Elem1->Weight();

            if (Trans.Elem2No >= 0)
            {
                Trans.Loc2.Transform(ip, eip2);
                Trans.Elem2->SetIntPoint(&eip2);

                double u_val = 0.5 * (u->Eval(*Trans.Elem1, eip1)
                                      - u->Eval(*Trans.Elem2, eip2));
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * u_val;
                double j2 = Trans.Elem2->Weight();

                el2.CalcDShape(eip2, dshape2);
                CalcAdjugate(Trans.Elem2->Jacobian(), adjJ2);

                Vector dummy1(ndof1), dummy2(ndof2);

                adjJ1.Mult(nor, tmp1);
                dshape1.Mult(tmp1, dummy1);

                adjJ2.Mult(nor, tmp2);
                dshape2.Mult(tmp2, dummy2);

                for (int i=0; i<ndof1; ++i)
                    elvect(i) += w * dummy1(i) / j1;

                for (int i=0; i<ndof2; ++i)
                    elvect(i + ndof1) += w * dummy2(i) / j2;
            }
            else
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1)
                           * u->Eval(*Trans.Elem1, eip1) / j1;

                adjJ1.Mult(nor, tmp1);
                Vector dummy(ndof1);
                dshape1.Mult(tmp1, dummy);
                elvect.Add(w, dummy);
            }
        }

    }
};



/* 计算区域边界积分
 *
 *     q<u_D, v grad(w).n>_E,
 *
 * v is Test function,
 * q and u_D are Coefficient, w is GridFunction.
 * */
class DGEdgeLFIntegrator2: public LinearFormIntegrator
{
protected:
    Coefficient *q, *u_D;
    GradientGridFunctionCoefficient *gradw;

    Vector nor, shape, grad_w;

public:
    DGEdgeLFIntegrator2(Coefficient *q_, Coefficient *u_D_, GridFunction* w)
            : q(q_), u_D(u_D_) { gradw = new GradientGridFunctionCoefficient(w); }

    // 单元积分: (. , .)
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("Not Support");
    }

    // 单元边界积分: <. , .>
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
            Tr.SetAllIntPoints(&ip);

            const IntegrationPoint &eip = Tr.GetElement1IntPoint();

            if (dim == 1)
                nor(0) = 2*eip.x - 1.0; //1维参考单元的法向量(不是单位的)
            else
                CalcOrtho(Tr.Face->Jacobian(), nor);

            el.CalcShape(eip, shape);

            gradw->Eval(grad_w, *Tr.Elem1, eip);

            double val = ip.weight;
//            val *= q->Eval(*Tr.Face, ip);
//            val *= u_D->Eval(*Tr.Face, ip);
            val *= q->Eval(*Tr.Elem1, eip); // 与上面等价
            val *= u_D->Eval(*Tr.Elem1, eip);
            val *= (nor * grad_w);

            shape *= val;
            elvect += shape;
        }
    }
};


/* 计算单元边界积分
 *
 *    q <[Q1], {Q2 grad(u).n v}>
 *
 * u is Trial function, v is Test function,
 * q, Q1 and Q2 are Coefficient.
 * */
class DGEdgeIntegrator3: public BilinearFormIntegrator
{
protected:
    Coefficient *q, *Q1, *Q2;

    DenseMatrix dshape1, dshape2, dshape1dxt, dshape2dxt;
    Vector nor, shape1, shape2, vec1, vec2;

public:
    DGEdgeIntegrator3(Coefficient* q_, Coefficient *Q1_, Coefficient *Q2_): q(q_), Q1(Q1_), Q2(Q2_) {}
    ~DGEdgeIntegrator3() {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement& el1,
                                    const FiniteElement& el2,
                                    FaceElementTransformations& Trans,
                                    DenseMatrix& elmat)
    {
        int dim, ndof1, ndof2(0), ndofs;

        dim = el1.GetDim();
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        vec1.SetSize(ndof1);
        dshape1.SetSize(ndof1, dim);
        dshape1dxt.SetSize(ndof1, dim);

        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
            vec2.SetSize(ndof2);
            dshape2.SetSize(ndof2, dim);
            dshape2dxt.SetSize(ndof2, dim);
        }

        ndofs = ndof1 + ndof2;
        elmat.SetSize(ndofs);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (ndof2)
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

        for (int p=0; p<ir->GetNPoints(); ++p) {
            const IntegrationPoint &ip = ir->IntPoint(p);
            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            if (dim == 1) {
                nor(0) = 2 * eip1.x - 1.0;
            }
            else {
                CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量
            }

            el1.CalcShape(eip1, shape1);
            el1.CalcDShape(eip1, dshape1);
            Mult(dshape1, Trans.Elem1->AdjugateJacobian(), dshape1dxt);
            dshape1dxt.Mult(nor, vec1);

            if (Trans.Elem2No >= 0)
            {
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                Mult(dshape2, Trans.Elem2->AdjugateJacobian(), dshape2dxt);
                dshape2dxt.Mult(nor, vec2);

                double Q1_jump = Q1->Eval(*Trans.Elem1, eip1) - Q1->Eval(*Trans.Elem2, eip2);

                double w = ip.weight * q->Eval(*Trans.Elem1, eip1) * Q1_jump / 2; // ip.weight q [Q1] / 2

                for (int i=0; i<ndof1; ++i)
                {
                    for (int j=0; j<ndof1; ++j)
                    {
                        elmat(i, j) += shape1(i) * Q2->Eval(*Trans.Elem1, eip1) * vec1(j) / Trans.Elem1->Weight();
                    }
                }
                for (int i=0; i<ndof2; ++i)
                {
                    for (int j=0; j<ndof2; ++j)
                    {
                        elmat(i+ndof1, j+ndof1) += shape2(i) * Q2->Eval(*Trans.Elem2, eip2) * vec2(j) / Trans.Elem2->Weight();
                    }
                }
            }
            else
            {
                double w = ip.weight * q->Eval(*Trans.Elem1, eip1)
                                     * Q1->Eval(*Trans.Elem1, eip1)
                                     * Q2->Eval(*Trans.Elem1, eip1) / Trans.Elem1->Weight();

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * vec1(j);
            }
        }
    }
};




namespace _DGEdgeIntegrator
{
    double sin_cfun(const Vector& x)
    {
        return sin(x[0]) * sin(x[1]); // sin(x) * sin(y)
    }
    double cos_cfun(const Vector& x)
    {
        return cos(x[0]) * cos(x[1]);
    }
    FunctionCoefficient sin_coeff(sin_cfun);
    ConstantCoefficient one(1.0);
    ConstantCoefficient neg(-1.0);

    bool TEST_ALL = true;

    void Test_DGEdgeBLFIntegrator1_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);
        GradientGridFunctionCoefficient grad_rand(&rand_gf);

        // 只在区域内部边积分
        if (TEST_ALL) {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=sin
            blf1.AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i) {
                assert(abs(out1[i] - out2[i]) < 1E-10);
            }
        }

        // 只在区域内部边积分
        if (TEST_ALL) {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=rand_gf
            blf1.AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg, rand_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i) {
                assert(abs(out1[i] - out2[i]) < 1E-10);
            }
        }

        // 只在区域边界积分
        if (TEST_ALL) {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            //  q <{u grad(w).n}, [v]>, q=neg, w=sin_gf
            blf1.AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i) {
                assert(abs(out1[i] - out2[i]) < 1E-10);
            }
        }

        // 在区域内部边界和区域外部边界积分
        if (TEST_ALL) {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=rand_gf
            blf1.AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg, rand_gf));
            blf1.AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg, rand_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i) {
                assert(abs(out1[i] - out2[i]) < 1E-10);
            }
        }

        delete mesh;
    }

    void Test_DGEdgeBLFIntegrator2_and_DGEdgeBLFIntegrator3_1()
    {
        Mesh* mesh = new Mesh(20, 20, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);
        GradientGridFunctionCoefficient grad_rand(&rand_gf);

        // 测试区域内部边界和区域外部边界
        if (TEST_ALL)
        {
            Vector out1(size), out2(size);

            BilinearForm blf1(&fsp);
            // <[u], {q v grad(w).n}>
            blf1.AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(one, rand_gf));
            blf1.AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(one, rand_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // <[rand], {v grad(rand).n}>

            BilinearForm blf2(&fsp);
            // <[q1 u], {v q2.n}>
            blf2.AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator3(one, grad_rand));
            blf2.AddBdrFaceIntegrator(new DGEdgeBLFIntegrator3(one, grad_rand));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2); // <[rand], {v grad_rand.n}>

            for (int i=0; i<size; ++i)
            {
                assert(abs(out1[i] - out2[i]) < 1E-10);
            }
        }

        delete mesh;
    }



    void Test_DGEdgeLFIntegrator1_1()
    {
        Mesh* mesh = new Mesh(100, 100, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);

        // 区域内部边界积分
        if (TEST_ALL) {
            Vector out(size);

            BilinearForm blf(&fsp);
            // sigma <[u], {q grad(v).n}>
            blf.AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(one, 1.0));
            blf.Assemble();
            blf.Finalize();
            blf.Mult(rand_gf, out); // <[rand], {grad(v).n}>

            LinearForm lf(&fsp);
            // <[u], {q grad(v).n}>
            lf.AddInteriorFaceIntegrator(new DGEdgeLFIntegrator1(one,rand_coeff));
            lf.Assemble(); // <[rand], {grad(v).n}>

            for (int i=0; i<size; ++i)
            {
                assert (fabs(out[i] - lf[i]) < 1E-10);
            }
        }

        // 区域外部边界积分
        if (TEST_ALL) {
            Vector out(size);

            BilinearForm blf(&fsp);
            // sigma <[u], {q grad(v).n}>
            blf.AddBdrFaceIntegrator(new DGDiffusion_Symmetry(one, 1.0));
            blf.Assemble();
            blf.Finalize();
            blf.Mult(rand_gf, out); // <[rand], {grad(v).n}>

            LinearForm lf(&fsp);
            // <[u], {q grad(v).n}>
            lf.AddBdrFaceIntegrator(new DGEdgeLFIntegrator1(one,rand_coeff));
            lf.Assemble(); // <[rand], {grad(v).n}>

            for (int i=0; i<size; ++i)
            {
                assert (fabs(out[i] - lf[i]) < 1E-10);
            }
        }

        // 区域内部边界和外部边界积分
        if (TEST_ALL) {
            Vector out(size);

            BilinearForm blf(&fsp);
            // sigma <[u], {q grad(v).n}>
            blf.AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(one, 1.0));
            blf.AddBdrFaceIntegrator(new DGDiffusion_Symmetry(one, 1.0));
            blf.Assemble();
            blf.Finalize();
            blf.Mult(rand_gf, out); // <[rand], {grad(v).n}>

            LinearForm lf(&fsp);
            // <[u], {q grad(v).n}>
            lf.AddInteriorFaceIntegrator(new DGEdgeLFIntegrator1(one,rand_coeff));
            lf.AddBdrFaceIntegrator(new DGEdgeLFIntegrator1(one,rand_coeff));
            lf.Assemble(); // <[rand], {grad(v).n}>

            for (int i=0; i<size; ++i)
            {
                assert(abs(out[i] - lf[i]) < 1E-7);
            }
        }
    }

}


void Test_DGEdgeIntegrator()
{
    using namespace _DGEdgeIntegrator;

    Test_DGEdgeBLFIntegrator1_1();
    Test_DGEdgeBLFIntegrator2_and_DGEdgeBLFIntegrator3_1();

    Test_DGEdgeLFIntegrator1_1();

    /* ----------  ------------ */
}
#endif
