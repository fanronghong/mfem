
#ifndef MFEM_DGDIFFUSION_EDGE_SYMMETRY_PENALTY_HPP
#define MFEM_DGDIFFUSION_EDGE_SYMMETRY_PENALTY_HPP

#include <iostream>
#include "mfem.hpp"
using namespace std;
using namespace mfem;


/** Integrator for the DG form:
    * ref: DGDiffusionIntegrator, 下面我们把这个积分子拆分成3个单独的积分子. 注意: 我们可以把 symmetry 看出 edge 的转置.
    * 下面只把DGDiffusionIntegrator里面的部分代码修改掉, 理论上来讲应该不会有bug.

    - < {(Q grad(u)).n}, [v] >           ------> edge
    + sigma < [u], {(Q grad(v)).n} >     ------> symmetry
    + kappa < {h^{-1} Q} [u], [v] >,     ------> penalty

    where Q is a scalar or matrix diffusion coefficient and u, v are the trial
    and test spaces, respectively. The parameters sigma and kappa determine the
    DG method to be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
    * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
    * sigma = +1, kappa = 0: the method of Baumann and Oden. */
class DGDiffusion_Edge : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    // 只修改了下面三个构造函数: 把sigma, kappa设成0.0
    // AssembleFaceMatrix()里面没有任何改变, 所以不可能出错
    DGDiffusion_Edge()
            : Q(NULL), MQ(NULL), sigma(0.0), kappa(0.0) { }
    DGDiffusion_Edge(Coefficient &q)
            : Q(&q), MQ(NULL), sigma(0.0), kappa(0.0) { }
    DGDiffusion_Edge(MatrixCoefficient &q)
            : Q(NULL), MQ(&q), sigma(0.0), kappa(0.0) { }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)

    {
        int dim, ndof1, ndof2, ndofs;
        bool kappa_is_nonzero = (kappa != 0.);
        double w, wq = 0.0;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

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
        if (kappa_is_nonzero)
        {
            jmat.SetSize(ndofs);
            jmat = 0.;
        }

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

        // assemble: < {(Q \nabla u).n},[v] >      --> elmat
        //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

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
            w = ip.weight/Trans.Elem1->Weight();
            if (ndof2)
            {
                w /= 2;
            }
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Trans.Elem1, eip1);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Trans.Elem1, eip1);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            if (kappa_is_nonzero)
            {
                wq = ni * nor;
            }
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
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                w = ip.weight/2/Trans.Elem2->Weight();
                if (!MQ)
                {
                    if (Q)
                    {
                        w *= Q->Eval(*Trans.Elem2, eip2);
                    }
                    ni.Set(w, nor);
                }
                else
                {
                    nh.Set(w, nor);
                    MQ->Eval(mq, *Trans.Elem2, eip2);
                    mq.MultTranspose(nh, ni);
                }
                CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
                adjJ.Mult(ni, nh);
                if (kappa_is_nonzero)
                {
                    wq += ni * nor;
                }

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

            if (kappa_is_nonzero)
            {
                // only assemble the lower triangular part of jmat
                wq *= kappa;
                for (int i = 0; i < ndof1; i++)
                {
                    const double wsi = wq*shape1(i);
                    for (int j = 0; j <= i; j++)
                    {
                        jmat(i, j) += wsi * shape1(j);
                    }
                }
                if (ndof2)
                {
                    for (int i = 0; i < ndof2; i++)
                    {
                        const int i2 = ndof1 + i;
                        const double wsi = wq*shape2(i);
                        for (int j = 0; j < ndof1; j++)
                        {
                            jmat(i2, j) -= wsi * shape1(j);
                        }
                        for (int j = 0; j <= i; j++)
                        {
                            jmat(i2, ndof1 + j) += wsi * shape2(j);
                        }
                    }
                }
            }
        }

        // elmat := -elmat + sigma*elmat^t + jmat
        if (kappa_is_nonzero)
        {
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
                    elmat(i,j) = sigma*aji - aij + mij;
                    elmat(j,i) = sigma*aij - aji + mij;
                }
                elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
            }
        }
        else
        {
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i);
                    elmat(i,j) = sigma*aji - aij;
                    elmat(j,i) = sigma*aij - aji;
                }
                elmat(i,i) *= (sigma - 1.);
            }
        }
    }
};
class DGDiffusion_Symmetry : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    // 第一处修改: 把下面三个构造函数的kappa全部设为0.0
    DGDiffusion_Symmetry(const double s)
            : Q(NULL), MQ(NULL), sigma(s), kappa(0.0) { }
    DGDiffusion_Symmetry(Coefficient &q, const double s)
            : Q(&q), MQ(NULL), sigma(s), kappa(0.0) { }
    DGDiffusion_Symmetry(MatrixCoefficient &q, const double s)
            : Q(NULL), MQ(&q), sigma(s), kappa(0.0) { }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)

    {
        int dim, ndof1, ndof2, ndofs;
        bool kappa_is_nonzero = (kappa != 0.);
        double w, wq = 0.0;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

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
        if (kappa_is_nonzero)
        {
            jmat.SetSize(ndofs);
            jmat = 0.;
        }

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

        // assemble: < {(Q \nabla u).n},[v] >      --> elmat
        //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

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
            w = ip.weight/Trans.Elem1->Weight();
            if (ndof2)
            {
                w /= 2;
            }
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Trans.Elem1, eip1);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Trans.Elem1, eip1);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            if (kappa_is_nonzero)
            {
                wq = ni * nor;
            }
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
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                w = ip.weight/2/Trans.Elem2->Weight();
                if (!MQ)
                {
                    if (Q)
                    {
                        w *= Q->Eval(*Trans.Elem2, eip2);
                    }
                    ni.Set(w, nor);
                }
                else
                {
                    nh.Set(w, nor);
                    MQ->Eval(mq, *Trans.Elem2, eip2);
                    mq.MultTranspose(nh, ni);
                }
                CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
                adjJ.Mult(ni, nh);
                if (kappa_is_nonzero)
                {
                    wq += ni * nor;
                }

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

            if (kappa_is_nonzero)
            {
                // only assemble the lower triangular part of jmat
                wq *= kappa;
                for (int i = 0; i < ndof1; i++)
                {
                    const double wsi = wq*shape1(i);
                    for (int j = 0; j <= i; j++)
                    {
                        jmat(i, j) += wsi * shape1(j);
                    }
                }
                if (ndof2)
                {
                    for (int i = 0; i < ndof2; i++)
                    {
                        const int i2 = ndof1 + i;
                        const double wsi = wq*shape2(i);
                        for (int j = 0; j < ndof1; j++)
                        {
                            jmat(i2, j) -= wsi * shape1(j);
                        }
                        for (int j = 0; j <= i; j++)
                        {
                            jmat(i2, ndof1 + j) += wsi * shape2(j);
                        }
                    }
                }
            }
        }

        // elmat := -elmat + sigma*elmat^t + jmat
        // 第二处修改: 只保留 sigma*elemat^t. 注: 下面注释的内容为原本内容, 后面紧接着是修改后的内容
        if (kappa_is_nonzero)
        {
            MFEM_ABORT("kappa must be zero for this integrator. Added by fan.")
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
                    elmat(i,j) = sigma*aji - aij + mij;
                    elmat(j,i) = sigma*aij - aji + mij;
                }
                elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
            }
        }
        else
        {
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i);
//                    elmat(i,j) = sigma*aji - aij;
//                    elmat(j,i) = sigma*aij - aji;
                    elmat(i,j) = sigma*aji;
                    elmat(j,i) = sigma*aij;
                }
//                elmat(i,i) *= (sigma - 1.);
                elmat(i,i) *= (sigma - 0.);
            }
        }
    }
};
class DGDiffusion_Penalty : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    // 第一处修改: 把sigma全部变成0.0
    DGDiffusion_Penalty(const double k)
            : Q(NULL), MQ(NULL), sigma(0.0), kappa(k) { }
    DGDiffusion_Penalty(Coefficient &q, const double k)
            : Q(&q), MQ(NULL), sigma(0.0), kappa(k) { }
    DGDiffusion_Penalty(MatrixCoefficient &q, const double k)
            : Q(NULL), MQ(&q), sigma(0.0), kappa(k) { }
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)

    {
        int dim, ndof1, ndof2, ndofs;
        bool kappa_is_nonzero = (kappa != 0.);
        double w, wq = 0.0;

        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

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
        if (kappa_is_nonzero)
        {
            jmat.SetSize(ndofs);
            jmat = 0.;
        }
        else return; // 第二处修改: kappa为0就直接退出

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

        // assemble: < {(Q \nabla u).n},[v] >      --> elmat
        //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

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
            w = ip.weight/Trans.Elem1->Weight();
            if (ndof2)
            {
                w /= 2;
            }
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Trans.Elem1, eip1);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Trans.Elem1, eip1);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            if (kappa_is_nonzero)
            {
                wq = ni * nor;
            }
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
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                w = ip.weight/2/Trans.Elem2->Weight();
                if (!MQ)
                {
                    if (Q)
                    {
                        w *= Q->Eval(*Trans.Elem2, eip2);
                    }
                    ni.Set(w, nor);
                }
                else
                {
                    nh.Set(w, nor);
                    MQ->Eval(mq, *Trans.Elem2, eip2);
                    mq.MultTranspose(nh, ni);
                }
                CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
                adjJ.Mult(ni, nh);
                if (kappa_is_nonzero)
                {
                    wq += ni * nor;
                }

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

            if (kappa_is_nonzero)
            {
                // only assemble the lower triangular part of jmat
                wq *= kappa;
                for (int i = 0; i < ndof1; i++)
                {
                    const double wsi = wq*shape1(i);
                    for (int j = 0; j <= i; j++)
                    {
                        jmat(i, j) += wsi * shape1(j);
                    }
                }
                if (ndof2)
                {
                    for (int i = 0; i < ndof2; i++)
                    {
                        const int i2 = ndof1 + i;
                        const double wsi = wq*shape2(i);
                        for (int j = 0; j < ndof1; j++)
                        {
                            jmat(i2, j) -= wsi * shape1(j);
                        }
                        for (int j = 0; j <= i; j++)
                        {
                            jmat(i2, ndof1 + j) += wsi * shape2(j);
                        }
                    }
                }
            }
        }

        // elmat := -elmat + sigma*elmat^t + jmat
        // 第三处修改: 只保留 jmat. 注: 下面注释的内容为原本内容, 后面紧接着是修改后的内容
        if (kappa_is_nonzero)
        {
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
//                    elmat(i,j) = sigma*aji - aij + mij;
//                    elmat(j,i) = sigma*aij - aji + mij;
                    elmat(i,j) = mij;
                    elmat(j,i) = mij;
                }
//                elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
                elmat(i,i) = jmat(i,i);
            }
        }
        else
        {
            for (int i = 0; i < ndofs; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    double aij = elmat(i,j), aji = elmat(j,i);
//                    elmat(i,j) = sigma*aji - aij;
//                    elmat(j,i) = sigma*aij - aji;
                    elmat(i,j) = 0.0;
                    elmat(j,i) = 0.0;
                }
//                elmat(i,i) *= (sigma - 1.);
                elmat(i,i) *= (0.0);
            }
        }
    }
};
class DGDiffusion_Penalty2: public BilinearFormIntegrator
{
/* 似乎不太确定 DGDiffusion_Penalty 是否有bug, 所以自己写了这个积分子, 以此来互相验证
 * */
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double kappa;

    // these are not thread-safe!
    Vector shape1, shape2, nor;

public:
    DGDiffusion_Penalty2(const double k)
            : Q(NULL), MQ(NULL), kappa(k) { }
    DGDiffusion_Penalty2(Coefficient &q, const double k)
    : Q(&q), MQ(NULL), kappa(k) { }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        bool kappa_is_nonzero = (kappa != 0.);
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }
        else ndof2 = 0;

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
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            // Access the neighboring elements' integration points
            // Note: eip2 will only contain valid data if Elem2 exists
            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            if (dim == 1)
            {
                nor(0) = 2*eip1.x - 1.0;
            }
            else
            {
                CalcOrtho(Trans.Jacobian(), nor);
            }

            el1.CalcShape(eip1, shape1);

            if (Trans.Elem2No >= 0) // 内部边界
            {
                el2.CalcShape(eip2, shape2);
                Trans.SetAllIntPoints(&ip);

                double h1_E = (nor * nor) / Trans.Elem1->Weight();
                if (Q) h1_E *= Q->Eval(*Trans.Elem1, eip1);
                Trans.SetAllIntPoints(&ip);

                double h2_E = (nor * nor) / Trans.Elem2->Weight();
                if (Q) h2_E *= Q->Eval(*Trans.Elem2, eip2);

                double w = h1_E + h2_E;
                w *= ip.weight * kappa / 2;

                for (int i=0; i<ndof1; ++i)
                    for (int j = 0; j < ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * shape1(j);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, j+ndof1) -= w * shape1(i) * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i + ndof1, j) -= w * shape2(i) * shape1(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i + ndof1, j+ndof1) += w * shape2(i) * shape2(j);
            }
            else // 区域边界
            {
                double h_E = (nor * nor) / Trans.Elem1->Weight();

                double w = ip.weight * kappa * h_E;
                if (Q) w *= Q->Eval(*Trans.Elem1, eip1);

                for (int i=0; i<ndof1; ++i)
                {
                    for (int j=0; j<ndof1; ++j)
                    {
                        elmat(i, j) += w * shape1(i) * shape1(j);
                    }
                }
            }
        }
    }
};


/** Boundary linear integrator for imposing non-zero Dirichlet boundary
    conditions, to be used in conjunction with DGDiffusionIntegrator.
    Specifically, given the Dirichlet data u_D, the linear form assembles the
    following integrals on the boundary:

    ref: DGDirichletLFIntegrator, 下面我们把这个积分子拆分成2个单独的积分子.
    下面只把DGDirichletLFIntegrator里面的部分代码修改掉, 理论上不应该会有bug.

    sigma < u_D, (Q grad(v)).n >      ------> symmetry
    + kappa < {h^{-1} Q} u_D, v >,    ------> penalty

    where Q is a scalar or matrix diffusion coefficient and v is the test
    function. The parameters sigma and kappa should be the same as the ones
    used in the DGDiffusionIntegrator. */
class DGDirichletLF_Symmetry : public LinearFormIntegrator
{
protected:
    Coefficient *uD, *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape, dshape_dn, nor, nh, ni;
    DenseMatrix dshape, mq, adjJ;

public:
    // 只修改一个地方: 把下面三个构造函数的参数kappa设为0.0
    DGDirichletLF_Symmetry(Coefficient &u, const double s)
            : uD(&u), Q(NULL), MQ(NULL), sigma(s), kappa(0.0) { }
    DGDirichletLF_Symmetry(Coefficient &u, Coefficient &q, const double s)
            : uD(&u), Q(&q), MQ(NULL), sigma(s), kappa(0.0) { }
    DGDirichletLF_Symmetry(Coefficient &u, MatrixCoefficient &q,
                            const double s)
            : uD(&u), Q(NULL), MQ(&q), sigma(s), kappa(0.0) { }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect)

    {
        int dim, ndof;
        bool kappa_is_nonzero = (kappa != 0.);
        double w;

        dim = el.GetDim();
        ndof = el.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

        shape.SetSize(ndof);
        dshape.SetSize(ndof, dim);
        dshape_dn.SetSize(ndof);

        elvect.SetSize(ndof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order = 2*el.GetOrder();
            ir = &IntRules.Get(Tr.GetGeometryType(), order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring element
            Tr.SetAllIntPoints(&ip);

            // Access the neighboring element's integration point
            const IntegrationPoint &eip = Tr.GetElement1IntPoint();

            if (dim == 1)
            {
                nor(0) = 2*eip.x - 1.0;
            }
            else
            {
                CalcOrtho(Tr.Jacobian(), nor);
            }

            el.CalcShape(eip, shape);
            el.CalcDShape(eip, dshape);

            // compute uD through the face transformation
            w = ip.weight * uD->Eval(Tr, ip) / Tr.Elem1->Weight();
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Tr.Elem1, eip);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Tr.Elem1, eip);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);

            dshape.Mult(nh, dshape_dn);
            elvect.Add(sigma, dshape_dn);

            if (kappa_is_nonzero)
            {
                elvect.Add(kappa*(ni*nor), shape);
            }
        }
    }
};
class DGDirichletLF_Penalty : public LinearFormIntegrator
{
protected:
    Coefficient *uD, *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape, dshape_dn, nor, nh, ni;
    DenseMatrix dshape, mq, adjJ;

public:
    // 只修改一个地方: 把下面3个构造函数的参数sigma设为0.0
    DGDirichletLF_Penalty(Coefficient &u, const double k)
            : uD(&u), Q(NULL), MQ(NULL), sigma(0.0), kappa(k) { }
    DGDirichletLF_Penalty(Coefficient &u, Coefficient &q, const double k)
            : uD(&u), Q(&q), MQ(NULL), sigma(0.0), kappa(k) { }
    DGDirichletLF_Penalty(Coefficient &u, MatrixCoefficient &q, const double k)
            : uD(&u), Q(NULL), MQ(&q), sigma(0.0), kappa(k) { }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect)

    {
        int dim, ndof;
        bool kappa_is_nonzero = (kappa != 0.);
        double w;

        dim = el.GetDim();
        ndof = el.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

        shape.SetSize(ndof);
        dshape.SetSize(ndof, dim);
        dshape_dn.SetSize(ndof);

        elvect.SetSize(ndof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order = 2*el.GetOrder();
            ir = &IntRules.Get(Tr.GetGeometryType(), order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring element
            Tr.SetAllIntPoints(&ip);

            // Access the neighboring element's integration point
            const IntegrationPoint &eip = Tr.GetElement1IntPoint();

            if (dim == 1)
            {
                nor(0) = 2*eip.x - 1.0;
            }
            else
            {
                CalcOrtho(Tr.Jacobian(), nor);
            }

            el.CalcShape(eip, shape);
            el.CalcDShape(eip, dshape);

            // compute uD through the face transformation
            w = ip.weight * uD->Eval(Tr, ip) / Tr.Elem1->Weight();
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Tr.Elem1, eip);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Tr.Elem1, eip);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);

            dshape.Mult(nh, dshape_dn);
            elvect.Add(sigma, dshape_dn);

            if (kappa_is_nonzero)
            {
                elvect.Add(kappa*(ni*nor), shape);
            }
        }
    }
};


/* 由上面的 DGDirichletLF_Symmetry 改变而来: 把sigma始终设为1.0
 *
 *     < u_D, (Q grad(v)).n >, 只在Dirichlet边界积分
 *
 * 所以理论上来讲不会有bug
 * */
class DGWeakDirichlet_LFIntegrator : public LinearFormIntegrator
{
protected:
    Coefficient *uD, *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;

    // these are not thread-safe!
    Vector shape, dshape_dn, nor, nh, ni;
    DenseMatrix dshape, mq, adjJ;

public:
    // 只修改一个地方: 把下面三个构造函数的参数sigma设为1.0
    DGWeakDirichlet_LFIntegrator(Coefficient &u)
            : uD(&u), Q(NULL), MQ(NULL), sigma(1.0), kappa(0.0) { }
    DGWeakDirichlet_LFIntegrator(Coefficient &u, Coefficient &q)
            : uD(&u), Q(&q), MQ(NULL), sigma(1.0), kappa(0.0) { }
    DGWeakDirichlet_LFIntegrator(Coefficient &u, MatrixCoefficient &q)
            : uD(&u), Q(NULL), MQ(&q), sigma(1.0), kappa(0.0) { }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect)

    {
        int dim, ndof;
        bool kappa_is_nonzero = (kappa != 0.);
        double w;

        dim = el.GetDim();
        ndof = el.GetDof();

        nor.SetSize(dim);
        nh.SetSize(dim);
        ni.SetSize(dim);
        adjJ.SetSize(dim);
        if (MQ)
        {
            mq.SetSize(dim);
        }

        shape.SetSize(ndof);
        dshape.SetSize(ndof, dim);
        dshape_dn.SetSize(ndof);

        elvect.SetSize(ndof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order = 2*el.GetOrder();
            ir = &IntRules.Get(Tr.GetGeometryType(), order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring element
            Tr.SetAllIntPoints(&ip);

            // Access the neighboring element's integration point
            const IntegrationPoint &eip = Tr.GetElement1IntPoint();

            if (dim == 1)
            {
                nor(0) = 2*eip.x - 1.0;
            }
            else
            {
                CalcOrtho(Tr.Jacobian(), nor);
            }

            el.CalcShape(eip, shape);
            el.CalcDShape(eip, dshape);

            // compute uD through the face transformation
            w = ip.weight * uD->Eval(Tr, ip) / Tr.Elem1->Weight();
            if (!MQ)
            {
                if (Q)
                {
                    w *= Q->Eval(*Tr.Elem1, eip);
                }
                ni.Set(w, nor);
            }
            else
            {
                nh.Set(w, nor);
                MQ->Eval(mq, *Tr.Elem1, eip);
                mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);

            dshape.Mult(nh, dshape_dn);
            elvect.Add(sigma, dshape_dn);

            if (kappa_is_nonzero)
            {
                elvect.Add(kappa*(ni*nor), shape);
            }
        }
    }
};
/* 参考上面的 DGWeakDirichlet_LFIntegrator 修改而来: 只要替换u_D为u即可
 *
 *     <u, (Q grad(v)).n >, 只在Dirichlet边界积分
 *
 * */
class DGWeakDirichlet_BLFIntegrator : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    DenseMatrix dshape1, adjJ;
    Vector nor, shape1, work1, work2;

public:
    DGWeakDirichlet_BLFIntegrator() : Q(NULL) {}
    DGWeakDirichlet_BLFIntegrator(Coefficient &q) : Q(&q) {}
    ~DGWeakDirichlet_BLFIntegrator() { }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1; // 不应该出现el2, 因为是对边界积分

        dim = el1.GetDim();
        nor.SetSize(dim);
        work1.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        dshape1.SetSize(ndof1, dim);
        adjJ.SetSize(dim);
        work2.SetSize(ndof1);

        elmat.SetSize(ndof1);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();;
            // Assuming order(u)==order(mesh)

            if (el1.Space() == FunctionSpace::Pk) {
                order++;
            }
            ir = &IntRules.Get(Trans.FaceGeom, order); //得到face上的积分规则(里面包含积分点)
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);

            // Set the integration point in the face and the neighboring elements
            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

            el1.CalcShape(eip1, shape1);
            el1.CalcDShape(eip1, dshape1);
            CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);

            if (dim == 1) nor(0) = 2*eip1.x - 1.0;
            else CalcOrtho(Trans.Face->Jacobian(), nor); // 计算Face的法向量

            double val = ip.weight / Trans.Elem1->Weight();
            if (Q) {
                val *= Q->Eval(*Trans.Elem1, eip1);
            }

            adjJ.Mult(nor, work1);
            dshape1.Mult(work1, work2);

            for (int i=0; i<ndof1; ++i)
                for (int j=0; j<ndof1; ++j)
                    elmat(i, j) += val * work2(i) * shape1(j);
        }
    }
};


namespace _DGDiffusion_Edge_Symmetry_Penalty
{
    double sin_cfun(const Vector& x)
    {
        return sin(x[0]) * sin(x[1]); // sin(x) * sin(y)
    }

    // 同时测试 DGDiffusion_Edge, DGDiffusion_Symmetry, DGDiffusion_Penalty 三者
    void Test_DGDiffusion_Edge_Symmetry_Penalty_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fes(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);

        GridFunction rand_gf(&fes);
        for (int i=0; i<fes.GetNDofs(); ++i)
        {
            rand_gf[i] = rand() % 10;
        }

        GridFunctionCoefficient rand_coeff(&rand_gf);
        double sigma = 3.14159268888, kappa=0.3216549877777;
        {
            BilinearForm blf1(&fes);
            blf1.AddInteriorFaceIntegrator(new DGDiffusion_Edge(rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGDiffusion_Edge(rand_coeff));
            blf1.AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(rand_coeff, sigma));
            blf1.AddBdrFaceIntegrator(new DGDiffusion_Symmetry(rand_coeff, sigma));
            blf1.AddInteriorFaceIntegrator(new DGDiffusion_Penalty(rand_coeff, kappa));
            blf1.AddBdrFaceIntegrator(new DGDiffusion_Penalty(rand_coeff, kappa));
            blf1.Assemble();

            BilinearForm blf2(&fes);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, sigma, kappa));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, sigma, kappa));
            blf2.Assemble();

            Vector worker1(fes.GetVSize()), worker2(fes.GetVSize());
            blf1.Mult(rand_gf, worker1);
            blf2.Mult(rand_gf, worker2);

//            worker1.Print(cout << "worker1:\n", 1);
//            worker2.Print(cout << "worker2:\n", 1);
            worker1 -= worker2;

            double TOL = 1E-10;
            for (int i=0; i<fes.GetVSize(); ++i)
            {
                if (abs(worker1[i]) > TOL)
                MFEM_ABORT("Error in DGDiffusion_Edge_Symmetry_Penalty");
            }
        }
    }

    // 对比测试 class DGWeakDirichlet_LFIntegrator 和 class DGWeakDirichlet_BLFIntegrator
    void Test_DGWeakDirichlet_BLFIntegrator_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fes(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);

        GridFunction rand_gf(&fes);
        for (int i=0; i<fes.GetNDofs(); ++i)
        {
            rand_gf[i] = rand() % 10;
        }

        GridFunctionCoefficient rand_coeff(&rand_gf);

        {
            LinearForm lf(&fes);
            lf.AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(rand_coeff, rand_coeff));
            lf.Assemble();

            BilinearForm blf(&fes);
            blf.AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(rand_coeff));
            blf.Assemble();

            Vector worker1(fes.GetVSize());
            blf.Mult(rand_gf, worker1);

            worker1 -= lf;

            double TOL = 1E-10;
            for (int i=0; i<fes.GetVSize(); ++i)
            {
                if (abs(worker1[i]) > TOL)
                    MFEM_ABORT("Error in DGWeakDirichlet_BLFIntegrator");
            }
        }
    }

    void Test_DGDiffusion_Penalty2()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fes(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);

        GridFunction rand_gf(&fes);
        for (int i=0; i<fes.GetNDofs(); ++i)
        {
            rand_gf[i] = rand() % 10;
        }

        GridFunctionCoefficient rand_coeff(&rand_gf);

        {
            BilinearForm blf0(&fes);
            blf0.AddInteriorFaceIntegrator(new DGDiffusion_Penalty2(rand_coeff, 3.1415926));
            blf0.AddBdrFaceIntegrator(new DGDiffusion_Penalty2(rand_coeff, 3.1415926));
            blf0.Assemble();

            BilinearForm blf1(&fes);
            blf1.AddInteriorFaceIntegrator(new DGDiffusion_Penalty(rand_coeff, 3.1415926));
            blf1.AddBdrFaceIntegrator(new DGDiffusion_Penalty(rand_coeff, 3.1415926));
            blf1.Assemble();

            ProductCoefficient neg_rand(-1.0, rand_coeff);

            BilinearForm blf2(&fes);
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 3.1415926, 3.1415926));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 3.1415926, 3.1415926));
            // 抵消 edge 积分
            blf2.AddInteriorFaceIntegrator(new DGDiffusion_Edge(neg_rand));
            blf2.AddBdrFaceIntegrator(new DGDiffusion_Edge(neg_rand));
            // 抵消 penalty 积分
            blf2.AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(neg_rand, 3.1415926));
            blf2.AddBdrFaceIntegrator(new DGDiffusion_Symmetry(neg_rand, 3.1415926));
            blf2.Assemble();

            Vector worker0(fes.GetVSize()), worker1(fes.GetVSize()), worker2(fes.GetVSize());
            blf0.Mult(rand_gf, worker0);
            blf1.Mult(rand_gf, worker1);
            blf2.Mult(rand_gf, worker2);

            worker0 -= worker2;
            worker1 -= worker2;
//            worker0.Print(cout << "worker0: \n", 1);
//            worker1.Print(cout << "worker1: \n", 1);

            double TOL = 1E-10;
            for (int i=0; i<fes.GetVSize(); ++i)
            {
                if (abs(worker0[i]) > TOL)
                MFEM_ABORT("Error in DGDiffusion_Penalty2");
            }
            for (int i=0; i<fes.GetVSize(); ++i)
            {
                if (abs(worker1[i]) > TOL)
                MFEM_ABORT("Error in DGDiffusion_Penalty");
            }

        }
    }

    // 把 symmetry 看出 edge 的转置来测试
    void Test_DGDiffusion_Symmetry1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fes(mesh, &fec);

        FunctionCoefficient sin_coeff(sin_cfun);

        GridFunction rand_gf(&fes);
        for (int i=0; i<fes.GetNDofs(); ++i)
            rand_gf[i] = rand() % 10;

        GridFunctionCoefficient rand_coeff(&rand_gf);

        {
            BilinearForm blf1(&fes);
            blf1.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0, 0));
            blf1.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0, 0));
            blf1.Assemble();

            BilinearForm blf2(&fes);
            blf2.AddInteriorFaceIntegrator(new TransposeIntegrator(
            new DGDiffusion_Symmetry(rand_coeff, -1.0)));
            blf2.AddBdrFaceIntegrator(new TransposeIntegrator(
            new DGDiffusion_Symmetry(rand_coeff, -1.0)));
            blf2.Assemble();

            Vector worker1(fes.GetVSize()), worker2(fes.GetVSize());
            blf1.Mult(rand_gf, worker1);
            blf2.Mult(rand_gf, worker2);

            worker1 -= worker2;
    //        worker1.Print(cout << "work1:(SHOULD BE ZERO)\n", 1);

            double TOL = 1E-10;
            for (int i=0; i<fes.GetVSize(); ++i)
            {
                if (abs(worker1[i]) > TOL)
                MFEM_ABORT("Error in DGWeakDirichlet_BLFIntegrator");
            }
        }

    }
}


void Test_DGDiffusion_Edge_Symmetry_Penalty()
{
    using namespace _DGDiffusion_Edge_Symmetry_Penalty;

    Test_DGDiffusion_Penalty2();

    Test_DGDiffusion_Edge_Symmetry_Penalty_1();

    Test_DGWeakDirichlet_BLFIntegrator_1();

    Test_DGDiffusion_Symmetry1();

    /* ---------- 完成本文件中所有类的全部测试 ------------ */
};

#endif //MFEM_DGDIFFUSION_EDGE_SYMMETRY_PENALTY_HPP
