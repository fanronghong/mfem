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
#include "./GradConvection_Integrator.hpp"
using namespace std;
using namespace mfem;


/** Integrator for the DG form: (只对标记为marker的单元的内部边界积分)

    - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >
    + kappa < {h^{-1} Q} [u], [v] >,

    where Q is a scalar or matrix diffusion coefficient and u, v are the trial
    and test spaces, respectively. The parameters sigma and kappa determine the
    DG method to be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
    * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
    * sigma = +1, kappa = 0: the method of Baumann and Oden. */
class selfDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;
    const Mesh* mesh;
    int marker;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    selfDGDiffusionIntegrator(Coefficient &q, const double s, const double k, const Mesh* mesh_, int marker_)
            : Q(&q), MQ(NULL), sigma(s), kappa(k), mesh(mesh_), marker(marker_) { }

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

        {
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
                Trans.Loc2.Transform(ip, eip2);
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
class selfDGDiffusionSymmetryIntegrator : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;
    const Mesh* mesh;
    int marker;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    selfDGDiffusionSymmetryIntegrator(Coefficient* q, const double s, const Mesh* mesh_, int marker_)
            : Q(q), MQ(NULL), sigma(s), kappa(0.0), mesh(mesh_), marker(marker_) { }

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

        {
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
                Trans.Loc2.Transform(ip, eip2);
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
class selfDGDiffusionPenaltyIntegrator : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    MatrixCoefficient *MQ;
    double sigma, kappa;
    const Mesh* mesh;
    int marker;

    // these are not thread-safe!
    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    selfDGDiffusionPenaltyIntegrator(Coefficient* q, const double k, const Mesh* mesh_, int marker_)
            : Q(q), MQ(NULL), sigma(0.0), kappa(k), mesh(mesh_), marker(marker_) { }

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

        {
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
                Trans.Loc2.Transform(ip, eip2);
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


// 计算: Q*(w, grad(v))_{\Omega}, NOT test. Same with GradConvectionIntegrator2.
// Q is Coefficient, w is VectorCoefficient w
class GradGradConvectionIntegrator: public LinearFormIntegrator
{
protected:
    VectorCoefficient* w;
    Coefficient* q;

    DenseMatrix adjJ, dshape, tmp;
    Vector w_val, tmp_vec;

public:
    GradGradConvectionIntegrator(Coefficient* q_, VectorCoefficient* w_): q(q_), w(w_) {}
    ~GradGradConvectionIntegrator() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        int nd = el.GetDof();
        int dim = el.GetDim();

        adjJ.SetSize(dim);
        w_val.SetSize(dim);

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
            el.CalcDShape(ip, dshape);

            Tr.SetIntPoint(&ip);
            CalcAdjugate(Tr.Jacobian(), adjJ);

            w->Eval(w_val, Tr, ip);
            double wi = ip.weight * q->Eval(Tr, ip);

            Mult(dshape, adjJ, tmp);
            tmp.Mult(w_val, tmp_vec);
            elvect.Add(wi, tmp_vec);
        }
    }
};


// 被替换成: DGEdgeLFIntegrator2
// 计算(区域边界积分): q<u_D, v grad(w).n>_E,
// q, u_D are Coefficient, w is GridFunction
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


// 被替换成: DGEdgeBLFIntegrator1
// 计算(边界或者内部Face都可以): q <{u grad(w).n}, [v]>_E,
// u is trial function, v is test function; q are Coefficient, q在边E的两边连续; w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
class DGSelfTraceIntegrator_1 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;
    const Mesh* mesh;
    int marker;

    Vector nor, shape1, shape2, grad_w;
    double val1, val2;

public:
    DGSelfTraceIntegrator_1(Coefficient &q, GridFunction &w)
            : Q(&q), mesh(NULL)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    DGSelfTraceIntegrator_1(Coefficient &q, GridFunction &w, const Mesh* mesh_, int marker_)
            : Q(&q), mesh(mesh_), marker(marker_)
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


// 被替换成: DGEdgeBLFIntegrator2
// 计算(边界或者内部Face都可以): <[u], {q v grad(w).n}>_E,
// u is trial function, v is test function; q is Coefficient, w is GridFunction */
class DGSelfTraceIntegrator_2 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;
    const Mesh* mesh;
    int marker;

    Vector nor, shape1, shape2, grad_w;

public:
    DGSelfTraceIntegrator_2(Coefficient &q, GridFunction &w)
            : Q(&q), mesh(NULL)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    DGSelfTraceIntegrator_2(Coefficient &q, GridFunction &w, const Mesh* mesh_, int marker_)
            : Q(&q), mesh(mesh_), marker(marker_)
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
        shape1.SetSize(ndof1);

        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }
        else { ndof2 = 0; }

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
            IntegrationPoint eip1, eip2;

            Trans.Loc1.Transform(ip, eip1);
            Trans.Elem1->SetIntPoint(&eip1);
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

            gradw->Eval(grad_w, *Trans.Elem1, eip1);

            double val = ip.weight * Q->Eval(*Trans.Elem1, eip1)
                                   * (grad_w * nor);

            if (!ndof2) // on boundary
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val * shape1(i) * shape1(j);
            }
            else
            {
                Trans.Elem2->SetIntPoint(&eip2);
                gradw->Eval(grad_w, *Trans.Elem2, eip2);
                val += ip.weight * Q->Eval(*Trans.Elem2, eip2)
                                 * (grad_w * nor);
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


// 计算(边界或者内部Face都可以): <{h^{-1} q} [u], [v]>_E,
// u is trial function, v is test function; q are Coefficient
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

            el1.CalcShape(eip1, shape1);

            if (ndof2 == 0)
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1)
                           * nor.Norml2() / h_E;

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * shape1(j);
            }
            else
            {
                Trans.Loc2.Transform(ip, eip2);
                Trans.Elem2->SetIntPoint(&eip2);
                el2.CalcShape(eip2, shape2);

                double w = ip.weight
                            * 0.5 * (Q->Eval(*Trans.Elem1, eip1) + Q->Eval(*Trans.Elem2, eip2))
                            * nor.Norml2() / h_E;

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * shape1(j);

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
// 计算(边界或者内部Face都可以): <{h^{-1} q} [u], [v]>_E, 只在标记为marker的单元边界积分
// u is trial function, v is test function; q are Coefficient
class DGSelfTraceIntegrator_3_1: public BilinearFormIntegrator
{
protected:
    Coefficient* Q;
    Vector shape1, shape2, nor;
    Mesh* mesh;
    int marker;

public:
    DGSelfTraceIntegrator_3_1(Coefficient& q, Mesh* mesh_, int marker_)
        : Q(&q), mesh(mesh_), marker(marker_) {}

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
        else ndof2 = 0;

        ndofs = ndof1 + ndof2;
        elmat.SetSize(ndofs);
        elmat = 0.0;

        {
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

            el1.CalcShape(eip1, shape1);

            if (ndof2 == 0)
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1)
                           * nor.Norml2() / h_E;

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * shape1(j);
            }
            else
            {
                Trans.Loc2.Transform(ip, eip2);
                Trans.Elem2->SetIntPoint(&eip2);
                el2.CalcShape(eip2, shape2);

                double w = ip.weight
                           * 0.5 * (Q->Eval(*Trans.Elem1, eip1) + Q->Eval(*Trans.Elem2, eip2))
                           * nor.Norml2() / h_E;

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += w * shape1(i) * shape1(j);

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


// 计算(边界或者内部Face都可以): <{h^{-1} q} [u], [v]>_E,
// u is Coefficient, v is test function; q are Coefficient */
class DGSelfTraceIntegrator_4: public LinearFormIntegrator
{
protected:
    Coefficient *Q, *u;
    Vector shape1, shape2, nor;

public:
    DGSelfTraceIntegrator_4(Coefficient* q, Coefficient* u_): Q(q), u(u_) {}
    ~DGSelfTraceIntegrator_4() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;
        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        shape1.SetSize(ndof1);
        ndof2 = 0;

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

            el1.CalcShape(eip1, shape1);
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * nor.Norml2() / h_E;

                double u_val = u->Eval(*Trans.Elem1, eip1);

                elvect.Add(w * u_val, shape1);
            }
        }
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el1,
                                        const FiniteElement& el2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;
        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }
        else ndof2 = 0;

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


            el1.CalcShape(eip1, shape1);
            if (ndof2 > 0)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
                Trans.Elem2->SetIntPoint(&eip2);

                double w = ip.weight
                           * 0.5 * (Q->Eval(*Trans.Elem1, eip1) + Q->Eval(*Trans.Elem2, eip2))
                           * nor.Norml2() / h_E;
                double u_val = u->Eval(*Trans.Elem1, eip1) - u->Eval(*Trans.Elem2, eip2);

                for (int i=0; i<ndof1; ++i)
                    elvect(i) += w * u_val * shape1(i);

                for (int j=0; j<ndof2; ++j)
                    elvect(j + ndof1) -= w * u_val * shape2(j);
            }
            else
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * nor.Norml2() / h_E;

                double u_val = u->Eval(*Trans.Elem1, eip1);

                elvect.Add(w * u_val, shape1);
            }
        }
    }
};
// 计算(边界或者内部Face都可以): <{h^{-1} q} [u], [v]>_E, 只在标记为marker的单元边界积分
// u is Coefficient, v is test function; q are Coefficient */
class DGSelfTraceIntegrator_4_1: public LinearFormIntegrator
{
protected:
    Coefficient *Q, *u;
    Vector shape1, shape2, nor;
    Mesh* mesh;
    int marker;

public:
    DGSelfTraceIntegrator_4_1(Coefficient* q, Coefficient* u_, Mesh* mesh_, int marker_)
        : Q(q), u(u_), mesh(mesh_), marker(marker_) {}
    ~DGSelfTraceIntegrator_4_1() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;
        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        shape1.SetSize(ndof1);
        ndof2 = 0;

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

            el1.CalcShape(eip1, shape1);
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * nor.Norml2() / h_E;

                double u_val = u->Eval(*Trans.Elem1, eip1);

                elvect.Add(w * u_val, shape1);
            }
        }
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el1,
                                        const FiniteElement& el2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;
        dim = el1.GetDim();
        ndof1 = el1.GetDof();

        nor.SetSize(dim);
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0)
        {
            ndof2 = el2.GetDof();
            shape2.SetSize(ndof2);
        }
        else ndof2 = 0;

        ndofs = ndof1 + ndof2;
        elvect.SetSize(ndofs);
        elvect = 0.0;
        if (ndof2 == 0) return;

        {
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


            el1.CalcShape(eip1, shape1);
            if (ndof2 > 0)
            {
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
                Trans.Elem2->SetIntPoint(&eip2);

                double w = ip.weight
                           * 0.5 * (Q->Eval(*Trans.Elem1, eip1) + Q->Eval(*Trans.Elem2, eip2))
                           * nor.Norml2() / h_E;
                double u_val = u->Eval(*Trans.Elem1, eip1) - u->Eval(*Trans.Elem2, eip2);

                for (int i=0; i<ndof1; ++i)
                    elvect(i) += w * u_val * shape1(i);

                for (int j=0; j<ndof2; ++j)
                    elvect(j + ndof1) -= w * u_val * shape2(j);
            }
            else
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1) * nor.Norml2() / h_E;

                double u_val = u->Eval(*Trans.Elem1, eip1);

                elvect.Add(w * u_val, shape1);
            }
        }
    }
};


// 计算(边界或者内部Face都可以): <{q grad(u).n}, [v]>_E,
// u is GridFunction, v is test function; q are Coefficient */
class DGSelfTraceIntegrator_5 : public LinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient *gradu;

    Vector shape1, shape2, nor, grad_u;

public:
    DGSelfTraceIntegrator_5(Coefficient* q, GridFunction* u): Q(q)
    {
        gradu = new GradientGridFunctionCoefficient(u);
    }
    ~DGSelfTraceIntegrator_5() { delete gradu; }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        grad_u.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        ndof2 = 0;

        ndofs = ndof1 + ndof2;
        elvect.SetSize(ndofs);
        elvect = 0.0;

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
            el1.CalcShape(eip1, shape1);

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

            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1);
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                w *= (grad_u * nor);
                elvect.Add(w, shape1);
            }
        }
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el1,
                                        const FiniteElement& el2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        grad_u.SetSize(dim);
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
            el1.CalcShape(eip1, shape1);

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

            if (ndof2 > 0)
            {
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                double tmp = (grad_u * nor) * Q->Eval(*Trans.Elem1, eip1);

                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);

                Trans.Elem2->SetIntPoint(&eip2);
                gradu->Eval(grad_u, *Trans.Elem2, eip2);
                tmp += (grad_u * nor) * Q->Eval(*Trans.Elem2, eip2);

                double w = 0.5 * tmp * ip.weight;

                for (int i=0; i<ndof1; ++i)
                    elvect(i) += w * shape1(i);

                for (int i=0; i<ndof2; ++i)
                    elvect(i + ndof1) -= w * shape2(i);
            }
            else
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1);
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                w *= (grad_u * nor);
                elvect.Add(w, shape1);
            }
        }
    }
};
// 计算(边界或者内部Face都可以): <{q grad(u).n}, [v]>_E, 只在标记为marker的单元边界积分
// u is GridFunction, v is test function; q are Coefficient */
class DGSelfTraceIntegrator_5_1 : public LinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient *gradu;
    Mesh* mesh;
    int marker;
    Vector shape1, shape2, nor, grad_u;

public:
    DGSelfTraceIntegrator_5_1(Coefficient* q, GridFunction* u, Mesh* mesh_, int marker_)
        : Q(q), mesh(mesh_), marker(marker_)
    {
        gradu = new GradientGridFunctionCoefficient(u);
    }
    ~DGSelfTraceIntegrator_5_1() { delete gradu; }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        grad_u.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        shape1.SetSize(ndof1);
        ndof2 = 0;

        ndofs = ndof1 + ndof2;
        elvect.SetSize(ndofs);
        elvect = 0.0;

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
            el1.CalcShape(eip1, shape1);

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

            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1);
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                w *= (grad_u * nor);
                elvect.Add(w, shape1);
            }
        }
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el1,
                                        const FiniteElement& el2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        int dim, ndof1, ndof2, ndofs;

        dim = el1.GetDim();
        grad_u.SetSize(dim);
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
        elvect.SetSize(ndofs);
        elvect = 0.0;
        if (ndof2 == 0) return;


        {
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
            el1.CalcShape(eip1, shape1);

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

            if (ndof2 > 0)
            {
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                double tmp = (grad_u * nor) * Q->Eval(*Trans.Elem1, eip1);

                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);

                Trans.Elem2->SetIntPoint(&eip2);
                gradu->Eval(grad_u, *Trans.Elem2, eip2);
                tmp += (grad_u * nor) * Q->Eval(*Trans.Elem2, eip2);

                double w = 0.5 * tmp * ip.weight;

                for (int i=0; i<ndof1; ++i)
                    elvect(i) += w * shape1(i);

                for (int i=0; i<ndof2; ++i)
                    elvect(i + ndof1) -= w * shape2(i);
            }
            else
            {
                double w = ip.weight * Q->Eval(*Trans.Elem1, eip1);
                gradu->Eval(grad_u, *Trans.Elem1, eip1);
                w *= (grad_u * nor);
                elvect.Add(w, shape1);
            }
        }
    }
};


// 计算(边界或者内部Face都可以): - <{Q grad(u).n}, [v]>_E,
// u is trial function, v is test function, Q is Coefficient */
class DGSelfTraceIntegrator_6 : public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    double sigma=0.0, kappa=0.0;

    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
    DGSelfTraceIntegrator_6(Coefficient* Q_): Q(Q_) {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    { // copy from DGDiffusionIntegrator::AssembleFaceMatrix() exactly
        int dim, ndof1, ndof2, ndofs;
        bool kappa_is_nonzero = (kappa != 0.);
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
            {
                if (Q)
                {
                    w *= Q->Eval(*Trans.Elem1, eip1);
                }
                ni.Set(w, nor);
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
                Trans.Loc2.Transform(ip, eip2);
                el2.CalcShape(eip2, shape2);
                el2.CalcDShape(eip2, dshape2);
                Trans.Elem2->SetIntPoint(&eip2);
                w = ip.weight/2/Trans.Elem2->Weight();
                {
                    if (Q)
                    {
                        w *= Q->Eval(*Trans.Elem2, eip2);
                    }
                    ni.Set(w, nor);
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


// 计算(边界或者内部Face都可以): <[u], {q grad(v).n}>_E,
// u is Coefficient, v is test function; q is Coefficient */
class DGSelfTraceIntegrator_7 : public LinearFormIntegrator
{
protected:
    Coefficient *Q, *u;
    Vector nor, shape1, shape2, tmp1, tmp2;
    DenseMatrix adjJ1, adjJ2, dshape1, dshape2;

public:
    DGSelfTraceIntegrator_7(Coefficient &q, Coefficient &u_) : Q(&q), u(&u_) {}
    ~DGSelfTraceIntegrator_7() {}

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
// 计算(边界或者内部Face都可以): <[u], {q grad(v).n}>_E, 只在标记为marker的单元边界积分
// u is Coefficient, v is test function; q is Coefficient */
class DGSelfTraceIntegrator_7_1 : public LinearFormIntegrator
{
protected:
    Coefficient *Q, *u;
    Vector nor, shape1, shape2, tmp1, tmp2;
    DenseMatrix adjJ1, adjJ2, dshape1, dshape2;
    Mesh* mesh;
    int marker;

public:
    DGSelfTraceIntegrator_7_1(Coefficient &q, Coefficient &u_, Mesh* mesh_, int marker_)
        : Q(&q), u(&u_), mesh(mesh_), marker(marker_) {}
    ~DGSelfTraceIntegrator_7_1() {}

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


        {
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


// 计算(边界或者内部Face都可以): <{q u}, [v]>_E,
// u is trial function, v is test function; q are Coefficient */
class DGSelfTraceIntegrator_8 : public BilinearFormIntegrator
{
protected:
    VectorCoefficient *Q;
    Vector nor, shape1, shape2, vec1, vec2;
    double val1, val2;

public:
    DGSelfTraceIntegrator_8(VectorCoefficient &q): Q(&q) {}
    ~DGSelfTraceIntegrator_8() {}

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1, ndof2;

        dim = el1.GetDim();
        nor.SetSize(dim);
        vec1.SetSize(dim);
        vec2.SetSize(dim);

        ndof1 = el1.GetDof();
        if (Trans.Elem2No >= 0)
            ndof2 = el2.GetDof();
        else ndof2 = 0;

        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);
        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule;
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
            Q->Eval(vec1, *Trans.Elem1, eip1);

            if (!ndof2)
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += ip.weight * (vec1 * nor) * shape1(i) * shape1(j);
            }
            else
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += 0.5 * ip.weight * (vec1 * nor) * shape1(i) * shape1(j);

                Trans.Elem2->SetIntPoint(&eip2);
                Q->Eval(vec2, *Trans.Elem2, eip2);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(ndof1 + i, j) -= 0.5 * ip.weight * (vec1 * nor) * shape2(i) * shape1(j);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, ndof1+j) += 0.5 * ip.weight * (vec2 * nor) * shape1(i) * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(ndof1 + i, ndof1 + j) -= 0.5 * ip.weight * (vec2 * nor) * shape2(i) * shape2(j);
            }
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

    void Test_DGSelfBdrFaceIntegrator_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        FunctionCoefficient cos_coeff(cos_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);
        GradientGridFunctionCoefficient grad_sin(&sin_gf);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);
        GradientGridFunctionCoefficient grad_rand(&rand_gf);

        {
            LinearForm lf1(&fsp);
            // (alpha/2) < (u.n) f, w > - beta < |u.n| f, w >, f=cos, u = grad(sin)
            // i.e., 1 <grad(sin).n cos, w>
            lf1.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(cos_coeff, grad_sin, 2.0, 0.0));
            lf1.Assemble();

            LinearForm lf2(&fsp);
            // one <cos, v grad(sin).n>
            lf2.AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&one, &cos_coeff, &sin_gf));
            lf2.Assemble();

//            lf1.Print(cout << "lf1: ", fsp.GetVSize());
//            lf2.Print(cout << "lf2: ", fsp.GetVSize());
            for (int i=0; i<fsp.GetVSize(); ++i)
            {
                assert(abs(lf1[i] - lf2[i]) < 1E-10);
            }
        }

        {
            LinearForm lf1(&fsp);
            // (alpha/2) < (u.n) f, w > - beta < |u.n| f, w >,
            // f=rand_coeff, u = grad(rand_gf), i.e., 1 <grad(rand_gf).n rand_coeff, w>
            lf1.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(rand_coeff, grad_rand, 2.0, 0.0));
            lf1.Assemble();

            LinearForm lf2(&fsp);
            // one <rand_coeff, v grad(sin).n>
            lf2.AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&one, &rand_coeff, &rand_gf));
            lf2.Assemble();

//            lf1.Print(cout << "lf1: ", fsp.GetVSize());
//            lf2.Print(cout << "lf2: ", fsp.GetVSize());
            for (int i=0; i<fsp.GetVSize(); ++i)
            {
                assert(abs(lf1[i] - lf2[i]) < 1E-10);
            }
        }

        delete mesh;
    }

    void Test_DGSelfTraceIntegrator_1()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);
        GradientGridFunctionCoefficient grad_rand(&rand_gf);

        { // 只在内部边积分, test DGSelfTraceIntegrator_1
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=sin
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
        { // 只在内部边积分, test DGSelfTraceIntegrator_1
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=rand_gf
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg, rand_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2);

//            out1.Print(cout << "out1: ", fsp.GetVSize());
//            out2.Print(cout << "out2: ", fsp.GetVSize());
            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
        { // 只在边界积分, test DGSelfTraceIntegrator_1
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            //  q <{u grad(w).n}, [v]>, q=neg, w=sin_gf
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg, sin_gf));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(one_gf, out1);

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=one
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(sin_gf, out2);

            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
        {
            Vector out1(fsp.GetVSize()), out2(fsp.GetVSize());

            BilinearForm blf1(&fsp);
            // q <{u grad(w).n}, [v]>_E, q=neg, w=rand_gf
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg, rand_gf));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg, rand_gf));
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

//            out1.Print(cout << "out1: ", fsp.GetVSize());
//            out2.Print(cout << "out2: ", fsp.GetVSize());
            for (int i=0; i<fsp.GetVSize(); ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }

        delete mesh;
    }

    void Test_DGSelfTraceIntegrator_2()
    {
        Mesh* mesh = new Mesh(20, 20, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);
        GradientGridFunctionCoefficient grad_rand(&rand_gf);

        { // test: DGSelfTraceIntegrator_2
            Vector out1(size), out2(size), out3(size);

            BilinearForm blf1(&fsp);
            // q <[u], {v grad(w).n}>, q=neg, w=sin_gf
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(one, sin_gf));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(one, sin_gf));
            blf1.Assemble(); blf1.Finalize();
            blf1.Mult(sin_gf, out1); // <[sin], {v grad(sin).n}>

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >, Q=one, sigma=1.0,
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.Assemble(); blf2.Finalize();
            blf2.Mult(sin_gf, out2); // -<{grad(sin).n}, [v]> + <[sin], {grad(v).n}>

            BilinearForm blf3(&fsp);
            // q <{u grad(w).n}, [v]>, q=one, w=sin_gf
            blf3.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.Assemble(); blf3.Finalize();
            blf3.Mult(one_gf, out3); // <{grad(sin).n}, [v]>

            out3 += out2; // <[sin], {grad(v).n}>
            for (int i=0; i<size; ++i)
            {
//            assert(abs(out1[i] - out2[i]) < 1E-10);
//            cout << "hh" << endl;
            }
        }
        delete mesh;
        cout << "     Needs more tests here!" << endl;
    }

    void Test_DGSelfTraceIntegrator_3()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);

        Vector out1(size), out2(size);
        {
            BilinearForm blf1(&fsp);
            // <{h^{-1} q} [u], [v]>, q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            // - < {(Q grad(u)).n}, [v] >, Q=rand
            blf1.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0, 0));
            blf1.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0, 0));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // <{h^{-1} rand} [rand], [v]> - <{rand grad(rand).n}, [v]>

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] > + + kappa < {h^{-1} Q} [u], [v] >, Q=rand
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 1.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 1.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2); // -<{rand grad(rand).n}, [v]> + <{h^{-1} rand} [rand], [v]>

//            out1.Print(cout << "out1: ", size);
//            out2.Print(cout << "out2: ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }

        {
            BilinearForm blf1(&fsp);
            // <{h^{-1} q} [u], [v]>, q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            // - <{Q grad(u).n}, [v]>, Q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // <{h^{-1} rand} [rand], [v]> - <{rand grad(rand).n}, [v]>

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] > + kappa < {h^{-1} Q} [u], [v] >, Q=rand, kappa=1
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 1.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 1.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2); // - <{(rand grad(rand)).n}, [v]> + < {h^{-1} rand} [rand], [v] >

//            out1.Print(cout << "out1: ", size);
//            out2.Print(cout << "out2: ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
    }

    void Test_DGSelfTraceIntegrator_4()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);

        Vector out1(size), out2(size);
        {
            LinearForm lf(&fsp);
            // <{h^{-1} q} [u], [v]>, q=rand, u=rand
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&rand_coeff, &rand_coeff));
            lf.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&rand_coeff, &rand_coeff));
            lf.Assemble();

            BilinearForm blf1(&fsp);
            // <{h^{-1} q} [u], [v]>, q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(rand_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // <{h^{-1} sin} [rand], [v]>

//            out1.Print(cout << "out1: ", size);
//            lf  .Print(cout << "lf  : ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - lf[i]) < 1E-10);
        }
    }

    void Test_DGSelfTraceIntegrator_6()
    {
        Mesh* mesh = new Mesh(60, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);

        GridFunction rand_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        GridFunctionCoefficient rand_coeff(&rand_gf);

        Vector out1(size), out2(size);
        {
            BilinearForm blf1(&fsp);
            // - <{Q grad(u).n}, [v]>, Q=sin
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&sin_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&sin_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // - <{sin grad(rand).n}, [v]>

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=sin
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sin_coeff, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2); // - <{(sin grad(rand)).n}, [v]>

    //        out1.Print(cout << "out1: ", fsp.GetTrueVSize());
    //        out2.Print(cout << "out2: ", fsp.GetTrueVSize());
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
        {
            BilinearForm blf1(&fsp);
            // - <{Q grad(u).n}, [v]>, Q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // - <{rand grad(rand).n}, [v]>

            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] >, Q=rand
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(rand_coeff, 0.0, 0.0));
            blf2.Assemble();
            blf2.Finalize();
            blf2.Mult(rand_gf, out2); // - <{(rand grad(rand)).n}, [v]>

//            out1.Print(cout << "out1: ", fsp.GetTrueVSize());
//            out2.Print(cout << "out2: ", fsp.GetTrueVSize());
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - out2[i]) < 1E-10);
        }
    }

    void Test_DGSelfTraceIntegrator_5()
    {
        Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        FunctionCoefficient sin_coeff(sin_cfun);
        FunctionCoefficient cos_coeff(cos_cfun);
        ProductCoefficient sin_square(sin_coeff, sin_coeff);
        ProductCoefficient sin_cos(sin_coeff, cos_coeff);
        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        ProductCoefficient neg_sin_squre(neg, sin_square);
        ProductCoefficient neg_sin_cos(neg, sin_cos);

        GridFunction rand_gf(&fsp), sin_gf(&fsp);
        for (int i=0; i<fsp.GetNDofs(); ++i) rand_gf[i] = rand() % 10;
        sin_gf.ProjectCoefficient(sin_coeff);
        GridFunctionCoefficient rand_coeff(&rand_gf);

        Vector out1(size);
        {
            LinearForm lf(&fsp);
            // <{q grad(u).n}, [v]>, q=neg_sin_cos, u=rand
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_sin_cos, &rand_gf));
            lf.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_sin_cos, &rand_gf));
            lf.Assemble(); // <{- sin cos grad(rand).n}, [v]>

            BilinearForm blf1(&fsp);
            // - <{Q grad(u).n}, [v]>, Q=sin_cos
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&sin_cos));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&sin_cos));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // -<{sin cos grad(rand).n}, [v]>

//            out1.Print(cout << "out1: ", size);
//            lf  .Print(cout << "lf  : ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - lf[i]) < 1E-10);
        }
        {
            LinearForm lf(&fsp);
            // <{q grad(u).n}, [v]>, q=rand, u=rand
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&rand_coeff, &rand_gf));
            lf.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&rand_coeff, &rand_gf));
            lf.Assemble(); // <{rand grad(rand).n}, [v]>

            BilinearForm blf1(&fsp);
            // - <{Q grad(u).n}, [v]>, Q=rand
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&rand_coeff));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(rand_gf, out1); // -<{sin cos grad(rand).n}, [v]>

            out1 += lf;

//            out1.Print(cout << "out1: ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i]) < 1E-10);
        }
    }

    void Test_DGSelfTraceIntegrator_several()
    {
        Mesh* mesh = new Mesh(100, 100, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(2, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        Vector out1(size), out2(size), out3(size);
        { // 在区域边界估计不准确很正常ffff
            BilinearForm blf1(&fsp);
            // (Q grad(u), grad(v))
            blf1.AddDomainIntegrator(new DiffusionIntegrator(one));
            // - <{(Q grad(u)).n}, [v]> + sigma <[u], {(Q grad(v)).n}> + kappa <{h^{-1} Q} [u], [v]>
            // i.e., (grad(u), grad(v)) - <{grad(u).n}, [v]> + <[u], {grad(v).n}> + <h^{-1}, [u], [v]>
            blf1.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 1.0));
//            blf1.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 1.0));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(sin_gf, out1); // (grad(sin), grad(v)) - <{grad(sin).n}, [v]> + <[sin], {grad(v).n}> + <h^{-1} [sin], [v]>

            LinearForm lf(&fsp);
            // Q (grad(u), grad(v)), Q=one, u=sin
            lf.AddDomainIntegrator(new GradConvection_LFIntegrator(&one, &sin_gf));
            // <{q grad(u).n}, [v]>, q=neg, u=sin
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg, &sin_gf));
            // q <[u], {grad(v).n}>, q=neg, u=sin, i.e., -<[sin], {grad(v).n}>
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(neg,sin_coeff));
            // <{h^{-1} q} [u], [v]>, q=one, u=sin
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&one, &sin_coeff));
            lf.Assemble();

            out1 -= lf;
//            out1.Print(cout << "out1: ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i]) < 1E-6);
        }
    }

    void Test_DGSelfTraceIntegrator_7()
    {
        Mesh* mesh = new Mesh(100, 100, Element::TRIANGLE, true, 1.0, 1.0);

        DG_FECollection fec(1, mesh->Dimension());
        FiniteElementSpace fsp(mesh, &fec);
        int size = fsp.GetVSize();

        ConstantCoefficient one(1.0);
        ConstantCoefficient neg(-1.0);
        FunctionCoefficient sin_coeff(sin_cfun);
        GridFunction sin_gf(&fsp), one_gf(&fsp);
        sin_gf.ProjectCoefficient(sin_coeff);
        one_gf.ProjectCoefficient(one);

        Vector out1(size), out2(size), out3(size);
        {
            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >, Q=one, sigma=1.0,
            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
//            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.Assemble(); blf2.Finalize();
            blf2.Mult(sin_gf, out2); // -<{grad(sin).n}, [v]> + <[sin], {grad(v).n}>

            BilinearForm blf3(&fsp);
            // q <{u grad(w).n}, [v]>, q=one, w=sin_gf
            blf3.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
//            blf3.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.Assemble(); blf3.Finalize();
            blf3.Mult(one_gf, out3); // <{grad(sin).n}, [v]>

            out3 += out2; // <[sin], {grad(v).n}>

            LinearForm lf(&fsp);
            // q <[u], {grad(v).n}>, q=one, u=sin, i.e., <[sin], {grad(v).n}>
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(one,sin_coeff));
//            lf.AddFaceIntegrator(new DGSelfTraceIntegrator_7_bdr(one,sin_coeff));
            lf.Assemble();

//            out3.Print(cout << "out3: ", size);
//            lf.Print(cout << "lf  : ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out3[i] - lf[i]) < 1E-7); // 内部计算的精度比边界高
        }
        {
            BilinearForm blf2(&fsp);
            // - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >, Q=one, sigma=1.0,
//            blf2.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, 1.0, 0.0));
            blf2.Assemble(); blf2.Finalize();
            blf2.Mult(sin_gf, out2); // -<{grad(sin).n}, [v]> + <[sin], {grad(v).n}>

            BilinearForm blf3(&fsp);
            // q <{u grad(w).n}, [v]>, q=one, w=sin_gf
//            blf3.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(one, sin_gf));
            blf3.Assemble(); blf3.Finalize();
            blf3.Mult(one_gf, out3); // <{grad(sin).n}, [v]>

            out3 += out2; // <[sin], {grad(v).n}>

            LinearForm lf(&fsp);
            // q <[u], {grad(v).n}>, q=one, u=sin, i.e., <[sin], {grad(v).n}>
//            lf.AddFaceIntegrator(new DGSelfTraceIntegrator_7_int(one,sin_coeff));
            lf.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(one,sin_coeff));
            lf.Assemble();

//            out3.Print(cout << "out3: ", size);
//            lf.Print(cout << "lf  : ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out3[i] - lf[i]) < 1E-4); // 内部计算的精度比边界高
        }
        {
            BilinearForm blf1(&fsp);
            // - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >
            // Q=one, sigma=-1.0, i.e., -<{grad(u).n}, [v]> - <[u], {grad(v).n}>
            blf1.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, 0.0));
            blf1.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, 0.0));
            // - <{Q grad(u).n}, [v]>, Q=neg, i.e., <{grad(u).n}j, [v]>
            blf1.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&neg));
            blf1.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&neg));
            blf1.Assemble();
            blf1.Finalize();
            blf1.Mult(sin_gf, out1); // -<[sin], {grad(v).n}>

            LinearForm lf(&fsp);
            // q <[u], {grad(v).n}>, q=neg, u=sin, i.e., -<[sin], {grad(v).n}>
            lf.AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(neg,sin_coeff));
            lf.AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(neg,sin_coeff));
            lf.Assemble();

//            out1.Print(cout << "out1: ", size);
//            lf.Print(cout << "lf  : ", size);
            for (int i=0; i<size; ++i)
                assert(abs(out1[i] - lf[i]) < 1E-4);
        }
    }

}

void Test_DGSelfTraceIntegrator()
{
    using namespace _DGSelfTraceIntegrator;
    Test_DGSelfBdrFaceIntegrator_1();

    Test_DGSelfTraceIntegrator_1();
//    Test_DGSelfTraceIntegrator_2(); // no tests
    Test_DGSelfTraceIntegrator_3();
    Test_DGSelfTraceIntegrator_4();
    Test_DGSelfTraceIntegrator_5();
    Test_DGSelfTraceIntegrator_6();
    Test_DGSelfTraceIntegrator_7();
    Test_DGSelfTraceIntegrator_several();
    cout << "===> Test Pass: DGSelfTraceIntegrator.hpp" << endl;
}

#endif //LEARN_MFEM_DGSELFTRACEINTEGRATOR_HPP
