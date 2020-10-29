#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

/* Compute Facet integral:
 *
 *     q <{u grad(w).n}, [v]>,
 *
 * u is Trial function, v is Test function
 * q is given Coefficient, w is GridFunction
 * */
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

class Demo
{
private:
    ParFiniteElementSpace* fes;
    ParBilinearForm* w;

public:
    Demo(ParFiniteElementSpace* fes_): fes(fes_)
    {
        w = NULL;
    }
    ~Demo() { delete w; }

    void build_w(Coefficient& coeff)
    {
        if (w != NULL) { delete w; }

        auto* w = new ParBilinearForm(fes);
        w->AddInteriorFaceIntegrator(new DGDiffusion_Edge(coeff));
        w->AddBdrFaceIntegrator(new DGDiffusion_Edge(coeff));
        w->Assemble(0);
        w->Finalize(0);

        auto* w_tdof = w->ParallelAssemble();
    }

    void Solve(Coefficient& coeff)
    {
        build_w(coeff);

        auto* lf = new ParLinearForm(fes);
        *lf = 2.0;

        auto* gf = new ParGridFunction(fes);
        *gf = 3.0;

        // 1st method
//        w->AddMult(*gf, *lf, 1.0);

        // 2nd method goon
        auto* w_tdof = w->ParallelAssemble();
        auto* Restriction = w->GetRestriction();
        auto* lf_tdof = lf->ParallelAssemble();

        w_tdof->Mult(1.0, gf->GetTrueVector(), 1.0, *lf_tdof);
        Restriction->MultTranspose(*lf_tdof, *lf);

        delete w_tdof; delete Restriction; delete lf_tdof;

        cout << "l2 norm of lf: " << lf->Norml2() << endl;

    }

};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    Mesh* mesh = new Mesh(4, 4, 4, Element::TETRAHEDRON);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    auto* fec = new DG_FECollection(1, pmesh->Dimension());
    auto* fes = new ParFiniteElementSpace(pmesh, fec);

    ConstantCoefficient one(1.0);

    Demo* demo = new Demo(fes);
    demo->Solve(one);

    MPI_Finalize();
}