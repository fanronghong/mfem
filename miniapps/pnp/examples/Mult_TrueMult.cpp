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
        shape1.SetSize(ndof1);
        if (Trans.Elem2No >= 0) // interior facet
        {
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
            ir = &IntRules.Get(Trans.FaceGeom, order);
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);

            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            el1.CalcShape(eip1, shape1);

            if (dim == 1) {
                nor(0) = 2 * eip1.x - 1.0;
            }
            else {
                CalcOrtho(Trans.Jacobian(), nor);
            }

            gradw->Eval(grad_w, *Trans.Elem1, eip1);
            val1 = ip.weight * (grad_w * nor);
            if (Q) {
                val1 *= Q->Eval(*Trans.Elem1, eip1);
            }

            if (Trans.Elem2No >= 0) // interior facet
            {
                el2.CalcShape(eip2, shape2);

                val1 *= 0.5;
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);

                gradw->Eval(grad_w, *Trans.Elem2, eip2);

                val2 = ip.weight * (grad_w * nor) * 0.5;
                if (Q) {
                    val2 *= Q->Eval(*Trans.Elem2, eip2);
                }

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


int main(int argc, char **argv)
{
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mesh* mesh = new Mesh(4, 4, 4, Element::TETRAHEDRON);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    auto* fec = new DG_FECollection(1, pmesh->Dimension());
    auto* fes = new ParFiniteElementSpace(pmesh, fec);

    auto* u = new ParGridFunction(fes);
    *u = 1.0;

    auto* blf = new ParBilinearForm(fes);
    u->ExchangeFaceNbrData();
    blf->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(*u));
    blf->Assemble(0);
    blf->Finalize(0);

    auto* lf = new ParLinearForm(fes);
    *lf = 2.0;

    auto* gf = new ParGridFunction(fes);
    *gf = 3.0;

    /* ref: https://github.com/mfem/mfem/issues/1830
     * In general, yes, *A maps primal tdofs to dual tdofs. However, for DG FE spaces,
     * there is no difference between dofs (ParGridFunctions) and tdofs.
     * That's why the above code is fine in this case.
     * */
//    auto *A = blf->ParallelAssemble();
//    A->Mult(1.0, *gf, 1.0, *lf); // Mult(a, x, b, y) performs: a*A*x + b*y -> y
    blf->AddMult(*gf, *lf);

    gf->ParallelAssemble();

    cout.precision(14);
    cout << "l2 norm of lf: " << lf->Norml2() << endl;

    MPI_Finalize();
}