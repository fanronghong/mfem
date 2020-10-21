#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;


/* Integral on a face:
 *
 *     q <{u grad(w).n}, [v]>_E,
 *
 * u is Trial function, v is Test function
 * q is given Coefficient
 * w is GridFunction
 * */
class DGEdgeIntegrator1: public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    ParGridFunction& w;
    GradientGridFunctionCoefficient* gradw;

    Vector nor, shape1, shape2, grad_w;
    double val1, val2;

public:
    DGEdgeIntegrator1(ParGridFunction &w_) : Q(NULL), w(w_)
    { gradw = new GradientGridFunctionCoefficient(&w); }

    DGEdgeIntegrator1(Coefficient &q, ParGridFunction &w_) : Q(&q), w(w_)
    { gradw = new GradientGridFunctionCoefficient(&w); }

    ~DGEdgeIntegrator1() { delete gradw; }

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
        if (Trans.Elem2No >= 0) // interior boundary
        {
            ndof2 = el2.GetDof();
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

            gradw->Eval(grad_w, *Trans.Elem1, eip1); // OK: parallel and serial
            val1 = ip.weight * Q->Eval(*Trans.Elem1, eip1) * (grad_w * nor);

            if (Trans.Elem2No >= 0) // interior boundary
            {
                val1 *= 0.5;
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);

                if (Trans.Elem2No >= w.FESpace()->GetNE())
                {
                    cout << "Elem2NO: " << Trans.Elem2No << ", " << w.FESpace()->GetNE() << ", " << w.ParFESpace()->GetNE() << endl;
                }
                gradw->Eval(grad_w, *Trans.Elem2, eip2); // Error: parallel; OK: serial
//                w.GetGradient(*Trans.Elem2, grad_w); // same as above

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

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Mesh* mesh = new Mesh(8, 8, 8, Element::TETRAHEDRON);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    DG_FECollection* fec = new DG_FECollection(1, 3);
    ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, fec);

    ParGridFunction* phi_gf = new ParGridFunction(fes);
    *phi_gf = 0.0;

    ParBilinearForm* e1 = new ParBilinearForm(fes);
    ConstantCoefficient one(1.0);
    phi_gf->ExchangeFaceNbrData(); // OK without this line
    e1->AddInteriorFaceIntegrator(new DGEdgeIntegrator1(one, *phi_gf));
    e1->AddBdrFaceIntegrator(new DGEdgeIntegrator1(one, *phi_gf));
    e1->Assemble(0);

    delete e1;
    MPI_Finalize();
}
