
#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "../utils/PrintMesh.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/python_utils.hpp"
#include "../utils/LocalConservation.hpp"
using namespace std;
using namespace mfem;

double sin_func(const Vector& x)
{
    return sin(x[0]) * sin(x[1]);
}

int main()
{
    Mesh mesh(2, 2, 2, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);

    H1_FECollection h1_fec(1, mesh.Dimension());
    FiniteElementSpace h1_fes(&mesh, &h1_fec); // byVDIM, byNODES

    DG_FECollection dg_fec(1, mesh.Dimension());
    FiniteElementSpace dg_fes(&mesh, &dg_fec);

    BilinearForm blf(&h1_fes);
    blf.AddDomainIntegrator(new DiffusionIntegrator);
    for (int i=0; i<2; ++i)
    {
        Array<int> vdofs;
        DenseMatrix elmat;
        blf.ComputeElementMatrix(i, elmat);
        blf.AssembleElementMatrix(i, elmat, vdofs, 1);
    }
    blf.Finalize();
    SparseMatrix spmat = blf.SpMat();
    spmat.Print(cout << "spmat:\n");




    cout << "all good" << endl;
}