
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
    Mesh mesh(4, 4, 4, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);

    H1_FECollection h1_fec(1, mesh.Dimension());
    FiniteElementSpace h1_fes(&mesh, &h1_fec);

    DG_FECollection dg_fec(1, mesh.Dimension());
    FiniteElementSpace dg_fes(&mesh, &dg_fec);

    FunctionCoefficient sin_coeff(sin_func);
    ConstantCoefficient one(1.0);

    GridFunction h1_gf(&h1_fes);
    h1_gf.ProjectCoefficient(sin_coeff);

    GridFunction dg_gf(&dg_fes);
    dg_gf.ProjectGridFunction(h1_gf);

    Vector error;
    ComputeLocalConservation(one, dg_gf, error);
    error.Print(cout << "error:\n", 1);

    cout << "all good" << endl;
}