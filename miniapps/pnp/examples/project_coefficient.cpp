#include <iostream>
#include "mfem.hpp"
using namespace std;
using namespace mfem;

double boundary(const Vector& x)
{
    if (abs(x[2] - 1.0) < 1E-10) return 1.0;
    else if (abs(x[2] - 0.0) < 1E-10) return 0.0;
    else return -1.0;
}

int main()
{
    Mesh mesh(4, 4, 4, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);

    H1_FECollection h1_fec(1, mesh.Dimension());
    FiniteElementSpace h1_fes(&mesh, &h1_fec);

    FunctionCoefficient bdc_coeff(boundary);

    GridFunction gf(&h1_fes);
    gf.ProjectCoefficient(bdc_coeff);
    gf.Print(cout << "gf:\n", h1_fes.GetVSize());

    cout << "all good" << endl;
}