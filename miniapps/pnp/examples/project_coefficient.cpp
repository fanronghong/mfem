#include <iostream>
#include "mfem.hpp"
#include "../utils/mfem_utils.hpp"
using namespace std;
using namespace mfem;

double boundary(const Vector& x)
{
    return -1.0;
}

int main()
{
    Mesh mesh(4, 4, 4, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);

    mesh.bdr_attributes.Print(cout << "bdr_attributes: ", 10);
    Array<int> bdr_attr(mesh.bdr_attributes.Max());
    bdr_attr = 0;
    bdr_attr[1 - 1] = 1;
    bdr_attr[2 - 1] = 1;
    bdr_attr[3 - 1] = 1;

    H1_FECollection h1_fec(1, mesh.Dimension());
    FiniteElementSpace h1_fes(&mesh, &h1_fec);

    DG_FECollection dg_fec(1, mesh.Dimension());
    FiniteElementSpace dg_fes(&mesh, &dg_fec);

    FunctionCoefficient bdc_coeff(boundary);

    GridFunction gf(&h1_fes);
    gf.ProjectBdrCoefficient(bdc_coeff, bdr_attr);
    gf.Print(cout << "gf:\n", h1_fes.GetVSize());

    GridFunction dg_gf(&dg_fes);
    dg_gf.ProjectGridFunction(gf);


    VisItDataCollection dc("data collection", &mesh);
    dc.RegisterField("gf", &gf);
    dc.RegisterField("dg_gf",   &dg_gf);

    Visualize(dc, "gf", "gf");
    Visualize(dc, "dg_gf", "dg_gf");

    cout << "all good" << endl;
}