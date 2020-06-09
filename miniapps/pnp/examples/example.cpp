
#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "../utils/PrintMesh.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/python_utils.hpp"
using namespace std;
using namespace mfem;

double sin_func(const Vector& x) {
    return sin(x[0]) * sin(x[1]) * sin(x[2]);
}
double cos_func(const Vector& x) {
    return cos(x[0]) * cos(x[1]) * cos(x[2]);
}

int main()
{
    const int interface_marker = 9;
    Mesh mesh("../pnp_protein/1MAG_2.msh");

    Array<int> marker;
    marker.SetSize(mesh.bdr_attributes.Max());
    marker = 0;
    marker[interface_marker - 1] = 1;

    H1_FECollection fec(1, 3);
    FiniteElementSpace fes(&mesh, &fec);
    int size = fes.GetTrueVSize();

    ConstantCoefficient rand(3.1415926);
    ConstantCoefficient neg(-1.0);
    ProductCoefficient neg_rand(neg, rand);
    FunctionCoefficient sin_coeff(sin_func);
    FunctionCoefficient cos_coeff(cos_func);

    BilinearForm blf(&fes);
    blf.AddBdrFaceIntegrator(new DGDiffusionIntegrator(neg_rand, 0.0, 0.0), marker);
    blf.Assemble();

    GridFunction sin_gf(&fes), cos_gf(&fes), u(&fes);
    sin_gf.ProjectCoefficient(sin_coeff);
    cos_gf.ProjectCoefficient(cos_coeff);
    u = sin_gf;
    u += cos_gf;
    GradientGridFunctionCoefficient gradu(&u);

    Vector out1(size);
    blf.Mult(u, out1);

    LinearForm lf(&fes);
    lf.AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator(&rand, &gradu, &mesh, 1, 2));
    lf.Assemble();

    out1.Print(cout << "out1: ", size);
    lf  .Print(cout << "lf  : ", size);
}