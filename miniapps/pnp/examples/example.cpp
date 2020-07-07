
#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

double sin_cos(const Vector& x)
{
    return sin(x[0]) * cos(x[1]);
}


int main(int argc, char *argv[])
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Mesh *mesh = new Mesh(16, 16, Element::TRIANGLE, true, 1.0, 1.0);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    int dim = pmesh->Dimension();

    H1_FECollection cg_fec(1, dim);
    DG_FECollection dg_fec(1, dim);
    ParFiniteElementSpace cg(pmesh, &cg_fec, dim);
    ParFiniteElementSpace dg(pmesh, &dg_fec, dim);

    FunctionCoefficient sin_cos_coeff(sin_cos);
    ConstantCoefficient zero(0.0);

    ParGridFunction gf_cg(&cg), gf_dg(&dg), gf_temp(&dg);
    gf_cg.ProjectCoefficient(sin_cos_coeff);
    gf_dg.ProjectCoefficient(sin_cos_coeff);
    gf_temp.ProjectGridFunction(gf_cg);

    cout << "L2 norm of   gf_cg: " << gf_cg.ComputeL2Error(zero) << endl;
    cout << "L2 norm of   gf_dg: " << gf_dg.ComputeL2Error(zero) << endl;
    cout << "L2 norm of gf_temp: " << gf_temp.ComputeL2Error(zero) << endl;
}