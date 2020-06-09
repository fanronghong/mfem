#include <iostream>
#include "mfem.hpp"
#include "../utils/mfem_utils.hpp"
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

    Mesh *mesh = new Mesh(8, 8, Element::TRIANGLE, true, 1.0, 1.0);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    int dim = pmesh->Dimension();

    H1_FECollection cg_fec(1, dim);
    DG_FECollection dg_fec(1, dim);
    ParFiniteElementSpace cg(pmesh, &cg_fec, dim);
    ParFiniteElementSpace dg(pmesh, &dg_fec, dim);

    FunctionCoefficient sin_cos_coeff(sin_cos);

    if (0)
    {   // wrong, ref: https://github.com/mfem/mfem/issues/1526
        ParDiscreteLinearOperator proj(&cg, &dg);
        {
            cout << "Height: " << proj.Height() << '\n'
                 << "Width : " << proj.Width() << endl;
            cout << "dg ndofs: " << dg.GetTrueVSize() << endl;
            cout << "cg ndofs: " << cg.GetTrueVSize() << endl;
        }
        proj.AddDomainInterpolator(new IdentityInterpolator());
        proj.Assemble();

        ParGridFunction gf_trial(&cg);
        gf_trial.ProjectCoefficient(sin_cos_coeff);
        gf_trial.SetTrueVector();
        gf_trial.SetFromTrueVector();

        ParGridFunction gf_test(&dg);
        proj.Mult(gf_trial, gf_test);
    }

    {
        ParGridFunction gf_trial(&cg);
        gf_trial.ProjectCoefficient(sin_cos_coeff);
        gf_trial.SetTrueVector();
        gf_trial.SetFromTrueVector();

        ParGridFunction gf_test(&dg);
        gf_test.ProjectGridFunction(gf_trial);

        VisItDataCollection* dc = new VisItDataCollection("data collection", pmesh);
        dc->RegisterField("trial", &gf_trial);
        dc->RegisterField("test", &gf_test);

        Visualize(*dc, "trial", "sin_cos on CG");
        Visualize(*dc, "test", "sin_cos on DG");
    }

    MPI_Finalize();

    return 0;
}