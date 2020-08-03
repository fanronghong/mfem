#include <iostream>
#include "mfem.hpp"
using namespace std;
using namespace mfem;

int Wx = 0, Wy = 0;            // window position
int Ww = 350, Wh = 350;        // window size
int offx = Ww+5, offy = Wh+25; // window offsets
void Visualize(VisItDataCollection& dc, string field, string title="No title", string caption="No caption", int x=Wx, int y=Wy)
{
    int w = Ww, h = Wh;

    char vishost[] = "localhost";
    int  visport   = 19916;

    socketstream sol_sockL2(vishost, visport);
    sol_sockL2.precision(10);
    sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField(field)
               << "window_geometry " << x << " " << y << " " << w << " " << h
               << "window_title '" << title << "'"
               << "plot_caption '" << caption << "'" << flush; // 主要单引号和单引号前面(关键字window_title, plot_caption后面)的空格
}

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

    Mesh *mesh = new Mesh(4, 4, Element::TRIANGLE, true, 1.0, 1.0);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    int dim = pmesh->Dimension();
    Array<int> Dirichlet;
    {
        int bdr_size = pmesh->bdr_attributes.Max();
        Dirichlet.SetSize(bdr_size);
        Dirichlet = 1;
    }

    H1_FECollection* cg_fec = new H1_FECollection(1, dim);
    DG_FECollection* dg_fec = new DG_FECollection(1, dim);
    ParFiniteElementSpace* cg = new ParFiniteElementSpace(pmesh, cg_fec);
    ParFiniteElementSpace* dg = new ParFiniteElementSpace(pmesh, dg_fec);

    FunctionCoefficient sin_cos_coeff(sin_cos);

    ParGridFunction gf1(cg), gf2(dg);

    gf1 = 0.0;
//    gf1.ProjectCoefficient(sin_cos_coeff); // OK
    gf1.ProjectBdrCoefficient(sin_cos_coeff, Dirichlet); // Seg fault

    gf2 = 0.0;
    gf2.ProjectCoefficient(sin_cos_coeff); // OK
//    gf1.ProjectBdrCoefficient(sin_cos_coeff, Dirichlet); // Seg fault

    VisItDataCollection* dc = new VisItDataCollection("data collection", pmesh);
    dc->RegisterField("gf1", &gf1);
    dc->RegisterField("gf2", &gf2);
    Visualize(*dc, "gf1", "CG");
    Visualize(*dc, "gf2", "DG");

    MPI_Finalize();
    return 0;
}
