/*
 * 稳态的PNP方程，用Gummel迭代
 *
 * */
#include <iostream>
#include <numeric>

#include "mfem.hpp"
#include "./pnp_steadystate_protein.hpp"
#include "./pnp_steadystate_protein_solvers.hpp"

using namespace std;
using namespace mfem;

int main(int args, char **argv)
{
    int num_procs, myid;
    MPI_Init(&args, &argv);
    MFEMInitializePetsc(NULL, NULL, options_src, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef SELF_DEBUG
    Test_ReadPQR();
    Test_PhysicalParameters();
    Test_G_gradG_cfun(); // slow
#endif

    Mesh mesh(mesh_file, 1, 1);
    int mesh_dim = mesh.Dimension(); //网格的维数:1D,2D,3D
    for (int i=0; i<refine_times; ++i) mesh.UniformRefinement();
#ifdef SELF_DEBUG
    mesh.PrintInfo(cout);
#endif

//    PNP_Gummel_Solver* solver = new PNP_Gummel_Solver(mesh);
//    PNP_Gummel_Solver_par* solver = new PNP_Gummel_Solver_par(&mesh);
//    PNP_Newton_Solver* solver = new PNP_Newton_Solver(&mesh);
    PNP_Newton_Solver_par* solver = new PNP_Newton_Solver_par(&mesh);
    solver->Solve();
    delete solver;


    MFEMFinalizePetsc();
    MPI_Finalize();
    return 0;
}
