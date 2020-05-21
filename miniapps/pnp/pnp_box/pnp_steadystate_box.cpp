/*
 * 稳态的PNP方程，用Gummel迭代
 *
 * */
#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "./pnp_steadystate_box_solvers.hpp"
using namespace std;
using namespace mfem;

int main(int args, char **argv)
{
    int num_procs, myid;
    MPI_Init(&args, &argv);
    MFEMInitializePetsc(NULL, NULL, options_src, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Mesh mesh(mesh_file);
    Array<double> phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes;

    for (int i=0; i<refine_times; i++) mesh.UniformRefinement();

//    PNP_CG_Gummel_Solver_par* solver = new PNP_CG_Gummel_Solver_par(mesh);
//    PNP_DG_Gummel_Solver_par* solver = new PNP_DG_Gummel_Solver_par(mesh);
//    PNP_CG_Newton_box_Solver_par* solver = new PNP_CG_Newton_box_Solver_par(&mesh);
    PNP_DG_Newton_box_Solver_par* solver = new PNP_DG_Newton_box_Solver_par(mesh);
    solver->Solve(phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes);
    delete solver;

#ifndef PhysicalModel
    meshsizes.Print(cout << "\nMesh sizes: \n", meshsizes.Size());

    phi3L2errornorms.Print(cout << "\nL2 errornorms of |phi3 - phi3_h|: \n", phi3L2errornorms.Size());
    Array<double> phi3rates = compute_convergence(phi3L2errornorms, meshsizes);

    c1L2errornorms.Print(cout << "\nL2 errornorms of |c1 - c1_h|: \n", c1L2errornorms.Size());
    Array<double> c1rates = compute_convergence(c1L2errornorms, meshsizes);

    c2L2errornorms.Print(cout << "\nL2 errornorms of |c2 - c2_h|: \n", c2L2errornorms.Size());
    Array<double> c2rates = compute_convergence(c2L2errornorms, meshsizes);

    phi3rates.Print(cout << "\nphi3 convergence rate: \n", phi3rates.Size());
    c1rates  .Print(cout << "c1 convergence rate: \n", c1rates.Size());
    c2rates  .Print(cout << "c2 convergence rate: \n", c2rates.Size());
#endif
    MFEMFinalizePetsc();
    MPI_Finalize();
    cout << "------------------------------ All Good! -------------------------" << endl;
}

