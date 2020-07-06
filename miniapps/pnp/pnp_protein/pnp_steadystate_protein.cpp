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

int main(int argc, char **argv)
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Choose a mesh");
    args.AddOption(&refine_times, "-r", "--refine", "Refine mesh times", true);
    args.AddOption(&pqr_file, "-pqr", "--pqr", "Select a PQR file");
    args.AddOption(&p_order, "-p", "--p_order", "Polynomial order of basis function.");
    args.AddOption(&Linearize, "-lin", "--linearize", "Linearization method: choose: cg, dg", true);
    args.AddOption(&Discretize, "-dis", "--discretization", "Descretization method, choose: newton, gummel", true);
    args.AddOption(&self_debug, "-debug", "--self_debug", "-nodebug", "--no_self_debug", "Run many asserts to debug");
    args.AddOption(&verbose, "-ver", "--verbose", "-nover", "--noverbose", "Verbose for more outputs");
    args.AddOption(&visualize, "-v", "--vis", "-nov", "--novis", "Visualize outputs");
    args.AddOption(&prec_type, "-prec", "--prec_type", "Preconditioner type for Newton disretization, choose: block, uzawa, simple");
    args.AddOption(&options_src, "-opts", "--petscopts", "Petsc options file", true);
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }

    MFEMInitializePetsc(NULL, NULL, options_src, NULL);

    if (self_debug)
    {
        Test_ReadPQR();
        Test_PhysicalParameters();
        Test_G_gradG_cfun(); // slow
    }

    Mesh mesh(mesh_file, 1, 1);
    for (int i=0; i<refine_times; ++i) mesh.UniformRefinement();

    if (strcmp(Linearize, "gummel") == 0 && strcmp(Discretize, "cg") == 0)
    {
        PNP_Gummel_CG_Solver_par* solver = new PNP_Gummel_CG_Solver_par(&mesh);
        solver->Solve();
        delete solver;
    }
    else if (strcmp(Linearize, "gummel") == 0 && strcmp(Discretize, "dg") == 0)
    {
        PNP_Gummel_DG_Solver_par* solver = new PNP_Gummel_DG_Solver_par(&mesh);
        solver->Solve();
        delete solver;
    }
    else if (strcmp(Linearize, "newton") == 0 && strcmp(Discretize, "cg") == 0)
    {
        PNP_Newton_CG_Solver_par* solver = new PNP_Newton_CG_Solver_par(&mesh);
        solver->Solve();
        delete solver;
    }
    else if (strcmp(Linearize, "newton") == 0 && strcmp(Discretize, "dg") == 0)
    {
        PNP_Newton_DG_Solver_par* solver = new PNP_Newton_DG_Solver_par(mesh);
        solver->Solve();
        delete solver;
    }

    MFEMFinalizePetsc();
    MPI_Finalize();
    cout << "------------------------------ All Good! -------------------------\n\n" << endl;
}
