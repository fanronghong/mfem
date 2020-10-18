#include <iostream>
#include <numeric>
#include "mfem.hpp"
#include "./pnp_protein_timedependent_solvers.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char **argv)
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    OptionsParser args(argc, argv);
    args.AddOption(&output, "-out", "--output", "Just for showing all commands in cluster, like bsub -o ...");
    args.AddOption(&refine_times, "-r", "--refine", "Refine mesh times");
    args.AddOption(&p_order, "-p", "--p_order", "Polynomial order of basis function.");
    args.AddOption(&Linearize, "-lin", "--linearize", "Linearization method: choose: cg, dg");
    args.AddOption(&relax, "-relax", "--relax", "Relax parameter: (0.0, 1.0)");
    args.AddOption(&Discretize, "-dis", "--discretization", "Descretization method, choose: newton, gummel");
    args.AddOption(&AdvecStable, "-stab", "--stable", "Choose Stabilization method: none, supg, eafe");
    args.AddOption(&self_debug, "-debug", "--self_debug", "-nodebug", "--no_self_debug", "Run many asserts to debug");
    args.AddOption(&local_conservation, "-conserv", "--conservation", "-noconserv", "--noconservation", "Show local conservation");
    args.AddOption(&show_peclet, "-peclet", "--peclet", "-nopeclet", "--nopeclet", "Show Peclet numbers");
    args.AddOption(&verbose, "-verb", "--verbose", "Print Level: 1,2");
    args.AddOption(&visualize, "-v", "--vis", "-nov", "--novis", "Visualize outputs");
    args.AddOption(&prec_type, "-prec", "--prec_type", "Preconditioner type for Newton disretization, choose: block, uzawa, simple");
    args.AddOption(&schur_alpha1, "-schur1", "--schur1", "1st parameter for Schur Complement");
    args.AddOption(&schur_alpha2, "-schur2", "--schur2", "2nd parameter for Schur Complement");
    args.AddOption(&options_src, "-opts", "--petscopts", "Petsc options file");
    args.AddOption(&paraview, "-para", "--paraview", "-nopara", "--noparaview", "Save time-dependent results");
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

    Mesh* mesh = new Mesh(mesh_file, 1, 1);
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    for (int i=0; i<refine_times; ++i) pmesh->UniformRefinement();

    if (strcmp(Linearize, "gummel") == 0 && strcmp(Discretize, "cg") == 0)
    {
        PNP_Protein_TimeDependent_Solver* solver = new PNP_Protein_TimeDependent_Solver(pmesh, 1);
        solver->Solve();
        delete solver;
    }

    delete pmesh;

    MFEMFinalizePetsc();
    MPI_Finalize();
}
