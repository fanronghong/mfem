#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "./pnp_box_timedependent_solvers.hpp"
using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    OptionsParser args(argc, argv);
    args.AddOption(&p_order, "-p", "--p_order", "Polynomial order of basis function.");
    args.AddOption(&Linearize, "-lin", "--linearize", "Linearization method.");
    args.AddOption(&relax, "-relax", "--relax", "Relax parameter: (0.0, 1.0)");
    args.AddOption(&max_newton, "-maxNewton", "--maxNewton", "Newton max iterations");
    args.AddOption(&zero_initial, "-zero", "--zero_initial", "-nonzero", "--nonzero_initial", "Choose zero or nonzero for nonlinear iteration initial value");
    args.AddOption(&initTol, "-initTol", "--initTol", "For obtaining initial value, Gummel iteration to satisfy the Tol");
    args.AddOption(&Discretize, "-dis", "--discretization", "Descretization method.");
    args.AddOption(&AdvecStable, "-stable", "--stable", "Choose stabilization: none, supg, eafe");
    args.AddOption(&ode_type, "-ode", "--ode", "Use ODE Solver");
    args.AddOption(&t_init, "-t_init", "--t_init", "Initial time");
    args.AddOption(&t_final, "-t_final", "--t_final", "Final time");
    args.AddOption(&t_stepsize, "-dt", "--dt", "Time Step");
    args.AddOption(&SpaceConvergRate, "-space_rate", "--space_rate", "-nospace_rate", "--nospace_rate", "Compute space convergence rate by using analytic solutions");
    args.AddOption(&SpaceConvergRate_Change_dt, "-change_dt", "--change_dt", "-nochange_dt", "--nochange_dt", "Change dt to compute (c1 dt + c2 h^2)");
    args.AddOption(&refine_time, "-ref_dt", "--refine_dt", "Refine the initial time-step times.");
    args.AddOption(&time_scale, "-dt_scale", "--dt_scale", "Time-step scale factor");
    args.AddOption(&refine_mesh, "-ref_h", "--refine_h", "Refine the initial mesh times.");
    args.AddOption(&TimeConvergRate, "-time_rate", "--time_rate", "-notime_rate", "--notime_rate", "Compute time convergence rate by using analytic solutions");
    args.AddOption(&options_src, "-opts", "--petscopts", "Petsc options file");
    args.AddOption(&paraview, "-para", "--paraview", "-nopara", "--noparaview", "Save time-dependent results");
    args.AddOption(&output, "-out", "--output", "File name to save outputs", false);
    args.AddOption(&verbose, "-verb", "--verbose", "Print Level: 1,2");
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

    Array<double> phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes, timesteps;
    if (SpaceConvergRate)
    {
        MFEM_ASSERT(!TimeConvergRate, "SpaceConvergRate and TimeConvergRate cannot exist simultaneously");

        int origin_refine_mesh = refine_mesh; // save refine_mesh temporarily
        for (int i=0; i<refine_mesh+1; ++i)
        {
            Mesh* mesh = new Mesh(mesh_file);
            ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
            delete mesh;
            for (int j=0; j<i; ++j) pmesh->UniformRefinement();

            refine_mesh = i; // for cout right verbose outputs

            PNP_Box_TimeDependent_Solver* solver = new PNP_Box_TimeDependent_Solver(pmesh, ode_type);
            solver->Solve(phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes, timesteps);
            delete solver;
            delete pmesh;

            refine_mesh = origin_refine_mesh; // reset real refine_mesh
        }

        if (myid == 0)
        {
            meshsizes.Print(cout << "\nMesh sizes: \n", meshsizes.Size());

            phi3L2errornorms.Print(cout << "\nL2 errornorms of |phi3 - phi3_h|: \n", phi3L2errornorms.Size());
            Array<double> phi3rates = compute_convergence(phi3L2errornorms, meshsizes);

            c1L2errornorms.Print(cout << "\nL2 errornorms of |c1 - c1_h|: \n", c1L2errornorms.Size());
            Array<double> c1rates = compute_convergence(c1L2errornorms, meshsizes);

            c2L2errornorms.Print(cout << "\nL2 errornorms of |c2 - c2_h|: \n", c2L2errornorms.Size());
            Array<double> c2rates = compute_convergence(c2L2errornorms, meshsizes);

            phi3rates.Print(cout << "\nphi3 L2 convergence rate: \n", phi3rates.Size());
            c1rates  .Print(cout << "c1   L2 convergence rate: \n", c1rates.Size());
            c2rates  .Print(cout << "c2   L2 convergence rate: \n", c2rates.Size());
        }
    }
    else if (TimeConvergRate)
    {
        MFEM_ASSERT(!SpaceConvergRate, "SpaceConvergRate and TimeConvergRate cannot exist simultaneously");

        double origin_t_stepsize = t_stepsize;
        int origin_refine_time = refine_time;
        for (int i=0; i<refine_time+1; ++i) // 对时间步长进行"加密"
        {
            Mesh* mesh = new Mesh(mesh_file);
            ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
            delete mesh;
            for (int k=0; k<refine_mesh; k++) pmesh->UniformRefinement();

            for (int j=0; j<i; ++j) t_stepsize *= time_scale;

            refine_time = i; // for cout right verbose outputs

            PNP_Box_TimeDependent_Solver* solver = new PNP_Box_TimeDependent_Solver(pmesh, ode_type);
            solver->Solve(phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes, timesteps);
            delete solver;
            delete pmesh;

            refine_time = origin_refine_time; // reset real refine_time and t_stepsize
            t_stepsize = origin_t_stepsize;
        }

        if (myid == 0)
        {
            timesteps.Print(cout << "\nTime-step sizes: \n", timesteps.Size());

            phi3L2errornorms.Print(cout << "\nL2 errornorms of |phi3 - phi3_h|: \n", phi3L2errornorms.Size());
            Array<double> phi3rates = compute_convergence(phi3L2errornorms, timesteps);

            c1L2errornorms.Print(cout << "\nL2 errornorms of |c1 - c1_h|: \n", c1L2errornorms.Size());
            Array<double> c1rates = compute_convergence(c1L2errornorms, timesteps);

            c2L2errornorms.Print(cout << "\nL2 errornorms of |c2 - c2_h|: \n", c2L2errornorms.Size());
            Array<double> c2rates = compute_convergence(c2L2errornorms, timesteps);

            phi3rates.Print(cout << "\nphi3 Time convergence rate: \n", phi3rates.Size());
            c1rates  .Print(cout << "c1   Time convergence rate: \n", c1rates.Size());
            c2rates  .Print(cout << "c2   Time convergence rate: \n", c2rates.Size());
        }
    }
    else
    {
        Mesh* mesh = new Mesh(mesh_file);
        ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        delete mesh;
        for (int i=0; i<refine_mesh; i++) pmesh->UniformRefinement();

        PNP_Box_TimeDependent_Solver* solver = new PNP_Box_TimeDependent_Solver(pmesh, ode_type);
        solver->Solve(phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes, timesteps);
        delete solver;
        delete pmesh;
    }

    MFEMFinalizePetsc();
    MPI_Finalize();
}

