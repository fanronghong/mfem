#include <iostream>
#include <numeric>
#include "mfem.hpp"
#include "./pnp_protein_timedependent_solvers.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char **argv)
{
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    OptionsParser args(argc, argv);
    args.AddOption(&output, "-out", "--output", "Just for showing all commands in cluster, like bsub -o ...");
    args.AddOption(&mesh_file, "-msh", "--msh", "Protein mesh file.");
    args.AddOption(&pqr_file, "-pqr", "--pqr", "PQR file");
    args.AddOption(&p_order, "-p", "--p_order", "Polynomial order of basis function.");
    args.AddOption(&Linearize, "-lin", "--linearize", "Linearization method: choose: cg, dg");
    args.AddOption(&nonzero_NewtonInitial, "-nonzero_init", "--nonzero_NewtonInit", "-zero_init", "--zero_NewtonInit", "Use Gummel iteration for Newton initial?");
    args.AddOption(&nonzero_maxGummel, "-nonzero_maxGummel", "--nonzero_maxGummel", "Max Gummel iterations for providing Newton initial.");
    args.AddOption(&relax, "-relax", "--relax", "Relax parameter: (0.0, 1.0)");
    args.AddOption(&Discretize, "-dis", "--discretization", "Descretization method, choose: newton, gummel");
    args.AddOption(&AdvecStable, "-stab", "--stable", "Choose Stabilization method: none, supg, eafe");
    args.AddOption(&self_debug, "-debug", "--self_debug", "-nodebug", "--no_self_debug", "Run many asserts to debug");
    args.AddOption(&local_conservation, "-conserv", "--conservation", "-noconserv", "--noconservation", "Show local conservation");
    args.AddOption(&ode_type, "-ode", "--ode", "Use ODE Solver");
    args.AddOption(&SpaceConvergRate, "-space_rate", "--space_rate", "-nospace_rate", "--nospace_rate", "Compute space convergence rate by using analytic solutions");
    args.AddOption(&SpaceConvergRate_Change_dt, "-change_dt", "--change_dt", "-nochange_dt", "--nochange_dt", "Change dt to compute (c1 dt + c2 h^2)");
    args.AddOption(&Change_dt_factor, "-change_dt_factor", "--change_dt_factor", "Set dt = factor * h^2");
    args.AddOption(&refine_time, "-ref_dt", "--refine_dt", "Refine the initial time-step times.");
    args.AddOption(&time_scale, "-dt_scale", "--dt_scale", "Time-step scale factor");
    args.AddOption(&refine_mesh, "-ref_h", "--refine_h", "Refine the initial mesh times.");
    args.AddOption(&TimeConvergRate, "-time_rate", "--time_rate", "-notime_rate", "--notime_rate", "Compute time convergence rate by using analytic solutions");
    args.AddOption(&show_peclet, "-peclet", "--peclet", "-nopeclet", "--nopeclet", "Show Peclet numbers");
    args.AddOption(&verbose, "-verb", "--verbose", "Print Level: 1,2");
    args.AddOption(&visualize, "-v", "--vis", "-nov", "--novis", "Visualize outputs");
    args.AddOption(&prec_type, "-prec", "--prec_type", "Preconditioner type for Newton disretization, choose: block, lower, upper, blockschur, lowerblockschur, upperblockschur");
    args.AddOption(&schur_alpha1, "-schur1", "--schur1", "1st parameter for Schur Complement");
    args.AddOption(&schur_alpha2, "-schur2", "--schur2", "2nd parameter for Schur Complement");
    args.AddOption(&t_init, "-t_init", "--t_init", "Initial time");
    args.AddOption(&t_final, "-t_final", "--t_final", "Final time");
    args.AddOption(&t_stepsize, "-dt", "--dt", "Time Step");
    args.AddOption(&options_src, "-opts", "--petscopts", "Petsc options file");
    args.AddOption(&paraview, "-para", "--paraview", "-nopara", "--noparaview", "Save time-dependent results");
    args.AddOption(&paraview_dir, "-para_dir", "--paraview_directory", "Directory name for saving Paraview outputs.");
    args.Parse();
    if (!args.Good())
    {
        if (rank == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
    {
        args.PrintOptions(cout);
    }

    MFEMInitializePetsc(NULL, NULL, options_src, NULL);

    if (self_debug)
    {
        Test_ReadPQR();
        Test_PhysicalParameters();
        Test_G_gradG_cfun(); // slow
    }

    Array<double> phi3L2errornorms, c1L2errornorms, c2L2errornorms, meshsizes, timesteps;
    if (SpaceConvergRate) // dt 不变, 改变 h
    {
        MFEM_ASSERT(!TimeConvergRate, "SpaceConvergRate and TimeConvergRate cannot be true simultaneously");

        for (int i=0; i<refine_time; ++i) t_stepsize *= time_scale; // 先把时间步长 dt 确定下来

        Mesh* mesh = new Mesh(mesh_file);
        Array<ParMesh*> pmeshes;
        Array<Return*> rets;
        Array<PNP_Protein_TimeDependent_Solver*> solvers;

        int origin_refine_mesh = refine_mesh; // save refine_mesh temporarily
        for (int i=0; i <= refine_mesh; ++i) // 对网格加密多次
        {
            ParMesh *pmesh;
            if (i == 0) {
                pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
            } else {
                pmesh = new ParMesh(*pmeshes[i - 1]);
                pmesh->UniformRefinement();
            }
            pmeshes.Append(pmesh);

            refine_mesh = i; // for cout right verbose outputs

            auto* solver = new PNP_Protein_TimeDependent_Solver(pmesh, ode_type);
            Return* ret = solver->Solve(meshsizes, timesteps);
            solvers.Append(solver);
            rets.Append(ret);

            refine_mesh = origin_refine_mesh; // reset real refine_mesh
        }

        for (int i=0; i<rets.Size()-1; i++)
        {
            GridTransfer* gt = new InterpolationGridTransfer(*rets[i]->fes, *rets[i+1]->fes);
            const Operator& Prolongate = gt->ForwardOperator();
            const Operator& Restrict   = gt->BackwardOperator();

            ParGridFunction phi3_f2c(rets[i]->fes), c1_f2c(rets[i]->fes), c2_f2c(rets[i]->fes); // fine to coarse

            Restrict.Mult(*rets[i+1]->phi3, phi3_f2c);
            GridFunctionCoefficient phi3_f2c_coeff(&phi3_f2c);
            phi3L2errornorms.Append(rets[i]->phi3->ComputeL2Error(phi3_f2c_coeff));

            Restrict.Mult(*rets[i+1]->c1, c1_f2c);
            GridFunctionCoefficient c1_f2c_coeff(&c1_f2c);
            c1L2errornorms.Append(rets[i]->c1->ComputeL2Error(c1_f2c_coeff));

            Restrict.Mult(*rets[i+1]->c2, c2_f2c);
            GridFunctionCoefficient c2_f2c_coeff(&c2_f2c);
            c2L2errornorms.Append(rets[i]->c2->ComputeL2Error(c2_f2c_coeff));
        }

        for (int i=0; i<pmeshes.Size(); ++i) delete pmeshes[i];
        for (int i=0; i<solvers.Size(); ++i) delete solvers[i];
        delete mesh;
    }
    else if (TimeConvergRate) // h 不变, 改变 dt
    {
        MFEM_ASSERT(!SpaceConvergRate, "SpaceConvergRate and TimeConvergRate cannot be true simultaneously");

    }
    else
    {
        Mesh* mesh = new Mesh(mesh_file);
        auto* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        delete mesh;

        for (int i=0; i<refine_mesh; i++) pmesh->UniformRefinement(); // 确定计算网格
        for (int i=0; i<refine_time; i++) t_stepsize *= time_scale;   // 确定时间步长

        auto* solver = new PNP_Protein_TimeDependent_Solver(pmesh, ode_type);
        solver->Solve(meshsizes, timesteps);

        delete solver;
        delete pmesh;
    }

    if (rank == 0 && (SpaceConvergRate || TimeConvergRate))
    {
        meshsizes.Print(cout << "\nMesh sizes: \n", meshsizes.Size());
        timesteps.Print(cout << "Time-step sizes: \n", timesteps.Size());

        phi3L2errornorms.Print(cout << "\nL2 errornorms of |phi3 - phi3_h|: \n", phi3L2errornorms.Size());
        c1L2errornorms.Print(cout << "\nL2 errornorms of |c1 - c1_h|: \n", c1L2errornorms.Size());
        c2L2errornorms.Print(cout << "\nL2 errornorms of |c2 - c2_h|: \n", c2L2errornorms.Size());

        Array<double> phi3rates, c1rates, c2rates;
        if (SpaceConvergRate)
        {
            phi3rates = compute_convergence(phi3L2errornorms, meshsizes);
            c1rates = compute_convergence(c1L2errornorms, meshsizes);
            c2rates = compute_convergence(c2L2errornorms, meshsizes);
        }
        else
        {
            MFEM_ASSERT(TimeConvergRate, "SpaceConvergRate or TimeConvergRate: must choose one.");

            phi3rates = compute_convergence(phi3L2errornorms, timesteps);
            c1rates = compute_convergence(c1L2errornorms, timesteps);
            c2rates = compute_convergence(c2L2errornorms, timesteps);
        }

        phi3rates.Print(cout << "\nphi3 L2 convergence rate: \n", phi3rates.Size());
        c1rates  .Print(cout << "c1   L2 convergence rate: \n", c1rates.Size());
        c2rates  .Print(cout << "c2   L2 convergence rate: \n", c2rates.Size());
    }

    MFEMFinalizePetsc();
    MPI_Finalize();
}
