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
    args.AddOption(&p_order, "-p", "--p_order", "Polynomial order of basis function.");
    args.AddOption(&Linearize, "-lin", "--linearize", "Linearization method: choose: cg, dg");
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
    args.AddOption(&prec_type, "-prec", "--prec_type", "Preconditioner type for Newton disretization, choose: block, uzawa, simple");
    args.AddOption(&schur_alpha1, "-schur1", "--schur1", "1st parameter for Schur Complement");
    args.AddOption(&schur_alpha2, "-schur2", "--schur2", "2nd parameter for Schur Complement");
    args.AddOption(&options_src, "-opts", "--petscopts", "Petsc options file");
    args.AddOption(&paraview, "-para", "--paraview", "-nopara", "--noparaview", "Save time-dependent results");
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

    MFEMInitializePetsc(NULL, NULL, options_src, NULL);

    if (self_debug)
    {
        Test_ReadPQR();
        Test_PhysicalParameters();
        Test_G_gradG_cfun(); // slow
    }

    Array<Return*> rets;
    if (SpaceConvergRate) // dt 不变, 改变 h
    {
        MFEM_ASSERT(!TimeConvergRate, "SpaceConvergRate and TimeConvergRate cannot be true simultaneously");

        for (int i=0; i<refine_time; ++i) t_stepsize *= time_scale; // 先把时间步长 dt 确定下来

        Mesh* mesh = new Mesh(mesh_file);
        Array<ParMesh*> pmeshes;

        int origin_refine_mesh = refine_mesh; // save refine_mesh temporarily
        for (int i=0; i <= refine_mesh; ++i) // 对网格加密多次
        {
            ParMesh *pmesh;
            if (i == 0) {
                pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
            }
            else {
                pmesh = new ParMesh(*pmeshes[i]);
                pmesh->UniformRefinement();
            }
            pmeshes.Append(pmesh);

            refine_mesh = i; // for cout right verbose outputs

            PNP_Protein_TimeDependent_Solver* solver = new PNP_Protein_TimeDependent_Solver(pmesh, ode_type);
            Return* ret = solver->Solve();
            rets.Append(ret);

            refine_mesh = origin_refine_mesh; // reset real refine_mesh
        }

        if (rank == 0) cout << "finish computing" << endl;

        if (rets.Size() > 1)
        {
            for (int i=0; i<rets.Size()-1; i++)
            {
                if (rank == 0) cout << "before forming GridTransfer" << endl;

                GridTransfer* gt = new InterpolationGridTransfer(*rets[i]->fes, *rets[i+1]->fes);
//                const Operator& Prolongate = gt->ForwardOperator();
                const Operator& Restrict   = gt->BackwardOperator();

                if (rank == 0) cout << "after forming GridTransfer" << endl;

                ParGridFunction temp_f2c(rets[i]->fes); // fine to coarse
                Restrict.Mult(*rets[i+1]->phi3, temp_f2c);
                
                GridFunctionCoefficient temp_f2c_coeff(&temp_f2c);
                double L2err = rets[i]->phi3->ComputeL2Error(temp_f2c_coeff);

                if (rank == 0) cout << "fffffffffffffffffff: " << L2err << endl;
            }
        }

        delete mesh;

    }
    else if (TimeConvergRate) // h 不变, 改变 dt
    {
        MFEM_ASSERT(!SpaceConvergRate, "SpaceConvergRate and TimeConvergRate cannot be true simultaneously");

    }
    else
    {
        Mesh* mesh = new Mesh(mesh_file);
        ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        delete mesh;

        for (int i=0; i<refine_mesh; i++) pmesh->UniformRefinement(); // 确定计算网格
        for (int i=0; i<refine_time; i++) t_stepsize *= time_scale;   // 确定时间步长

        PNP_Protein_TimeDependent_Solver* solver = new PNP_Protein_TimeDependent_Solver(pmesh, ode_type);
        solver->Solve();

        delete solver;
        delete pmesh;
    }

    MFEMFinalizePetsc();
    MPI_Finalize();
}
