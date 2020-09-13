#ifndef _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_
#define _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_

#include "./pnp_box_timedependent.hpp"
#include <map>
#include "../utils/EAFE_ModifyStiffnessMatrix.hpp"
#include "../utils/GradConvection_Integrator.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/SUPG_Integrator.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "petsc.h"
#include "../utils/petsc_utils.hpp"
#include "../utils/LocalConservation.hpp"
//#include "../utils/python_utils.hpp" // not work in computer cluster


/* Poisson Equation:
 *     div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
 * NP Equation:
 *     dc_i / dt = div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * */
class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* h1;

    HypreParMatrix *A, *M1, *M2, *B1, *B2;
    PetscLinearSolver *A_solver, *M1_solver, *M2_solver;

    mutable Vector z;
    mutable HypreParVector *b, *b1, *b2;
    mutable HypreParMatrix *A1, *A2;

    int true_size;
    Array<int> true_offset, ess_tdof_list;
    int num_procs, myid;

public:
    PNP_Box_Gummel_CG_TimeDependent(HypreParMatrix* A_, HypreParMatrix* M1_, HypreParMatrix* M2_,
                                    HypreParMatrix* B1_, HypreParMatrix* B2_,
                                    int truesize, Array<int>& offset, Array<int>& ess_list,
                                    ParFiniteElementSpace* fsp, double time)
        : TimeDependentOperator(3*truesize, time), A(A_), M1(M1_), M2(M2_), B1(B1_), B2(B2_),
          true_size(truesize), true_offset(offset), ess_tdof_list(ess_list), h1(fsp)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        A_solver  = new PetscLinearSolver(*A,  false, "phi_");
        M1_solver = new PetscLinearSolver(*M1, false, "np1_");
        M2_solver = new PetscLinearSolver(*M2, false, "np2_");
    }
    virtual ~PNP_Box_Gummel_CG_TimeDependent()
    {
        delete A_solver;
        delete M1_solver;
        delete M2_solver;
    }

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
        Vector phi(phic1c2.GetData() + 0*true_size, true_size);
        Vector c1 (phic1c2.GetData() + 1*true_size, true_size);
        Vector c2 (phic1c2.GetData() + 2*true_size, true_size);
        Vector dphi_dt(dphic1c2_dt.GetData() + 0*true_size, true_size);
        Vector dc1_dt (dphic1c2_dt.GetData() + 1*true_size, true_size);
        Vector dc2_dt (dphic1c2_dt.GetData() + 2*true_size, true_size);

        // 首先求解Poisson方程
        ParGridFunction new_phi(h1);
        new_phi.SetFromTrueDofs(phi); // 让new_phi满足essential bdc

        if (myid == 0) cout << "l2 norm of phi: " << phi.Norml2() << endl;

        ParLinearForm *l = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l->Assemble();
        b = l->ParallelAssemble(); // 一定要自己delete b

        // 在求解器求解的外面所使用的Vector，Matrix全部是Hypre类型的，在给PETSc的Krylov求解器传入参数
        // 时也是传入的Hypre类型的(因为求解器内部会将Hypre的矩阵和向量转化为PETSc的类型)
        B1->Mult(1.0, c1, 1.0, *b); // B1 c1 + b -> b
        B2->Mult(1.0, c2, 1.0, *b); // B1 c1 + B2 c2 + b -> b
        b->SetSubVector(ess_tdof_list, 0.0); // 给定essential bdc
        A_solver->Mult(*b, new_phi);
        add(1.0, new_phi, -1.0, phi, dphi_dt);
        dphi_dt /= t_stepsize; // fff应该是dt_real

        // 然后求解NP1方程
        ParBilinearForm *a22 = new ParBilinearForm(h1);
        a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        a22->AddDomainIntegrator(new GradConvectionIntegrator(new_phi, &D_K_prod_v_K));
        a22->Assemble(skip_zero_entries);
        a22->Finalize(skip_zero_entries);
        A1 = a22->ParallelAssemble(); // 一定要自己delete A1
        A1->EliminateRowsCols(ess_tdof_list);

        ParLinearForm *l1 = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        b1 = l1->ParallelAssemble(); // 一定要自己delete b1
        b1->SetSubVector(ess_tdof_list, 0.0);

        A1->Mult(1.0, c1, 1.0, *b1); // A1 c1 + b1 -> b1
        M1_solver->Mult(*b1, dc1_dt); // solve M1 dc1_dt = A1 c1 + b1


        // 然后求解NP2方程
        ParBilinearForm *a33 = new ParBilinearForm(h1);
        a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        a33->AddDomainIntegrator(new GradConvectionIntegrator(new_phi, &D_Cl_prod_v_Cl));
        a33->Assemble(skip_zero_entries);
        a33->Finalize(skip_zero_entries);
        A2 = a33->ParallelAssemble(); // 一定要自己delete A2
        A2->EliminateRowsCols(ess_tdof_list);

        ParLinearForm *l2 = new ParLinearForm(h1);
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();
        b2 = l2->ParallelAssemble(); // 一定要自己delete b2
        b2->SetSubVector(ess_tdof_list, 0.0);

        A2->Mult(1.0, c2, 1.0, *b2); // A2 c2 + b2 -> b2
        M2_solver->Mult(*b2, dc2_dt); // solve M2 dc2_dt = A2 c2 + b2

        delete l;
        delete a22;
        delete l1;
        delete a33;
        delete l2;
        delete b;
        delete A1;
        delete b1;
        delete A2;
        delete b2;
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        Vector phi(phic1c2.GetData() + 0*true_size, true_size);
        Vector c1 (phic1c2.GetData() + 1*true_size, true_size);
        Vector c2 (phic1c2.GetData() + 2*true_size, true_size);
        Vector dphi_dt(dphic1c2_dt.GetData() + 0*true_size, true_size);
        Vector dc1_dt (dphic1c2_dt.GetData() + 1*true_size, true_size);
        Vector dc2_dt (dphic1c2_dt.GetData() + 2*true_size, true_size);

        // 首先求解Poisson方程
        ParGridFunction new_phi(h1);
        new_phi.SetFromTrueDofs(phi); // 让new_phi满足essential bdc

        ParLinearForm *l = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l->Assemble();
        b = l->ParallelAssemble();

        // 在求解器求解的外面所使用的Vector，Matrix全部是Hypre类型的，在给PETSc的Krylov求解器传入参数
        // 时也是传入的Hypre类型的(因为求解器内部会将Hypre的矩阵和向量转化为PETSc的类型)
        B1->Mult(1.0, c1, 1.0, *b); // B1 c1 + b -> b
        B2->Mult(1.0, c2, 1.0, *b); // B1 c1 + B2 c2 + b -> b
        b->SetSubVector(ess_tdof_list, 0.0); // 给定essential bdc
        A_solver->Mult(*b, new_phi);
        dphi_dt = (new_phi - phi) / t_stepsize; // fff应该是dt_real

        // 然后求解NP1方程
        ParBilinearForm *a22 = new ParBilinearForm(h1);
        a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        a22->AddDomainIntegrator(new GradConvectionIntegrator(new_phi, &D_K_prod_v_K));
        a22->Assemble(skip_zero_entries); // keep sparsity pattern of A1 and M1 the same
        a22->Finalize(skip_zero_entries);
        A1 = a22->ParallelAssemble();
        A1->EliminateRowsCols(ess_tdof_list);

        ParLinearForm *l1 = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        b1 = l1->ParallelAssemble();
        b1->SetSubVector(ess_tdof_list, 0.0);

        A1->Mult(1.0, c1, 1.0, *b1); // A1 c1 + b1 -> b1
        HypreParMatrix* temp_A1 = Add(1.0, *M1, -1.0*t_stepsize, *A1); // gurantee M1 and A1 with same sparsity pattern
        PetscLinearSolver* A1_solver = new PetscLinearSolver(*temp_A1, false, "np1_");
        A1_solver->Mult(*b1, dc1_dt);

        // 然后求解NP2方程
        ParBilinearForm *a33 = new ParBilinearForm(h1);
        a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        a33->AddDomainIntegrator(new GradConvectionIntegrator(new_phi, &D_Cl_prod_v_Cl));
        a33->Assemble(skip_zero_entries); // keep sparsity pattern of A2 and M2 the same
        a33->Finalize(skip_zero_entries);
        A2 = a33->ParallelAssemble();
        A2->EliminateRowsCols(ess_tdof_list);

        ParLinearForm *l2 = new ParLinearForm(h1);
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();
        b2 = l2->ParallelAssemble();
        b2->SetSubVector(ess_tdof_list, 0.0);

        A2->Mult(1.0, c2, 1.0, *b2); // A2 c2 + b2 -> b2
        HypreParMatrix* temp_A2 = Add(1.0, *M2, -1.0*t_stepsize, *A2); // guarantee M2 and A2 with same sparsity pattern
        PetscLinearSolver* A2_solver = new PetscLinearSolver(*temp_A2, false, "np2_");
        A2_solver->Mult(*b2, dc2_dt);

        delete l;
        delete a22;
        delete l1;
        delete a33;
        delete l2;
        delete temp_A1, temp_A2;
        delete A1_solver, A2_solver;
    }
};
class PNP_Box_Gummel_CG_TimeDependent_Solver
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    ParBilinearForm *a11, *a12, *a13, *m1, *m2;
    HypreParMatrix *A, *B1, *B2, *M1, *M2;
    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    PNP_Box_Gummel_CG_TimeDependent* oper;
    double t; // 当前时间
    Vector init_value;
    ODESolver *ode_solver;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr, ess_tdof_list;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_Solver(Mesh& mesh_, int ode_solver_type): mesh(mesh_)
    {
        t = t_init;

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的
        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        a11 = new ParBilinearForm(h1);
        // (epsilon_s grad(phi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a11->Assemble(skip_zero_entries);
        a11->Finalize(skip_zero_entries);
        A = a11->ParallelAssemble();
        A->EliminateRowsCols(ess_tdof_list); // fff边界条件对吗

        a12 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z1 c1, psi)
        a12->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(skip_zero_entries);
        a12->Finalize(skip_zero_entries);
        B1 = a12->ParallelAssemble(); // fff不需要设置essential bdc吗

        a13 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z2 c2, psi)
        a13->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(skip_zero_entries);
        a13->Finalize(skip_zero_entries);
        B2 = a13->ParallelAssemble(); // fff不需要设置essential bdc吗

        m1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1->AddDomainIntegrator(new MassIntegrator);
        m1->Assemble(skip_zero_entries); // keep sparsity pattern of A1 and M1 the same
        m1->Finalize(skip_zero_entries);
        M1 = m1->ParallelAssemble();
        M1->EliminateRowsCols(ess_tdof_list); // fffbdc对吗

        m2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2->AddDomainIntegrator(new MassIntegrator);
        m2->Assemble(skip_zero_entries); // keep sparsity pattern of A2 and M2 the same
        m2->Finalize(skip_zero_entries);
        M2 = m2->ParallelAssemble();
        M2->EliminateRowsCols(ess_tdof_list); // fffbdc对吗

        phi_gf = new ParGridFunction(h1); *phi_gf = 0.0;
        c1_gf  = new ParGridFunction(h1); *c1_gf  = 0.0;
        c2_gf  = new ParGridFunction(h1); *c2_gf  = 0.0;

        true_size = h1->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_size;
        true_offset[2] = true_size * 2;
        true_offset[3] = true_size * 3;

        phic1c2 = new BlockVector(true_offset); *phic1c2 = 0.0;
        phi_gf->MakeTRef(h1, *phic1c2, true_offset[0]);
        c1_gf ->MakeTRef(h1, *phic1c2, true_offset[1]);
        c2_gf ->MakeTRef(h1, *phic1c2, true_offset[2]);

        // 设定初值
        phi_exact.SetTime(t);
        phi_gf->ProjectCoefficient(phi_exact);
        phi_gf->SetTrueVector();
        phi_gf->SetFromTrueVector();

        c1_exact.SetTime(t);
        c1_gf->ProjectCoefficient(c1_exact);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_exact.SetTime(t);
        c2_gf->ProjectCoefficient(c2_exact);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        oper = new PNP_Box_Gummel_CG_TimeDependent(A, M1, M2, B1, B2, true_size,
                                            true_offset, ess_tdof_list, h1, t);

        switch (ode_solver_type)
        {
            // Implicit L-stable methods
            case 1:  ode_solver = new BackwardEulerSolver; break;
            case 2:  ode_solver = new SDIRK23Solver(2); break;
            case 3:  ode_solver = new SDIRK33Solver; break;
                // Explicit methods
            case 11: ode_solver = new ForwardEulerSolver; break;
            case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
            case 13: ode_solver = new RK3SSPSolver; break;
            case 14: ode_solver = new RK4Solver; break;
                // Implicit A-stable methods (not L-stable)
            case 22: ode_solver = new ImplicitMidpointSolver; break;
            case 23: ode_solver = new SDIRK23Solver; break;
            case 24: ode_solver = new SDIRK34Solver; break;
            default:
            MFEM_ABORT("Not support ODE solver.");
        }

        oper->SetTime(t);
        ode_solver->Init(*oper);

        if (paraview)
        {
            pd = new ParaViewDataCollection("PNP_CG_Gummel_Time_Dependent", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);
        }
    }
    ~PNP_Box_Gummel_CG_TimeDependent_Solver()
    {
        delete pmesh;
        delete fec;
        delete h1;
        delete a11;
        delete a12;
        delete a13;
        delete m1;
        delete m2;
        delete phi_gf;
        delete c1_gf;
        delete c2_gf;
        delete phic1c2;
        delete oper;
        delete ode_solver;
        if (paraview) delete pd;
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        if (myid == 0) {
            cout << '\n';
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times << '\n'
                 << ", " << options_src << '\n'
                 << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" \
                                                                       : "wrong type")) << ", " << "time step: " << t_stepsize
                 << endl;
        }

        int gdb_break = 0;
        while(gdb_break) {};

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();

        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(t_stepsize, t_final - t);

            ode_solver->Step(*phic1c2, t, dt_real); // 进过这一步之后phic1c2和t都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            if (paraview)
            {
                pd->SetCycle(ti); // 第 i 个时间步
                pd->SetTime(t); // 第i个时间步所表示的时间
                pd->Save();
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Stop();

        {
            phi_exact.SetTime(t);
            phi_gf->SetFromTrueVector();

            c1_exact.SetTime(t);
            c1_gf->SetFromTrueVector();

            c2_exact.SetTime(t);
            c2_gf->SetFromTrueVector();

            // 计算误差范数只能是在所有进程上都运行，输出误差范数可以只在root进程
            double phiL2err = phi_gf->ComputeL2Error(phi_exact);
            double c1L2err = c1_gf->ComputeL2Error(c1_exact);
            double c2L2err = c2_gf->ComputeL2Error(c2_exact);

            if (myid == 0) {
                cout << "ODE solver taking " << chrono.RealTime() << " s." << endl;
                cout.precision(14);
                cout << "At final time: " << t << '\n'
                     << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
                     << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
                     << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;
            }

            if (ComputeConvergenceRate)
            {
                double totle_size = 0.0;
                for (int i = 0; i < mesh.GetNE(); i++)
                    totle_size += mesh.GetElementSize(0, 1);
                meshsizes_.Append(totle_size / mesh.GetNE());

                phiL2errornorms_.Append(phiL2err);
                c1L2errornorms_.Append(c1L2err);
                c2L2errornorms_.Append(c2L2err);
            }
        }
    }
};


/* Poisson Equation:
 *     div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
 * NP Equation:
 *     dc_i / dt = div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * 使用Gummel线性化方法: 给定 phi^n, c1^n, c2^n, 求 phi^n+1, c1^n+1, c2^n+1
 *     A phi^n+1 = B1 c1^n + B2 c2^n + b
 *     M1 c1^n+1 = M1 c1^n + dt A1 c1^n + dt b1
 *     M2 c2^n+1 = M2 c2^n + dt A2 c2^n + dt b2
 * */
class PNP_Box_Gummel_CG_TimeDependent_ForwardEuler
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    ParBilinearForm *a11, *a12, *a13, *m1, *m2;
    HypreParMatrix *A, *B1, *B2, *M1, *M2, *M1_no_bc, *M2_no_bc;
    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    double current_t;
    mutable Vector z;
    mutable HypreParVector *b, *b1, *b2;
    mutable HypreParMatrix *A1, *A2;
    PetscLinearSolver *A_solver, *M1_solver, *M2_solver;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr, ess_tdof_list;
    socketstream vis_ostream;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_ForwardEuler(Mesh& mesh_): mesh(mesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的
        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        a11 = new ParBilinearForm(h1);
        // (epsilon_s grad(phi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a11->Assemble(skip_zero_entries);
        a11->Finalize(skip_zero_entries);
        // 这里暂时不形成最终的刚度矩阵, 后面在形成的时候同时设定essential bc

        a12 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z1 c1, psi)
        a12->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(skip_zero_entries);
        a12->Finalize(skip_zero_entries);
        B1 = a12->ParallelAssemble();

        a13 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z2 c2, psi)
        a13->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(skip_zero_entries);
        a13->Finalize(skip_zero_entries);
        B2 = a13->ParallelAssemble();

        m1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1->AddDomainIntegrator(new MassIntegrator);
        m1->Assemble(skip_zero_entries); // keep sparsity pattern of A1 and M1 the same
        m1->Finalize(skip_zero_entries);
        M1_no_bc = m1->ParallelAssemble();

        m2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2->AddDomainIntegrator(new MassIntegrator);
        m2->Assemble(skip_zero_entries); // keep sparsity pattern of A2 and M2 the same
        m2->Finalize(skip_zero_entries);
        M2_no_bc = m2->ParallelAssemble();

        A  = new HypreParMatrix();
        M1 = new HypreParMatrix();
        M2 = new HypreParMatrix();

        phi_gf = new ParGridFunction(h1); *phi_gf = 0.0;
        c1_gf  = new ParGridFunction(h1); *c1_gf  = 0.0;
        c2_gf  = new ParGridFunction(h1); *c2_gf  = 0.0;

        true_size = h1->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_size;
        true_offset[2] = true_size * 2;
        true_offset[3] = true_size * 3;

        phic1c2  = new BlockVector(true_offset); *phic1c2 = 0.0;
        phi_gf->MakeTRef(h1, *phic1c2, true_offset[0]);
        c1_gf ->MakeTRef(h1, *phic1c2, true_offset[1]);
        c2_gf ->MakeTRef(h1, *phic1c2, true_offset[2]);

        // 设定初始条件，同时满足了边界条件
        current_t = t_init;
        phi_exact.SetTime(current_t);
        phi_gf->ProjectCoefficient(phi_exact);
        phi_gf->SetTrueVector();
        phi_gf->SetFromTrueVector();

        c1_exact.SetTime(current_t);
        c1_gf->ProjectCoefficient(c1_exact);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_exact.SetTime(current_t);
        c2_gf->ProjectCoefficient(c2_exact);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        {
            double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
            double  c1L2errornorm =  c1_gf->ComputeL2Error(c1_exact);
            double  c2L2errornorm =  c2_gf->ComputeL2Error(c2_exact);
            if (myid == 0) {
                cout << "After set initial conditions: \n"
                     << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
            }
        }

        if (paraview)
        {
            pd = new ParaViewDataCollection("PNP_Box_Gummel_CG_TimeDependent_ForwardEuler", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY32);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);

            pd->SetCycle(0);
            pd->SetTime(current_t);
            pd->Save();
        }
    }
    ~PNP_Box_Gummel_CG_TimeDependent_ForwardEuler()
    {
        delete pmesh;
        delete fec;
        delete h1;

        delete a11;
        delete a12;
        delete B1;
        delete a13;
        delete B2;
        delete m1;
        delete M1_no_bc;
        delete m2;
        delete M2_no_bc;

        delete A;
        delete M1;
        delete M2;

        delete phi_gf;
        delete c1_gf;
        delete c2_gf;
        delete phic1c2;

        if (paraview) delete pd;

        delete b;
        delete A1;
        delete b1;
        delete A2;
        delete b2;

        delete A_solver;
        delete M1_solver;
        delete M2_solver;
    }

    void Step()
    {
        // 获得已知的上一时刻的解的TrueVector
        Vector phi(phic1c2->GetData() + 0*true_size, true_size);
        Vector c1 (phic1c2->GetData() + 1*true_size, true_size);
        Vector c2 (phic1c2->GetData() + 2*true_size, true_size);

        // new_*_gf 用在Gummel迭代中
        ParGridFunction new_phi_gf(h1), phi_Gummel(h1), new_c1_gf(h1), new_c2_gf(h1);
        new_phi_gf = 0.0;
        new_c1_gf  = 0.0;
        new_c2_gf  = 0.0;
        phi_exact.SetTime(current_t);
        c1_exact .SetTime(current_t);
        c2_exact .SetTime(current_t);
        new_phi_gf.ProjectBdrCoefficient(phi_exact, ess_bdr); // 需要满足下一时刻的边界条件
        new_c1_gf .ProjectBdrCoefficient( c1_exact, ess_bdr);
        new_c2_gf .ProjectBdrCoefficient( c2_exact, ess_bdr);

        const Operator &R = *h1->GetRestrictionMatrix(); // Operator要不要换成SparseMatrixfff
        const Operator &P = *h1->GetProlongationMatrix();
        f1_analytic.SetTime(current_t);
        f2_analytic.SetTime(current_t);

        Vector diff(h1->GetVSize());
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // new_* 就是对应 new_*_gf 的TrueVector, 用在Gummel迭代中
            new_phi_gf.SetTrueVector();
            new_c1_gf .SetTrueVector();
            new_c2_gf .SetTrueVector();
            HypreParVector* new_phi = new_phi_gf.GetTrueDofs();
            HypreParVector* new_c1  = new_c1_gf.GetTrueDofs();
            HypreParVector* new_c2  = new_c2_gf.GetTrueDofs();
            phi_Gummel = new_phi_gf;

            {
            /// ------------------------------ 求解Poisson方程 ------------------------------
                // 下一时刻的Poisson方程的右端项
                ParLinearForm *l = new ParLinearForm(h1);
                l->Assemble();
                delete b;
                b = l->ParallelAssemble(); // 一定要自己delete b

                // 在求解器求解的外面所使用的Vector，Matrix全部是Hypre类型的，在给PETSc的Krylov求解器传入参数时也是传入的Hypre类型的(因为求解器内部会将Hypre的矩阵和向量转化为PETSc的类型)
                B1->Mult(1.0, *new_c1, 1.0, *b); // B1 c1^k + b -> b. k表示Gummel迭代次数, 下同
                B2->Mult(1.0, *new_c2, 1.0, *b); // B1 c1^k + B2 c2^k + b -> b

                R.MultTranspose(*b, *l);
                a11->FormLinearSystem(ess_tdof_list, new_phi_gf, *l, *A, *new_phi, *b);
                A_solver = new PetscLinearSolver(*A, false, "phi_");
                A_solver->Mult(*b, *new_phi);

                new_phi_gf = *new_phi; // 更新phi
                delete l;

                diff = 0.0;
                diff += new_phi_gf;
                diff -= phi_Gummel;
                double tol = diff.Norml2() / new_phi_gf.Norml2();
                phi_Gummel = new_phi_gf; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
                if (myid == 0) {
                    cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
                }
                if (tol < Gummel_rel_tol) { // Gummel迭代停止
                    last_gummel_step = true;
                }
            }

            {
            /// ------------------------------ 求解NP1方程 ------------------------------
                ParBilinearForm *a22 = new ParBilinearForm(h1);
                // D1 (grad(c1^n), grad(v1))
                a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
                // D1 z1 c1^n (grad(phi^n+1), grad(v1)), 这里phi^n+1就是new_phi
                a22->AddDomainIntegrator(new GradConvectionIntegrator(new_phi_gf, &D_K_prod_v_K));
                a22->Assemble(skip_zero_entries);
                a22->Finalize(skip_zero_entries);
                A1 = a22->ParallelAssemble(); // 一定要自己delete A1

                ParLinearForm *l1 = new ParLinearForm(h1);
                // (f1, v1)
                l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
                l1->Assemble();
                b1 = l1->ParallelAssemble(); // 一定要自己delete b1

//                A1->Mult(t_stepsize, *new_c1, t_stepsize, *b1); // dt (A1 c1^k + b1) -> b1
                M1_no_bc->Mult(1.0, c1, t_stepsize, *b1); // M1 c1 + dt b1 -> b1

                R.MultTranspose(*b1, *l1);
                m1->FormLinearSystem(ess_tdof_list, new_c1_gf, *l1, *M1, *new_c1, *b1); // goon
                M1_solver = new PetscLinearSolver(*M1, false, "np1_");
                M1_solver->Mult(*b1, *new_c1);

                new_c1_gf = *new_c1;
                delete a22;
                delete l1;
            }
            {
            /// ------------------------------ 求解NP2方程 ------------------------------
                ParBilinearForm *a33 = new ParBilinearForm(h1);
                // D2 (grad(c2^n), grad(v2))
                a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
                // D2 z2 c2^n (grad(phi^n+1), grad(v2)), 这里phi^n+1就是new_phi
                a33->AddDomainIntegrator(new GradConvectionIntegrator(new_phi_gf, &D_Cl_prod_v_Cl));
                a33->Assemble(skip_zero_entries);
                a33->Finalize(skip_zero_entries);
                A2 = a33->ParallelAssemble(); // 一定要自己delete A2

                ParLinearForm *l2 = new ParLinearForm(h1);
                // (f2, v2)
                l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
                l2->Assemble();
                b2 = l2->ParallelAssemble(); // 一定要自己delete b2

//                A2->Mult(t_stepsize, *new_c2, t_stepsize, *b2); // dt (A2 c2^k + b2) -> b2
                M2_no_bc->Mult(1.0, c2, t_stepsize, *b2); // M2 c2 + dt b2 -> b2

                R.MultTranspose(*b2, *l2);
                m2->FormLinearSystem(ess_tdof_list, new_c2_gf, *l2, *M2, *new_c2, *b2);
                M2_solver = new PetscLinearSolver(*M2, false, "np2_");
                M2_solver->Mult(*b2, *new_c2);

                new_c2_gf = *new_c2;
                delete a33;
                delete l2;
            }
        }

        {
            phi_exact.SetTime(current_t);
            c1_exact.SetTime(current_t);
            c2_exact.SetTime(current_t);
            double phiL2errornorm = new_phi_gf.ComputeL2Error(phi_exact);
            double  c1L2errornorm =  new_c1_gf.ComputeL2Error(c1_exact);
            double  c2L2errornorm =  new_c2_gf.ComputeL2Error(c2_exact);
            if (myid == 0) {
                cout << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
            }
        }
    }

    void Solve()
    {
        bool last_step = false;
        for (int step=1; !last_step; ++step)
        {
            current_t += t_stepsize; // 更新时间：即将求的下一个时刻的物理量
            if (current_t + t_stepsize >= t_final - t_stepsize/2) last_step = true;

            Step(); // 更新解向量TrueVector

            phi_gf->SetFromTrueVector();
            c1_gf ->SetFromTrueVector();
            c2_gf ->SetFromTrueVector();
            {
                phi_exact.SetTime(current_t);
                c1_exact.SetTime(current_t);
                c2_exact.SetTime(current_t);
                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm =  c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm =  c2_gf->ComputeL2Error(c2_exact);
                if (myid == 0) {
                    cout << "\nStep: " << step << '\n'
                         << "phi L2 errornorm: " << phiL2errornorm << '\n'
                         << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                         << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
                }
            }

            if (paraview)
            {
                pd->SetCycle(step);
                pd->SetTime(current_t);
                pd->Save();
            }
        }
    }
};
#endif
