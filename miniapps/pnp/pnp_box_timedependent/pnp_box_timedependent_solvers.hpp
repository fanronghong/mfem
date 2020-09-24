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
 *     dc_i / dt = div( D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * */
class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* h1;

    mutable ParBilinearForm *a0, *a1, *a2, *b1, *b2, *m, *m1_dta1, *m2_dta2;
    mutable HypreParMatrix *A0, *M, *M1_dtA1, *M2_dtA2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;
    int true_size;
    mutable Array<int> true_offset, ess_bdr, ess_tdof_list;
    int num_procs, myid;

public:
    PNP_Box_Gummel_CG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_,
                                    ParFiniteElementSpace* fsp, double time)
            : TimeDependentOperator(3*truesize, time), true_size(truesize), true_offset(offset), ess_bdr(ess_bdr_), h1(fsp),
              b1(NULL), b2(NULL), a0(NULL), a1(NULL), m(NULL), a2(NULL), m1_dta1(NULL), m2_dta2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        A0 = new HypreParMatrix;
        M = new HypreParMatrix;
        M1_dtA1 = new HypreParMatrix;
        M2_dtA2 = new HypreParMatrix;

        temp_x0 = new Vector;
        temp_b0 = new Vector;
        temp_x1 = new Vector;
        temp_b1 = new Vector;
        temp_x2 = new Vector;
        temp_b2 = new Vector;

    }
    virtual ~PNP_Box_Gummel_CG_TimeDependent()
    {
        delete m;
        delete m1_dta1;
        delete m2_dta2;
        delete a0;
        delete a1;
        delete a2;
        delete b1;
        delete b2;

        delete temp_x0;
        delete temp_b0;
        delete temp_x1;
        delete temp_b1;
        delete temp_x2;
        delete temp_b2;
    }

    // (ci, vi), i=1,2
    void buildm() const
    {
        if (m != NULL) { delete m; }
        
        m = new ParBilinearForm(h1);
        m->AddDomainIntegrator(new MassIntegrator);
        
        m->Assemble(skip_zero_entries);
    }
    
    // (c1, v1) + dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given dt and phi
    void buildm1_dta1(double dt, ParGridFunction& phi) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);

        m1_dta1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1_dta1->AddDomainIntegrator(new MassIntegrator);
        // dt D1 (grad(c1), grad(v1))
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        // dt D1 z1 (c1 grad(phi), grad(v1))
        m1_dta1->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // (c2, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given dt and phi
    void buildm2_dta2(double dt, ParGridFunction& phi) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);

        m2_dta2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2_dta2->AddDomainIntegrator(new MassIntegrator);
        // dt D2 (grad(c2), grad(v2))
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        // dt D2 z2 (c2 grad(phi), grad(v2))
        m2_dta2->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D2_z2));

        m2_dta2->Assemble(skip_zero_entries);
    }

    // epsilon_s (grad(phi), grad(psi))
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(h1);
        a0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        a0->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
    void builda1(ParGridFunction& phi) const
    {
        if (a1 != NULL) { delete a1; }

        a1 = new ParBilinearForm(h1);
        // D1 (grad(c1), grad(v1))
        a1->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi), grad(v1))
        a1->AddDomainIntegrator(new GradConvectionIntegrator(phi, &D_K_prod_v_K));

        a1->Assemble(skip_zero_entries);
    }

    // D2 ( grad(c2) + z2 c2 grad(phi), grad(v2) )
    void builda2(ParGridFunction& phi) const
    {
        if (a2 != NULL) { delete a2; }

        a2 = new ParBilinearForm(h1);
        // D2 (grad(c2), grad(v2))
        a2->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi), grad(v2))
        a2->AddDomainIntegrator(new GradConvectionIntegrator(phi, &D_Cl_prod_v_Cl));

        a2->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(h1);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
        b1->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(h1);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
        b2->Finalize(skip_zero_entries);
    }

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
//        int gdb_break=1;
//        while (gdb_break && abs(t - 0.39372532809215) < 1E-10) {}
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(h1, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(h1, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(h1, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt;
        dc1dt.MakeTRef(h1, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(h1, dphic1c2_dt, true_offset[2]);
        dc1dt.SetFromTrueVector();
        
        ParGridFunction temp_phi(h1); // 保留初始的phi, 后面要用(old_phi在Poisson solver中会被改变)
        temp_phi = old_phi;

        // 求解 Poisson
        ParLinearForm *l0 = new ParLinearForm(h1);
        // (f0, psi)
        f0_analytic.SetTime(t);
        l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
        l0->Assemble();

        buildb1();
        b1->AddMult(old_c1, *l0, 1.0); // l0 = l0 + b1 c1
        buildb2();
        b2->AddMult(old_c2, *l0, 1.0); // l0 = l0 + b1 c1 + b2 c2

        builda0();
        phi_exact.SetTime(t);
        old_phi.ProjectBdrCoefficient(phi_exact, ess_bdr);
        a0->FormLinearSystem(ess_tdof_list, old_phi, *l0, *A0, *temp_x0, *temp_b0);

        PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi_");
        poisson_solver->Mult(*temp_b0, *temp_x0);
        a0->RecoverFEMSolution(*temp_x0, *l0, old_phi); // 更新old_phi
        delete l0;
        delete poisson_solver;

        // 求解 NP1
        ParLinearForm *l1 = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        cout.precision(14);

        builda1(temp_phi);
        a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1

        buildm();
        dc1dt_exact.SetTime(t);
        dc1dt.ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
        m->FormLinearSystem(ess_tdof_list, dc1dt, *l1, *M, *temp_x1, *temp_b1);

        PetscLinearSolver* np1_solver = new PetscLinearSolver(*M, false, "np1_");
        np1_solver->Mult(*temp_b1, *temp_x1);
        m->RecoverFEMSolution(*temp_x1, *l1, dc1dt); // 更新 dc1dt
        delete l1;
        delete np1_solver;

        // 求解 NP2
        ParLinearForm *l2 = new ParLinearForm(h1);
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();

        builda2(temp_phi);
        a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2

        buildm();
        dc2dt_exact.SetTime(t);
        dc2dt.ProjectBdrCoefficient(dc2dt_exact, ess_bdr);
        m->FormLinearSystem(ess_tdof_list, dc2dt, *l2, *M, *temp_x2, *temp_b2);

        PetscLinearSolver* np2_solver = new PetscLinearSolver(*M, false, "np2_");
        np2_solver->Mult(*temp_b2, *temp_x2);
        m->RecoverFEMSolution(*temp_x2, *l2, dc2dt); // 更新 dc2dt
        delete l2;
        delete np2_solver;

        // 上面我们求解了3个未知量的PrimalVector: old_phi, dc1dt, dc2dt. 但返回值必须是TrueVector
        old_phi.SetTrueVector();
        dc1dt  .SetTrueVector();
        dc2dt  .SetTrueVector();
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(h1, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(h1, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(h1, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt;
        dc1dt.MakeTRef(h1, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(h1, dphic1c2_dt, true_offset[2]);

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi_Gummel(h1), dc1dt_Gummel(h1), dc2dt_Gummel(h1);
        phi_Gummel = 0.0; dc1dt_Gummel = 0.0; dc2dt_Gummel = 0.0;
        phi_exact.SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);

        ParGridFunction diff(h1);
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // 求解 Poisson
            ParLinearForm *l0 = new ParLinearForm(h1);
            // (f0, psi)
            f0_analytic.SetTime(t);
            l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
            l0->Assemble();

            buildb1();
            buildb2();
            b1->AddMult(old_c1, *l0, 1.0);    // l0 = l0 + b1 c1
            b2->AddMult(old_c2, *l0, 1.0);    // l0 = l0 + b1 c1 + b2 c2
            b1->AddMult(dc1dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
            b2->AddMult(dc2dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt

            builda0();
            phi_Gummel.ProjectBdrCoefficient(phi_exact, ess_bdr);
            a0->FormLinearSystem(ess_tdof_list, phi_Gummel, *l0, *A0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel); // 更新old_phi
            delete l0;
            delete poisson_solver;

            diff = 0.0;
            diff += phi_Gummel;
            diff -= old_phi; // 用到的是old_phi的PrimalVector
            double tol = diff.ComputeL2Error(zero) / phi_Gummel.ComputeL2Error(zero);
            old_phi = phi_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
            if (myid == 0 && verbose >= 2) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
            }
            if (tol < Gummel_rel_tol) { // Gummel迭代停止
                last_gummel_step = true;
            }

            // 求解 NP1
            ParLinearForm *l1 = new ParLinearForm(h1);
            // (f1, v1)
            f1_analytic.SetTime(t);
            l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            l1->Assemble();

            builda1(phi_Gummel);
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1

            buildm1_dta1(dt, phi_Gummel);
            dc1dt_Gummel.ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
            m1_dta1->FormLinearSystem(ess_tdof_list, dc1dt_Gummel, *l1, *M1_dtA1, *temp_x1, *temp_b1);

            PetscLinearSolver* np1_solver = new PetscLinearSolver(*M1_dtA1, false, "np1_");
            np1_solver->Mult(*temp_b1, *temp_x1);
            m1_dta1->RecoverFEMSolution(*temp_x1, *l1, dc1dt_Gummel); // 更新 dc1dt
            delete l1;
            delete np1_solver;

            // 求解 NP2
            ParLinearForm *l2 = new ParLinearForm(h1);
            // (f2, v2)
            f2_analytic.SetTime(t);
            l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
            l2->Assemble();

            builda2(phi_Gummel);
            a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2

            buildm2_dta2(dt, phi_Gummel);
            dc2dt_Gummel.ProjectBdrCoefficient(dc2dt_exact, ess_bdr);
            m2_dta2->FormLinearSystem(ess_tdof_list, dc2dt_Gummel, *l2, *M2_dtA2, *temp_x2, *temp_b2);

            PetscLinearSolver* np2_solver = new PetscLinearSolver(*M2_dtA2, false, "np2_");
            np2_solver->Mult(*temp_b2, *temp_x2);
            m2_dta2->RecoverFEMSolution(*temp_x2, *l2, dc2dt_Gummel); // 更新 dc2dt
            delete l2;
            delete np2_solver;
        }

        // 用最终Gummel迭代的解更新要求解的3个未知量
        old_phi = phi_Gummel;
        dc1dt = dc1dt_Gummel;
        dc2dt = dc2dt_Gummel;
        // 而我们要返回的TrueVector, 而不是PrimalVector
        old_phi.SetTrueVector();
        dc1dt  .SetTrueVector();
        dc2dt  .SetTrueVector();
    }
};
class PNP_Box_Gummel_CG_TimeDependent_Solver
{
private:
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    PNP_Box_Gummel_CG_TimeDependent* oper;
    double t; // 当前时间
    Vector init_value;
    ODESolver *ode_solver;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_Solver(ParMesh* pmesh_, int ode_solver_type): pmesh(pmesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        t = t_init;

        fec   = new H1_FECollection(p_order, pmesh->Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的

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

        oper = new PNP_Box_Gummel_CG_TimeDependent(true_size, true_offset, ess_bdr, h1, t);

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
        delete fec;
        delete h1;
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
        double mesh_size=0.0;
        {
            double max_size=0;
            for (int i=0; i<pmesh->GetNE(); ++i)
            {
                double size = pmesh->GetElementSize(i);
                if (size > max_size) max_size = size;
            }

            MPI_Allreduce(&max_size, &mesh_size, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }
        // 时间离散误差加上空间离散误差: error = c1 dt + c2 h^2
        // 如果收敛, 向前向后Euler格式都是1阶, 下面算空间L^2误差范数
        if(ComputeConvergenceRate) {
            t_stepsize = mesh_size * mesh_size;
        }
        if (myid == 0) {
            cout << "\n======> ";
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times << ", mesh size: " << mesh_size << '\n'
                 << options_src << ", DOFs: " << h1->GlobalTrueVSize() * 3<< ", Cores: " << num_procs << '\n'
                 << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" \
                                                                       : "wrong type")) << ", " << "time step: " << t_stepsize
                 << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();

        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(t_stepsize, t_final - t);

            ode_solver->Step(*phic1c2, t, dt_real); // 进过这一步之后phic1c2和t都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            phi_gf->SetFromTrueVector();
            c1_gf->SetFromTrueVector();
            c2_gf->SetFromTrueVector();

            if (paraview)
            {
                pd->SetCycle(ti); // 第 i 个时间步
                pd->SetTime(t); // 第i个时间步所表示的时间
                pd->Save();
            }
            if (verbose)
            {
                phi_exact.SetTime(t);
                c1_exact.SetTime(t);
                c2_exact.SetTime(t);

                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm = c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm = c2_gf->ComputeL2Error(c2_exact);
                if (myid == 0) {
                    cout << "\nTime: " << t << '\n'
                         << "phi L2 errornorm: " << phiL2errornorm << '\n'
                         << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                         << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
                }
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
                     << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << '\n'
                     << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << '\n'
                     << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;

                if (ComputeConvergenceRate)
                {
                    meshsizes_.Append(mesh_size);
                    phiL2errornorms_.Append(phiL2err);
                    c1L2errornorms_.Append(c1L2err);
                    c2L2errornorms_.Append(c2L2err);
                }
            }
        }
    }
};



#endif
