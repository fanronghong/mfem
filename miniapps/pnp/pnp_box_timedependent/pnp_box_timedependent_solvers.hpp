/* Poisson Equation:
 *     div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
 * NP Equation:
 *     dc_i / dt = div( D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * */
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
#include "../utils/DGDiffusion_Edge_Symmetry_Penalty.hpp"
#include "../utils/DGEdgeIntegrator.hpp"
#include "../utils/PNP_Preconditioners.hpp"
#include "../utils/PNP_BCHandler.hpp"


class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;

    mutable ParBilinearForm *a0, *a1, *a2, *b1, *b2, *m, *m1_dta1, *m2_dta2;
    mutable HypreParMatrix *A0, *M, *M1_dtA1, *M2_dtA2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;
    int true_vsize;
    mutable Array<int> true_offset, ess_bdr, ess_tdof_list;
    int num_procs, rank;

public:
    PNP_Box_Gummel_CG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fsp, double time)
        : TimeDependentOperator(3*truesize, time), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_), fes(fsp),
                                   b1(NULL), b2(NULL), a0(NULL), a1(NULL), m(NULL), a2(NULL), m1_dta1(NULL), m2_dta2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        A0      = new HypreParMatrix;
        M       = new HypreParMatrix;
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

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
//        int gdb_break=1;
//        while (gdb_break && abs(t - 0.39372532809215) < 1E-10) {}
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt;
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);
        dc1dt.SetFromTrueVector();

        ParGridFunction temp_phi(fes); // 保留初始的phi, 后面要用(old_phi在Poisson solver中会被改变)
        temp_phi = old_phi;

        // 求解 Poisson
        ParLinearForm *l0 = new ParLinearForm(fes);
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
        ParLinearForm *l1 = new ParLinearForm(fes);
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
        ParLinearForm *l2 = new ParLinearForm(fes);
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
        // 求解新的 old_phi 从而更新 phic1c2_ptr, 最终更新 phic1c2
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi_dt
        // 下面通过求解 dc1dt, dc2dt 从而更新 dphic1c2_dt
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi_Gummel(fes), dc1dt_Gummel(fes), dc2dt_Gummel(fes);
        phi_Gummel   = 0.0; // 这里暂不设定边界条件, 后面在计算的时候直接设定essential边界条件
        dc1dt_Gummel = 0.0;
        dc2dt_Gummel = 0.0;
        phi_exact  .SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);

        ParGridFunction diff(fes);
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // **************************************************************************************
            //                                1. 求解 Poisson
            // **************************************************************************************
            ParLinearForm *l0 = new ParLinearForm(fes);
            // b0: (f0, psi)
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
            phi_Gummel.ProjectBdrCoefficient(phi_exact, ess_bdr); // 设定解的边界条件
            a0->FormLinearSystem(ess_tdof_list, phi_Gummel, *l0, *A0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel);
            delete l0;
            delete poisson_solver;

            if (visualization)
            {
                VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
                dc->RegisterField("phi_Gummel", &phi_Gummel);

                Visualize(*dc, "phi_Gummel", "phi_Gummel_DG");

                delete dc;
            }


            // **************************************************************************************
            //                                2. 计算Gummel迭代相对误差
            // **************************************************************************************
            diff = 0.0;
            diff += phi_Gummel;
            diff -= old_phi; // 用到的是old_phi的PrimalVector
            double tol = diff.ComputeL2Error(zero) / phi_Gummel.ComputeL2Error(zero); // 这里不能把diff设为Vector类型, 如果是Vector类型, 这里计算Norml2()时各个进程得到的值不一样
            old_phi = phi_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
            if (rank == 0 && verbose >= 2) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
            }
            if (tol < Gummel_rel_tol) { // Gummel迭代停止
                last_gummel_step = true;
            }


            // **************************************************************************************
            //                                3. 求解 NP1
            // **************************************************************************************
            ParLinearForm *l1 = new ParLinearForm(fes);
            // b1: (f1, v1)
            f1_analytic.SetTime(t);
            l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            l1->Assemble();

            builda1(phi_Gummel);
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1

            buildm1_dta1(dt, phi_Gummel);
            dc1dt_Gummel.ProjectBdrCoefficient(dc1dt_exact, ess_bdr); // 设定未知量的边界条件
            m1_dta1->FormLinearSystem(ess_tdof_list, dc1dt_Gummel, *l1, *M1_dtA1, *temp_x1, *temp_b1);

            PetscLinearSolver* np1_solver = new PetscLinearSolver(*M1_dtA1, false, "np1_");
            np1_solver->Mult(*temp_b1, *temp_x1);
            m1_dta1->RecoverFEMSolution(*temp_x1, *l1, dc1dt_Gummel); // 更新 dc1dt
            delete l1;
            delete np1_solver;


            // **************************************************************************************
            //                                4. 求解 NP2
            // **************************************************************************************
            ParLinearForm *l2 = new ParLinearForm(fes);
            // b2: (f2, v2)
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
        dc1dt   = dc1dt_Gummel;
        dc2dt   = dc2dt_Gummel;
        // 而我们要返回的TrueVector, 而不是PrimalVector
        old_phi.SetTrueVector();
        dc1dt  .SetTrueVector();
        dc2dt  .SetTrueVector();
    }

private:
    // (ci, vi), i=1,2
    void buildm() const
    {
        if (m != NULL) { delete m; }
        
        m = new ParBilinearForm(fes);
        m->AddDomainIntegrator(new MassIntegrator);
        
        m->Assemble(skip_zero_entries);
    }
    
    // (c1, v1) + dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given dt and phi
    void buildm1_dta1(double dt, ParGridFunction& phi) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);

        m1_dta1 = new ParBilinearForm(fes);
        // (c1, v1)
        m1_dta1->AddDomainIntegrator(new MassIntegrator);
        // dt D1 (grad(c1), grad(v1))
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        // dt D1 z1 (c1 grad(phi), grad(v1))
        m1_dta1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // (c2, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given dt and phi
    void buildm2_dta2(double dt, ParGridFunction& phi) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);

        m2_dta2 = new ParBilinearForm(fes);
        // (c2, v2)
        m2_dta2->AddDomainIntegrator(new MassIntegrator);
        // dt D2 (grad(c2), grad(v2))
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        // dt D2 z2 (c2 grad(phi), grad(v2))
        m2_dta2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D2_z2));

        m2_dta2->Assemble(skip_zero_entries);
    }

    // epsilon_s (grad(phi), grad(psi))
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(fes);
        a0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        a0->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
    void builda1(ParGridFunction& phi) const
    {
        if (a1 != NULL) { delete a1; }

        a1 = new ParBilinearForm(fes);
        // D1 (grad(c1), grad(v1))
        a1->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi), grad(v1))
        a1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &D_K_prod_v_K));

        a1->Assemble(skip_zero_entries);
    }

    // D2 ( grad(c2) + z2 c2 grad(phi), grad(v2) )
    void builda2(ParGridFunction& phi) const
    {
        if (a2 != NULL) { delete a2; }

        a2 = new ParBilinearForm(fes);
        // D2 (grad(c2), grad(v2))
        a2->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi), grad(v2))
        a2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &D_Cl_prod_v_Cl));

        a2->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(fes);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
        b1->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(fes);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
        b2->Finalize(skip_zero_entries);
    }
};


class PNP_Box_Gummel_DG_Operator: public Operator
{
private:
    ParFiniteElementSpace* fes;
//    // 用来给DG空间的GridFunction设定边界条件: 如果gf属于DG空间的GridFunction, 则gf.ProjectBdrCoefficient()会出错
//    H1_FECollection* h1_fec;
//    ParFiniteElementSpace* h1;

    ParGridFunction *c1, *c2;
    double t, dt;
    int true_vsize;
    mutable Array<int> true_offset, ess_bdr, null_array; // 在H1空间中存在ess_tdof_list, 在DG空间中不存在

    mutable ParBilinearForm *a0_e0_s0_p0, *m1_dta1_dte1_dts1_dtp1, *m2_dta2_dte2_dts2_dtp2,
            *a1, *e1, *s1, *p1,
            *a2, *e2, *s2, *p2,
            *b1, *b2;
    mutable HypreParMatrix *A0_E0_S0_P0,
            *M1_dtA1_dtE1_dtS1_dtP1,
            *M2_dtA2_dtE2_dtS2_dtP2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;

    int num_procs, rank;

public:
    PNP_Box_Gummel_DG_Operator(ParFiniteElementSpace* fes_, int truevsize, Array<int>& offset, Array<int>& ess_bdr_)
            : Operator(3 * truevsize), fes(fes_), true_vsize(truevsize), true_offset(offset), ess_bdr(ess_bdr_),
              a0_e0_s0_p0(NULL), b1(NULL), b2(NULL),
              a1(NULL), e1(NULL), s1(NULL), p1(NULL), m1_dta1_dte1_dts1_dtp1(NULL),
              a2(NULL), e2(NULL), s2(NULL), p2(NULL), m2_dta2_dte2_dts2_dtp2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // 在H1空间中存在, 在DG空间中不存在
//        h1_fec = new H1_FECollection(p_order, fes->GetParMesh()->Dimension());
//        h1 = new ParFiniteElementSpace(fes->GetParMesh(), h1_fec);

        A0_E0_S0_P0            = new HypreParMatrix;
        M1_dtA1_dtE1_dtS1_dtP1 = new HypreParMatrix;
        M2_dtA2_dtE2_dtS2_dtP2 = new HypreParMatrix;

        temp_x0 = new Vector;
        temp_b0 = new Vector;
        temp_x1 = new Vector;
        temp_b1 = new Vector;
        temp_x2 = new Vector;
        temp_b2 = new Vector;
    }
    ~PNP_Box_Gummel_DG_Operator()
    {
        delete b1; delete b2; delete a1; delete a2;
        delete e1; delete e2;
        delete s1; delete s2;
        delete p1; delete p2;
        delete a0_e0_s0_p0; delete m1_dta1_dte1_dts1_dtp1; delete m2_dta2_dte2_dts2_dtp2;

        delete A0_E0_S0_P0;
        delete M1_dtA1_dtE1_dtS1_dtP1;
        delete M2_dtA2_dtE2_dtS2_dtP2;

        delete temp_x0; delete temp_b0;
        delete temp_x1; delete temp_b1;
        delete temp_x2; delete temp_b2;

    }

    void UpdateParameters(double t_, double dt_, ParGridFunction* c1_, ParGridFunction* c2_)
    {
        t = t_;
        dt = dt_;
        c1 = c1_;
        c2 = c2_;
    }

    virtual void Mult(const Vector& b, Vector& phi_dc1dt_dc2dt) const
    {
        ParGridFunction phi, dc1dt, dc2dt;
        phi  .MakeTRef(fes, phi_dc1dt_dc2dt, true_offset[0]);
        dc1dt.MakeTRef(fes, phi_dc1dt_dc2dt, true_offset[1]);
        dc2dt.MakeTRef(fes, phi_dc1dt_dc2dt, true_offset[2]);
        phi  .SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt.SetFromTrueVector();
        dc2dt.SetFromTrueVector();

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi_Gummel(fes), dc1dt_Gummel(fes), dc2dt_Gummel(fes);
        phi_Gummel   = 0.0; // 这里需要设定边界条件吗fff
        dc1dt_Gummel = 0.0;
        dc2dt_Gummel = 0.0;

        phi_exact.SetTime(t);
        c1_exact.SetTime(t);
        c2_exact.SetTime(t);
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);
        f0_analytic.SetTime(t);
        f1_analytic.SetTime(t);
        f2_analytic.SetTime(t);

        ParGridFunction diff(fes);
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // **************************************************************************************
            //                                1. 求解 Poisson
            // **************************************************************************************
            c1->ProjectCoefficient(c1_exact); // fff
            c2->ProjectCoefficient(c2_exact); // fff
            ParLinearForm *l0 = new ParLinearForm(fes);
            // b0: (f0, psi)
            l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
            if (! (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) ) // weak Dirichlet boundary condition. 如果对称项和惩罚项都没有添加边界积分项, 就必须额外添加weak Dirichlet边界条件
            {
                l0->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(phi_exact, epsilon_water), ess_bdr);
            }
            if (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) // 添加对称项
            {
                // g0: sigma <phi_D, epsilon_s grad(psi).n>
                l0->AddBdrFaceIntegrator(new DGDirichletLF_Symmetry(phi_exact, epsilon_water, sigma), ess_bdr);
            }
            if (abs(kappa - 0.0) > 1E-10 && penalty_with_boundary) // 添加惩罚项
            {
                // q0: kappa <{h^{-1} epsilon_s} phi_D, psi>
                l0->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(phi_exact, epsilon_water, kappa), ess_bdr);
            }
            l0->Assemble();
//
            buildb1();
            buildb2();
            b1->AddMult(*c1, *l0, 1.0);    // l0 = l0 + b1 c1
            b2->AddMult(*c2, *l0, 1.0);    // l0 = l0 + b1 c1 + b2 c2
            b1->AddMult(dc1dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
            b2->AddMult(dc2dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt

            builda0_e0_s0_p0();
            a0_e0_s0_p0->FormLinearSystem(null_array, phi_Gummel, *l0, *A0_E0_S0_P0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0_E0_S0_P0, false, "phi_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0_e0_s0_p0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel); // fff不能用迭代法amg求解, 只能用直接法. 经检测: 应该是edge和symmetry导致矩阵难解
            delete l0;
            delete poisson_solver;


            // **************************************************************************************
            //                                2. 计算Gummel迭代相对误差
            // **************************************************************************************
            diff  = 0.0;
            diff += phi_Gummel;
            diff -= phi; // 用到的是old_phi的PrimalVector
            double tol = diff.ComputeL2Error(zero) / phi_Gummel.ComputeL2Error(zero); // 这里不能把diff设为Vector类型, 如果是Vector类型, 这里计算Norml2()时各个进程得到的值不一样
            phi = phi_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
            if (rank == 0 && verbose >= 2) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
            }
            if (tol < Gummel_rel_tol || gummel_step >= Gummel_max_iters) { // Gummel迭代停止
                last_gummel_step = true;
                if (gummel_step >= Gummel_max_iters && rank == 0)
                MFEM_ABORT("Gummel iteration not converge!!!");
            }

            phi_Gummel.ExchangeFaceNbrData(); // 后面有可能利用其在内部边界积分, 而相邻单元有可能不再同一个进程


            // **************************************************************************************
            //                                3. 求解 NP1
            // **************************************************************************************
            ParLinearForm *l1 = new ParLinearForm(fes);
            // b1: (f1, v1)
            l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            // weak Dirichlet boundary condition
            l1->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(dc1dt_exact, D_K_), ess_bdr);
            if (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) // 添加对称项
            {
                // -g1: sigma <c1_D, D1(grad(v1) + z1 v1 grad(phi)).n> = sigma <c1_D, D1 grad(v1).n> + sigma <c1_D, D1 z1 v1 grad(phi).n>
                l1->AddBdrFaceIntegrator(new DGDirichletLF_Symmetry(c1_exact, D_K_, sigma), ess_bdr);
                l1->AddBdrFaceIntegrator(new DGEdgeLFIntegrator2(&sigma_D_K_v_K, &c1_exact, &phi_Gummel), ess_bdr); // fff no test
            }
            if (abs(kappa - 0.0) > 1E-10 && penalty_with_boundary) // 添加惩罚项
            {
                // -q1: kappa <{h^{-1}} c1_D, v1>
                l1->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c1_exact, kappa * D_K), ess_bdr); // fff
            }
            l1->Assemble();

            builda1(phi_Gummel);
            builde1(phi_Gummel);
            a1->AddMult(*c1, *l1, -1.0); // l1 = l1 - a1 c1
//            e1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1 - e1 c1. error(下同): https://github.com/mfem/mfem/issues/1830
            {
                auto *e1_tdof = e1->ParallelAssemble();
                auto *l1_tdof = l1->ParallelAssemble();
                auto *Restriction = e1->GetRestriction(); // ref: https://mfem.org/pri-dual-vec/

                e1_tdof->Mult(-1.0, c1->GetTrueVector(), 1.0, *l1_tdof);
                Restriction->MultTranspose(*l1_tdof, *l1);

                delete e1_tdof;
                delete l1_tdof;
            }

            if (abs(sigma - 0.0) > 1E-10) // 添加对称项
            {
                builds1(phi_Gummel);
//                s1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1
                auto* s1_tdof = s1->ParallelAssemble();
                auto* l1_tdof = l1->ParallelAssemble();
                auto* Restriction = s1->GetRestriction();

                s1_tdof->Mult(1.0, c1->GetTrueVector(), 1.0, *l1_tdof);
                Restriction->MultTranspose(*l1_tdof, *l1);

                delete s1_tdof;
                delete l1_tdof;
            }
            if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
            {
                buildp1();
//                p1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1 + p1 c1
                auto* p1_tdof = p1->ParallelAssemble();
                auto* l1_tdof = l1->ParallelAssemble();
                auto* Restriction = p1->GetRestriction();

                p1_tdof->Mult(1.0, c1->GetTrueVector(), 1.0, *l1_tdof);
                Restriction->MultTranspose(*l1_tdof, *l1);

                delete p1_tdof;
                delete l1_tdof;
            }

            buildm1_dta1_dte1_dts1_dtp1(phi_Gummel);
            m1_dta1_dte1_dts1_dtp1->FormLinearSystem(null_array, dc1dt_Gummel, *l1, *M1_dtA1_dtE1_dtS1_dtP1, *temp_x1, *temp_b1);

            PetscLinearSolver* np1_solver = new PetscLinearSolver(*M1_dtA1_dtE1_dtS1_dtP1, false, "np1_");
            np1_solver->Mult(*temp_b1, *temp_x1);
            m1_dta1_dte1_dts1_dtp1->RecoverFEMSolution(*temp_x1, *l1, dc1dt_Gummel); // 更新 dc1dt
            delete l1;
            delete np1_solver;


            // **************************************************************************************
            //                                4. 求解 NP2
            // **************************************************************************************
            ParLinearForm *l2 = new ParLinearForm(fes);
            // b2: (f2, v2)
            l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
            // weak Dirichlet boundary condition
            l2->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(dc2dt_exact, D_Cl_), ess_bdr);
            if (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) // 添加对称项
            {
                // -g2: sigma <c2_D, D2(grad(v2) + z2 v2 grad(phi)).n>
                l2->AddBdrFaceIntegrator(new DGDirichletLF_Symmetry(c2_exact, D_Cl_, sigma), ess_bdr);
                l2->AddBdrFaceIntegrator(new DGEdgeLFIntegrator2(&sigma_D_Cl_v_Cl, &c2_exact, &phi_Gummel), ess_bdr);
            }
            if (abs(kappa - 0.0) > 1E-10 && penalty_with_boundary) // 添加惩罚项
            {
                // -q2: kappa <{h^{-1}} c2_D, v2>
                l2->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c2_exact, kappa*D_Cl), ess_bdr);
            }
            l2->Assemble();

            builda2(phi_Gummel);
            builde2(phi_Gummel);
            a2->AddMult(*c2, *l2, -1.0); // l2 = l2 - a2 c2
//            e2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2 - e2 c2
            {
                auto *e2_tdof = e2->ParallelAssemble();
                auto *l2_tdof = l2->ParallelAssemble();
                auto *Restriction = e2->GetRestriction(); // ref: https://mfem.org/pri-dual-vec/

                e2_tdof->Mult(-1.0, c2->GetTrueVector(), 1.0, *l2_tdof);
                Restriction->MultTranspose(*l2_tdof, *l2);

                delete e2_tdof;
                delete l2_tdof;
            }

            if (abs(sigma - 0.0) > 1E-10) // 添加对称项
            {
                builds2(phi_Gummel);
//                s2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2
                auto* s2_tdof = s2->ParallelAssemble();
                auto* l2_tdof = l2->ParallelAssemble();
                auto* Restriction = s2->GetRestriction();

                s2_tdof->Mult(1.0, c2->GetTrueVector(), 1.0, *l2_tdof);
                Restriction->MultTranspose(*l2_tdof, *l2);

                delete s2_tdof;
                delete l2_tdof;
            }
            if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
            {
                buildp2();
//                p2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2 + p2 c2
                auto* p2_tdof = p2->ParallelAssemble();
                auto* l2_tdof = l2->ParallelAssemble();
                auto* Restriction = p2->GetRestriction();

                p2_tdof->Mult(1.0, c2->GetTrueVector(), 1.0, *l2_tdof);
                Restriction->MultTranspose(*l2_tdof, *l2);

                delete p2_tdof;
                delete l2_tdof;
            }

            buildm2_dta2_dte2_dts2_dtp2(phi_Gummel);
            m2_dta2_dte2_dts2_dtp2->FormLinearSystem(null_array, dc2dt_Gummel, *l2, *M2_dtA2_dtE2_dtS2_dtP2, *temp_x2, *temp_b2);

            PetscLinearSolver* np2_solver = new PetscLinearSolver(*M2_dtA2_dtE2_dtS2_dtP2, false, "np2_");
            np2_solver->Mult(*temp_b2, *temp_x2);
            m2_dta2_dte2_dts2_dtp2->RecoverFEMSolution(*temp_x2, *l2, dc2dt_Gummel); // 更新 dc2dt
            delete l2;
            delete np2_solver;
        }

        // 用最终Gummel迭代的解更新要求解的3个未知量
        phi   = phi_Gummel; // 这3步可以放到Gummel迭代里面去
        dc1dt = dc1dt_Gummel;
        dc2dt = dc2dt_Gummel;
        // 而我们要返回的TrueVector, 而不是PrimalVector
        phi  .SetTrueVector();
        dc1dt.SetTrueVector();
        dc2dt.SetTrueVector();
    }

private:
    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(fes);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(fes);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
    void builda1(ParGridFunction& phi) const
    {
        if (a1 != NULL) { delete a1; }

        phi.ExchangeFaceNbrData();

        a1 = new ParBilinearForm(fes);
        // D1 (grad(c1), grad(v1))
        a1->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi), grad(v1))
        a1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &D_K_prod_v_K));

        a1->Assemble(skip_zero_entries);
    }

    // D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
    void builda2(ParGridFunction& phi) const
    {
        if (a2 != NULL) { delete a2; }

        a2 = new ParBilinearForm(fes);
        // D2 (grad(c2), grad(v2))
        a2->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi), grad(v2))
        phi.ExchangeFaceNbrData();
        a2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &D_Cl_prod_v_Cl));

        a2->Assemble(skip_zero_entries);
    }

    // -<{D1 (grad(c1) + z1 c1 grad(phi))}, [v1]>, given phi
    void builde1(ParGridFunction& phi) const
    {
        if (e1 != NULL) { delete e1; }

        e1 = new ParBilinearForm(fes);
        // -<{D1 grad(c1)}, [v1]>
        e1->AddInteriorFaceIntegrator(new DGDiffusion_Edge(D_K_));
        e1->AddBdrFaceIntegrator(new DGDiffusion_Edge(D_K_), ess_bdr);

        // -<{D1 z1 c1 grad(phi)}, [v1]>
        e1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_K_v_K, phi));
        e1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_K_v_K, phi), ess_bdr);

        e1->Assemble(skip_zero_entries);
        e1->Finalize(skip_zero_entries);
    }

    // -<{D2 (grad(c2) + z2 c2 grad(phi))}, [v2]>, given phi
    void builde2(ParGridFunction& phi) const
    {
        if (e2 != NULL) { delete e2; }

        e2 = new ParBilinearForm(fes);
        // -<{D2 grad(c2)}, [v2]>
        e2->AddInteriorFaceIntegrator(new DGDiffusion_Edge(D_Cl_));
        e2->AddBdrFaceIntegrator(new DGDiffusion_Edge(D_Cl_), ess_bdr);

        // -<{D2 z2 c2 grad(phi)}, [v2]>
        phi.ExchangeFaceNbrData();
        e2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_Cl_v_Cl, phi));
        e2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_Cl_v_Cl, phi), ess_bdr);

        e2->Assemble(skip_zero_entries);
        e2->Finalize(skip_zero_entries);
    }

    // - sigma <[c1], {D1 (grad(v1) + z1 v1 grad(phi))}>, given phi
    void builds1(ParGridFunction& phi) const
    {
        if (s1 != NULL) { delete s1; }

        s1 = new ParBilinearForm(fes);
        // -sigma <[c1], {D1 grad(v1)}>
        s1->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(neg_D1, sigma));
        if (symmetry_with_boundary)
        {
            s1->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(neg_D1, sigma), ess_bdr);
        }

        // -sigma <[c1], {D1 z1 v1 grad(phi)}>
        phi.ExchangeFaceNbrData();
        s1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_K_v_K, phi));
        if (symmetry_with_boundary)
        {
            s1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_K_v_K, phi), ess_bdr);
        }

        s1->Assemble(skip_zero_entries);
        s1->Finalize(skip_zero_entries);
    }

    // -sigma <[c2], {D2 (grad(v2) + z2 v2 grad(phi))}>, given phi
    void builds2(ParGridFunction& phi) const
    {
        if (s2 != NULL) { delete s2; }

        s2 = new ParBilinearForm(fes);
        // -sigma <[c2], {D2 grad(v2)}>
        s2->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(neg_D2, sigma));
        if (symmetry_with_boundary)
        {
            s2->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(neg_D2, sigma), ess_bdr);
        }

        // -sigma <[c2], {D2 z2 v2 grad(phi)}>
        phi.ExchangeFaceNbrData();
        s2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_Cl_v_Cl, phi));
        if (symmetry_with_boundary)
        {
            s2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_Cl_v_Cl, phi), ess_bdr);
        }

        s2->Assemble(skip_zero_entries);
        s2->Finalize(skip_zero_entries);
    }

    // -kappa <{h^{-1}} [c1], [v1]>
    void buildp1() const
    {
        if (p1 != NULL) { delete p1; }

        p1 = new ParBilinearForm(fes);
        // -kappa <{h^{-1}} [c1], [v1]> 对单元内部边界和区域外部边界积分
        p1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa* D_K));
        if (penalty_with_boundary)
        {
            p1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa* D_K), ess_bdr);
        }

        p1->Assemble(skip_zero_entries);
        p1->Finalize(skip_zero_entries);
    }

    // -kappa <{h^{-1}} [c2], [v2]>
    void buildp2() const
    {
        if (p2 != NULL) { delete p2; }

        p2 = new ParBilinearForm(fes);
        // -kappa <{h^{-1}} [c2], [v2]> 对单元内部边界和区域外部边界积分
        p2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa*D_Cl));
        if (penalty_with_boundary)
        {
            p2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa*D_Cl), ess_bdr);
        }

        p2->Assemble(skip_zero_entries);
        p2->Finalize(skip_zero_entries);
    }

    void builda0_e0_s0_p0() const
    {
        if (a0_e0_s0_p0 != NULL) { delete a0_e0_s0_p0; }

        a0_e0_s0_p0 = new ParBilinearForm(fes);

        // (epsilon_s grad(phi), grad(psi))
        a0_e0_s0_p0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        // -<{epsilon_s grad(phi)}, [psi]>
        a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Edge(epsilon_water));
        a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Edge(epsilon_water), ess_bdr);

        // weak Dirichlet boundary condition. 如果对称项和惩罚项都没有添加边界积分项, 就必须额外添加weak Dirichlet边界条件
        if (! (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) )
        {
            a0_e0_s0_p0->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(epsilon_water), ess_bdr);
        }

        // sigma <[phi], {epsilon_s grad(psi)}>
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma));
            if (symmetry_with_boundary)
            {
                a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma), ess_bdr);
            }
        }

        // kappa <{h^{-1} epsilon_s} [phi], [psi]>
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(epsilon_water, kappa)); // fff
            if (penalty_with_boundary)
            {
                a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Penalty(epsilon_water, kappa), ess_bdr); // fff
            }
        }

        a0_e0_s0_p0->Assemble(skip_zero_entries);
    }

    void buildm1_dta1_dte1_dts1_dtp1(ParGridFunction& phi) const
    {
        if (m1_dta1_dte1_dts1_dtp1 != NULL) { delete m1_dta1_dte1_dts1_dtp1; }

        phi.ExchangeFaceNbrData();

        m1_dta1_dte1_dts1_dtp1 = new ParBilinearForm(fes);

        // (dc1_dt, v1)
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new MassIntegrator);

        // dt D1 (grad(dc1_dt) + z1 dc1_dt grad(phi), grad(v1))
        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D1_z1));

        // - dt <{D1 grad(dc1_dt) + D1 z1 dc1_dt grad(phi)}, [v1]>
        ProductCoefficient dt_neg_D_K_v_K(dt, neg_D_K_v_K);
        m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Edge(dt_D1));
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Edge(dt_D1), ess_bdr);
        m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_K_v_K, phi));
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_K_v_K, phi), ess_bdr);

        // weak Dirichlet boundary condition fff
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(D_K_), ess_bdr);

        // dt sigma <[dc1_dt], {D1 (grad(v1) + z1 v1 grad(phi))}>
        ProductCoefficient dt_sigma_D1_z1(dt, sigma_D1_z1);
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_K_, dt * sigma));
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D1_z1, phi));

            if (symmetry_with_boundary)
            {
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_K_, dt * sigma), ess_bdr);
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D1_z1, phi), ess_bdr);
            }
        }

        // dt kappa <{h^{-1}} [dc1_dt], [v1]>
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(dt * kappa*D_K));

            if (penalty_with_boundary)
            {
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(dt * kappa*D_K), ess_bdr);
            }
        }

        m1_dta1_dte1_dts1_dtp1->Assemble(skip_zero_entries);
    }

    void buildm2_dta2_dte2_dts2_dtp2(ParGridFunction& phi) const
    {
        if (m2_dta2_dte2_dts2_dtp2 != NULL) { delete m2_dta2_dte2_dts2_dtp2; }
        phi.ExchangeFaceNbrData();

        m2_dta2_dte2_dts2_dtp2 = new ParBilinearForm(fes);

        // (dc2_dt, v2)
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new MassIntegrator);

        // dt D2 (grad(dc2_dt) + z2 dc2_dt grad(phi), grad(v2))
        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D2_z2));

        // - dt <{D2 grad(dc2_dt) + D2 z2 dc2_dt grad(phi)}, [v2]>
        ProductCoefficient dt_neg_D_Cl_v_Cl(dt, neg_D_Cl_v_Cl);
        m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Edge(dt_D2));
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Edge(dt_D2), ess_bdr);
        m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_Cl_v_Cl, phi));
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_Cl_v_Cl, phi), ess_bdr);

        // weak Dirichlet boundary condition fff
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(D_Cl_), ess_bdr);

        // dt sigma <[dc1_dt], {D1 (grad(v1) + z1 v1 grad(phi))}>
        ProductCoefficient dt_sigma_D2_z2(dt, sigma_D2_z2);
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, dt * sigma));
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D2_z2, phi));

            if (symmetry_with_boundary)
            {
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, dt * sigma), ess_bdr);
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D2_z2, phi), ess_bdr);
            }
        }

        // dt kappa <{h^{-1}} [dc2_dt], [v2]>
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(dt * kappa*D_Cl));

            if (penalty_with_boundary)
            {
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(dt * kappa*D_Cl), ess_bdr);
            }
        }

        m2_dta2_dte2_dts2_dtp2->Assemble(skip_zero_entries);
    }
};
class PNP_Box_Gummel_DG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;
    PNP_Box_Gummel_DG_Operator* oper;

    mutable Array<int> true_offset;

public:
    PNP_Box_Gummel_DG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fes_, double time)
            : TimeDependentOperator(3*truesize, time), true_offset(offset), fes(fes_)
    {
        oper = new PNP_Box_Gummel_DG_Operator(fes_, truesize, offset, ess_bdr_);

    }
    virtual ~PNP_Box_Gummel_DG_TimeDependent()
    {
        delete oper;
    }

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
        MFEM_ABORT("Not supported now.");
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2; // 上一个时间步的解(已知)
        // 求解新的 old_phi 从而更新 phic1c2_ptr, 最终更新 phic1c2
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 后面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi_dt
        // 下面通过求解 dc1dt, dc2dt 从而更新 dphic1c2_dt
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);
        dc1dt.SetFromTrueVector();
        dc2dt.SetFromTrueVector();

        auto* phi_dc1dt_dc2dt = new BlockVector(true_offset);
        *phi_dc1dt_dc2dt = 0.0;
        old_phi.SetFromTrueVector();
        dc1dt.SetTrueVector();
        dc2dt.SetTrueVector();
        phi_dc1dt_dc2dt->SetVector(old_phi.GetTrueVector(), true_offset[0]);
        phi_dc1dt_dc2dt->SetVector(  dc1dt.GetTrueVector(), true_offset[1]);
        phi_dc1dt_dc2dt->SetVector(  dc2dt.GetTrueVector(), true_offset[2]);

        oper->UpdateParameters(t, dt, &old_c1, &old_c2); // 传入当前解

        Vector zero_vec;
        oper->Mult(zero_vec, *phi_dc1dt_dc2dt);

        phic1c2_ptr->SetVector(phi_dc1dt_dc2dt->GetBlock(0), true_offset[0]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(1), true_offset[1]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(2), true_offset[2]);
        delete phi_dc1dt_dc2dt;
    }
};


class PNP_Box_Newton_CG_Operator: public Operator
{
private:
    ParFiniteElementSpace* fes;

    mutable ParBilinearForm *a0, *b1, *b2, *m1_dta1, *m2_dta2, *g1_, *g2_, *h1, *h2, *h1_dth1, *h2_dth2;
    mutable ParLinearForm *l0, *l1, *l2;
    HypreParMatrix *A0, *B1, *B2, *M1_dtA1, *M2_dtA2, *G1, *G2, *H1, *H2, *H1_dtH1, *H2_dtH2;

    mutable BlockOperator *jac_k; // Jacobian at current solution
    ParGridFunction *phi, *dc1dt, *dc2dt;

    ParGridFunction *c1, *c2;
    double t, dt;

    int true_vsize;
    Array<int> &true_offset, &ess_tdof_list, null_array;
    int num_procs, rank;

public:
    PNP_Box_Newton_CG_Operator(ParFiniteElementSpace* fes_, int truevsize, Array<int>& offset, Array<int>& ess_tdof_list_)
        : Operator(3*truevsize), fes(fes_), true_vsize(truevsize), true_offset(offset), ess_tdof_list(ess_tdof_list_),
          a0(NULL), b1(NULL), b2(NULL), m1_dta1(NULL), m2_dta2(NULL),
          g1_(NULL), g2_(NULL), h1(NULL), h2(NULL), h1_dth1(NULL), h2_dth2(NULL)
    {
        MPI_Comm_size(fes->GetComm(), &num_procs);
        MPI_Comm_rank(fes->GetComm(), &rank);

        phi   = new ParGridFunction;
        dc1dt = new ParGridFunction;
        dc2dt = new ParGridFunction;

        A0 = new HypreParMatrix;
        B1 = new HypreParMatrix;
        B2 = new HypreParMatrix;
        G1 = new HypreParMatrix;
        G2 = new HypreParMatrix;
        H1 = new HypreParMatrix;
        H2 = new HypreParMatrix;
        H1_dtH1 = new HypreParMatrix;
        H2_dtH2 = new HypreParMatrix;
        M1_dtA1 = new HypreParMatrix;
        M2_dtA2 = new HypreParMatrix;

        l0 = new ParLinearForm(fes);
        l1 = new ParLinearForm(fes);
        l2 = new ParLinearForm(fes);

        jac_k = new BlockOperator(true_offset);

    }
    ~PNP_Box_Newton_CG_Operator()
    {
        delete phi; delete dc1dt; delete dc2dt;

        delete a0; delete b1; delete b2;
        delete g1_; delete g2_;
        delete h1; delete h2;
        delete h1_dth1; delete h2_dth2;
        delete m1_dta1; delete m2_dta2;

        delete A0; delete B1; delete B2;
        delete G1; delete G2;
        delete H1; delete H2;
        delete M1_dtA1; delete M2_dtA2;

        delete l0; delete l1; delete l2;

        delete jac_k;
    }

    void UpdateParameters(double current, double dt_, ParGridFunction* c1_, ParGridFunction* c2_)
    {
        t  = current;
        dt = dt_;
        c1 = c1_;
        c2 = c2_;

        c1->SetTrueVector(); // 后面有可能要用到TrueVector
        c2->SetTrueVector();
    }

    // 以PrimalVector为计算核心, 与 以TrueVector为计算核心 等价
    virtual void Mult_(const Vector& phi_dc1dt_dc2dt, Vector& residual) const
    {
        Vector& phi_dc1dt_dc2dt_ = const_cast<Vector&>(phi_dc1dt_dc2dt);

        phi  ->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[0]);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[1]);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[2]);
        phi  ->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();

        residual = 0.0;
        Vector y0(residual.GetData() + 0 * true_vsize, true_vsize);
        Vector y1(residual.GetData() + 1 * true_vsize, true_vsize);
        Vector y2(residual.GetData() + 2 * true_vsize, true_vsize);


        // **************************************************************************************
        //                                1. Poisson 方程 Residual
        // **************************************************************************************
        delete l0;
        l0 = new ParLinearForm(fes);
        // b0: (f0, psi)
        f0_analytic.SetTime(t);
        l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
        l0->Assemble();

        buildb1();
        buildb2();
        builda0();
        b1->AddMult(*c1, *l0, 1.0);   // l0 = l0 + b1 c1
        b2->AddMult(*c2, *l0, 1.0);   // l0 = l0 + b1 c1 + b2 c2
        b1->AddMult(*dc1dt, *l0, dt);    // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
        b2->AddMult(*dc2dt, *l0, dt);    // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt
        a0->AddMult(*phi, *l0, -1.0); // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt - a0 phi

        l0->ParallelAssemble(y0); // PrimalVector转换为TrueVector
        y0.SetSubVector(ess_tdof_list, 0.0);


        // **************************************************************************************
        //                                2. NP1 方程 Residual
        // **************************************************************************************
        delete l1;
        l1 = new ParLinearForm(fes);
        // b1: (f1, v1)
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();

        buildg1_();
        buildh1();
        buildm1_dta1(phi);
        g1_->AddMult(*c1, *l1, -1.0);        // l1 = l1 - g1 c1
        h1->AddMult(*phi, *l1, -1.0);        // l1 = l1 - g1 c1 - h1 phi
        m1_dta1->AddMult(*dc1dt, *l1, -1.0); // l1 = l1 - g1 c1 - h1 phi - m1_dta1 dc1dt

        l1->ParallelAssemble(y1); // PrimalVector转换为TrueVectorfff, 经检验正确
        y1.SetSubVector(ess_tdof_list, 0.0);


        // **************************************************************************************
        //                                3. NP2 方程 Residual
        // **************************************************************************************
        delete l2;
        l2 = new ParLinearForm(fes);
        // b2: (f2, v2)
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();

        buildg2_();
        buildh2();
        buildm2_dta2(phi);
        g2_->AddMult(*c2, *l2, -1.0);        // l2 = l2 - g2 c2
        h2->AddMult(*phi, *l2, -1.0);        // l2 = l2 - g2 c2 - h2 phi
        m2_dta2->AddMult(*dc2dt, *l2, -1.0); // l2 = l2 - g2 c2 - h2 phi - m2_dta2 dc2dt

        l2->ParallelAssemble(y2); // PrimalVector转换为TrueVector fff
        y2.SetSubVector(ess_tdof_list, 0.0);

        residual.Neg();
    }

    // 以TrueVector为计算核心, 与 以PrimalVector为计算核心 等价
    virtual void Mult(const Vector& phi_dc1dt_dc2dt, Vector& residual) const
    {
        Vector& phi_dc1dt_dc2dt_ = const_cast<Vector&>(phi_dc1dt_dc2dt);
        phi  ->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[0]);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[1]);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[2]);
        phi  ->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();

        Vector   phi_tdof(phi_dc1dt_dc2dt.GetData() + 0*true_vsize, true_vsize);
        Vector dc1dt_tdof(phi_dc1dt_dc2dt.GetData() + 1*true_vsize, true_vsize);
        Vector dc2dt_tdof(phi_dc1dt_dc2dt.GetData() + 2*true_vsize, true_vsize);

        Vector y0(residual.GetData() + 0 * true_vsize, true_vsize);
        Vector y1(residual.GetData() + 1 * true_vsize, true_vsize);
        Vector y2(residual.GetData() + 2 * true_vsize, true_vsize);


        // **************************************************************************************
        //                                1. Poisson 方程 Residual
        // **************************************************************************************
        delete l0;
        l0 = new ParLinearForm(fes);
        // b0: (f0, psi)
        f0_analytic.SetTime(t);
        l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
        l0->Assemble();
        l0->ParallelAssemble(y0);

        buildb1();
        buildb2();
        builda0();
        b1->TrueAddMult(c1->GetTrueVector(), y0, 1.0);   // l0 = l0 + b1 c1
        b2->TrueAddMult(c2->GetTrueVector(), y0, 1.0);   // l0 = l0 + b1 c1 + b2 c2
        b1->TrueAddMult(dc1dt_tdof, y0, dt);    // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
        b2->TrueAddMult(dc2dt_tdof, y0, dt);    // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt
        a0->TrueAddMult(phi_tdof, y0, -1.0); // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt - a0 phi

        y0.SetSubVector(ess_tdof_list, 0.0); // fff这样设定边界对吗?经检验正确


        // **************************************************************************************
        //                                2. NP1 方程 Residual
        // **************************************************************************************
        delete l1;
        l1 = new ParLinearForm(fes);
        // b1: (f1, v1)
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        l1->ParallelAssemble(y1);

        buildg1_();
        buildh1();
        buildm1_dta1(phi);
        g1_->TrueAddMult(c1->GetTrueVector(), y1, -1.0); // l1 = l1 - g1 c1
        h1->TrueAddMult(phi_tdof, y1, -1.0);        // l1 = l1 - g1 c1 - h1 phi
        m1_dta1->TrueAddMult(dc1dt_tdof, y1, -1.0); // l1 = l1 - g1 c1 - h1 phi - m1_dta1 dc1dt

        y1.SetSubVector(ess_tdof_list, 0.0); // fff


        // **************************************************************************************
        //                                3. NP2 方程 Residual
        // **************************************************************************************
        delete l2;
        l2 = new ParLinearForm(fes);
        // b2: (f2, v2)
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();
        l2->ParallelAssemble(y2);

        buildg2_();
        buildh2();
        buildm2_dta2(phi);
        g2_->TrueAddMult(c2->GetTrueVector(), y2, -1.0);        // l2 = l2 - g2 c2
        h2->TrueAddMult(phi_tdof, y2, -1.0);        // l2 = l2 - g2 c2 - h2 phi
        m2_dta2->TrueAddMult(dc2dt_tdof, y2, -1.0); // l2 = l2 - g2 c2 - h2 phi - m2_dta2 dc2dt

        y2.SetSubVector(ess_tdof_list, 0.0); // fff

        residual.Neg(); // 残量取负
    }

    virtual Operator &GetGradient(const Vector& phi_dc1dt_dc2dt) const
    {
        Vector& phi_dc1dt_dc2dt_ = const_cast<Vector&>(phi_dc1dt_dc2dt);

        phi  ->MakeTRef(fes, phi_dc1dt_dc2dt_, 0*true_vsize);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, 1*true_vsize);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, 2*true_vsize);
        phi  ->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();


        // **************************************************************************************
        //                                1. Poisson 方程的 Jacobian
        // **************************************************************************************
        builda0();
        a0->FormSystemMatrix(ess_tdof_list, *A0);

        buildb1();
        b1->FormSystemMatrix(null_array, *B1);
        *B1 *= -1.0*dt;

        buildb2();
        b2->FormSystemMatrix(null_array, *B2);
        *B2 *= -1.0*dt;


        // **************************************************************************************
        //                                2. NP1 方程的 Jacobian
        // **************************************************************************************
        buildh1_dth1(dc1dt);
        h1_dth1->FormSystemMatrix(null_array, *H1_dtH1);

        buildm1_dta1(phi);
        m1_dta1->FormSystemMatrix(ess_tdof_list, *M1_dtA1);


        // **************************************************************************************
        //                                3. NP2 方程的 Jacobian
        // **************************************************************************************
        buildh2_dth2(dc2dt);
        h2_dth2->FormSystemMatrix(null_array, *H2_dtH2);

        buildm2_dta2(phi);
        m2_dta2->FormSystemMatrix(ess_tdof_list, *M2_dtA2);


        delete jac_k;
        jac_k = new BlockOperator(true_offset);
        jac_k->SetBlock(0, 0, A0);
        jac_k->SetBlock(0, 1, B1);
        jac_k->SetBlock(0, 2, B2);
        jac_k->SetBlock(1, 0, H1_dtH1);
        jac_k->SetBlock(1, 1, M1_dtA1);
        jac_k->SetBlock(2, 0, H2_dtH2);
        jac_k->SetBlock(2, 2, M2_dtA2);
        return *jac_k;
    }

private:
    // epsilon_s (grad(phi), grad(psi))
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(fes);
        a0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        a0->Assemble(skip_zero_entries);
        a0->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(fes);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
        b1->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(fes);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
        b2->Finalize(skip_zero_entries);
    }

    // (c1, v1) + dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
    void buildm1_dta1(const ParGridFunction* phi_) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt, D_K_prod_v_K);

        m1_dta1 = new ParBilinearForm(fes);
        // (c1, v1)
        m1_dta1->AddDomainIntegrator(new MassIntegrator);
        // dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        m1_dta1->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi_, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // (c2, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
    void buildm2_dta2(const ParGridFunction* phi_) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt, D_Cl_prod_v_Cl);

        m2_dta2 = new ParBilinearForm(fes);
        // (c2, v2)
        m2_dta2->AddDomainIntegrator(new MassIntegrator);
        // dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        m2_dta2->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi_, &dt_D2_z2));

        m2_dta2->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1), grad(v1))
    void buildg1_() const
    {
        if (g1_ != NULL) { delete g1_; }

        g1_ = new ParBilinearForm(fes);
        // D1 (grad(c1), grad(v1))
        g1_->AddDomainIntegrator(new DiffusionIntegrator(D_K_));

        g1_->Assemble(skip_zero_entries);
    }

    // D2 (grad(c2), grad(v2))
    void buildg2_() const
    {
        if (g2_ != NULL) { delete g2_; }

        g2_ = new ParBilinearForm(fes);
        // D2 (grad(c2), grad(v2))
        g2_->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));

        g2_->Assemble(skip_zero_entries);
    }

    // D1 (z1 c1 grad(dphi), grad(v1)), given c1
    void buildh1() const
    {
        if (h1 != NULL) { delete h1; }

        GridFunctionCoefficient c1_coeff(c1);
        ProductCoefficient D1_z1_c1_coeff(D_K_prod_v_K, c1_coeff);

        h1 = new ParBilinearForm(fes);
        // D1 (z1 c1 grad(dphi), grad(v1)), given c1
        h1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_c1_coeff));

        h1->Assemble(skip_zero_entries);
    }

    // D1 (z1 (c1 + dt dc1dt) grad(dphi), grad(v1)), given c1 and dc1dt
    void buildh1_dth1(const ParGridFunction* dc1dt_) const
    {
        if (h1_dth1 != NULL) { delete h1_dth1; }

        GridFunctionCoefficient c1_coeff(c1), dc1dt_coeff(dc1dt_);
        ProductCoefficient D1_z1_c1_coeff(D_K_prod_v_K, c1_coeff);
        ProductCoefficient dt_dc1dt_coeff(dt, dc1dt_coeff);
        ProductCoefficient D1_z1_dt_dc1dt_coeff(D_K_prod_v_K, dt_dc1dt_coeff);

        h1_dth1 = new ParBilinearForm(fes);
        // D1 (z1 c1 grad(dphi), grad(v1)), given c1
        h1_dth1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_c1_coeff));
        // D1 (z1 dt dc1dt grad(dphi), grad(v1)), given dc1dt
        h1_dth1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_dt_dc1dt_coeff));

        h1_dth1->Assemble(skip_zero_entries);
    }

    // D2 (z2 c2 grad(dphi), grad(v2)), given c2
    void buildh2() const
    {
        if (h2 != NULL) { delete h2; }

        GridFunctionCoefficient c2_coeff(c2);
        ProductCoefficient D2_z2_c2_coeff(D_Cl_prod_v_Cl, c2_coeff);

        h2 = new ParBilinearForm(fes);
        // D2 (z2 c2 grad(dphi), grad(v2)), given c2
        h2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_c2_coeff));

        h2->Assemble(skip_zero_entries);
    }

    // D2 (z2 (c2 + dt dc2dt) grad(dphi), grad(v2)), given c2 and dc2dt
    void buildh2_dth2(const ParGridFunction* dc2dt_) const
    {
        if (h2_dth2 != NULL) { delete h2_dth2; }

        GridFunctionCoefficient c2_coeff(c2), dc2dt_coeff(dc2dt_);
        ProductCoefficient D2_z2_c2_coeff(D_Cl_prod_v_Cl, c2_coeff);
        ProductCoefficient dt_dc2dt_coeff(dt, dc2dt_coeff);
        ProductCoefficient D2_z2_dt_dc2dt_coeff(D_Cl_prod_v_Cl, dt_dc2dt_coeff);

        h2_dth2 = new ParBilinearForm(fes);
        // D2 (z2 c2 grad(dphi), grad(v2)), given c2
        h2_dth2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_c2_coeff));
        // D2 (z2 dt dc2dt grad(dphi), grad(v2)), given dc2dt
        h2_dth2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_dt_dc2dt_coeff));

        h2_dth2->Assemble(skip_zero_entries);
    }
};
class PNP_Box_Newton_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;

    PNP_Box_Newton_CG_Operator* oper;
//    PNP_Newton_BCHandler* bchandler;
    PetscNonlinearSolver* newton_solver;
    PetscPreconditionerFactory *jac_factory;

    ParGridFunction old_phi, old_c1, old_c2; // 上一个时间步的解(已知)

    int true_vsize;
    Array<int> true_offset, ess_bdr, ess_tdof_list; // 在H1空间中存在ess_tdof_list, 在DG空间中不存在
    int num_procs, rank;
    StopWatch chrono;

public:
    PNP_Box_Newton_CG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fes_, double time)
            : TimeDependentOperator(3*truesize, time), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_), fes(fes_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        oper          = new PNP_Box_Newton_CG_Operator(fes, true_vsize, true_offset, ess_tdof_list);
//        bchandler     = new PNP_Newton_BCHandler(PetscBCHandler::CONSTANT, ess_tdof_list, 3, true_vsize); // 3表示3个变量:phi,c1,c2
        jac_factory   = new PreconditionerFactory(*oper, prec_type);
        newton_solver = new PetscNonlinearSolver(fes->GetComm(), *oper, "newton_");
        newton_solver->SetPreconditionerFactory(jac_factory);
//        newton_solver->SetBCHandler(bchandler);
        newton_solver->iterative_mode = true;
    }
    ~PNP_Box_Newton_CG_TimeDependent()
    {
        delete oper;
        delete jac_factory;
        delete newton_solver;
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        // 求解新的 old_phi 从而更新 phic1c2_ptr, 最终更新 phic1c2
        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        // 下面通过求解 dc1dt, dc2dt 从而更新 dphic1c2_dt
        ParGridFunction dc1dt, dc2dt;
        dphic1c2_dt = 0.0;
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);
        dc1dt.SetFromTrueVector();
        dc2dt.SetFromTrueVector();

        phi_exact  .SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);
        old_phi.ProjectBdrCoefficient(  phi_exact, ess_bdr); // 设定解的边界条件
        dc1dt  .ProjectBdrCoefficient(dc1dt_exact, ess_bdr); // 这里做ProjectBdrCoefficient()是把dc1dt当成PrimalVector, 所以上面一定要有dc1dt.SetFromTrueVector()
        dc2dt  .ProjectBdrCoefficient(dc2dt_exact, ess_bdr);

        // !!!引用 phi, dc1dt, dc2dt 的 TrueVector, 使得 phi_dc1dt_dc2dt 所指的内存块就是phi, dc1dt, dc2dt的内存块.
        // 从而在Newton求解器中对 phi_dc1dt_dc2dt 的修改就等同于对phi, dc1dt, dc2dt的修改, 最终达到了更新解的目的.
        auto* phi_dc1dt_dc2dt = new BlockVector(true_offset);
        *phi_dc1dt_dc2dt = 0.0;
        old_phi.SetTrueVector();
        dc1dt.SetTrueVector();
        dc2dt.SetTrueVector();
        phi_dc1dt_dc2dt->SetVector(old_phi.GetTrueVector(), true_offset[0]);
        phi_dc1dt_dc2dt->SetVector(  dc1dt.GetTrueVector(), true_offset[1]);
        phi_dc1dt_dc2dt->SetVector(  dc2dt.GetTrueVector(), true_offset[2]);

        oper->UpdateParameters(t, dt, &old_c1, &old_c2); // 传入当前解
//        bchandler->SetBoundarValues(*phi_dc1dt_dc2dt); // 设定BCHandler

        Vector zero_vec;
        newton_solver->Mult(zero_vec, *phi_dc1dt_dc2dt);
        if (!newton_solver->GetConverged()) MFEM_ABORT("Newton solver did not converge!!!");

        // 设定新的解向量
        phic1c2_ptr->SetVector(phi_dc1dt_dc2dt->GetBlock(0), true_offset[0]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(1), true_offset[1]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(2), true_offset[2]);
        delete phi_dc1dt_dc2dt;
    }
};


class PNP_Box_Newton_DG_Operator: public Operator {
private:
    ParFiniteElementSpace* fes;
    const Operator* Restriction; // 不能delete,它是FiniteElementSpace的成员变量

    mutable ParBilinearForm *a0_e0_s0_p0, *a0, *b1, *b2, *m1_dta1, *m2_dta2, *g1_, *g2_,
                            *h1, *h2, *h1_dth1, *h2_dth2, *k1, *k2, *r1, *r2, *t1, *t2, *L1, *L2, *w1, *w2,
                            *e1, *e2, *s1, *s2, *p1, *p2, *h1_k1_r1_l1, *h2_k2_r2_l2, *m1_a1_e1_s1_p1, *m2_a2_e2_s2_p2,
                            *np1_c1, *np1_phi, *np2_c2, *np2_phi;
    mutable ParLinearForm *l0, *l1, *l2;
    HypreParMatrix *A0_E0_S0_P0, *A0, *B1, *B2, *M1_dtA1, *M2_dtA2, *G1, *G2, *H1, *H2, *H1_dtH1, *H2_dtH2,
            *H1_K1_R1_L1, *H2_K2_R2_L2, *M1_A1_E1_S1_P1, *M2_A2_E2_S2_P2;;

    mutable BlockOperator *jac_k; // Jacobian at current solution
    ParGridFunction *phi, *dc1dt, *dc2dt;

    ParGridFunction *c1, *c2;
    double t, dt;

    int true_vsize;
    Array<int> &true_offset, &ess_bdr, null_array;
    int num_procs, rank;

public:
    PNP_Box_Newton_DG_Operator(ParFiniteElementSpace *fes_, int truevsize, Array<int> &offset, Array<int> &ess_bdr_)
            : Operator(3 * truevsize), fes(fes_), true_vsize(truevsize), true_offset(offset), ess_bdr(ess_bdr_) {
        MPI_Comm_size(fes->GetComm(), &num_procs);
        MPI_Comm_rank(fes->GetComm(), &rank);

        Restriction = fes->GetRestrictionMatrix();

        phi   = new ParGridFunction;
        dc1dt = new ParGridFunction;
        dc2dt = new ParGridFunction;

        A0_E0_S0_P0 = new HypreParMatrix;
        B1 = new HypreParMatrix;
        B2 = new HypreParMatrix;

        H1_K1_R1_L1 = new HypreParMatrix;
        M1_A1_E1_S1_P1 = new HypreParMatrix;

        H2_K2_R2_L2 = new HypreParMatrix;
        M2_A2_E2_S2_P2 = new HypreParMatrix;

        l0 = new ParLinearForm(fes);
        l1 = new ParLinearForm(fes);
        l2 = new ParLinearForm(fes);

        np1_c1 = NULL;
        np1_phi = NULL;
        np2_c2 = NULL;
        np2_phi = NULL;

        jac_k = new BlockOperator(true_offset);

        b1 = NULL;
        b2 = NULL;
        a0_e0_s0_p0 = NULL;

        np1c1 = NULL;np1phi = NULL;np1dc1dt = NULL;np1dtdc1dt = NULL;np2c2 = NULL;np2phi = NULL;np2dc2dt = NULL;np2dtdc2dt = NULL;

        a0_e0_s0_p0 = NULL;
        h1_k1_r1_l1 = NULL;
        m1_a1_e1_s1_p1 = NULL;
        h2_k2_r2_l2 = NULL;
        m2_a2_e2_s2_p2 = NULL;
    }

    ~PNP_Box_Newton_DG_Operator() {
        delete phi;
        delete dc1dt;
        delete dc2dt;

        delete b1;
        delete b2;
        delete a0_e0_s0_p0;

        delete a0_e0_s0_p0;
        delete h1_k1_r1_l1;
        delete m1_a1_e1_s1_p1;
        delete h2_k2_r2_l2;
        delete m2_a2_e2_s2_p2;

        delete A0_E0_S0_P0;
        delete H1_K1_R1_L1;
        delete H2_K2_R2_L2;
        delete A0;
        delete B1;
        delete B2;
        delete G1;
        delete G2;
        delete H1;
        delete H2;
        delete M1_dtA1;
        delete M2_dtA2;

        delete l0;
        delete l1;
        delete l2;

        delete np1c1; delete np1phi; delete np1dc1dt; delete np1dtdc1dt; delete np2c2; delete np2phi; delete np2dc2dt; delete np2dtdc2dt;

        delete jac_k;
    }

    void UpdateParameters(double current, double dt_, ParGridFunction *c1_, ParGridFunction *c2_) {
        t = current;
        dt = dt_;
        c1 = c1_;
        c2 = c2_;

        c1->SetTrueVector(); // 后面有可能要用到TrueVector
        c2->SetTrueVector();
    }

    virtual void Mult(const Vector &phi_dc1dt_dc2dt, Vector &residual) const
    {
        Vector &phi_dc1dt_dc2dt_ = const_cast<Vector &>(phi_dc1dt_dc2dt);
        phi->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[0]);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[1]);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[2]);
        phi->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();

        Vector   phi_tdof(phi_dc1dt_dc2dt.GetData() + 0*true_vsize, true_vsize);
        Vector dc1dt_tdof(phi_dc1dt_dc2dt.GetData() + 1*true_vsize, true_vsize);
        Vector dc2dt_tdof(phi_dc1dt_dc2dt.GetData() + 2*true_vsize, true_vsize);

        residual = 0.0;
        Vector y0(residual.GetData() + 0 * true_vsize, true_vsize);
        Vector y1(residual.GetData() + 1 * true_vsize, true_vsize);
        Vector y2(residual.GetData() + 2 * true_vsize, true_vsize);

        if (0) {
            cout << endl;
            if (rank == 0) {
                cout << "In compute Residual(), l2 norm of   phi: " << phi  ->Norml2() << endl;
                cout << "In compute Residual(), l2 norm of dc1dt: " << dc1dt->Norml2() << endl;
                cout << "In compute Residual(), l2 norm of dc2dt: " << dc2dt->Norml2() << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 1) {
                cout << "In compute Residual(), l2 norm of   phi: " << phi  ->Norml2() << endl;
                cout << "In compute Residual(), l2 norm of dc1dt: " << dc1dt->Norml2() << endl;
                cout << "In compute Residual(), l2 norm of dc2dt: " << dc2dt->Norml2() << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (1) {
            if (0)
            {
                cout << "l2 norm of y0: " << y0.Norml2() << endl;
                cout << "l2 norm of y1: " << y1.Norml2() << endl;
                cout << "l2 norm of y2: " << y2.Norml2() << endl;
                residual.Neg(); // 残量取负
                return;
            }

            if (1)
            {
                cout << "Only obtain Poisson Residual." << endl;
                // **************************************************************************************
                //                                1. Poisson 方程 Residual
                // **************************************************************************************
                delete l0;
                l0 = new ParLinearForm(fes);
                // b0: (f0, psi)
                f0_analytic.SetTime(t);
                l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
                // q0: kappa <{h^{-1} epsilon_s} phi_D, psi>
                phi_exact.SetTime(t);
                l0->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, 0.0, kappa));
                l0->Assemble();
                {
                    l0->ParallelAssemble(y0);
                    return;
                }

                buildb1();
                buildb2();
                b1->AddMult(*c1, *l0, 1.0);  // l0 = l0 + b1 c1
                b2->AddMult(*c2, *l0, 1.0);  // l0 = l0 + b1 c1 + b2 c2
                b1->AddMult(*dc1dt, *l0, dt);   // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
                b2->AddMult(*dc2dt, *l0, dt);   // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt
                builda0_e0_s0_p0();
                { // ref: https://github.com/mfem/mfem/issues/1830
                    auto* blf_tdof = a0_e0_s0_p0->ParallelAssemble();
                    auto* l0_tdof  = l0->ParallelAssemble();
                    auto* Restriction = a0_e0_s0_p0->GetRestriction();

                    blf_tdof->Mult(-1.0, phi->GetTrueVector(), 1.0, *l0_tdof); // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt - a0_e0_s0_p0 phi
                    Restriction->MultTranspose(*l0_tdof, *l0);

                    delete blf_tdof; delete l0_tdof;
                }

                l0->ParallelAssemble(y0);

                cout << "l2 norm of y0: " << y0.Norml2() << endl;
                cout << "l2 norm of y1: " << y1.Norml2() << endl;
                cout << "l2 norm of y2: " << y2.Norml2() << endl;
                cout << "L2 norm of   phi: " << phi->ComputeL2Error(zero) << endl;
                cout << "L2 norm of dc1dt: " << dc1dt->ComputeL2Error(zero) << endl;
                cout << "L2 norm of dc2dt: " << dc2dt->ComputeL2Error(zero) << endl;
//                residual.Neg(); // 残量取负
                cout << "l2 norm of residual: " << residual.Norml2() << endl;
                return;
            }

            if (0)
            {
                // **************************************************************************************
                //                                2. NP1 方程 Residual
                // **************************************************************************************
                delete l1;
                l1 = new ParLinearForm(fes);
                // b1: (f1, v1)
                f1_analytic.SetTime(t);
                l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
//                // -q1: kappa <{h^{-1} D1 } c1_D, v1>
//                c1_exact.SetTime(t);
//                l1->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c1_exact, -1.0*kappa * D_K), ess_bdr);
                l1->Assemble();

                buildnp1_c1();
                {
                    auto* np1_c1_tdof = np1_c1->ParallelAssemble();
                    auto* l1_tdof  = l1->ParallelAssemble();

                    np1_c1_tdof->Mult(-1.0, c1->GetTrueVector(), 1.0, *l1_tdof);
                    np1_c1_tdof->Mult(-1.0*dt, dc1dt_tdof, 1.0, *l1_tdof);

                    Restriction->MultTranspose(*l1_tdof, *l1);
                    delete np1_c1_tdof;
                    delete l1_tdof;
                }

                buildnp1_phi(c1);
                {
                    auto* np1_phi_tdof = np1_phi->ParallelAssemble();
                    auto* l1_tdof  = l1->ParallelAssemble();

                    np1_phi_tdof->Mult(-1.0, phi_tdof, 1.0, *l1_tdof);

                    Restriction->MultTranspose(*l1_tdof, *l1);
                    delete np1_phi_tdof;
                    delete l1_tdof;
                }

                buildnp1_phi(dc1dt);
                {
                    auto* np1_phi_tdof = np1_phi->ParallelAssemble();
                    auto* l1_tdof  = l1->ParallelAssemble();

                    np1_phi_tdof->Mult(-1.0*dt, phi_tdof, 1.0, *l1_tdof);

                    Restriction->MultTranspose(*l1_tdof, *l1);
                    delete np1_phi_tdof;
                    delete l1_tdof;
                }

                l1->ParallelAssemble(y1);

//                cout.precision(14);
//                cout << endl;
//                cout << "l2 norm of dc1dt_tdof: " << dc1dt_tdof.Norml2() << endl;
                cout << "l2 norm of y0: " << y0.Norml2() << endl;
                cout << "l2 norm of y1: " << y1.Norml2() << endl;
                cout << "l2 norm of y2: " << y2.Norml2() << endl;
//                cout << "l2 norm of Residual: " << residual.Norml2() << endl;
//                cout << "L2 norm of phi: " << phi->ComputeL2Error(zero) << endl;
//                cout << "L2 norm of dc1dt: " << dc1dt->ComputeL2Error(zero) << endl;
//                cout << "L2 norm of dc2dt: " << dc2dt->ComputeL2Error(zero) << endl;
                residual.Neg(); // 残量取负
//                MFEM_ABORT("F");
                return;
            }
        }


        // **************************************************************************************
        //                                1. Poisson 方程 Residual
        // **************************************************************************************
        delete l0;
        l0 = new ParLinearForm(fes);
        f0_analytic.SetTime(t);
        phi_exact.SetTime(t);
        // b0: (f0, psi)
        l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
        l0->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, sigma, kappa), ess_bdr);
        l0->Assemble();

        buildb1();
        buildb2();
        builda0_e0_s0_p0();
        b1->AddMult(*c1, *l0, 1.0);            // l0 = l0 + b1 c1
        b2->AddMult(*c2, *l0, 1.0);            // l0 = l0 + b1 c1 + b2 c2
        b1->AddMult(*dc1dt, *l0, dt);             // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
        b2->AddMult(*dc2dt, *l0, dt);             // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt
        builda0_e0_s0_p0();
        { // ref: https://github.com/mfem/mfem/issues/1830
            auto* blf_tdof = a0_e0_s0_p0->ParallelAssemble();
            auto* l0_tdof  = l0->ParallelAssemble();
            auto* Restriction = a0_e0_s0_p0->GetRestriction();

            blf_tdof->Mult(-1.0, phi->GetTrueVector(), 1.0, *l0_tdof); // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt - a0_e0_s0_p0 phi
            Restriction->MultTranspose(*l0_tdof, *l0);

            delete blf_tdof; delete l0_tdof;
        }

        l0->ParallelAssemble(y0);


        // **************************************************************************************
        //                                2. NP1 方程 Residual
        // **************************************************************************************
        delete l1;
        l1 = new ParLinearForm(fes);
        f1_analytic.SetTime(t);
        c1_exact.SetTime(t);
        // b1: (f1, v1)
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();

        buildnp1_c1();
        {
            auto* np1_c1_tdof = np1_c1->ParallelAssemble();
            auto* l1_tdof  = l1->ParallelAssemble();

            np1_c1_tdof->Mult(-1.0, c1->GetTrueVector(), 1.0, *l1_tdof);
            np1_c1_tdof->Mult(-1.0*dt, dc1dt_tdof, 1.0, *l1_tdof);

            Restriction->MultTranspose(*l1_tdof, *l1);
            delete np1_c1_tdof;
            delete l1_tdof;
        }

        buildnp1_phi(c1);
        {
            auto* np1_phi_tdof = np1_phi->ParallelAssemble();
            auto* l1_tdof  = l1->ParallelAssemble();

            np1_phi_tdof->Mult(-1.0, phi_tdof, 1.0, *l1_tdof);

            Restriction->MultTranspose(*l1_tdof, *l1);
            delete np1_phi_tdof;
            delete l1_tdof;
        }

        buildnp1_phi(dc1dt);
        {
            auto* np1_phi_tdof = np1_phi->ParallelAssemble();
            auto* l1_tdof  = l1->ParallelAssemble();

            np1_phi_tdof->Mult(-1.0*dt, phi_tdof, 1.0, *l1_tdof);

            Restriction->MultTranspose(*l1_tdof, *l1);
            delete np1_phi_tdof;
            delete l1_tdof;
        }

        l1->ParallelAssemble(y1);


        // **************************************************************************************
        //                                3. NP2 方程 Residual
        // **************************************************************************************
        delete l2;
        l2 = new ParLinearForm(fes);
        f2_analytic.SetTime(t);
        c2_exact.SetTime(t);
        // b2: (f2, v2)
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();

        buildnp2_c2();
        {
            auto* np2_c2_tdof = np2_c2->ParallelAssemble();
            auto* l2_tdof  = l2->ParallelAssemble();

            np2_c2_tdof->Mult(-1.0, c2->GetTrueVector(), 1.0, *l2_tdof);
            np2_c2_tdof->Mult(-1.0*dt, dc2dt_tdof, 1.0, *l2_tdof);

            Restriction->MultTranspose(*l2_tdof, *l2);
            delete np2_c2_tdof;
            delete l2_tdof;
        }

        buildnp2_phi(c2);
        {
            auto* np2_phi_tdof = np2_phi->ParallelAssemble();
            auto* l2_tdof  = l2->ParallelAssemble();

            np2_phi_tdof->Mult(-1.0, phi_tdof, 1.0, *l2_tdof);

            Restriction->MultTranspose(*l2_tdof, *l2);
            delete np2_phi_tdof;
            delete l2_tdof;
        }

        buildnp2_phi(dc2dt);
        {
            auto* np2_phi_tdof = np2_phi->ParallelAssemble();
            auto* l2_tdof  = l2->ParallelAssemble();

            np2_phi_tdof->Mult(-1.0*dt, phi_tdof, 1.0, *l2_tdof);

            Restriction->MultTranspose(*l2_tdof, *l2);
            delete np2_phi_tdof;
            delete l2_tdof;
        }

        l2->ParallelAssemble(y2);

        residual.Neg(); // 残量取负

        if (1)
        {
            cout << "l2 norm of y0: " << y0.Norml2() << endl;
            cout << "l2 norm of y1: " << y1.Norml2() << endl;
            cout << "l2 norm of y2: " << y2.Norml2() << endl;
        }
    }

    virtual Operator &GetGradient(const Vector &phi_dc1dt_dc2dt) const {
        Vector &phi_dc1dt_dc2dt_ = const_cast<Vector &>(phi_dc1dt_dc2dt);

        phi->MakeTRef(fes, phi_dc1dt_dc2dt_, 0 * true_vsize);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, 1 * true_vsize);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, 2 * true_vsize);
        phi->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();

        if (1)
        {
            if (0) {
                auto *temp0 = new ParBilinearForm(fes);
                auto *temp1 = new ParBilinearForm(fes);
                auto *temp2 = new ParBilinearForm(fes);

                temp0->AddDomainIntegrator(new MassIntegrator(zero));
                temp0->Assemble();
                temp0->FormSystemMatrix(null_array, *A0_E0_S0_P0);
                A0_E0_S0_P0->EliminateZeroRows();

                temp1->AddDomainIntegrator(new MassIntegrator(zero));
                temp1->Assemble();
                temp1->FormSystemMatrix(null_array, *M1_A1_E1_S1_P1);
                M1_A1_E1_S1_P1->EliminateZeroRows();

                temp2->AddDomainIntegrator(new MassIntegrator(zero));
                temp2->Assemble();
                temp2->FormSystemMatrix(null_array, *M2_A2_E2_S2_P2);
                M2_A2_E2_S2_P2->EliminateZeroRows();

                delete jac_k;
                jac_k = new BlockOperator(true_offset);
                jac_k->SetBlock(0, 0, A0_E0_S0_P0);
                jac_k->SetBlock(1, 1, M1_A1_E1_S1_P1);
                jac_k->SetBlock(2, 2, M2_A2_E2_S2_P2);
                return *jac_k;
            }

            if (1) {
                cout << "Only obtain Poisson Jacobian." << endl;
                // **************************************************************************************
                //                                1. Poisson 方程的 Jacobian
                // **************************************************************************************
                builda0_e0_s0_p0();
                a0_e0_s0_p0->FormSystemMatrix(null_array, *A0_E0_S0_P0);

                buildb1();
                b1->FormSystemMatrix(null_array, *B1);
                *B1 *= -1.0*dt;

                buildb2();
                b2->FormSystemMatrix(null_array, *B2);
                *B2 *= -1.0*dt;

                // 把NP方程的部分变成单位阵，那么在Newton迭代过程中就相当于不求解NP方程
                delete m1_a1_e1_s1_p1;
                m1_a1_e1_s1_p1 = new ParBilinearForm(fes);

                m1_a1_e1_s1_p1->AddDomainIntegrator(new MassIntegrator(zero));
                m1_a1_e1_s1_p1->Assemble(skip_zero_entries);
                m1_a1_e1_s1_p1->Finalize(skip_zero_entries);
                m1_a1_e1_s1_p1->FormSystemMatrix(null_array, *M1_A1_E1_S1_P1);
                M1_A1_E1_S1_P1->EliminateZeroRows();

                delete m2_a2_e2_s2_p2;
                m2_a2_e2_s2_p2 = new ParBilinearForm(fes);

                m2_a2_e2_s2_p2->AddDomainIntegrator(new MassIntegrator(zero));
                m2_a2_e2_s2_p2->Assemble(skip_zero_entries);
                m2_a2_e2_s2_p2->Finalize(skip_zero_entries);
                m2_a2_e2_s2_p2->FormSystemMatrix(null_array, *M2_A2_E2_S2_P2);
                M2_A2_E2_S2_P2->EliminateZeroRows();


                delete jac_k;
                jac_k = new BlockOperator(true_offset);
                jac_k->SetBlock(0, 0, A0_E0_S0_P0);
                jac_k->SetBlock(0, 1, B1);
                jac_k->SetBlock(0, 2, B2);
                jac_k->SetBlock(1, 1, M1_A1_E1_S1_P1);
                jac_k->SetBlock(2, 2, M2_A2_E2_S2_P2);
                return *jac_k;
            }

            if (0)
            {
                // **************************************************************************************
                //                                2. NP1 方程的 Jacobian
                // **************************************************************************************
                buildh1_k1_r1_l1(c1, dc1dt);
                h1_k1_r1_l1->FormSystemMatrix(null_array, *H1_K1_R1_L1);

                buildm1_a1_e1_s1_p1(phi);
                m1_a1_e1_s1_p1->FormSystemMatrix(null_array, *M1_A1_E1_S1_P1);

                auto *temp0 = new ParBilinearForm(fes);
                auto *temp2 = new ParBilinearForm(fes);

                temp0->AddDomainIntegrator(new MassIntegrator(zero));
                temp0->Assemble();
                temp0->FormSystemMatrix(null_array, *A0_E0_S0_P0);
                A0_E0_S0_P0->EliminateZeroRows();

                temp2->AddDomainIntegrator(new MassIntegrator(zero));
                temp2->Assemble();
                temp2->FormSystemMatrix(null_array, *M2_A2_E2_S2_P2);
                M2_A2_E2_S2_P2->EliminateZeroRows();

                delete jac_k;
                jac_k = new BlockOperator(true_offset);
                jac_k->SetBlock(0, 0, A0_E0_S0_P0);
                jac_k->SetBlock(1, 0, H1_K1_R1_L1);
                jac_k->SetBlock(1, 1, M1_A1_E1_S1_P1);
                jac_k->SetBlock(2, 2, M2_A2_E2_S2_P2);
                return *jac_k;
            }
        }


        // **************************************************************************************
        //                                1. Poisson 方程的 Jacobian
        // **************************************************************************************
        builda0_e0_s0_p0();
        a0_e0_s0_p0->FormSystemMatrix(null_array, *A0_E0_S0_P0);

        buildb1();
        b1->FormSystemMatrix(null_array, *B1);
        *B1 *= -1.0 * dt;

        buildb2();
        b2->FormSystemMatrix(null_array, *B2);
        *B2 *= -1.0 * dt;


        // **************************************************************************************
        //                                2. NP1 方程的 Jacobian
        // **************************************************************************************
        buildh1_k1_r1_l1(c1, dc1dt);
        h1_k1_r1_l1->FormSystemMatrix(null_array, *H1_K1_R1_L1);

        buildm1_a1_e1_s1_p1(phi);
        m1_a1_e1_s1_p1->FormSystemMatrix(null_array, *M1_A1_E1_S1_P1);


        // **************************************************************************************
        //                                3. NP2 方程的 Jacobian
        // **************************************************************************************
        buildh2_k2_r2_l2(c2, dc2dt);
        h2_k2_r2_l2->FormSystemMatrix(null_array, *H2_K2_R2_L2);

        buildm2_a2_e2_s2_p2(phi);
        m2_a2_e2_s2_p2->FormSystemMatrix(null_array, *M2_A2_E2_S2_P2);


        delete jac_k;
        jac_k = new BlockOperator(true_offset);
        jac_k->SetBlock(0, 0, A0_E0_S0_P0);
        jac_k->SetBlock(0, 1, B1);
        jac_k->SetBlock(0, 2, B2);
        jac_k->SetBlock(1, 0, H1_K1_R1_L1);
        jac_k->SetBlock(1, 1, M1_A1_E1_S1_P1);
        jac_k->SetBlock(2, 0, H2_K2_R2_L2);
        jac_k->SetBlock(2, 2, M2_A2_E2_S2_P2);
        return *jac_k;
    }

private:
    void builda0_e0_s0_p0() const
    {
        if (a0_e0_s0_p0 != NULL) { delete a0_e0_s0_p0; }

        a0_e0_s0_p0 = new ParBilinearForm(fes);

        // (epsilon_s grad(phi), grad(psi))
        a0_e0_s0_p0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        // -<{epsilon_s grad(phi)}, [psi]>
        a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, 0.0, kappa));
        a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, 0.0, kappa), ess_bdr);

        a0_e0_s0_p0->Assemble(skip_zero_entries);
        a0_e0_s0_p0->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(fes);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
        b1->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(fes);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
        b2->Finalize(skip_zero_entries);
    }

    void buildh1_k1_r1_l1(ParGridFunction *c1_, ParGridFunction *dc1dt_) const
    {
        if (h1_k1_r1_l1 != NULL) { delete h1_k1_r1_l1; }

        h1_k1_r1_l1 = new ParBilinearForm(fes);

        GridFunctionCoefficient c1_coeff(c1_), dc1dt_coeff(dc1dt_);
        ProductCoefficient D1_z1_c1_coeff(D_K_prod_v_K, c1_coeff);
        ProductCoefficient dt_dc1dt(dt, dc1dt_coeff);
        ProductCoefficient D1_z1_dt_dc1dt_coeff(D_K_prod_v_K, dt_dc1dt);

        // D1 (z1 c1 grad(dphi), grad(v1)) - <{D1 z1 c1 grad(dphi).n}, [v1]>, given c1
        h1_k1_r1_l1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_c1_coeff));
        h1_k1_r1_l1->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D1_z1_c1_coeff, 0.0, 0.0));
        h1_k1_r1_l1->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D1_z1_c1_coeff, 0.0, 0.0), ess_bdr);

        // D1 (z1 dt dc1dt grad(dphi), grad(v1)) - <{D1 z1 dt dc1dt grad(dphi).n}, [v1]>, given c1
        h1_k1_r1_l1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_dt_dc1dt_coeff));
        h1_k1_r1_l1->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D1_z1_dt_dc1dt_coeff, 0.0, 0.0));
        h1_k1_r1_l1->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D1_z1_dt_dc1dt_coeff, 0.0, 0.0), ess_bdr);

        h1_k1_r1_l1->Assemble(skip_zero_entries);
    }

    void buildh2_k2_r2_l2(ParGridFunction *c2_, ParGridFunction *dc2dt_) const
    {
        if (h2_k2_r2_l2 != NULL) { delete h2_k2_r2_l2; }

        h2_k2_r2_l2 = new ParBilinearForm(fes);

        GridFunctionCoefficient c2_coeff(c2_), dc2dt_coeff(dc2dt_);
        ProductCoefficient D2_z2_c2_coeff(D_Cl_prod_v_Cl, c2_coeff);
        ProductCoefficient dt_dc2dt(dt, dc2dt_coeff);
        ProductCoefficient D2_z2_dt_dc2dt_coeff(D_Cl_prod_v_Cl, dt_dc2dt);

        // D2 (z2 c2 grad(dphi), grad(v2)) - <{D2 z2 c2 grad(dphi).n}, [v2]>, given c2
        h2_k2_r2_l2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_c2_coeff));
        h2_k2_r2_l2->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D2_z2_c2_coeff, 0.0, 0.0));
        h2_k2_r2_l2->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D2_z2_c2_coeff, 0.0, 0.0), ess_bdr);

        // D2 (z2 dt dc2dt grad(dphi), grad(v2)) - <{D2 z2 dt dc2dt grad(dphi).n}, [v2]>, given c2
        h2_k2_r2_l2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_dt_dc2dt_coeff));
        h2_k2_r2_l2->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D2_z2_dt_dc2dt_coeff, 0.0, 0.0));
        h2_k2_r2_l2->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D2_z2_dt_dc2dt_coeff, 0.0, 0.0), ess_bdr);

        h2_k2_r2_l2->Assemble(skip_zero_entries);
    }

    void buildm1_a1_e1_s1_p1(ParGridFunction *phi_) const
    {
        if (m1_a1_e1_s1_p1 != NULL) { delete m1_a1_e1_s1_p1; }
        phi->ExchangeFaceNbrData();

        phi->ExchangeFaceNbrData();
        ProductCoefficient neg_dt_D_K_v_K(dt, neg_D_K_v_K);
        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt, D_K_prod_v_K);
        ProductCoefficient neg_dt_D1_z1(neg, dt_D1_z1);
        ProductCoefficient dt_sigma_D_K_v_K(dt, neg_D_K_v_K);

        m1_a1_e1_s1_p1 = new ParBilinearForm(fes);

        // (c1, v1)
        m1_a1_e1_s1_p1->AddDomainIntegrator(new MassIntegrator);

        // dt D1 (grad(c1), grad(v1)) - <{dt D1 grad(c1).n}, [v1]> - kappa <{h^{-1} dt D1} [c1], [v1]>
        m1_a1_e1_s1_p1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        m1_a1_e1_s1_p1->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_D1, 0.0, 0.0));
        m1_a1_e1_s1_p1->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_D1, 0.0, 0.0), ess_bdr);

        // dt D1 z1 (c1 grad(phi), grad(v1)), given phi
        m1_a1_e1_s1_p1->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi_, &dt_D1_z1));

        // - <{dt D1 z1 c1 grad(phi).n}, [v1]>, given phi
        m1_a1_e1_s1_p1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg_dt_D_K_v_K, *phi_));
        m1_a1_e1_s1_p1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg_dt_D_K_v_K, *phi_), ess_bdr);

        m1_a1_e1_s1_p1->Assemble(skip_zero_entries);
    }

    void buildm2_a2_e2_s2_p2(ParGridFunction *phi_) const
    {
        if (m2_a2_e2_s2_p2 != NULL) { delete m2_a2_e2_s2_p2; }
        phi->ExchangeFaceNbrData();

        m2_a2_e2_s2_p2 = new ParBilinearForm(fes);

        phi->ExchangeFaceNbrData();
        ProductCoefficient neg_dt_D_Cl_v_Cl(dt, neg_D_Cl_v_Cl);
        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt, D_Cl_prod_v_Cl);
        ProductCoefficient neg_dt_D2_z2(neg, dt_D2_z2);
        ProductCoefficient dt_sigma_D_Cl_v_Cl(dt, neg_D_Cl_v_Cl);

        m2_a2_e2_s2_p2 = new ParBilinearForm(fes);

        // (c2, v2)
        m2_a2_e2_s2_p2->AddDomainIntegrator(new MassIntegrator);

        // dt D2 (grad(c2), grad(v2)) - <{dt D2 grad(c2).n}, [v2]> - kappa <{h^{-1} dt D2} [c2], [v2]>
        m2_a2_e2_s2_p2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        m2_a2_e2_s2_p2->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(dt_D2, 0.0, 0.0));
        m2_a2_e2_s2_p2->AddBdrFaceIntegrator(new DGDiffusionIntegrator(dt_D2, 0.0, 0.0), ess_bdr);

        // dt D2 z2 (c2 grad(phi), grad(v2)), given phi
        m2_a2_e2_s2_p2->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi_, &dt_D2_z2));

        // - <{dt D2 z2 c2 grad(phi).n}, [v2]>, given phi
        m2_a2_e2_s2_p2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg_dt_D_Cl_v_Cl, *phi_));
        m2_a2_e2_s2_p2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg_dt_D_Cl_v_Cl, *phi_), ess_bdr);

        m2_a2_e2_s2_p2->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1), grad(v1)) - <{D1 grad(c1).n}, [v1]> - kappa <{h^{-1} D1} [c1], [v1]>
    void buildnp1_c1() const
    {
        if (np1_c1 != NULL) { delete np1_c1; }

        np1_c1 = new ParBilinearForm(fes);

        // D1 (grad(c1), grad(v1))
        np1_c1->AddDomainIntegrator(new DiffusionIntegrator(D_K_));

        // - <{D1 grad(c1).n}, [v1]> - kappa <{h^{-1} D1} [c1], [v1]>
        np1_c1->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_K_, 0.0, 0.0));
        np1_c1->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_K_, 0.0, 0.0), ess_bdr);

        np1_c1->Assemble(skip_zero_entries);
        np1_c1->Finalize(skip_zero_entries);
    }

    // D1 z1 c1 (grad(phi), grad(v1)) - <{D1 z1 c1 grad(phi).n}, [v1]>, given c1
    void buildnp1_phi(ParGridFunction* c1_) const
    {
        if (np1_phi != NULL) { delete np1_phi; }

        GridFunctionCoefficient c1_coeff(c1_);
        ProductCoefficient D1_z1_c1(D_K_prod_v_K, c1_coeff);

        np1_phi = new ParBilinearForm(fes);

        // D1 z1 c1 (grad(phi), grad(v1))
        np1_phi->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_c1));

        // - <{D1 z1 c1 grad(phi).n}, [v1]>
        np1_phi->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D1_z1_c1, 0.0, 0.0));
        np1_phi->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D1_z1_c1, 0.0, 0.0), ess_bdr);

        np1_phi->Assemble(skip_zero_entries);
        np1_phi->Finalize(skip_zero_entries);
    }

    // D2 (grad(c2), grad(v2)) - <{D2 grad(c2).n}, [v2]> - kappa <{h^{-1} D2} [c2], [v2]>
    void buildnp2_c2() const
    {
        if (np2_c2 != NULL) { delete np2_c2; }

        np2_c2 = new ParBilinearForm(fes);

        // D2 (grad(c2), grad(v2))
        np2_c2->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));

        // - <{D2 grad(c2).n}, [v2]> - kappa <{h^{-1} D2} [c2], [v2]>
        np2_c2->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, 0.0, 0.0));
        np2_c2->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, 0.0, 0.0), ess_bdr);

        np2_c2->Assemble(skip_zero_entries);
        np2_c2->Finalize(skip_zero_entries);
    }

    // D2 z2 c2 (grad(phi), grad(v2)) - <{D2 z2 c2 grad(phi).n}, [v2]>, given c2
    void buildnp2_phi(ParGridFunction* c2_) const
    {
        if (np2_phi != NULL) { delete np2_phi; }

        GridFunctionCoefficient c2_coeff(c2_);
        ProductCoefficient D2_z2_c2(D_Cl_prod_v_Cl, c2_coeff);

        np2_phi = new ParBilinearForm(fes);

        // D2 z2 c2 (grad(phi), grad(v2))
        np2_phi->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_c2));

        // - <{D2 z2 c2 grad(phi).n}, [v2]>
        np2_phi->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D2_z2_c2, 0.0, 0.0));
        np2_phi->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D2_z2_c2, 0.0, 0.0), ess_bdr);

        np2_phi->Assemble(skip_zero_entries);
        np2_phi->Finalize(skip_zero_entries);
    }

};
class PNP_Box_Newton_DG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;

    PNP_Box_Newton_DG_Operator* oper;
    PetscNonlinearSolver* newton_solver;
    PetscPreconditionerFactory *jac_factory;

    ParGridFunction old_phi, old_c1, old_c2; // 上一个时间步的解(已知)

    int true_vsize;
    Array<int> true_offset, ess_bdr; // 在H1空间中存在ess_tdof_list, 在DG空间中不存在
    int num_procs, rank;

public:
    PNP_Box_Newton_DG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fes_, double time)
            : TimeDependentOperator(3*truesize, time), fes(fes_), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_)
    {
        MPI_Comm_size(fes->GetComm(), &num_procs);
        MPI_Comm_rank(fes->GetComm(), &rank);

        oper = new PNP_Box_Newton_DG_Operator(fes, true_vsize, true_offset, ess_bdr);

        jac_factory   = new PreconditionerFactory(*oper, prec_type);
        newton_solver = new PetscNonlinearSolver(fes->GetComm(), *oper, "newton_");
        newton_solver->SetPreconditionerFactory(jac_factory);
        newton_solver->iterative_mode = true;
    }

    ~PNP_Box_Newton_DG_TimeDependent()
    {
        delete oper;
        delete jac_factory;
        delete newton_solver;
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        // 求解新的 old_phi 从而更新 phic1c2_ptr, 最终更新 phic1c2
        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        if (0) {
            phi_exact.SetTime(t-dt);
            c1_exact.SetTime(t-dt);
            c2_exact.SetTime(t-dt);

            double phiL2errornorm = old_phi.ComputeL2Error(phi_exact);
            double  c1L2errornorm = old_c1 .ComputeL2Error(c1_exact);
            double  c2L2errornorm = old_c2 .ComputeL2Error(c2_exact);
            cout << "1. ImplicitSolve(), phi L2 errornorm: " << phiL2errornorm << '\n'
                 << "1. ImplicitSolve(),  c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                 << "1. ImplicitSolve(),  c2 L2 errornorm: " <<  c2L2errornorm << '\n' << endl;
        }


        // 下面通过求解 dc1dt, dc2dt 从而更新 dphic1c2_dt
        ParGridFunction dc1dt(fes), dc2dt(fes);
        dphic1c2_dt = 0.0;
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);
        dc1dt.SetFromTrueVector();
        dc2dt.SetFromTrueVector();

        { // fff
//            dc1dt_exact.SetTime(t);
//            dc2dt_exact.SetTime(t);
//            dc1dt.ProjectCoefficient(dc1dt_exact);
//            dc2dt.ProjectCoefficient(dc2dt_exact);
        }

        /* DG 不需要设定强的边界条件, 都是weak Dirichlet边界条件. */
        if (0) {
            phi_exact.SetTime(t);
            dc1dt_exact.SetTime(t);
            dc2dt_exact.SetTime(t);
            c1_exact.SetTime(t);
            c2_exact.SetTime(t);

            old_phi.ProjectCoefficient(phi_exact);
            dc1dt.ProjectCoefficient(dc1dt_exact);
            dc2dt.ProjectCoefficient(dc2dt_exact);

//            old_c1.ProjectCoefficient(c1_exact);
//            old_c2.ProjectCoefficient(c2_exact);
            old_c1.Add(dt, dc1dt);
            old_c2.Add(dt, dc2dt);

            old_phi.SetTrueVector();
            dc1dt.SetTrueVector();
            dc2dt.SetTrueVector();

            {
                double phiL2errornorm = old_phi.ComputeL2Error(phi_exact);
                double  dc1dtL2errornorm = dc1dt.ComputeL2Error(dc1dt_exact);
                double  dc2dtL2errornorm = dc2dt.ComputeL2Error(dc2dt_exact);

                double c1L2errornorm = old_c1.ComputeL2Error(c1_exact);
                double c2L2errornorm = old_c2.ComputeL2Error(c2_exact);

                cout << "2. ImplicitSolve(),   phi L2 errornorm: " << phiL2errornorm << '\n'
                     << "2. ImplicitSolve(), dc1dt L2 errornorm: " <<  dc1dtL2errornorm << '\n'
                     << "2. ImplicitSolve(), dc2dt L2 errornorm: " <<  dc2dtL2errornorm << '\n'
                     << "2. ImplicitSolve(), old_c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << "2. ImplicitSolve(), old_c2 L2 errornorm: " <<  c2L2errornorm << '\n'
                        << endl;

            }

//            true_offset.Print(cout << "true_offset: ");
//            phic1c2_ptr->SetVector(old_phi.GetTrueVector(), true_offset[0]);
//            dphic1c2_dt .SetVector(dc1dt.GetTrueVector(), true_offset[1]);
//            dphic1c2_dt .SetVector(dc2dt.GetTrueVector(), true_offset[2]);
//            if (1) cout << "l2 norm of dphic1c2_dt: " << dphic1c2_dt.Norml2() << endl;
            return;
        }
        if (1) {
            phi_exact.SetTime(t);
            dc1dt_exact.SetTime(t);
            dc2dt_exact.SetTime(t);
            c1_exact.SetTime(t);
            c2_exact.SetTime(t);

            old_phi.ProjectCoefficient(phi_exact);
            dc1dt.ProjectCoefficient(dc1dt_exact);
            dc2dt.ProjectCoefficient(dc2dt_exact);
        }

        auto* phi_dc1dt_dc2dt = new BlockVector(true_offset);
        *phi_dc1dt_dc2dt = 0.0;
        old_phi.SetTrueVector();
        dc1dt.SetTrueVector();
        dc2dt.SetTrueVector();
        phi_dc1dt_dc2dt->SetVector(old_phi.GetTrueVector(), true_offset[0]);
        phi_dc1dt_dc2dt->SetVector(  dc1dt.GetTrueVector(), true_offset[1]);
        phi_dc1dt_dc2dt->SetVector(  dc2dt.GetTrueVector(), true_offset[2]);

        oper->UpdateParameters(t, dt, &old_c1, &old_c2); // 传入当前解
        if (0) {
            cout << "Before Newton::Mult(), L2 norm of  old_c1: " << old_c1.ComputeL2Error(zero) << endl;
            cout << "Before Newton::Mult(), L2 norm of  old_c2: " << old_c2.ComputeL2Error(zero) << endl;
            cout << "Before Newton::Mult(), L2 norm of old_phi: " << old_phi.ComputeL2Error(zero) << endl;
            cout << "Before Newton::Mult(), L2 norm of   dc1dt: " << dc1dt.ComputeL2Error(zero) << endl;
            cout << "Before Newton::Mult(), L2 norm of   dc2dt: " << dc2dt.ComputeL2Error(zero) << endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        Vector zero_vec;
        newton_solver->Mult(zero_vec, *phi_dc1dt_dc2dt);
        if (!newton_solver->GetConverged()) MFEM_ABORT("Newton solver did not converge!!!");

        // 设定新的解向量
        phic1c2_ptr->SetVector(phi_dc1dt_dc2dt->GetBlock(0), true_offset[0]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(1), true_offset[1]);
        dphic1c2_dt .SetVector(phi_dc1dt_dc2dt->GetBlock(2), true_offset[2]);
        if (0) {
            cout << "l2 norm of dphic1c2_dt: " << dphic1c2_dt.Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt->GetBlock(1): " << phi_dc1dt_dc2dt->GetBlock(1).Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt->GetBlock(2): " << phi_dc1dt_dc2dt->GetBlock(2).Norml2() << endl;
        }
        delete phi_dc1dt_dc2dt;
    }
};


class PNP_Box_TimeDependent_Solver
{
private:
    ParMesh* pmesh;
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fes;

    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    double t; // 当前时间
    TimeDependentOperator* oper;
    ODESolver *ode_solver;

    int true_vsize; // 有限元空间维数
    Array<int> true_offset, ess_bdr;
    ParaViewDataCollection* pd;
    int num_procs, rank;
    StopWatch chrono;

public:
    PNP_Box_TimeDependent_Solver(ParMesh* pmesh_, int ode_solver_type): pmesh(pmesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        t = t_init;

        if (strcmp(Discretize, "cg") == 0)
        {
            fec = new H1_FECollection(p_order, pmesh->Dimension());
        }
        else if (strcmp(Discretize, "dg") == 0)
        {
            fec = new DG_FECollection(p_order, pmesh->Dimension());
        }
        fes = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        ess_bdr = 1; // 对于H1空间, 设置所有边界都是essential的; 对DG空间, 边界条件都是weak的

        phi_gf = new ParGridFunction(fes); *phi_gf = 0.0;
        c1_gf  = new ParGridFunction(fes); *c1_gf  = 0.0;
        c2_gf  = new ParGridFunction(fes); *c2_gf  = 0.0;

        true_vsize = fes->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = true_vsize * 0;
        true_offset[1] = true_vsize * 1;
        true_offset[2] = true_vsize * 2;
        true_offset[3] = true_vsize * 3;

        phic1c2 = new BlockVector(true_offset); *phic1c2 = 0.0; // TrueVector, not PrimalVector
        phi_gf->MakeTRef(fes, *phic1c2, true_offset[0]);
        c1_gf ->MakeTRef(fes, *phic1c2, true_offset[1]);
        c2_gf ->MakeTRef(fes, *phic1c2, true_offset[2]);

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

        if (strcmp(Linearize, "gummel") == 0)
        {
            if (strcmp(Discretize, "cg") == 0)
            {
                oper = new PNP_Box_Gummel_CG_TimeDependent(true_vsize, true_offset, ess_bdr, fes, t);
            }
            else if (strcmp(Discretize, "dg") == 0)
            {
                oper = new PNP_Box_Gummel_DG_TimeDependent(true_vsize, true_offset, ess_bdr, fes, t);
            }
            else MFEM_ABORT("Not support discretization");
        }
        else
        {
            MFEM_ASSERT(strcmp(Linearize, "newton") == 0, "Linearizations: Gummel or Newton.");
            if (strcmp(Discretize, "cg") == 0)
            {
                oper = new PNP_Box_Newton_CG_TimeDependent(true_vsize, true_offset, ess_bdr, fes, t);
            }
            else if (strcmp(Discretize, "dg") == 0)
            {
                oper = new PNP_Box_Newton_DG_TimeDependent(true_vsize, true_offset, ess_bdr, fes, t);
            }
            else MFEM_ABORT("Not support discretization");
        }

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
            string paraview_title = string("PNP_Box_") + Discretize + "_" + Linearize + "_Time_Dependent";
            pd = new ParaViewDataCollection(paraview_title, pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);

            pd->SetCycle(0); // 第 0 个时间步
            pd->SetTime(t); // 第 0 个时间步所表示的时间
            pd->Save();
        }
    }
    ~PNP_Box_TimeDependent_Solver()
    {
        delete fec; delete fes;
        delete phi_gf; delete c1_gf; delete c2_gf; delete phic1c2;
        delete oper;
        delete ode_solver;
        if (paraview) delete pd;
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes, Array<double>& time_steps)
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
        if (SpaceConvergRate_Change_dt) {
            t_stepsize = mesh_size * mesh_size * Change_dt_factor;
        }
        time_steps.Append(t_stepsize);
        meshsizes.Append(mesh_size);

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();

        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(t_stepsize, t_final - t);

            { // fff做测试: 把(部分)真解带入计算
                phi_exact.SetTime(t);
                phi_gf->ProjectCoefficient(phi_exact);
                phi_gf->SetTrueVector();
                phi_gf->SetFromTrueVector();

                c1_exact.SetTime(t + dt_real);
                c1_gf->ProjectCoefficient(c1_exact);
                c1_gf->SetTrueVector();
                c1_gf->SetFromTrueVector();

                c2_exact.SetTime(t + dt_real);
                c2_gf->ProjectCoefficient(c2_exact);
                c2_gf->SetTrueVector();
                c2_gf->SetFromTrueVector();
            }
            ode_solver->Step(*phic1c2, t, dt_real); // 经过这一步之后 phic1c2(TrueVector, not PrimalVector) 和 t 都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            // 得到下一个时刻t的解, 这个t和执行上述Step()之前的t不一样, 差一个dt_real
            phi_gf->SetFromTrueVector();
            c1_gf->SetFromTrueVector();
            c2_gf->SetFromTrueVector();

            if (0) {
                phi_exact.SetTime(t);
                c1_exact.SetTime(t);
                c2_exact.SetTime(t);

                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm = c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm = c2_gf->ComputeL2Error(c2_exact);
                cout << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << '\n' << endl;

            }

            if (paraview)
            {
                pd->SetCycle(ti); // 第 i 个时间步. 注: ti为0的解就是初始时刻的解, 在构造函数中已经保存
                pd->SetTime(t); // 第 i 个时间步所表示的时间
                pd->Save();
            }
            if (verbose >= 1)
            {
                phi_exact.SetTime(t);
                c1_exact.SetTime(t);
                c2_exact.SetTime(t);

                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm = c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm = c2_gf->ComputeL2Error(c2_exact);
                if (rank == 0)
                {
                    cout << "Time: " << t << '\n'
                         << "phi L2 errornorm: " << phiL2errornorm << '\n'
                         << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                         << " c2 L2 errornorm: " <<  c2L2errornorm << '\n' << endl;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Stop();

        {
            // 计算最后一个时刻的解的误差范数
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

            if (rank == 0)
            {
                cout << "=========> ";
                cout << Discretize << p_order << ", " << Linearize << ", " << options_src << ", DOFs: " << fes->GlobalTrueVSize() * 3<< ", Cores: " << num_procs << ", "
                     << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" : "wrong type")) << '\n'
                     << mesh_file << ", refine mesh: " << refine_mesh << ", mesh size: " << mesh_size << '\n'
                     << "FiniteElementSpace size: " << fes->GlobalTrueVSize() << '\n'
                     << "t_init: "<< t_init << ", t_final: " << t_final << ", time step: " << t_stepsize << ", refine time: " << refine_time << ", time scale: " << time_scale
                     << endl;

                cout << "ODE solver taking " << chrono.RealTime() << " s." << endl;
                cout.precision(14);
                cout << "At final time: " << t << '\n'
                     << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << '\n'
                     << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << '\n'
                     << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << '\n' << endl;

                // 保留最后一个时间步的计算误差
                if (SpaceConvergRate || TimeConvergRate)
                {
                    phiL2errornorms_.Append(phiL2err);
                    c1L2errornorms_.Append(c1L2err);
                    c2L2errornorms_.Append(c2L2err);
                }
            }
        }
    }
};


#endif
