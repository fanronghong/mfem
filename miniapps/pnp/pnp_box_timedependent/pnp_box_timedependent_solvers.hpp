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
};


class PNP_Box_Gummel_DG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;
//    // 用来给DG空间的GridFunction设定边界条件: 如果gf属于DG空间的GridFunction, 则gf.ProjectBdrCoefficient()会出错
//    H1_FECollection* h1_fec;
//    ParFiniteElementSpace* h1;

    mutable ParBilinearForm *a0, *e0, *s0, *p0, *a0_e0_s0_p0,
                            *m1, *a1, *e1, *s1, *p1, *m1_dta1_dte1_dts1_dtp1,
                            *m2, *a2, *e2, *s2, *p2, *m2_dta2_dte2_dts2_dtp2,
                            *b1, *b2;
    mutable HypreParMatrix *A0_E0_S0_P0,
                           *M1_dtA1_dtE1_dtS1_dtP1,
                           *M2_dtA2_dtE2_dtS2_dtP2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;
    int true_vsize;
    mutable Array<int> true_offset, ess_bdr, null_array; // 在H1空间中存在ess_tdof_list, 在DG空间中不存在
    int num_procs, rank;

public:
    PNP_Box_Gummel_DG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fes_, double time)
            : TimeDependentOperator(3*truesize, time), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_), fes(fes_),
              a0(NULL), e0(NULL), s0(NULL), p0(NULL), a0_e0_s0_p0(NULL),
              m1(NULL), a1(NULL), e1(NULL), s1(NULL), p1(NULL), m1_dta1_dte1_dts1_dtp1(NULL),
              m2(NULL), a2(NULL), e2(NULL), s2(NULL), p2(NULL), m2_dta2_dte2_dts2_dtp2(NULL),
              b1(NULL), b2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//         fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // 在H1空间中存在, 在DG空间中不存在

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
    virtual ~PNP_Box_Gummel_DG_TimeDependent()
    {
        delete a0; delete e0; delete s0; delete p0; delete a0_e0_s0_p0;
        delete m1; delete a1; delete e1; delete s1; delete p1; delete m1_dta1_dte1_dts1_dtp1;
        delete m2; delete a2; delete e2; delete s2; delete p2; delete m2_dta2_dte2_dts2_dtp2;
        delete b1; delete b2;

        delete A0_E0_S0_P0;
        delete M1_dtA1_dtE1_dtS1_dtP1;
        delete M2_dtA2_dtE2_dtS2_dtP2;

        delete temp_x0; delete temp_b0;
        delete temp_x1; delete temp_b1;
        delete temp_x2; delete temp_b2;

//        delete h1_fec; delete h1;
    }

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
    // (c1, v1)
    void buildm1() const
    {
        if (m1 != NULL) { delete m1; }

        m1 = new ParBilinearForm(fes);
        m1->AddDomainIntegrator(new MassIntegrator);

        m1->Assemble(skip_zero_entries);
    }
    // (c2, v2)
    void buildm2() const
    {
        if (m2 != NULL) { delete m2; }

        BilinearFormIntegrator* np2_mass = new MassIntegrator;

        m2 = new ParBilinearForm(fes);
        m2->AddDomainIntegrator(np2_mass);

        m2->Assemble(skip_zero_entries);
    }

    // epsilon_s (grad(phi), grad(psi))
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(fes);
        // epsilon_s (grad(phi), grad(psi))
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

    // D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
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

    // -<{epsilon_s grad(phi)}, [psi]>
    void builde0() const
    {
        if (e0 != NULL) { delete e0; }

        e0 = new ParBilinearForm(fes);
        // -<{epsilon_s grad(phi)}, [psi]> 对单元内部边界和区域外部边界积分
        e0->AddInteriorFaceIntegrator(new DGDiffusion_Edge(epsilon_water));
        e0->AddBdrFaceIntegrator(new DGDiffusion_Edge(epsilon_water), ess_bdr);

        e0->Assemble(skip_zero_entries);
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
        e2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_Cl_v_Cl, phi));
        e2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(neg_D_Cl_v_Cl, phi), ess_bdr);

        e2->Assemble(skip_zero_entries);
    }

    // sigma <[phi], {epsilon_s grad(psi)}>
    void builds0() const
    {
        if (s0 != NULL) { delete s0; }

        s0 = new ParBilinearForm(fes);
        // sigma <[phi], {epsilon_s grad(psi)}> 对单元内部边界和区域外部边界积分
        s0->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma));
        s0->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma), ess_bdr);

        s0->Assemble(skip_zero_entries);
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
        s1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_K_v_K, phi));
        if (symmetry_with_boundary)
        {
            s1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_K_v_K, phi), ess_bdr);
        }

        s1->Assemble(skip_zero_entries);
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
        s2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_Cl_v_Cl, phi));
        if (symmetry_with_boundary)
        {
            s2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(neg_sigma_D_Cl_v_Cl, phi), ess_bdr);
        }

        s2->Assemble(skip_zero_entries);
    }

    // kappa <{h^{-1}} [phi], [psi]>
    void buildp0() const
    {
        if (p0 != NULL) { delete p0; }

        p0 = new ParBilinearForm(fes);
        // kappa <{h^{-1}} [phi], [psi]> 对单元内部边界和区域外部边界积分
        p0->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(kappa));
        p0->AddBdrFaceIntegrator(new DGDiffusion_Penalty(kappa), ess_bdr);

        p0->Assemble(skip_zero_entries);
    }

    // -kappa <{h^{-1}} [c1], [v1]>
    void buildp1() const
    {
        if (p1 != NULL) { delete p1; }

        p1 = new ParBilinearForm(fes);
        // -kappa <{h^{-1}} [c1], [v1]> 对单元内部边界和区域外部边界积分
        p1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa));
        if (penalty_with_boundary)
        {
            p1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa), ess_bdr);
        }

        p1->Assemble(skip_zero_entries);
    }

    // -kappa <{h^{-1}} [c2], [v2]>
    void buildp2() const
    {
        if (p2 != NULL) { delete p2; }

        p2 = new ParBilinearForm(fes);
        // -kappa <{h^{-1}} [c2], [v2]> 对单元内部边界和区域外部边界积分
        p2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa));
        if (penalty_with_boundary)
        {
            p2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0*kappa), ess_bdr);
        }

        p2->Assemble(skip_zero_entries);
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

        // kappa <{h^{-1}} [phi], [psi]>
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(kappa));
            if (penalty_with_boundary)
            {
                a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Penalty(kappa), ess_bdr);
            }
        }

        a0_e0_s0_p0->Assemble(skip_zero_entries);
    }

    void buildm1_dta1_dte1_dts1_dtp1(double dt, ParGridFunction& phi) const
    {
        if (m1_dta1_dte1_dts1_dtp1 != NULL) { delete m1_dta1_dte1_dts1_dtp1; }

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
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(dt * kappa));

            if (penalty_with_boundary)
            {
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(dt * kappa), ess_bdr);
            }
        }

        m1_dta1_dte1_dts1_dtp1->Assemble(skip_zero_entries);
    }

    void buildm2_dta2_dte2_dts2_dtp2(double dt, ParGridFunction& phi) const
    {
        if (m2_dta2_dte2_dts2_dtp2 != NULL) { delete m2_dta2_dte2_dts2_dtp2; }

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
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(dt * kappa));

            if (penalty_with_boundary)
            {
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(dt * kappa), ess_bdr);
            }
        }

        m2_dta2_dte2_dts2_dtp2->Assemble(skip_zero_entries);
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
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi_dt
        // 下面通过求解 dc1dt, dc2dt 从而更新 dphic1c2_dt
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);

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
                // q0: kappa <{h^{-1}} phi_D, psi>
                l0->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(phi_exact, kappa), ess_bdr);
            }
            l0->Assemble();

            buildb1();
            buildb2();
            b1->AddMult(old_c1, *l0, 1.0);    // l0 = l0 + b1 c1
            b2->AddMult(old_c2, *l0, 1.0);    // l0 = l0 + b1 c1 + b2 c2
            b1->AddMult(dc1dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
            b2->AddMult(dc2dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt

            builda0_e0_s0_p0();
            a0_e0_s0_p0->FormLinearSystem(null_array, phi_Gummel, *l0, *A0_E0_S0_P0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0_E0_S0_P0, false, "phi_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0_e0_s0_p0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel);
            delete l0;
            delete poisson_solver;
            

            // **************************************************************************************
            //                                2. 计算Gummel迭代相对误差
            // **************************************************************************************
            diff  = 0.0;
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
                l1->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c1_exact, kappa), ess_bdr);
            }
            l1->Assemble();

            builda1(phi_Gummel);
            builde1(phi_Gummel);
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1
            e1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1 - e1 c1
            if (abs(sigma - 0.0) > 1E-10) // 添加对称项
            {
                builds1(phi_Gummel);
                s1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1
            }
            if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
            {
                buildp1();
                p1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1 + p1 c1
            }

            buildm1_dta1_dte1_dts1_dtp1(dt, phi_Gummel);
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
                l2->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c2_exact, kappa), ess_bdr);
            }
            l2->Assemble();

            builda2(phi_Gummel);
            builde2(phi_Gummel);
            a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2
            e2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2 - e2 c2
            if (abs(sigma - 0.0) > 1E-10) // 添加对称项
            {
                builds2(phi_Gummel);
                s2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2
            }
            if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
            {
                buildp2();
                p2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2 + p2 c2
            }

            buildm2_dta2_dte2_dts2_dtp2(dt, phi_Gummel);
            m2_dta2_dte2_dts2_dtp2->FormLinearSystem(null_array, dc2dt_Gummel, *l2, *M2_dtA2_dtE2_dtS2_dtP2, *temp_x2, *temp_b2);

            PetscLinearSolver* np2_solver = new PetscLinearSolver(*M2_dtA2_dtE2_dtS2_dtP2, false, "np2_");
            np2_solver->Mult(*temp_b2, *temp_x2);
            m2_dta2_dte2_dts2_dtp2->RecoverFEMSolution(*temp_x2, *l2, dc2dt_Gummel); // 更新 dc2dt
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

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
        MFEM_ABORT("Not supported now.");
    }
};



class PNP_Box_Newton_CG_Operator: public Operator
{
private:
    ParFiniteElementSpace* fes;

    mutable ParBilinearForm *a0, *b1, *b2, *m1_dta1, *m2_dta2, *g1_, *g2_, *h1, *h2, *h1_dth1, *h2_dth2;
    mutable ParLinearForm *l0, *l1, *l2;
    HypreParMatrix *A0, *B1, *B2, *M1_dtA1, *M2_dtA2, *G1, *G2, *H1, *H2, *H1_dtH1, *H2_dtH2;

    const ParGridFunction *c1, *c2;
    ParGridFunction *phi, *dc1dt, *dc2dt;
    double t, dt;

    int true_vsize;
    Array<int> &true_offset, &ess_tdof_list, null_array;
    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution

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

        rhs_k = new BlockVector(true_offset);
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

        delete rhs_k; delete jac_k;
    }

    void UpdateParameters(double current, double dt_, const ParGridFunction* c1_, const ParGridFunction* c2_)
    {
        t  = current;
        dt = dt_;
        c1 = c1_;
        c2 = c2_;
    }

    virtual void Mult(const Vector& phi_dc1dt_dc2dt, Vector& residual) const
    {
        Vector& phi_dc1dt_dc2dt_ = const_cast<Vector&>(phi_dc1dt_dc2dt);

        phi  ->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[0]);
        dc1dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[1]);
        dc2dt->MakeTRef(fes, phi_dc1dt_dc2dt_, true_offset[2]);
        phi  ->SetFromTrueVector(); // 下面要用到 PrimalVector, 而不是 TrueVector
        dc1dt->SetFromTrueVector();
        dc2dt->SetFromTrueVector();

        if (hahahaha) {
            cout << "l2 norm of   phi: " <<   phi->Norml2() << endl;
            cout << "l2 norm of dc1dt: " << dc1dt->Norml2() << endl;
            cout << "l2 norm of dc2dt: " << dc2dt->Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt: " << phi_dc1dt_dc2dt.Norml2() << endl;
            cout << "l2 norm of residual: " << residual.Norml2() << '\n' << endl;
        }

        rhs_k->Update(residual.GetData(), true_offset); // update residual


        // **************************************************************************************
        //                                1. Poisson 方程 Residual
        // **************************************************************************************
        delete l0;
        l0 = new ParLinearForm(fes);
        l0->Update(fes, rhs_k->GetBlock(0), 0);
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
        l0->SetSubVector(ess_tdof_list, 0.0);


        // **************************************************************************************
        //                                2. NP1 方程 Residual
        // **************************************************************************************
        delete l1;
        l1 = new ParLinearForm(fes);
        l1->Update(fes, rhs_k->GetBlock(1), 0);
        // b1: (f1, v1)
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();

        buildg1_();
        buildh1(c1);
        buildm1_dta1(phi);
        g1_->AddMult(*c1, *l1, -1.0);        // l1 = l1 - g1 c1
        h1->AddMult(*phi, *l1, -1.0);        // l1 = l1 - g1 c1 - h1 phi
        m1_dta1->AddMult(*dc1dt, *l1, -1.0); // l1 = l1 - g1 c1 - h1 phi - m1_dta1 dc1dt
        l1->SetSubVector(ess_tdof_list, 0.0);


        // **************************************************************************************
        //                                3. NP2 方程 Residual
        // **************************************************************************************
        delete l2;
        l2 = new ParLinearForm(fes);
        l2->Update(fes, rhs_k->GetBlock(2), 0);
        // b2: (f2, v2)
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();

        buildg2_();
        buildh2(c2);
        buildm2_dta2(phi);
        g2_->AddMult(*c2, *l2, -1.0);        // l2 = l2 - g2 c2
        h2->AddMult(*phi, *l2, -1.0);        // l2 = l2 - g2 c2 - h2 phi
        m2_dta2->AddMult(*dc2dt, *l2, -1.0); // l2 = l2 - g2 c2 - h2 phi - m2_dta2 dc2dt
        l2->SetSubVector(ess_tdof_list, 0.0);

        if (hahahaha) {
            cout << "l2 norm of   phi: " <<   phi->Norml2() << endl;
            cout << "l2 norm of dc1dt: " << dc1dt->Norml2() << endl;
            cout << "l2 norm of dc2dt: " << dc2dt->Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt: " << phi_dc1dt_dc2dt.Norml2() << endl;
            cout << "l2 norm of residual: " << residual.Norml2() << '\n' << endl;
        }
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

        {
            cout << "l2 norm of   phi: " << phi->Norml2() << endl;
            cout << "l2 norm of dc1dt: " << dc1dt->Norml2() << endl;
            cout << "l2 norm of dc2dt: " << dc2dt->Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt: " << phi_dc1dt_dc2dt.Norml2() << '\n' << endl;
        }

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
        buildh1_dth1(c1, dc1dt);
        h1_dth1->FormSystemMatrix(null_array, *H1_dtH1);

        buildm1_dta1(phi);
        m1_dta1->FormSystemMatrix(ess_tdof_list, *M1_dtA1);


        // **************************************************************************************
        //                                3. NP2 方程的 Jacobian
        // **************************************************************************************
        buildh2_dth2(c2, dc2dt);
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
    void buildm1_dta1(ParGridFunction* phi) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt, D_K_prod_v_K);

        m1_dta1 = new ParBilinearForm(fes);
        // (c1, v1)
        m1_dta1->AddDomainIntegrator(new MassIntegrator);
        // dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        m1_dta1->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // (c2, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
    void buildm2_dta2(ParGridFunction* phi) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt, D_Cl_prod_v_Cl);

        m2_dta2 = new ParBilinearForm(fes);
        // (c2, v2)
        m2_dta2->AddDomainIntegrator(new MassIntegrator);
        // dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given phi
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        m2_dta2->AddDomainIntegrator(new GradConvection_BLFIntegrator(*phi, &dt_D2_z2));

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
    void buildh1(const ParGridFunction* c1) const
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
    void buildh1_dth1(const ParGridFunction* c1, ParGridFunction* dc1dt) const
    {
        if (h1_dth1 != NULL) { delete h1_dth1; }

        GridFunctionCoefficient c1_coeff(c1), dc1dt_coeff(dc1dt);
        ProductCoefficient D1_z1_c1_coeff(D_K_prod_v_K, c1_coeff);
        ProductCoefficient dt_dc1dt_coeff(dt, dc1dt_coeff);

        h1_dth1 = new ParBilinearForm(fes);
        // D1 (z1 c1 grad(dphi), grad(v1)), given c1
        h1_dth1->AddDomainIntegrator(new DiffusionIntegrator(D1_z1_c1_coeff));
        // D1 (z1 dt dc1dt grad(dphi), grad(v1)), given dc1dt
        h1_dth1->AddDomainIntegrator(new DiffusionIntegrator(dt_dc1dt_coeff));

        h1_dth1->Assemble(skip_zero_entries);
    }

    // D2 (z2 c2 grad(dphi), grad(v2)), given c2
    void buildh2(const ParGridFunction* c2) const
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
    void buildh2_dth2(const ParGridFunction* c2, ParGridFunction* dc2dt) const
    {
        if (h2_dth2 != NULL) { delete h2_dth2; }

        GridFunctionCoefficient c2_coeff(c2), dc2dt_coeff(dc2dt);
        ProductCoefficient D2_z2_c2_coeff(D_Cl_prod_v_Cl, c2_coeff);
        ProductCoefficient dt_dc2dt_coeff(dt, dc2dt_coeff);

        h2_dth2 = new ParBilinearForm(fes);
        // D2 (z2 c2 grad(dphi), grad(v2)), given c2
        h2_dth2->AddDomainIntegrator(new DiffusionIntegrator(D2_z2_c2_coeff));
        // D2 (z2 dt dc2dt grad(dphi), grad(v2)), given dc2dt
        h2_dth2->AddDomainIntegrator(new DiffusionIntegrator(dt_dc2dt_coeff));

        h2_dth2->Assemble(skip_zero_entries);
    }

};
class PNP_Box_Newton_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;

    BlockVector* phi_dc1dt_dc2dt;
    PNP_Box_Newton_CG_Operator* oper;
    PetscNonlinearSolver* newton_solver;
    PetscPreconditionerFactory *jac_factory;

    ParGridFunction old_phi, old_c1, old_c2; // 上一个时间步的解(已知)
    ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi_dt

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
        jac_factory   = new PreconditionerFactory(*oper, prec_type);
        newton_solver = new PetscNonlinearSolver(fes->GetComm(), *oper, "newton_");
        newton_solver->SetPreconditionerFactory(jac_factory);
        newton_solver->iterative_mode = true;
    }
    ~PNP_Box_Newton_CG_TimeDependent()
    {
        delete oper;
        delete newton_solver;
        delete jac_factory;
        delete phi_dc1dt_dc2dt;
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
        dphic1c2_dt = 0.0;
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);

        phi_exact  .SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);
        old_phi.ProjectBdrCoefficient(  phi_exact, ess_bdr); // 设定解的边界条件
        dc1dt  .ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
        dc2dt  .ProjectBdrCoefficient(dc2dt_exact, ess_bdr);
        old_phi.SetTrueVector();
        dc1dt  .SetTrueVector();
        dc2dt  .SetTrueVector();

        // !!!引用 phi, dc1dt, dc2dt 的 TrueVector, 使得 phi_dc1dt_dc2dt 所指的内存块就是phi, dc1dt, dc2dt的内存块.
        // 从而在Newton求解器中对 phi_dc1dt_dc2dt 的修改就等同于对phi, dc1dt, dc2dt的修改, 最终达到了更新解的目的.
        phi_dc1dt_dc2dt = new BlockVector(true_offset);
//        phi_dc1dt_dc2dt->MakeRef(old_phi.GetTrueVector(), true_offset[0], true_vsize); // fff 确保指向相同的内存,true_offset[0]为0
//        phi_dc1dt_dc2dt->MakeRef(  dc1dt.GetTrueVector(), true_offset[1], true_vsize);
//        phi_dc1dt_dc2dt->MakeRef(  dc2dt.GetTrueVector(), true_offset[2], true_vsize);
        phi_dc1dt_dc2dt->SetVector(old_phi.GetTrueVector(), true_offset[0]);
        phi_dc1dt_dc2dt->SetVector(  dc1dt.GetTrueVector(), true_offset[1]);
        phi_dc1dt_dc2dt->SetVector(  dc2dt.GetTrueVector(), true_offset[2]);

        oper->UpdateParameters(t, dt, &old_c1, &old_c2); // 传入当前解
        Vector zero_vec;
        if (hahahaha) {
            cout.precision(14);
            cout << "l2 norm of   phi: " <<old_phi.Norml2() << endl;
            cout << "l2 norm of dc1dt: " <<  dc1dt.Norml2() << endl;
            cout << "l2 norm of dc2dt: " <<  dc2dt.Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt: " << phi_dc1dt_dc2dt->Norml2() << '\n' << endl;
        }

        newton_solver->Mult(zero_vec, *phi_dc1dt_dc2dt);
        if (hahahaha) {
            cout.precision(14);
            cout << "l2 norm of   phi: " <<old_phi.Norml2() << endl;
            cout << "l2 norm of dc1dt: " <<  dc1dt.Norml2() << endl;
            cout << "l2 norm of dc2dt: " <<  dc2dt.Norml2() << endl;
            cout << "l2 norm of phi_dc1dt_dc2dt: " << phi_dc1dt_dc2dt->Norml2() << '\n' << endl;
        }
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
        true_offset[0] = 0;
        true_offset[1] = true_vsize;
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
        delete oper; delete ode_solver;
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

            ode_solver->Step(*phic1c2, t, dt_real); // 经过这一步之后 phic1c2(TrueVector, not PrimalVector) 和 t 都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            // 得到下一个时刻t的解, 这个t和执行上述Step()之前的t不一样, 差一个dt_real
            phi_gf->SetFromTrueVector();
            c1_gf->SetFromTrueVector();
            c2_gf->SetFromTrueVector();

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
