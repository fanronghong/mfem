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


class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;

    mutable ParBilinearForm *a0, *a1, *a2, *b1, *b2, *m, *m1_dta1, *m2_dta2;
    mutable HypreParMatrix *A0, *M, *M1_dtA1, *M2_dtA2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;
    int true_vsize;
    mutable Array<int> true_offset, ess_bdr, ess_tdof_list;
    int num_procs, myid;

public:
    PNP_Box_Gummel_CG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_,
                                    ParFiniteElementSpace* fsp, double time)
            : TimeDependentOperator(3*truesize, time), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_), fes(fsp),
              b1(NULL), b2(NULL), a0(NULL), a1(NULL), m(NULL), a2(NULL), m1_dta1(NULL), m2_dta2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

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

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi_Gummel(fes), dc1dt_Gummel(fes), dc2dt_Gummel(fes);
        phi_Gummel = 0.0; dc1dt_Gummel = 0.0; dc2dt_Gummel = 0.0;
        phi_exact.SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
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
            double tol = diff.ComputeL2Error(zero) / phi_Gummel.ComputeL2Error(zero);
            old_phi = phi_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
            if (myid == 0 && verbose >= 2) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
            }
            if (tol < Gummel_rel_tol) { // Gummel迭代停止
                last_gummel_step = true;
            }


            // **************************************************************************************
            //                                3. 求解 NP1
            // **************************************************************************************
            ParLinearForm *l1 = new ParLinearForm(fes);
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


            // **************************************************************************************
            //                                4. 求解 NP2
            // **************************************************************************************
            ParLinearForm *l2 = new ParLinearForm(fes);
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


class PNP_Box_Gummel_DG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* fes;
    // 用来给DG空间的GridFunction设定边界条件: 如果gf属于DG空间的GridFunction, 则gf.ProjectBdrCoefficient()会出错
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1;

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
    int num_procs, myid;

public:
    PNP_Box_Gummel_DG_TimeDependent(int truesize, Array<int>& offset, Array<int>& ess_bdr_, ParFiniteElementSpace* fes_, double time)
            : TimeDependentOperator(3*truesize, time), true_vsize(truesize), true_offset(offset), ess_bdr(ess_bdr_), fes(fes_),
              a0(NULL), e0(NULL), s0(NULL), p0(NULL), a0_e0_s0_p0(NULL),
              m1(NULL), a1(NULL), e1(NULL), s1(NULL), p1(NULL), m1_dta1_dte1_dts1_dtp1(NULL),
              m2(NULL), a2(NULL), e2(NULL), s2(NULL), p2(NULL), m2_dta2_dte2_dts2_dtp2(NULL),
              b1(NULL), b2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // 在H1空间中存在, 在DG空间中不存在

        h1_fec = new H1_FECollection(p_order, fes->GetParMesh()->Dimension());
        h1 = new ParFiniteElementSpace(fes->GetParMesh(), h1_fec);

        A0_E0_S0_P0 = new HypreParMatrix;
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

        delete h1_fec; delete h1;
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

        ProductCoefficient neg_D_Cl_v_Cl(neg, D_Cl_prod_v_Cl);

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
    // sigma <[c1], {D1 (grad(v1) + z1 v1 grad(phi))}>, given phi
    void builds1(ParGridFunction& phi) const
    {
        if (s1 != NULL) { delete s1; }

        s1 = new ParBilinearForm(fes);
        // sigma <[c1], {D1 z1 v1 grad(phi)}>
        s1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(one, phi));
        s1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(sigma_D_K_v_K, phi), ess_bdr);

        // sigma <[c1], {D1 grad(v1)}>
        s1->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_K_, sigma));
        s1->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_K_, sigma), ess_bdr);

        s1->Assemble(skip_zero_entries);
    }
    // sigma <[c2], {D2 (grad(v2) + z2 v2 grad(phi))}>, given phi
    void builds2(ParGridFunction& phi) const
    {
        if (s2 != NULL) { delete s2; }

        s2 = new ParBilinearForm(fes);
        // sigma <[c2], {D2 z2 v2 grad(phi)}>
        s2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(one, phi));
        s2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(sigma_D_Cl_v_Cl, phi), ess_bdr);

        // sigma <[c2], {D2 grad(v2)}>
        s2->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, sigma));
        s2->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, sigma), ess_bdr);

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
    // kappa <{h^{-1}} [c1], [v1]>
    void buildp1() const
    {
        if (p1 != NULL) { delete p1; }

        p1 = new ParBilinearForm(fes);
        // kappa <{h^{-1}} [c1], [v1]> 对单元内部边界和区域外部边界积分
        p1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(kappa));
        p1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(kappa), ess_bdr);

        p1->Assemble(skip_zero_entries);
    }
    // kappa <{h^{-1}} [c2], [v2]>
    void buildp2() const
    {
        if (p2 != NULL) { delete p2; }

        p2 = new ParBilinearForm(fes);
        // kappa <{h^{-1}} [c2], [v2]> 对单元内部边界和区域外部边界积分
        p2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(kappa));
        p2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(kappa), ess_bdr);

        p2->Assemble(skip_zero_entries);
    }

    void builda0_e0_s0_p0() const
    {
        if (a0_e0_s0_p0 != NULL) { delete a0_e0_s0_p0; }

        a0_e0_s0_p0 = new ParBilinearForm(fes);

        // copy from a0
        a0_e0_s0_p0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        // copy from e0
        a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Edge(epsilon_water));
        a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Edge(epsilon_water), ess_bdr);

        // copy from s0
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma));

            if (symmetry_with_boundary)
            {
                a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(epsilon_water, sigma), ess_bdr);
            }
        }

        // copy from p0
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            a0_e0_s0_p0->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(kappa));

            if (penalty_with_boundary)
            {
                a0_e0_s0_p0->AddBdrFaceIntegrator(new DGDiffusion_Penalty(kappa), ess_bdr);
            }
        }

        // weak Dirichlet boundary condition
        a0_e0_s0_p0->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(epsilon_water), ess_bdr);

        a0_e0_s0_p0->Assemble(skip_zero_entries);
    }

    void buildm1_dta1_dte1_dts1_dtp1(double dt, ParGridFunction& phi) const
    {
        if (m1_dta1_dte1_dts1_dtp1 != NULL) { delete m1_dta1_dte1_dts1_dtp1; }

        ProductCoefficient dt_one(-1.0 * dt, one);
        ProductCoefficient dt_sigma_D_K_v_K(-1.0 * dt, sigma_D_K_v_K);

        m1_dta1_dte1_dts1_dtp1 = new ParBilinearForm(fes);

        // copy from m1
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new MassIntegrator);

        // copy from a1, multiply dt
        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        m1_dta1_dte1_dts1_dtp1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D1_z1));

        // copy from e1, multiply dt
        ProductCoefficient dt_neg_D_K_v_K(dt, neg_D_K_v_K);
        m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Edge(dt_D1));
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Edge(dt_D1), ess_bdr);
        m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_K_v_K, phi));
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_K_v_K, phi), ess_bdr);

        // weak Dirichlet boundary condition
        m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(D_K_), ess_bdr);

        // copy from s1, multiply -dt
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(dt_one, phi));
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_K_, -1.0 * dt * sigma));

            if (symmetry_with_boundary)
            {
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D_K_v_K, phi), ess_bdr);
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_K_, -1.0 * dt * sigma), ess_bdr);
            }
        }

        // copy from p1, multiply -dt
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            m1_dta1_dte1_dts1_dtp1->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0 * dt * kappa));

            if (penalty_with_boundary)
            {
                m1_dta1_dte1_dts1_dtp1->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0 * dt * kappa), ess_bdr);
            }
        }

        m1_dta1_dte1_dts1_dtp1->Assemble(skip_zero_entries);
    }

    void buildm2_dta2_dte2_dts2_dtp2(double dt, ParGridFunction& phi) const
    {
        if (m2_dta2_dte2_dts2_dtp2 != NULL) { delete m2_dta2_dte2_dts2_dtp2; }

        ProductCoefficient dt_one(-1.0 * dt, one);
        ProductCoefficient dt_sigma_D_Cl_v_Cl(-1.0 * dt, sigma_D_Cl_v_Cl);

        m2_dta2_dte2_dts2_dtp2 = new ParBilinearForm(fes);

        // copy from m2
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new MassIntegrator);

        // copy from a2, multiply dt
        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        m2_dta2_dte2_dts2_dtp2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi, &dt_D2_z2));

        // copy from e2, multiply dt
        ProductCoefficient dt_neg_D_Cl_v_Cl(dt, neg_D_Cl_v_Cl);
        m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Edge(dt_D2));
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Edge(dt_D2), ess_bdr);
        m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_Cl_v_Cl, phi));
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator1(dt_neg_D_Cl_v_Cl, phi), ess_bdr);

        // weak Dirichlet boundary condition
        m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGWeakDirichlet_BLFIntegrator(D_Cl_), ess_bdr);

        // copy from s2, multiply -dt
        if (abs(sigma - 0.0) > 1E-10) // 添加对称项
        {
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGEdgeBLFIntegrator2(dt_one, phi));
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, -1.0 * dt * sigma));

            if (symmetry_with_boundary)
            {
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGEdgeBLFIntegrator2(dt_sigma_D_Cl_v_Cl, phi), ess_bdr);
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Symmetry(D_Cl_, -1.0 * dt * sigma), ess_bdr);
            }
        }

        // copy from p2, multiply -dt
        if (abs(kappa - 0.0) > 1E-10) // 添加惩罚项
        {
            m2_dta2_dte2_dts2_dtp2->AddInteriorFaceIntegrator(new DGDiffusion_Penalty(-1.0 * dt * kappa));

            if (penalty_with_boundary)
            {
                m2_dta2_dte2_dts2_dtp2->AddBdrFaceIntegrator(new DGDiffusion_Penalty(-1.0 * dt * kappa), ess_bdr);
            }
        }

        m2_dta2_dte2_dts2_dtp2->Assemble(skip_zero_entries);
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        // 上一个时间步的解(已知)
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(fes, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(fes, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(fes, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1 .SetFromTrueVector();
        old_c2 .SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi_dt
        dc1dt.MakeTRef(fes, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphic1c2_dt, true_offset[2]);

        phi_exact.SetTime(t); // t在ODE里面已经变成下一个时刻了(要求解的时刻)
        c1_exact.SetTime(t);
        c2_exact.SetTime(t);
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);
        f0_analytic.SetTime(t);
        f1_analytic.SetTime(t);
        f2_analytic.SetTime(t);

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi_Gummel(fes), dc1dt_Gummel(fes), dc2dt_Gummel(fes);
        phi_Gummel   = 0.0;
        dc1dt_Gummel = 0.0;
        dc2dt_Gummel = 0.0;

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
            // weak Dirichlet boundary condition
            l0->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(phi_exact, epsilon_water), ess_bdr);
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
            a0_e0_s0_p0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel); // 更新old_phi
            delete l0;
            delete poisson_solver;

            if (visualization && 0)
            {
                VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
                dc->RegisterField("phi_Gummel", &phi_Gummel);

                cout << "L2 norm of phi: " << phi_Gummel.ComputeL2Error(phi_exact) << endl;
//                phi_Gummel.ProjectCoefficient(phi_exact);
                Visualize(*dc, "phi_Gummel", "phi_Gummel_DG");

                delete dc;
                MFEM_ABORT("FFFF");
            }


            // **************************************************************************************
            //                                2. 计算Gummel迭代相对误差
            // **************************************************************************************
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


            // **************************************************************************************
            //                                3. 求解 NP1
            // **************************************************************************************
            ParLinearForm *l1 = new ParLinearForm(fes);
            // b1: (f1, v1)
            l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            // weak Dirichlet boundary condition
            l1->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(c1_exact, D_K_), ess_bdr);
            if (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) // 添加对称项
            {
                // -g1: -sigma <c1_D, D1(grad(v1) + z1 v1 grad(phi)).n>
                l1->AddBdrFaceIntegrator(new DGDirichletLF_Symmetry(c1_exact, D_K_, -1.0 * sigma), ess_bdr);
                l1->AddBdrFaceIntegrator(new DGEdgeLFIntegrator2(&neg_sigma_D_K_v_K, &c1_exact, &old_phi), ess_bdr);
            }
            if (abs(kappa - 0.0) > 1E-10 && penalty_with_boundary) // 添加惩罚项
            {
                // -q1: -kappa <{h^{-1}} c1_D, v1>
                l1->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c1_exact, -1.0 * kappa), ess_bdr);
            }
            l1->Assemble();

            builda1(phi_Gummel);
            builde1(phi_Gummel);
            builds1(phi_Gummel);
            buildp1();
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1
            e1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1 - e1 c1
            s1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1
            p1->AddMult(old_c1, *l1, 1.0);  // l1 = l1 - a1 c1 - e1 c1 + s1 c1 + p1 c1

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
            l2->AddBdrFaceIntegrator(new DGWeakDirichlet_LFIntegrator(c2_exact, D_Cl_), ess_bdr);
            if (abs(sigma - 0.0) > 1E-10 && symmetry_with_boundary) // 添加对称项
            {
                // -g2: -sigma <c2_D, D2(grad(v2) + z2 v2 grad(phi)).n>
                l2->AddBdrFaceIntegrator(new DGDirichletLF_Symmetry(c2_exact, D_Cl_, -1.0 * sigma), ess_bdr);
                l2->AddBdrFaceIntegrator(new DGEdgeLFIntegrator2(&neg_sigma_D_Cl_v_Cl, &c2_exact, &old_phi), ess_bdr);
            }
            if (abs(kappa - 0.0) > 1E-10 && penalty_with_boundary) // 添加惩罚项
            {
                // -q2: -kappa <{h^{-1}} c2_D, v2>
                l2->AddBdrFaceIntegrator(new DGDirichletLF_Penalty(c2_exact, -1.0 * kappa), ess_bdr);
            }
            l2->Assemble();

            builda2(phi_Gummel);
            builde2(phi_Gummel);
            builds2(phi_Gummel);
            buildp2();
            a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2
            e2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2 - e2 c2
            s2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2
            p2->AddMult(old_c2, *l2, 1.0);  // l2 = l2 - a2 c2 - e2 c2 + s2 c2 + p2 c2

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
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_TimeDependent_Solver(ParMesh* pmesh_, int ode_solver_type): pmesh(pmesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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

        phic1c2 = new BlockVector(true_offset); *phic1c2 = 0.0;
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
            MFEM_ABORT("Not support linearization");
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
            pd = new ParaViewDataCollection("PNP_DG_Gummel_Time_Dependent", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);
        }
    }
    ~PNP_Box_TimeDependent_Solver()
    {
        delete fec;
        delete fes;
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
            if (verbose >= 1)
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
                cout << "\n=========> ";
                cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine: " << refine_times << ", mesh size: " << mesh_size << '\n'
                     << options_src << ", DOFs: " << fes->GlobalTrueVSize() * 3<< ", Cores: " << num_procs << ", "
                     << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" \
                                                                       : "wrong type")) << '\n'
                     << "t_init: "<< t_init << ", t_final: " << t_final << ", time step: " << t_stepsize
                     << endl;

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
