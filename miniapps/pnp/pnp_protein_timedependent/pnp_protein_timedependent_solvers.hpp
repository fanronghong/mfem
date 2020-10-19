#ifndef MFEM_PNP_PROTEIN_TIMEDEPENDENT_SOLVERS_HPP
#define MFEM_PNP_PROTEIN_TIMEDEPENDENT_SOLVERS_HPP

#include <fstream>
#include <string>
#include <vector>
#include "petsc.h"
#include "mfem.hpp"
#include "../utils/GradConvection_Integrator.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/petsc_utils.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "../utils/LocalConservation.hpp"
#include "../utils/EAFE_ModifyStiffnessMatrix.hpp"
#include "../utils/SUPG_Integrator.hpp"
#include "./pnp_protein_timedependent.hpp"

using namespace std;
using namespace mfem;

struct Return
{
    ParFiniteElementSpace* fes;
    ParGridFunction* phi3;
    ParGridFunction* c1;
    ParGridFunction* c2;
};

class PNP_Protein_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParMesh* pmesh;
    ParFiniteElementSpace* fes;
    int true_vsize;
    Array<int>  &true_offset, &ess_bdr, &top_bdr, &bottom_bdr, &interface_bdr;
    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list;

    mutable ParBilinearForm *a0, *b1, *b2, *m1_dta1, *a1, *m2_dta2, *a2;
    mutable HypreParMatrix *A0, *M1_dtA1, *M2_dtA2;
    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;

    /* 将电势分解成3部分: 奇异电荷部分phi1, 调和部分phi2, 其余部分phi3,
    * ref: Poisson–Nernst–Planck equations for simulating biomolecular diffusion–reaction processes I: Finite element solutions
    * */
    ParGridFunction *phi1_gf, *phi2_gf;

    VectorCoefficient *grad_phi1, *grad_phi2, *grad_phi1_plus_grad_phi2; // grad(phi1 + phi2)

    StopWatch chrono;
    int num_procs, rank;

public:
    PNP_Protein_Gummel_CG_TimeDependent(ParFiniteElementSpace* fes_, double time, ParGridFunction* phi1_gf_, ParGridFunction* phi2_gf_,
                                        int truevsize, Array<int>& trueoffset, Array<int>& ess_bdr_, Array<int>& top_bdr_,
                                        Array<int>& bottom_bdr_, Array<int>& interface_bdr_)
        : TimeDependentOperator(truevsize*3, time), fes(fes_), phi1_gf(phi1_gf_), phi2_gf(phi2_gf_),
        true_vsize(truevsize), true_offset(trueoffset), ess_bdr(ess_bdr_), top_bdr(top_bdr_), bottom_bdr(bottom_bdr_), interface_bdr(interface_bdr_),
        a0(NULL), b1(NULL), b2(NULL), m1_dta1(NULL), m2_dta2(NULL), a1(NULL), a2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        pmesh = fes->GetParMesh();

        Solve_Phi1();
        Solve_Phi2();

        grad_phi1 = new GradientGridFunctionCoefficient(phi1_gf);
        grad_phi2 = new GradientGridFunctionCoefficient(phi2_gf);
        grad_phi1_plus_grad_phi2 = new VectorSumCoefficient(*grad_phi1, *grad_phi2);

        A0      = new HypreParMatrix;
        M1_dtA1 = new HypreParMatrix;
        M2_dtA2 = new HypreParMatrix;

        temp_x0 = new Vector;
        temp_b0 = new Vector;
        temp_x1 = new Vector;
        temp_b1 = new Vector;
        temp_x2 = new Vector;
        temp_b2 = new Vector;

        // DG has no "DOFs"
        fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
        fes->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);
        fes->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);
        fes->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

    }
    ~PNP_Protein_Gummel_CG_TimeDependent()
    {


        delete grad_phi1; delete grad_phi2; delete grad_phi1_plus_grad_phi2;

        delete a0; delete b1; delete b2;
        delete m1_dta1; delete a1;
        delete m2_dta2; delete a2;

        delete A0; delete M1_dtA1; delete M2_dtA2;
        delete temp_x0; delete temp_b0;
        delete temp_x1; delete temp_b1;
        delete temp_x2; delete temp_b2;

    }

    // 1.求解奇异电荷部分的电势
    void Solve_Phi1()
    {
        phi1_gf->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取

        double norm = phi1_gf->ComputeL2Error(zero);
        if (rank == 0) {
            cout << "L2 norm of phi1: " << norm << endl;
        }

        if (self_debug && strcmp(pqr_file, "./1MAG.pqr") == 0 && strcmp(mesh_file, "./1MAG_2.msh") == 0)
        {
            /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
             * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
             */
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1_gf->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            H1_FECollection*  temp_fec = new H1_FECollection(1, pmesh->Dimension());
            ParFiniteElementSpace* temp_fes = new ParFiniteElementSpace(pmesh, temp_fec, 3);

            GridFunction grad_phi1(temp_fes);
            grad_phi1.ProjectCoefficient(gradG_coeff);

            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
            delete temp_fec;
            delete temp_fes;
        }
    }

    // 2.求解调和方程部分的电势
    void Solve_Phi2()
    {
        // 为了简单, 我们只使用H1空间来计算phi2

        H1_FECollection* fec = new H1_FECollection(p_order, pmesh->Dimension());
        ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, fec);

        Array<int> interface_bdr, interface_ess_tdof_list, Gamma_m;
        {
            int size = pmesh->bdr_attributes.Max();

            Gamma_m.SetSize(size);
            Gamma_m                     = 0;
            Gamma_m[Gamma_m_marker - 1] = 1;

            interface_bdr.SetSize(size);
            interface_bdr                       = 0;
            interface_bdr[interface_marker - 1] = 1;
            fes->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);
        }

        ParBilinearForm blf(fes);
        // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m: protein domain
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        ParLinearForm lf(fes);
        // -<grad(G).n, psi2>_{\Gamma_M}, G is phi1
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m);
        lf.Assemble();

        phi2_gf->ProjectCoefficient(G_coeff);
        phi2_gf->Neg(); // 在interface(\Gamma)上是Dirichlet边界: -phi1

        HypreParMatrix* A = new HypreParMatrix;
        Vector *x = new Vector;
        Vector *b = new Vector;
        blf.FormLinearSystem(interface_ess_tdof_list, *phi2_gf, lf, *A, *x, *b);
        A->EliminateZeroRows(); // 设定所有的0行的主对角元为1

        PetscLinearSolver* solver = new PetscLinearSolver(*A, false, "phi2_");
        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf.RecoverFEMSolution(*x, lf, *phi2_gf);

        double norm = phi2_gf->ComputeL2Error(zero);
        if (rank == 0)
        {
            cout << "L2 norm of phi2: " << norm << endl;
        }

        if (verbose >= 2) {
            if (solver->GetConverged() == 1 && rank == 0)
                cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations()
                     << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi2 solver: failed to converged" << endl;
        }

        if (self_debug && strcmp(pqr_file, "./1MAG.pqr") == 0 && strcmp(mesh_file, "./1MAG_2.msh") == 0)
        {
            /* Only for pqr file "1MAG.pqr" and mesh file "1MAG_2.msh", we do below tests.
                Only need pqr file (to compute singluar electrostatic potential phi1)
                and mesh file, we can compute phi2.
                Data is provided by Zhang Qianru */
            double L2norm = phi2_gf->ComputeL2Error(zero);
            assert(abs(L2norm - 7.2139E+02) < 1); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi2 (no units)" << endl;
        }

        delete solver;
        delete fec;
        delete fes;
        delete A;
        delete x;
        delete b;
    }

    virtual void ImplicitSolve(const double dt, const Vector &phi3c1c2, Vector &dphi3c1c2_dt)
    {
        dphi3c1c2_dt = 0.0;

        Vector* phi3c1c2_ptr = (Vector*) &phi3c1c2;
        ParGridFunction old_phi3, old_c1, old_c2;
        // 求解新的 old_phi3 从而更新 phi3c1c2_ptr, 最终更新 phi3c1c2
        old_phi3.MakeTRef(fes, *phi3c1c2_ptr, true_offset[0]);
        old_c1  .MakeTRef(fes, *phi3c1c2_ptr, true_offset[1]);
        old_c2  .MakeTRef(fes, *phi3c1c2_ptr, true_offset[2]);
        old_phi3.SetFromTrueVector(); // 下面要用到PrimalVector, 而不是TrueVector
        old_c1  .SetFromTrueVector();
        old_c2  .SetFromTrueVector();

        if (0)
        {
            VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
            dc->RegisterField("old_phi3", &old_phi3);
            dc->RegisterField("old_c1", &old_c1);
            dc->RegisterField("old_c2", &old_c2);

            cout << "l2 norm of phi3: " << old_phi3.Norml2() << endl;
            cout << "l2 norm of   c1: " << old_c1.Norml2() << endl;
            cout << "l2 norm of   c2: " << old_c2.Norml2() << endl;

            Visualize(*dc, "old_phi3", "phi3");
            Visualize(*dc, "old_c1", "old_c1");
            Visualize(*dc, "old_c2", "old_c2");

            delete dc;
        }

        ParGridFunction dc1dt, dc2dt; // Poisson方程不是一个ODE, 所以不求dphi3_dt
        // 下面通过求解 dc1dt, dc2dt 从而更新 dphi3c1c2_dt
        dc1dt.MakeTRef(fes, dphi3c1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(fes, dphi3c1c2_dt, true_offset[2]);

        // 变量*_Gummel用于Gummel迭代过程中
        ParGridFunction phi3_Gummel(fes), dc1dt_Gummel(fes), dc2dt_Gummel(fes);
        phi3_Gummel  = 0.0; // 这里暂不设定边界条件, 后面在计算的时候直接设定essential边界条件
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
            // b0: - epsilon_m <grad(phi1 + phi2).n, psi3>_{\Gamma}
            l0->AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator(&neg_epsilon_protein, grad_phi1_plus_grad_phi2, pmesh, protein_marker, water_marker)); // fff
            // omit 0 Neumann bdc on \Gamma_N and \Gamma_M
            l0->Assemble();

            buildb1();
            buildb2();
            b1->AddMult(old_c1, *l0, 1.0);    // l0 = l0 + b1 c1
            b2->AddMult(old_c2, *l0, 1.0);    // l0 = l0 + b1 c1 + b2 c2
            b1->AddMult(dc1dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt
            b2->AddMult(dc2dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + b2 c2 + dt b1 dc1dt + dt b2 dc2dt

            builda0();
            phi3_Gummel.ProjectBdrCoefficient(phi_D_top_coeff, top_bdr); // 设定解的边界条件
            phi3_Gummel.ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
            a0->FormLinearSystem(ess_tdof_list, phi3_Gummel, *l0, *A0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi3_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0->RecoverFEMSolution(*temp_x0, *l0, phi3_Gummel);
            delete l0;
            delete poisson_solver;

            if (0)
            {
                VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
                dc->RegisterField("phi3_Gummel", &phi3_Gummel);

                Visualize(*dc, "phi3_Gummel", "phi3_Gummel_DG");

                delete dc;
            }


            // **************************************************************************************
            //                                2. 计算Gummel迭代相对误差
            // **************************************************************************************
            diff = 0.0;
            diff += phi3_Gummel;
            diff -= old_phi3; // 用到的是old_phi的PrimalVector
            double tol = diff.ComputeL2Error(zero) / phi3_Gummel.ComputeL2Error(zero); // 这里不能把diff设为Vector类型, 如果是Vector类型, 这里计算Norml2()时各个进程得到的值不一样
            old_phi3 = phi3_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
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
            *l1 = 0.0;

            builda1(phi3_Gummel);
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1

            buildm1_dta1(dt, phi3_Gummel);
//            dc1dt_Gummel.ProjectBdrCoefficient(zero, ess_bdr); // 边界条件为0
            m1_dta1->FormLinearSystem(ess_tdof_list, dc1dt_Gummel, *l1, *M1_dtA1, *temp_x1, *temp_b1);
            M1_dtA1->EliminateZeroRows(); // 把0行的主对角元素设为1

            PetscLinearSolver* np1_solver = new PetscLinearSolver(*M1_dtA1, false, "np1_");
            np1_solver->Mult(*temp_b1, *temp_x1);
            m1_dta1->RecoverFEMSolution(*temp_x1, *l1, dc1dt_Gummel); // 更新 dc1dt
            delete l1;
            delete np1_solver;

            if (0)
            {
                VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
                ParGridFunction c1_Gummel(fes);
                c1_Gummel = old_c1;
                c1_Gummel.Add(dt, dc1dt_Gummel);

                dc->RegisterField("c1_Gummel", &c1_Gummel);

                Visualize(*dc, "c1_Gummel", "c1_Gummel");

                delete dc;
            }


            // **************************************************************************************
            //                                4. 求解 NP2
            // **************************************************************************************
            ParLinearForm *l2 = new ParLinearForm(fes);
            *l2 = 0.0;

            builda2(phi3_Gummel);
            a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2

            buildm2_dta2(dt, phi3_Gummel);
//            dc2dt_Gummel.ProjectBdrCoefficient(zero, ess_bdr); // 边界条件为0
            m2_dta2->FormLinearSystem(ess_tdof_list, dc2dt_Gummel, *l2, *M2_dtA2, *temp_x2, *temp_b2);
            M2_dtA2->EliminateZeroRows(); // 把0行的主对角元素设为1

            PetscLinearSolver* np2_solver = new PetscLinearSolver(*M2_dtA2, false, "np2_");
            np2_solver->Mult(*temp_b2, *temp_x2);
            m2_dta2->RecoverFEMSolution(*temp_x2, *l2, dc2dt_Gummel); // 更新 dc2dt
            delete l2;
            delete np2_solver;

            if (0)
            {
                VisItDataCollection* dc = new VisItDataCollection("data collection", fes->GetMesh());
                ParGridFunction c2_Gummel(fes);
                c2_Gummel = old_c1;
                c2_Gummel.Add(dt, dc1dt_Gummel);

                dc->RegisterField("c2_Gummel", &c2_Gummel);

                Visualize(*dc, "c2_Gummel", "c2_Gummel");

                delete dc;
            }
        }

        // 用最终Gummel迭代的解更新要求解的3个未知量
        old_phi3 = phi3_Gummel;
        dc1dt    = dc1dt_Gummel;
        dc2dt    = dc2dt_Gummel;
        // 而我们要返回的TrueVector, 而不是PrimalVector
        old_phi3.SetTrueVector();
        dc1dt   .SetTrueVector();
        dc2dt   .SetTrueVector();
    }

private:
    // epsilon (grad(phi3), grad(psis))_{\Omega}, epsilon: epsilon_s, epsilon_m
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(fes);

        // epsilon (grad(phi3), grad(psis))_{\Omega}, epsilon: epsilon_s, epsilon_m
        a0->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));

        a0->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi3)_{\Omega_s}
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(fes);

        // alpha2 alpha3 z1 (c1, psi3)_{\Omega_s}
        b1->AddDomainIntegrator(new MassIntegrator(water_alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi3)_{\Omega_s}
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(fes);

        // alpha2 alpha3 z2 (c2, psi3)_{\Omega_s}
        b2->AddDomainIntegrator(new MassIntegrator(water_alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
    }

    // (c1, v1)_{\Omega_s} + dt D1 (grad(c1) + z1 c1 grad(phi3), grad(v1))_{\Omega_s}, given dt and phi3
    void buildm1_dta1(double dt, ParGridFunction& phi3) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D1_water);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);

        m1_dta1 = new ParBilinearForm(fes);

        // (c1, v1)_{\Omega_s}
        m1_dta1->AddDomainIntegrator(new MassIntegrator(mark_water_coeff));
        // dt D1 (grad(c1), grad(v1))_{\Omega_s}
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        // dt D1 z1 (c1 grad(phi3), grad(v1))_{\Omega_s}
        m1_dta1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi3, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1) + z1 c1 grad(phi3), grad(v1))_{\Omega_s}, given phi3
    void builda1(ParGridFunction& phi3) const
    {
        if (a1 != NULL) { delete a1; }

        a1 = new ParBilinearForm(fes);

        // D1 (grad(c1), grad(v1))_{\Omega_s}
        a1->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (c1 grad(phi3), grad(v1))_{\Omega_s}
        a1->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi3, &D1_prod_z1_water));

        a1->Assemble(skip_zero_entries);
    }

    // (c2, v2)_{\Omega_s} + dt D2 (grad(c2) + z2 c2 grad(phi3), grad(v2))_{\Omega_s}, given dt and phi3
    void buildm2_dta2(double dt, ParGridFunction& phi3) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D2_water);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);

        m2_dta2 = new ParBilinearForm(fes);

        // (c2, v2)_{\Omega_s}
        m2_dta2->AddDomainIntegrator(new MassIntegrator(mark_water_coeff));
        // dt D2 (grad(c2), grad(v2))_{\Omega_s}
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        // dt D2 z2 (c2 grad(phi3), grad(v2))_{\Omega_s}
        m2_dta2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi3, &dt_D2_z2));

        m2_dta2->Assemble(skip_zero_entries);
    }

    // D2 (grad(c2) + z2 c2 grad(phi3), grad(v2))_{\Omega_s}, given phi3
    void builda2(ParGridFunction& phi3) const
    {
        if (a2 != NULL) { delete a2; }

        a2 = new ParBilinearForm(fes);

        // D2 (grad(c2), grad(v2))_{\Omega_s}
        a2->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (c2 grad(phi3), grad(v2))_{\Omega_s}
        a2->AddDomainIntegrator(new GradConvection_BLFIntegrator(phi3, &D2_prod_z2_water));

        a2->Assemble(skip_zero_entries);
    }
};


class PNP_Protein_TimeDependent_Solver
{
private:
    ParMesh* pmesh;
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fes;

    BlockVector* phi3c1c2;
    ParGridFunction *phi1_gf, *phi2_gf, *phi3_gf, *c1_gf, *c2_gf;

    double t; // 当前时间
    TimeDependentOperator* oper;
    ODESolver *ode_solver;

    int true_vsize; // 有限元空间维数
    Array<int> true_offset, ess_bdr, top_bdr, bottom_bdr, interface_bdr, Gamma_m;

    int num_procs, rank;
    StopWatch chrono;
    ParaViewDataCollection* pd;
    Return* ret;

public:
    PNP_Protein_TimeDependent_Solver(ParMesh* pmesh_, int ode_solver_type): pmesh(pmesh_), ret(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        t = t_init;

        if (strcmp(Discretize, "cg") == 0)
        {
            fec = new H1_FECollection(p_order, pmesh->Dimension());
        }
        fes = new ParFiniteElementSpace(pmesh, fec);

        {
            int size = pmesh->bdr_attributes.Max();

            ess_bdr.SetSize(size);
            ess_bdr                    = 0;
            ess_bdr[top_marker    - 1] = 1;
            ess_bdr[bottom_marker - 1] = 1;

            top_bdr.SetSize(size);
            top_bdr                 = 0;
            top_bdr[top_marker - 1] = 1;

            bottom_bdr.SetSize(size);
            bottom_bdr                    = 0;
            bottom_bdr[bottom_marker - 1] = 1;

            interface_bdr.SetSize(size);
            interface_bdr                       = 0;
            interface_bdr[interface_marker - 1] = 1;

            Gamma_m.SetSize(size);
            Gamma_m                     = 0;
            Gamma_m[Gamma_m_marker - 1] = 1;
        }

        phi1_gf = new ParGridFunction(fes); *phi1_gf = 0.0;
        phi2_gf = new ParGridFunction(fes); *phi2_gf = 0.0;
        phi3_gf = new ParGridFunction(fes); *phi3_gf = 0.0;
        c1_gf   = new ParGridFunction(fes); *c1_gf   = 0.0;
        c2_gf   = new ParGridFunction(fes); *c2_gf   = 0.0;

        true_vsize = fes->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_vsize;
        true_offset[2] = true_vsize * 2;
        true_offset[3] = true_vsize * 3;

        phi3c1c2 = new BlockVector(true_offset); *phi3c1c2 = 0.0; // TrueVector, not PrimalVector
        phi3_gf->MakeTRef(fes, *phi3c1c2, true_offset[0]);
        c1_gf  ->MakeTRef(fes, *phi3c1c2, true_offset[1]);
        c2_gf  ->MakeTRef(fes, *phi3c1c2, true_offset[2]);

        // 设定初值
        phi3_gf->ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
        phi3_gf->ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
        phi3_gf->SetTrueVector();
        phi3_gf->SetFromTrueVector();

        c1_gf->ProjectBdrCoefficient( c1_D_top_coeff, top_bdr);
        c1_gf->ProjectBdrCoefficient( c1_D_bottom_coeff, bottom_bdr);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_gf->ProjectBdrCoefficient( c2_D_top_coeff, top_bdr);
        c2_gf->ProjectBdrCoefficient( c2_D_bottom_coeff, bottom_bdr);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        if (strcmp(Linearize, "gummel") == 0)
        {
            if (strcmp(Discretize, "cg") == 0)
            {
                oper = new PNP_Protein_Gummel_CG_TimeDependent(fes, t, phi1_gf, phi2_gf, true_vsize, true_offset, ess_bdr, top_bdr, bottom_bdr, interface_bdr);
            }
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
            string paraview_title = string("PNP_Protein_") + Discretize + "_" + Linearize + "_Time_Dependent";
            pd = new ParaViewDataCollection(paraview_title, pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi1", phi1_gf);
            pd->RegisterField("phi2", phi2_gf);
            pd->RegisterField("phi3", phi3_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);

            pd->SetCycle(0); // 第 0 个时间步
            pd->SetTime(t); // 第 0 个时间步所表示的时间
            pd->Save();
        }
    }
    ~PNP_Protein_TimeDependent_Solver()
    {
        delete fec;
        delete fes;
        delete phi1_gf;
        delete phi2_gf;
        delete phi3_gf;
        delete c1_gf;
        delete c2_gf;
        delete phi3c1c2;
        delete oper;
        delete ode_solver;
        if (ret) delete ret;
        if (paraview) delete pd;
    }

    Return* Solve(Array<double>& meshsizes, Array<double>& time_steps)
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

            ode_solver->Step(*phi3c1c2, t, dt_real); // 经过这一步之后 phi3c1c2(TrueVector, not PrimalVector) 和 t 都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            // 得到下一个时刻t的解, 这个t和执行上述Step()之前的t不一样, 差一个dt_real
            phi3_gf->SetFromTrueVector();
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
                double phiL2norm = phi3_gf->ComputeL2Error(zero);
                double  c1L2norm = c1_gf->ComputeL2Error(zero);
                double  c2L2norm = c2_gf->ComputeL2Error(zero);
                if (rank == 0)
                {
                    cout << "Time: " << t << '\n'
                         << "phi L2 norm: " << phiL2norm << '\n'
                         << " c1 L2 norm: " <<  c1L2norm << '\n'
                         << " c2 L2 norm: " <<  c2L2norm << '\n' << endl;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Stop();

        {
            // 计算最后一个时刻的解的范数. 计算误差范数只能是在所有进程上都运行，输出误差范数可以只在root进程
            double phiL2err = phi3_gf->ComputeL2Error(zero);
            double c1L2err = c1_gf->ComputeL2Error(zero);
            double c2L2err = c2_gf->ComputeL2Error(zero);

            if (rank == 0)
            {
                cout << "=========> ";
                cout << Discretize << p_order << ", " << Linearize << ", " << options_src << ", DOFs: " << fes->GlobalTrueVSize() * 3<< ", Cores: " << num_procs << ", "
                     << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" : "wrong type")) << '\n'
                     << mesh_file  << ", mesh size: " << mesh_size << '\n'
                     << "t_init: "<< t_init << ", t_final: " << t_final << ", time step: " << t_stepsize
                     << endl;

                cout << "ODE solver taking " << chrono.RealTime() << " s." << endl;
                cout.precision(14);
                cout << "At final time: " << t << '\n'
                     << "L2 norm of phi3_h: " << phiL2err << '\n'
                     << "L2 norm of  c1_h : " << c1L2err << '\n'
                     << "L2 norm of  c2_h : " << c2L2err << '\n' << endl;
            }
        }

        ret = new Return;
        ret->fes  = fes;
        ret->phi3 = phi3_gf;
        ret->c1   = c1_gf;
        ret->c2   = c2_gf;
    }
};


#endif //MFEM_PNP_PROTEIN_TIMEDEPENDENT_SOLVERS_HPP
