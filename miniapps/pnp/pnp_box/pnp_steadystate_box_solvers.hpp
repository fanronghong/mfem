#ifndef _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_
#define _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_

#include "../utils/EAFE_ModifyStiffnessMatrix.hpp"
#include "../utils/GradConvection_Integrator.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/python_utils.hpp"
#include "../utils/SUPG_Integrator.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "../utils/DGSelfBdrFaceIntegrator.hpp"
#include "pnp_steadystate_box.hpp"


class PNP_CG_Gummel_Solver
{
private:
    Mesh& mesh;
    FiniteElementCollection* fec;
    FiniteElementSpace* fsp;
    GridFunction *phi, *c1, *c2;       // FE解.
    GridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解
#ifndef PhysicalModel
    GridFunction *phi_exact_gf, *c1_exact_gf, *c2_exact_gf; // use to set Neumann bdc
#endif

    VisItDataCollection* dc;
    Array<int> ess_tdof_list, top_tdof_list, bottom_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet
    Array<int> Neumann_attr, Dirichlet_attr;

    StopWatch chrono;

public:
    PNP_CG_Gummel_Solver(Mesh& mesh_) : mesh(mesh_)
    {
        fec = new H1_FECollection(p_order, mesh.Dimension());
        fsp = new FiniteElementSpace(&mesh, fec);

        phi_n = new GridFunction(fsp); // Gummel 迭代当前解
        c1_n  = new GridFunction(fsp);
        c2_n  = new GridFunction(fsp);

        phi = new GridFunction(fsp); // Gummel 迭代下一步解
        c1  = new GridFunction(fsp);
        c2  = new GridFunction(fsp);

        int bdr_size = fsp->GetMesh()->bdr_attributes.Max();
        Neumann_attr  .SetSize(bdr_size);
        Dirichlet_attr.SetSize(bdr_size);
        {
            Neumann_attr = 0;
            Neumann_attr[front_attr - 1] = 1;
            Neumann_attr[back_attr  - 1] = 1;
            Neumann_attr[left_attr  - 1] = 1;
            Neumann_attr[right_attr - 1] = 1;

            Dirichlet_attr = 0;
            Dirichlet_attr[top_attr    - 1] = 1;
            Dirichlet_attr[bottom_attr - 1] = 1;
        }
        fsp->GetEssentialTrueDofs(Dirichlet_attr, ess_tdof_list);

        *phi_n = 0.0; // Gummel 迭代当前解满足Dirichlet边界条件
        *c1_n  = 0.0;
        *c2_n  = 0.0;
        *phi   = 0.0;
        *c1    = 0.0;
        *c2    = 0.0;
#if defined(PhysicalModel)
        Array<int> top_bdr(bdr_size);
        top_bdr               = 0;
        top_bdr[top_attr - 1] = 1;
        fsp->GetEssentialTrueDofs(top_bdr, top_tdof_list);

        Array<int> bottom_bdr(bdr_size);
        bottom_bdr                  = 0;
        bottom_bdr[bottom_attr - 1] = 1;
        fsp->GetEssentialTrueDofs(bottom_bdr, bottom_tdof_list);

        // essential边界条件
        for (int i=0; i<top_tdof_list.Size(); ++i)
        {
            (*phi)   [top_tdof_list[i]] = phi_top;
            (*c1)    [top_tdof_list[i]] = c1_top;
            (*c2)    [top_tdof_list[i]] = c2_top;
            (*phi_n) [top_tdof_list[i]] = phi_top;
            (*c1_n)  [top_tdof_list[i]] = c1_top;
            (*c2_n)  [top_tdof_list[i]] = c2_top;
        }
        for (int i=0; i<bottom_tdof_list.Size(); ++i)
        {
            (*phi)   [bottom_tdof_list[i]] = phi_bottom;
            (*c1)    [bottom_tdof_list[i]] = c1_bottom;
            (*c2)    [bottom_tdof_list[i]] = c2_bottom;
            (*phi_n) [bottom_tdof_list[i]] = phi_bottom;
            (*c1_n)  [bottom_tdof_list[i]] = c1_bottom;
            (*c2_n)  [bottom_tdof_list[i]] = c2_bottom;
        }
#else
        phi_n->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1_n ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2_n ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);
        phi->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1 ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2 ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);

        phi_exact_gf = new GridFunction(fsp);
        c1_exact_gf  = new GridFunction(fsp);
        c2_exact_gf  = new GridFunction(fsp);
        phi_exact_gf->ProjectCoefficient(phi_exact);
        c1_exact_gf->ProjectCoefficient(c1_exact);
        c2_exact_gf->ProjectCoefficient(c2_exact);
#endif

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_CG_Gummel_Solver()
    {
        delete fsp, phi, c1, c2, phi_n, c1_n, c2_n, dc, fec;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\n------> Begin Gummel iteration: CG1, box model\n";
        int iter = 0;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson();

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1();
            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter+1 << " times." << endl;
                break;
            }
            iter++;
            cout << endl;
        }
        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

        {
#ifndef PhysicalModel
            double phiL2err = phi->ComputeL2Error(phi_exact);
            double c1L2err = c1->ComputeL2Error(c1_exact);
            double c2L2err = c2->ComputeL2Error(c2_exact);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
#endif
        }

        (*phi) /= alpha1;
        (*c1)  /= alpha3;
        (*c2)  /= alpha3;

        Visualize(*dc, "phi", "phi");
        Visualize(*dc, "c1", "c1");
        Visualize(*dc, "c2", "c2");
//        cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;

        ofstream results("phi_c1_c2_cg.vtk");
        results.precision(14);
        int ref = 0;
        mesh.PrintVTK(results, ref);
        phi->SaveVTK(results, "phi", ref);
        c1->SaveVTK(results, "c1", ref);
        c2->SaveVTK(results, "c2", ref);

        (*phi) *= (alpha1);
        (*c1)  *= (alpha3);
        (*c2)  *= (alpha3);

#ifdef CLOSE
        {
            ShowMesh(mesh, "coarse mesh");
//            (*phi) += (*phi1); //把总的电势全部加到phi上面
//            (*phi) += (*phi2);
            (*phi) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi", "phi (with units, added by phi1 and phi2)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");
            cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);

            (*phi) *= (alpha1);
            (*c1)   *= (alpha3);
            (*c2)   *= (alpha3);

            // for Two-Grid algm
            mesh.UniformRefinement();
            ShowMesh(mesh, "fine mesh: after 1 uniform refinement");
            fsp->Update();
            phi->Update();
            c1->Update();
            c2->Update();
            Visualize(*dc, "phi", "phi: project from coarse mesh to fine mesh");
            Visualize(*dc, "c1", "c1: project from coarse mesh to fine mesh");
            Visualize(*dc, "c2", "c2: project from coarse mesh to fine mesh");
            cout << "solution vector size on fine mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;
        }
#endif
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson()
    {
        GridFunctionCoefficient* c1_n_coeff = new GridFunctionCoefficient(c1_n);
        GridFunctionCoefficient* c2_n_coeff = new GridFunctionCoefficient(c2_n);
#ifndef PhysicalModel
        c1_n->ProjectCoefficient(c1_exact); // for test Poisson convergence rate
        c2_n->ProjectCoefficient(c2_exact);
        phi_n->ProjectCoefficient(phi_exact); // for test
#endif

        BilinearForm *blf = new BilinearForm(fsp);
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        LinearForm *lf(new LinearForm(fsp)); //Poisson方程的右端项
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K , *c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, *c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
#ifndef PhysicalModel // for Physical model, it's zero Neumann bdc
        GradientGridFunctionCoefficient grad_phi_exact_coeff(phi_exact_gf);
        ScalarVectorProductCoefficient epsilon_s_prod_grad_phi(epsilon_water, grad_phi_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(epsilon_s_prod_grad_phi), Neumann_attr);
#endif
        lf->Assemble();

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new MINRESSolver;
            solver->SetAbsTol(phi_solver_atol);
            solver->SetRelTol(phi_solver_rtol);
            solver->SetMaxIter(phi_solver_maxiter);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(phi_solver_printlv);
            solver->SetOperator(A);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi);

        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

//        cout.precision(14);
//        cout << "l2 error norm of |phi_h - phi_e|: " << phi->ComputeL2Error(phi_exact) << endl;
//        MFEM_ABORT("Stop here for testing convergence rate!");

        (*phi_n) *= relax_phi;
        (*phi)   *= 1-relax_phi;
        (*phi)   += (*phi_n); // 利用松弛方法更新phi3
        (*phi_n) /= relax_phi+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

        delete blf;
        delete lf;
        delete solver;
        delete smoother;
        delete c1_n_coeff;
        delete c2_n_coeff;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1()
    {
        LinearForm *lf(new LinearForm(fsp)); //NP1方程的右端项
        *lf = 0.0;
#ifndef PhysicalModel // for PhysicalModel, set zero Neumann bdc
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        GradientGridFunctionCoefficient grad_c1_exact_coeff(c1_exact_gf);
        ScalarVectorProductCoefficient D_K_prod_grad_c1_exact(D_K_, grad_c1_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(D_K_prod_grad_c1_exact), Neumann_attr);
        lf->Assemble();
#endif

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(np1_solver_atol);
            solver->SetRelTol(np1_solver_rtol);
            solver->SetMaxIter(np1_solver_maxiter);
            solver->SetPrintLevel(np1_solver_printlv);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);
        if (solver->GetConverged() == 1)
            cout << "np1 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

        (*phi) /= alpha1; // goon
        (*c1)  /= alpha3;
        (*c2)  /= alpha3;
        Visualize(*dc, "phi", "phi");
        Visualize(*dc, "c1", "c1");
        Visualize(*dc, "c2", "c2");
        ofstream results("phi_c1_c2_cg.vtk");
        results.precision(14);
        int ref = 0;
        mesh.PrintVTK(results, ref);
        phi->SaveVTK(results, "phi", ref);
        c1->SaveVTK(results, "c1", ref);
        c2->SaveVTK(results, "c2", ref);
        (*phi) *= (alpha1);
        (*c1)  *= (alpha3);
        (*c2)  *= (alpha3);
        MFEM_ABORT("stop for visualize");

//        cout.precision(14);
//        cout << "l2 error norm of |c1_h - c1_e|: " << c1->ComputeL2Error(c1_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate!");

        (*c1_n) *= relax_c1;
        (*c1)   *= 1-relax_c1;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax_c1; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver, smoother;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2()
    {
        LinearForm *lf(new LinearForm(fsp)); //NP2方程的右端项
        *lf = 0.0;
#ifndef PhysicalModel // for PhysicalModel, set zero Neumann bdc
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        GradientGridFunctionCoefficient grad_c2_exact_coeff(c2_exact_gf);
        ScalarVectorProductCoefficient D_Cl_prod_grad_c2_exact(D_Cl_, grad_c2_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(D_Cl_prod_grad_c2_exact), Neumann_attr);
        lf->Assemble();
#endif

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(np2_solver_atol);
            solver->SetRelTol(np2_solver_rtol);
            solver->SetMaxIter(np2_solver_maxiter);
            solver->SetPrintLevel(np2_solver_printlv);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);
        if (solver->GetConverged() == 1)
            cout << "np2 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

//        cout.precision(14);
//        cout << "l2 error norm of |c2_h - c2_e|: " << c2->ComputeL2Error(c2_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate!");

        (*c2_n) *= relax_c2;
        (*c2)   *= 1-relax_c2;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax_c2+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver, smoother;
    }
};


class PNP_CG_Gummel_Solver_par
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* fsp;
    ParGridFunction *phi, *c1, *c2;       // FE解.
    ParGridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解
#ifndef PhysicalModel
    ParGridFunction *phi_exact_gf, *c1_exact_gf, *c2_exact_gf; // use to set Neumann bdc
#endif

    VisItDataCollection* dc;
    Array<int> ess_tdof_list, top_tdof_list, bottom_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet
    Array<int> Neumann_attr, Dirichlet_attr;

    StopWatch chrono;
    int num_procs, myid;

public:
    PNP_CG_Gummel_Solver_par(Mesh& mesh_) : mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

        fec = new H1_FECollection(p_order, mesh.Dimension());
        fsp = new ParFiniteElementSpace(pmesh, fec);

        phi_n = new ParGridFunction(fsp); // Gummel 迭代当前解
        c1_n  = new ParGridFunction(fsp);
        c2_n  = new ParGridFunction(fsp);

        phi = new ParGridFunction(fsp); // Gummel 迭代下一步解
        c1  = new ParGridFunction(fsp);
        c2  = new ParGridFunction(fsp);

        int bdr_size = fsp->GetMesh()->bdr_attributes.Max();
        Neumann_attr  .SetSize(bdr_size);
        Dirichlet_attr.SetSize(bdr_size);
        {
            Neumann_attr = 0;
            Neumann_attr[front_attr - 1] = 1;
            Neumann_attr[back_attr  - 1] = 1;
            Neumann_attr[left_attr  - 1] = 1;
            Neumann_attr[right_attr - 1] = 1;

            Dirichlet_attr = 0;
            Dirichlet_attr[top_attr    - 1] = 1;
            Dirichlet_attr[bottom_attr - 1] = 1;
        }
        fsp->GetEssentialTrueDofs(Dirichlet_attr, ess_tdof_list);

        *phi_n = 0.0; // Gummel 迭代当前解满足Dirichlet边界条件
        *c1_n  = 0.0;
        *c2_n  = 0.0;
        *phi   = 0.0;
        *c1    = 0.0;
        *c2    = 0.0;
#if defined(PhysicalModel)
        phi_n->ProjectCoefficient(phi_D_coeff);
        c1_n ->ProjectCoefficient(c1_D_coeff);
        c2_n ->ProjectCoefficient(c2_D_coeff);
        phi_n->SetTrueVector();
        c1_n ->SetTrueVector();
        c2_n ->SetTrueVector();
        phi_n->SetFromTrueVector();
        c1_n ->SetFromTrueVector();
        c2_n ->SetFromTrueVector();

        phi->ProjectCoefficient(phi_D_coeff);
        c1 ->ProjectCoefficient(c1_D_coeff);
        c2 ->ProjectCoefficient(c2_D_coeff);
        phi->SetTrueVector();
        c1 ->SetTrueVector();
        c2 ->SetTrueVector();
        phi->SetFromTrueVector();
        c1 ->SetFromTrueVector();
        c2 ->SetFromTrueVector();

//        Array<int> top_bdr(bdr_size);
//        top_bdr               = 0;
//        top_bdr[top_attr - 1] = 1;
//        fsp->GetEssentialTrueDofs(top_bdr, top_tdof_list);
//
//        Array<int> bottom_bdr(bdr_size);
//        bottom_bdr                  = 0;
//        bottom_bdr[bottom_attr - 1] = 1;
//        fsp->GetEssentialTrueDofs(bottom_bdr, bottom_tdof_list);
//
//        // essential边界条件
//        for (int i=0; i<top_tdof_list.Size(); ++i)
//        {
//            (*phi)   [top_tdof_list[i]] = phi_top;
//            (*c1)    [top_tdof_list[i]] = c1_top;
//            (*c2)    [top_tdof_list[i]] = c2_top;
//            (*phi_n) [top_tdof_list[i]] = phi_top;
//            (*c1_n)  [top_tdof_list[i]] = c1_top;
//            (*c2_n)  [top_tdof_list[i]] = c2_top;
//        }
//        for (int i=0; i<bottom_tdof_list.Size(); ++i)
//        {
//            (*phi)   [bottom_tdof_list[i]] = phi_bottom;
//            (*c1)    [bottom_tdof_list[i]] = c1_bottom;
//            (*c2)    [bottom_tdof_list[i]] = c2_bottom;
//            (*phi_n) [bottom_tdof_list[i]] = phi_bottom;
//            (*c1_n)  [bottom_tdof_list[i]] = c1_bottom;
//            (*c2_n)  [bottom_tdof_list[i]] = c2_bottom;
//        }
#else
        phi_n->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1_n ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2_n ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);
        phi->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1 ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2 ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);

        phi_exact_gf = new GridFunction(fsp);
        c1_exact_gf  = new GridFunction(fsp);
        c2_exact_gf  = new GridFunction(fsp);
        phi_exact_gf->ProjectCoefficient(phi_exact);
        c1_exact_gf->ProjectCoefficient(c1_exact);
        c2_exact_gf->ProjectCoefficient(c2_exact);
#endif

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_CG_Gummel_Solver_par()
    {
        delete fsp, phi, c1, c2, phi_n, c1_n, c2_n, dc, fec;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\n------> Begin Gummel iteration: CG1, box model\n";
        int iter = 0;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson();

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1();
            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter+1 << " times." << endl;
                break;
            }
            iter++;
            cout << endl;
        }
        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

#ifndef PhysicalModel
        {
            double phiL2err = phi->ComputeL2Error(phi_exact);
            double c1L2err = c1->ComputeL2Error(c1_exact);
            double c2L2err = c2->ComputeL2Error(c2_exact);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
        }
#endif

        cout << "L2 norm of phi: " << phi->ComputeL2Error(zero) << '\n'
             << "L2 norm of c1 : " << c1->ComputeL2Error(zero) << '\n'
             << "L2 norm of c2 : " << c2->ComputeL2Error(zero) << endl;

        (*phi) /= alpha1;
        (*c1)  /= alpha3;
        (*c2)  /= alpha3;

        Visualize(*dc, "phi", "phi");
        Visualize(*dc, "c1", "c1");
        Visualize(*dc, "c2", "c2");
        cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;

        ofstream results("phi_c1_c2_CG_Gummel.vtk");
        results.precision(14);
        int ref = 0;
        mesh.PrintVTK(results, ref);
        phi->SaveVTK(results, "phi", ref);
        c1->SaveVTK(results, "c1", ref);
        c2->SaveVTK(results, "c2", ref);

        (*phi) *= (alpha1);
        (*c1)  *= (alpha3);
        (*c2)  *= (alpha3);

#ifdef CLOSE
        {
            ShowMesh(mesh, "coarse mesh");
//            (*phi) += (*phi1); //把总的电势全部加到phi上面
//            (*phi) += (*phi2);
            (*phi) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi", "phi (with units, added by phi1 and phi2)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");
            cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);

            (*phi) *= (alpha1);
            (*c1)   *= (alpha3);
            (*c2)   *= (alpha3);

            // for Two-Grid algm
            mesh.UniformRefinement();
            ShowMesh(mesh, "fine mesh: after 1 uniform refinement");
            fsp->Update();
            phi->Update();
            c1->Update();
            c2->Update();
            Visualize(*dc, "phi", "phi: project from coarse mesh to fine mesh");
            Visualize(*dc, "c1", "c1: project from coarse mesh to fine mesh");
            Visualize(*dc, "c2", "c2: project from coarse mesh to fine mesh");
            cout << "solution vector size on fine mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;
        }
#endif
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson()
    {
        cout << "L2 norm of phi: " << phi->ComputeL2Error(zero) << '\n'
             << "L2 norm of c1 : " << c1->ComputeL2Error(zero) << '\n'
             << "L2 norm of c2 : " << c2->ComputeL2Error(zero) << endl;

        GridFunctionCoefficient* c1_n_coeff = new GridFunctionCoefficient(c1_n);
        GridFunctionCoefficient* c2_n_coeff = new GridFunctionCoefficient(c2_n);
#ifndef PhysicalModel
        c1_n->ProjectCoefficient(c1_exact); // for test Poisson convergence rate
        c2_n->ProjectCoefficient(c2_exact);
        phi_n->ProjectCoefficient(phi_exact); // for test
#endif

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf = new ParLinearForm(fsp); //Poisson方程的右端项
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K , *c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, *c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
#ifndef PhysicalModel // for Physical model, it's zero Neumann bdc
        GradientGridFunctionCoefficient grad_phi_exact_coeff(phi_exact_gf);
        ScalarVectorProductCoefficient epsilon_s_prod_grad_phi(epsilon_water, grad_phi_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(epsilon_s_prod_grad_phi), Neumann_attr);
#endif
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi_");
        solver->SetAbsTol(phi_solver_atol);
        solver->SetRelTol(phi_solver_rtol);
        solver->SetMaxIter(phi_solver_maxiter);
        solver->SetPrintLevel(phi_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *phi);
//        cout << "l2 norm of phi: " << phi->Norml2() << endl;
//        MFEM_ABORT("PNP_CG_Gummel_Solver_par: Stop!");

        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

//        cout.precision(14);
//        cout << "l2 error norm of |phi_h - phi_e|: " << phi->ComputeL2Error(phi_exact) << endl;
//        MFEM_ABORT("Stop here for testing convergence rate!");

        (*phi_n) *= relax_phi;
        (*phi)   *= 1-relax_phi;
        (*phi)   += (*phi_n); // 利用松弛方法更新phi3
        (*phi_n) /= relax_phi+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

        delete blf;
        delete lf;
        delete solver;
        delete c1_n_coeff;
        delete c2_n_coeff;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1()
    {
        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
        *lf = 0.0;
#ifndef PhysicalModel // for PhysicalModel, set zero Neumann bdc
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        GradientGridFunctionCoefficient grad_c1_exact_coeff(c1_exact_gf);
        ScalarVectorProductCoefficient D_K_prod_grad_c1_exact(D_K_, grad_c1_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(D_K_prod_grad_c1_exact), Neumann_attr);
        lf->Assemble();
#endif

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->Assemble(0);
        blf->Finalize(0);

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
        solver->SetAbsTol(np1_solver_atol);
        solver->SetRelTol(np1_solver_rtol);
        solver->SetMaxIter(np1_solver_maxiter);
        solver->SetPrintLevel(np1_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c1);

        if (solver->GetConverged() == 1)
            cout << "np1 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

//        cout.precision(14);
//        cout << "l2 error norm of |c1_h - c1_e|: " << c1->ComputeL2Error(c1_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate!");

        (*c1_n) *= relax_c1;
        (*c1)   *= 1-relax_c1;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax_c1; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2()
    {
        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
        *lf = 0.0;
#ifndef PhysicalModel // for PhysicalModel, set zero Neumann bdc
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        GradientGridFunctionCoefficient grad_c2_exact_coeff(c2_exact_gf);
        ScalarVectorProductCoefficient D_Cl_prod_grad_c2_exact(D_Cl_, grad_c2_exact_coeff);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(D_Cl_prod_grad_c2_exact), Neumann_attr);
        lf->Assemble();
#endif

        ParBilinearForm *blf(new ParBilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->Assemble(0);
        blf->Finalize(0);

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");
        solver->SetAbsTol(np2_solver_atol);
        solver->SetRelTol(np2_solver_rtol);
        solver->SetMaxIter(np2_solver_maxiter);
        solver->SetPrintLevel(np2_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c2);

        if (solver->GetConverged() == 1)
            cout << "np2 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;

//        cout.precision(14);
//        cout << "l2 error norm of |c2_h - c2_e|: " << c2->ComputeL2Error(c2_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate!");

        (*c2_n) *= relax_c2;
        (*c2)   *= 1-relax_c2;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax_c2+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver;
    }
};


void DiffusionTensor_K(const Vector &x, DenseMatrix &K) {
    K(0,0) = D_K;
    K(0,1) = 0;
    K(0,2) = 0;
    K(1,0) = 0;
    K(1,1) = D_K;
    K(1, 2) = 0;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = D_K;
}
void DiffusionTensor_Cl(const Vector &x, DenseMatrix &K) {
    K(0,0) = D_Cl;
    K(0,1) = 0;
    K(0,2) = 0;
    K(1,0) = 0;
    K(1,1) = D_Cl;
    K(1, 2) = 0;
    K(2, 0) = 0;
    K(2, 1) = 0;
    K(2, 2) = D_Cl;
}
void AdvectionVector_K(const Vector &x, Vector &advcoeff) {
    advcoeff[0] = 1;
    advcoeff[1] = 1;
    advcoeff[2] = 1;
}
class PNP_CG_Gummel_EAFE_Solver
{
private:
    Mesh& mesh;
    H1_FECollection* fec;
    FiniteElementSpace* fsp;
    GridFunction *phi, *c1, *c2;       // FE解.
    GridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    Array<int> ess_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet

    StopWatch chrono;

public:
    PNP_CG_Gummel_EAFE_Solver(Mesh& mesh_) : mesh(mesh_)
    {
        fec = new H1_FECollection(p_order, mesh.Dimension());
        fsp = new FiniteElementSpace(&mesh, fec);
        phi   = new GridFunction(fsp);
        c1     = new GridFunction(fsp);
        c2     = new GridFunction(fsp);
        phi_n = new GridFunction(fsp);
        c1_n   = new GridFunction(fsp);
        c2_n   = new GridFunction(fsp);

        if (mesh.bdr_attributes.Size())
        {
            Array<int> bdr_ess(mesh.bdr_attributes.Max());
            bdr_ess = 1;
            fsp->GetEssentialTrueDofs(bdr_ess, ess_tdof_list);
        }

#ifndef PhysicalModel
        phi->ProjectCoefficient(phi_exact);
        c1->ProjectCoefficient(c1_exact);
        c2->ProjectCoefficient(c2_exact);
#endif

        *phi_n = 0.0; // Gummel 迭代初值
        *c1_n = 0.0;
        *c2_n = 0.0;

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_CG_Gummel_EAFE_Solver()
    {
        delete phi, c1, c2, phi_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\n------> (EAFE) Begin Gummel iteration ..." << endl;
        int iter = 0;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson(*c1_n, *c2_n);

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1_EAFE(*phi_n);
            (*c1_n) = (*c1);

            Solve_NP2_EAFE(*phi_n);
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter+1 << " times." << endl;
                break;
            }
            iter++;
        }

        {
            int order_quad = max(9, 2*p_order + 1);
            const IntegrationRule* irs[Geometry::NumGeom];
            for (int i=0; i<Geometry::NumGeom; ++i)
            {
                irs[i] = &(IntRules.Get(i, order_quad));
            }

#ifndef PhysicalModel
            double phiL2err = phi_n->ComputeL2Error(phi_exact, irs);
            double c1L2err = c1_n->ComputeL2Error(c1_exact, irs);
            double c2L2err = c2_n->ComputeL2Error(c2_exact, irs);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
#endif
        }

#ifdef SELF_VERBOSE
        {
//            (*phi) += (*phi1); //把总的电势全部加到phi上面
//            (*phi) += (*phi2);
            (*phi) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi", "phi (with units, added by phi1 and phi2)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);

            (*phi) *= (alpha1);
            (*c1)   *= (alpha3);
            (*c2)   *= (alpha3);
        }
#endif
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson(GridFunction& c1_n_, GridFunction& c2_n_)
    {
        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        LinearForm *lf(new LinearForm(fsp)); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(&c1_n_), c2_n_coeff(&c2_n_);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
        lf->Assemble();

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new MINRESSolver;
            solver->SetAbsTol(1.e-20);
            solver->SetRelTol(1.e-14);
            solver->SetMaxIter(10000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete blf, lf, solver, smoother;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1_EAFE(GridFunction& phi_n_)
    {
        LinearForm *lf(new LinearForm(fsp)); //NP1方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->Assemble();
#endif

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator);
//        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
//        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_K_prod_v_K));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix& A = blf->SpMat();
        Vector& b = *lf;

        EAFE_Modify(mesh, A, DiffusionTensor_K, AdvectionVector_K);

        blf->EliminateVDofs(ess_tdof_list, *c1, *lf);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(1.e-10);
            solver->SetRelTol(1.e-10);
            solver->SetMaxIter(1000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        Vector x(lf->Size());
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np1 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete lf, blf, solver, smoother;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2_EAFE(GridFunction& phi_n_)
    {
        //  -------------------------- 最后求解NP2方程 ----------------------------------
        LinearForm *lf(new LinearForm(fsp)); //NP2方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->Assemble();
#endif

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_Cl_prod_v_Cl));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(1.e-10);
            solver->SetRelTol(1.e-10);
            solver->SetMaxIter(1000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np2 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete lf, blf, solver, smoother;
    }
};


class PNP_CG_Gummel_SUPG_Solver
{
private:
    Mesh& mesh;
    H1_FECollection* fec;
    FiniteElementSpace* fsp;
    GridFunction *phi, *c1, *c2;       // FE解.
    GridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    Array<int> ess_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet

    StopWatch chrono;

public:
    PNP_CG_Gummel_SUPG_Solver(Mesh& mesh_) : mesh(mesh_)
    {
        fec = new H1_FECollection(p_order, mesh.Dimension());
        fsp = new FiniteElementSpace(&mesh, fec);
        phi   = new GridFunction(fsp);
        c1     = new GridFunction(fsp);
        c2     = new GridFunction(fsp);
        phi_n = new GridFunction(fsp);
        c1_n   = new GridFunction(fsp);
        c2_n   = new GridFunction(fsp);

        if (mesh.bdr_attributes.Size())
        {
            Array<int> bdr_ess(mesh.bdr_attributes.Max());
            bdr_ess = 1;
            fsp->GetEssentialTrueDofs(bdr_ess, ess_tdof_list);
        }

#ifndef PhysicalModel
        phi->ProjectCoefficient(phi_exact);
        c1->ProjectCoefficient(c1_exact);
        c2->ProjectCoefficient(c2_exact);
#endif
        *phi_n = 0.0; // Gummel 迭代初值
        *c1_n = 0.0;
        *c2_n = 0.0;

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_CG_Gummel_SUPG_Solver()
    {
        delete phi, c1, c2, phi_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\n------> (SUPG) Begin Gummel iteration ..." << endl;
        int iter = 0;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson(*c1_n, *c2_n);

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1_SUPG(*phi_n);
            (*c1_n) = (*c1);

            Solve_NP2_SUPG(*phi_n);
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter+1 << " times." << endl;
                break;
            }
            iter++;
        }
        if (iter == Gummel_max_iters) cerr << "------> Gummel iteration Failed!!!" << endl;

        {
#ifndef PhysicalModel
            int order_quad = max(9, 2*p_order + 1);
            const IntegrationRule* irs[Geometry::NumGeom];
            for (int i=0; i<Geometry::NumGeom; ++i)
            {
                irs[i] = &(IntRules.Get(i, order_quad));
            }

            double phiL2err = phi_n->ComputeL2Error(phi_exact, irs);
            double c1L2err = c1_n->ComputeL2Error(c1_exact, irs);
            double c2L2err = c2_n->ComputeL2Error(c2_exact, irs);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
#endif
        }

#ifdef SELF_VERBOSE
        {
//            (*phi) += (*phi1); //把总的电势全部加到phi上面
//            (*phi) += (*phi2);
            (*phi) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi", "phi (with units, added by phi1 and phi2)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);

            (*phi) *= (alpha1);
            (*c1)   *= (alpha3);
            (*c2)   *= (alpha3);
        }
#endif
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson(GridFunction& c1_n_, GridFunction& c2_n_)
    {
        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        LinearForm *lf(new LinearForm(fsp)); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(&c1_n_), c2_n_coeff(&c2_n_);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
        lf->Assemble();

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new MINRESSolver;
            solver->SetAbsTol(1.e-20);
            solver->SetRelTol(1.e-14);
            solver->SetMaxIter(10000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete blf, lf, solver, smoother;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1_SUPG(GridFunction& phi_n_)
    {
        GenerateDiffusionMatrices();
        MatrixConstantCoefficient D_K_mat_coeff(D_K_mat);
        GradientGridFunctionCoefficient grad_phi(&phi_n_);
        ScalarVectorProductCoefficient adv_K(D_K_prod_v_K, grad_phi); // real advection: grad_phi * D_K_prod_v_K

        LinearForm *lf(new LinearForm(fsp)); //NP1方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        ProductCoefficient neg_f1(neg, f1_analytic);
        lf->AddDomainIntegrator(new SUPG_LinearFormIntegrator(D_K_mat_coeff, adv_K, one, neg_f1, mesh));
        lf->Assemble();
#endif
        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_K_prod_v_K));
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(D_K_mat_coeff, one, adv_K, one, zero, mesh));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(1.e-10);
            solver->SetRelTol(1.e-10);
            solver->SetMaxIter(1000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np1 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete lf, blf, solver, smoother;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2_SUPG(GridFunction& phi_n_)
    {
        GenerateDiffusionMatrices();
        MatrixConstantCoefficient D_Cl_mat_coeff(D_Cl_mat);
        GradientGridFunctionCoefficient grad_phi(&phi_n_);
        ScalarVectorProductCoefficient adv_Cl(D_Cl_prod_v_Cl, grad_phi); // real advection: grad_phi * D_Cl_prod_v_Cl

        LinearForm *lf(new LinearForm(fsp)); //NP2方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        ProductCoefficient neg_f2(neg, f2_analytic);
        lf->AddDomainIntegrator(new SUPG_LinearFormIntegrator(D_Cl_mat_coeff, adv_Cl, one, neg_f2, mesh));
        lf->Assemble();
#endif

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_Cl_prod_v_Cl));
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(D_Cl_mat_coeff, one, adv_Cl, one, zero, mesh));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, A, x, b);

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(1.e-10);
            solver->SetRelTol(1.e-10);
            solver->SetMaxIter(1000);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(0);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);
#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np2 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete lf, blf, solver, smoother;
    }
};


class PNP_DG_Gummel_Solver
{
private:
    Mesh& mesh;
    DG_FECollection* fec;
    FiniteElementSpace* fsp;
    GridFunction *phi, *c1, *c2;       // FE解.
    GridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    Array<int> ess_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet

    StopWatch chrono;

public:
    PNP_DG_Gummel_Solver(Mesh& mesh_) : mesh(mesh_)
    {
        fec = new DG_FECollection(p_order, mesh.Dimension());
        fsp = new FiniteElementSpace(&mesh, fec);
        
        phi   = new GridFunction(fsp);
        c1     = new GridFunction(fsp);
        c2     = new GridFunction(fsp);
        phi_n = new GridFunction(fsp);
        c1_n   = new GridFunction(fsp);
        c2_n   = new GridFunction(fsp);

        *phi = 0.0; // Gummel 迭代当前解
        *c1 = 0.0;
        *c2 = 0.0;

        *phi_n = 0.0; // Gummel 迭代下一步解
        *c1_n = 0.0;
        *c2_n = 0.0;

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_DG_Gummel_Solver()
    {
        delete phi, c1, c2, phi_n, c1_n, c2_n, dc, fec, fsp;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\n------> Begin Gummel iteration: DG1, box model, serial" << endl;
        int iter = 0;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson(*c1_n, *c2_n);

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1(*phi_n);
            (*c1_n) = (*c1);

            Solve_NP2(*phi_n);
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter+1 << " times." << endl;
                break;
            }
            iter++;
        }
        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

        {
#ifndef PhysicalModel
            double phiL2err = phi_n->ComputeL2Error(phi_exact);
            double c1L2err = c1_n->ComputeL2Error(c1_exact);
            double c2L2err = c2_n->ComputeL2Error(c2_exact);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
#endif
        }

#ifdef CLOSE
        {
            ShowMesh(mesh, "coarse mesh");
//            (*phi) += (*phi1); //把总的电势全部加到phi上面
//            (*phi) += (*phi2);
            (*phi) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi", "phi (with units, added by phi1 and phi2)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");
            cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);

            (*phi) *= (alpha1);
            (*c1)   *= (alpha3);
            (*c2)   *= (alpha3);

            mesh.UniformRefinement();
            ShowMesh(mesh, "fine mesh: after 1 uniform refinement");
            fsp->Update();
            phi->Update();
            c1->Update();
            c2->Update();
            Visualize(*dc, "phi", "phi: project from coarse mesh to fine mesh");
            Visualize(*dc, "c1", "c1: project from coarse mesh to fine mesh");
            Visualize(*dc, "c2", "c2: project from coarse mesh to fine mesh");
            cout << "solution vector size on fine mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;
        }
#endif
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson(GridFunction& c1_n_, GridFunction& c2_n_)
    {
//        c1_n_.ProjectCoefficient(c1_exact); // for test convergence rate
//        c2_n_.ProjectCoefficient(c2_exact);

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa)); // 后面两个参数分别是对称化参数, 惩罚参数
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa));
        blf->Assemble();
        blf->Finalize();
        SparseMatrix& A = blf->SpMat();

        // Poisson方程关于离子浓度的两项
        LinearForm *lf(new LinearForm(fsp)); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(&c1_n_), c2_n_coeff(&c2_n_);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
#ifndef PhysicalModel
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, sigma, kappa)); // 用真解构造Dirichlet边界条件
#endif
        lf->Assemble();

        IterativeSolver* solver;
        Solver* smoother = new GSSmoother(A);
        if (abs(sigma + 1.0) < 1E-10)
            solver = new CGSolver;
        else
            solver = new GMRESSolver;
        solver->SetAbsTol(phi_solver_atol);
        solver->SetRelTol(phi_solver_rtol);
        solver->SetMaxIter(phi_solver_maxiter);
        solver->SetPrintLevel(phi_solver_printlv);
        solver->SetOperator(A);
        solver->SetPreconditioner(*smoother);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*lf, *phi);
        chrono.Stop();

        #ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
        #endif

        delete blf, lf, solver, smoother;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1(GridFunction& phi_n_)
    {
//        phi_n_.ProjectCoefficient(phi_exact); // for test convergence rate

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_K_prod_v_K));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));

        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa));

        ProductCoefficient neg_D_K_v_K(neg, D_K_prod_v_K);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, phi_n_));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, phi_n_));

        ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, phi_n_));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, phi_n_));

        blf->Assemble(0);
        blf->Finalize(0);
        SparseMatrix& A = blf->SpMat();

        LinearForm *lf(new LinearForm(fsp)); //NP1方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_exact, D_K_, sigma, kappa));
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_exact, &phi_n_));
#endif
        lf->Assemble();

        IterativeSolver* solver;
        Solver* smoother = new GSSmoother(A);
        solver = new GMRESSolver;
        solver->SetAbsTol(np1_solver_atol);
        solver->SetRelTol(np1_solver_rtol);
        solver->SetMaxIter(np1_solver_maxiter);
        solver->SetPrintLevel(np1_solver_printlv);
        solver->SetOperator(A);
        solver->SetPreconditioner(*smoother);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*lf, *c1);
        chrono.Stop();

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np1 solver : Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif

        delete lf, blf, solver, smoother;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2(GridFunction& phi_n_)
    {
//        phi_n_.ProjectCoefficient(phi_exact); // for test convergence rate

        BilinearForm *blf(new BilinearForm(fsp));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi_n_, &D_Cl_prod_v_Cl));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));

        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa));

        ProductCoefficient neg_D_Cl_v_Cl(neg, D_Cl_prod_v_Cl);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, phi_n_));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, phi_n_));

        ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, phi_n_));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, phi_n_));

        blf->Assemble(0);
        blf->Finalize(0);
        SparseMatrix& A = blf->SpMat();

        LinearForm *lf(new LinearForm(fsp)); //NP2方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_exact, D_Cl_, sigma, kappa));
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_exact, &phi_n_));
#endif
        lf->Assemble();

        IterativeSolver* solver;
        Solver* smoother = new GSSmoother(A);
        solver = new GMRESSolver;
        solver->SetAbsTol(np2_solver_atol);
        solver->SetRelTol(np2_solver_rtol);
        solver->SetMaxIter(np2_solver_maxiter);
        solver->SetPrintLevel(np2_solver_printlv);
        solver->SetOperator(A);
        solver->SetPreconditioner(*smoother);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*lf, *c2);
        chrono.Stop();

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np2 solver : Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif
        delete lf, blf, solver, smoother;
    }
};


class PNP_DG_Gummel_Solver_par
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    FiniteElementCollection* fec;
    ParFiniteElementSpace* fsp;
    ParGridFunction *phi, *c1, *c2;       // FE 解
    ParGridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    Array<int> Dirichlet;
    Array<int> ess_tdof_list;
    StopWatch chrono;
    int num_procs, myid;

public:
    PNP_DG_Gummel_Solver_par(Mesh& mesh_): mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new DG_FECollection(p_order, mesh.Dimension());
        fsp   = new ParFiniteElementSpace(pmesh, fec);

        phi   = new ParGridFunction(fsp);
        c1    = new ParGridFunction(fsp);
        c2    = new ParGridFunction(fsp);
        phi_n = new ParGridFunction(fsp);
        c1_n  = new ParGridFunction(fsp);
        c2_n  = new ParGridFunction(fsp);

        *phi   = 0.0; phi  ->SetTrueVector(); phi  ->SetFromTrueVector();
        *phi_n = 0.0; phi_n->SetTrueVector(); phi_n->SetFromTrueVector();
        *c1    = 0.0; c1   ->SetTrueVector(); c1   ->SetFromTrueVector();
        *c1_n  = 0.0; c1_n ->SetTrueVector(); c1_n ->SetFromTrueVector();
        *c2    = 0.0; c2   ->SetTrueVector(); c2   ->SetFromTrueVector();
        *c2_n  = 0.0; c2_n ->SetTrueVector(); c2_n ->SetFromTrueVector();

#ifdef PhysicalModel
        Dirichlet.SetSize(fsp->GetMesh()->bdr_attributes.Max());
        Dirichlet = 0;
        Dirichlet[top_attr - 1] = 1;
        Dirichlet[bottom_attr - 1] = 1;
#endif

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi", phi);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);

    }
    ~PNP_DG_Gummel_Solver_par()
    {
        delete dc;

    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        // -------------------- 开始 Gummel 迭代 --------------------
        cout << "\n------> Begin Gummel iteration: DG1, box model, parallel\n";
        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson();

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1();
            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            cout << "======> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                break;
            }
            iter++;
            cout << endl;
        }
        if (iter == Gummel_max_iters)
            cerr << "===> Gummel Not converge!!!" << endl;
        {
#ifndef PhysicalModel
            double phiL2err = phi->ComputeL2Error(phi_exact);
            double c1L2err = c1->ComputeL2Error(c1_exact);
            double c2L2err = c2->ComputeL2Error(c2_exact);

            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);
            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++) {
                totle_size += mesh.GetElementSize(0, 1);
            }
            meshsizes_.Append(totle_size / mesh.GetNE());
#endif
        }

        cout << "L2 norm of phi: " << phi->ComputeL2Error(zero) << '\n'
             << "L2 norm of c1 : " << c1->ComputeL2Error(zero) << '\n'
             << "L2 norm of c2 : " << c2->ComputeL2Error(zero) << endl;
        (*phi) /= alpha1;
        (*c1)  /= alpha3;
        (*c2)  /= alpha3;
        Visualize(*dc, "phi", "phi");
        Visualize(*dc, "c1", "c1");
        Visualize(*dc, "c2", "c2");
        cout << "solution vector size on coarse mesh: phi, " << phi->Size() << "; c1, " << c1->Size() << "; c2, " << c2->Size() << endl;
        ofstream results("phi_c1_c2_DG_Gummel.vtk");
        results.precision(14);
        int ref = 0;
        mesh.PrintVTK(results, ref);
        phi->SaveVTK(results, "phi", ref);
        c1->SaveVTK(results, "c1", ref);
        c2->SaveVTK(results, "c2", ref);
        (*phi) *= (alpha1);
        (*c1)  *= (alpha3);
        (*c2)  *= (alpha3);

#ifdef CLOSE
        {
            (*phi)/= alpha1;
            (*c1) /= alpha3;
            (*c2) /= alpha3;
            Visualize(*dc, "phi", "phi (with units)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1  ->SaveVTK(results, "c1", ref);
            c2  ->SaveVTK(results, "c2", ref);
        }
#endif
    }

private:
    void Solve_Poisson()
    {
        cout << "L2 norm of phi: " << phi->ComputeL2Error(zero) << '\n'
             << "L2 norm of c1 : " << c1->ComputeL2Error(zero) << '\n'
             << "L2 norm of c2 : " << c2->ComputeL2Error(zero) << endl;

#ifndef PhysicalModel
        c1_n->ProjectCoefficient(c1_exact); // for test convergence rate
        c2_n->ProjectCoefficient(c2_exact);
#endif
        ParBilinearForm *blf = new ParBilinearForm(fsp);
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa)); // 后面两个参数分别是对称化参数, 惩罚参数
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa), Dirichlet);
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf = new ParLinearForm(fsp); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(c1_n), c2_n_coeff(c2_n);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
#ifndef PhysicalModel
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, sigma, kappa)); // 用真解构造Dirichlet边界条件
#else
        // zero Neumann bdc and below weak Dirichlet bdc
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_coeff, epsilon_water, sigma, kappa), Dirichlet); // 用真解构造Dirichlet边界条件
#endif
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi_");
        solver->SetAbsTol(phi_solver_atol);
        solver->SetRelTol(phi_solver_rtol);
        solver->SetMaxIter(phi_solver_maxiter);
        solver->SetPrintLevel(phi_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *phi);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "phi solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver: failed to converged" << endl;
#endif
        {
//            (*phi) /= alpha1;
//            (*c1)  /= alpha3;
//            (*c2)  /= alpha3;
//            ofstream results("phi_c1_c2_DG_Gummel.vtk");
//            results.precision(14);
//            int ref = 0;
//            mesh.PrintVTK(results, ref);
//            phi->SaveVTK(results, "phi", ref);
//            c1->SaveVTK(results, "c1", ref);
//            c2->SaveVTK(results, "c2", ref);
//            (*phi) *= (alpha1);
//            (*c1)  *= (alpha3);
//            (*c2)  *= (alpha3);
//            MFEM_ABORT("stop 1111");
        }
        delete blf;
        delete lf;
        delete solver;
        delete A;
        delete x;
        delete b;
    }

    void Solve_NP1()
    {
        ParBilinearForm *blf = new ParBilinearForm(fsp);
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));

        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa), Dirichlet);

        ProductCoefficient neg_D_K_v_K(neg, D_K_prod_v_K);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, *phi_n), Dirichlet);

        ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, *phi_n), Dirichlet);

        blf->Assemble(0);
        blf->Finalize(0);


        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_exact, D_K_, sigma, kappa));
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_exact, phi_n));
#else
        // zero Neumann bdc and below weak Dirichlet bdc
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_D_coeff, D_K_, sigma, kappa), Dirichlet);
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_D_coeff, phi_n), Dirichlet);
#endif
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
        solver->SetAbsTol(np1_solver_atol);
        solver->SetRelTol(np1_solver_rtol);
        solver->SetMaxIter(np1_solver_maxiter);
        solver->SetPrintLevel(np1_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c1);

#ifdef SELF_VERBOSE
//        cout << "            l2 norm of c1: " << c1->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np1 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : failed to converged" << endl;
#endif
#ifdef CLOSE
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete blf;
        delete lf;
        delete solver;
        delete A;
        delete x;
        delete b;
    }

    void Solve_NP2()
    {
        ParBilinearForm *blf = new ParBilinearForm(fsp);
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));

        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa), Dirichlet);

        ProductCoefficient neg_D_Cl_v_Cl(neg, D_Cl_prod_v_Cl);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, *phi_n), Dirichlet);

        ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi_n), Dirichlet);

        blf->Assemble(0);
        blf->Finalize(0);


        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
#ifndef PhysicalModel
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_exact, D_Cl_, sigma, kappa));
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_exact, phi_n));
#else
        // zero Neumann bdc and below weak Dirichlet bdc
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_D_coeff, D_K_, sigma, kappa), Dirichlet);
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_D_coeff, phi_n), Dirichlet);
#endif
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");
        solver->SetAbsTol(np2_solver_atol);
        solver->SetRelTol(np2_solver_rtol);
        solver->SetMaxIter(np2_solver_maxiter);
        solver->SetPrintLevel(np2_solver_printlv);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c2);

#ifdef SELF_VERBOSE
//        cout << "            l2 norm of c2: " << c2->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np2 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : failed to converged" << endl;
#endif
#ifdef CLOSE
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete blf;
        delete lf;
        delete solver;
        delete A;
        delete x;
        delete b;
    }
};


#endif
