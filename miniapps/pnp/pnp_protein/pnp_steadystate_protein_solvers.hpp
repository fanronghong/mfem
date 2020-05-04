//
// Created by fan on 2020/3/20.
//

#ifndef _PNP_GUMMEL_SOLVER_HPP_
#define _PNP_GUMMEL_SOLVER_HPP_

#include <fstream>
#include "petsc.h"
#include "mfem.hpp"
#include "../utils/GradConvection_Integrator.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/petsc_utils.hpp"
using namespace std;
using namespace mfem;

class PNP_Gummel_Solver
{
private:
    Mesh& mesh;
    H1_FECollection* h1_fec;
    FiniteElementSpace* h1_space;
    /* 将电势分解成3部分: 奇异电荷部分phi1, 调和部分phi2, 其余部分phi3,
    * ref: Poisson–Nernst–Planck equations for simulating biomolecular diffusion–reaction processes I: Finite element solutions
    * */
    GridFunction *phi1, *phi2;
    GridFunction *phi3, *c1, *c2;       // FE 解
    GridFunction *phi3_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    // protein_dofs和water_dofs里面不包含interface_ess_tdof_list
    Array<int> interface_ess_tdof_list, ess_tdof_list, protein_dofs, water_dofs;
    Array<int> top_ess_tdof_list, bottom_ess_tdof_list;

    StopWatch chrono;

public:
    PNP_Gummel_Solver(Mesh& mesh_) : mesh(mesh_)
    {
        h1_fec = new H1_FECollection(p_order, mesh.Dimension());
        h1_space = new FiniteElementSpace(&mesh, h1_fec);

        Array<int> top_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        Array<int> bottom_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        Array<int> ess_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        ess_bdr                    = 0;
        ess_bdr[top_marker - 1]    = 1;
        ess_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        Array<int> interface_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        interface_bdr = 0;
        interface_bdr[interface_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

        for (int i=0; i<h1_space->GetNE(); ++i)
        {
            Element* el = mesh.GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_space->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                h1_space->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        protein_dofs.Sort(); protein_dofs.Unique();
        water_dofs.Sort(); water_dofs.Unique();
        for (int i=0; i<interface_ess_tdof_list.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_ess_tdof_list[i]);
            water_dofs.DeleteFirst(interface_ess_tdof_list[i]);
        }

        phi1   = new GridFunction(h1_space);
        phi2   = new GridFunction(h1_space);
        phi3   = new GridFunction(h1_space);
        c1     = new GridFunction(h1_space);
        c2     = new GridFunction(h1_space);
        phi3_n = new GridFunction(h1_space);
        c1_n   = new GridFunction(h1_space);
        c2_n   = new GridFunction(h1_space);

        *phi3 = 0.0;
        *c1   = 0.0;
        *c2   = 0.0;
        for (int i=0; i<top_ess_tdof_list.Size(); ++i) // essential边界条件
        {
            (*phi3)  [top_ess_tdof_list[i]] = phi_top;
            (*c1)    [top_ess_tdof_list[i]] = c1_top;
            (*c2)    [top_ess_tdof_list[i]] = c2_top;
            (*phi3_n)[top_ess_tdof_list[i]] = phi_top;
            (*c1_n)  [top_ess_tdof_list[i]] = c1_top;
            (*c2_n)  [top_ess_tdof_list[i]] = c2_top;
        }
        for (int i=0; i<bottom_ess_tdof_list.Size(); ++i)
        {
            (*phi3)  [bottom_ess_tdof_list[i]] = phi_bottom;
            (*c1)    [bottom_ess_tdof_list[i]] = c1_bottom;
            (*c2)    [bottom_ess_tdof_list[i]] = c2_bottom;
            (*phi3_n)[bottom_ess_tdof_list[i]] = phi_bottom;
            (*c1_n)  [bottom_ess_tdof_list[i]] = c1_bottom;
            (*c2_n)  [bottom_ess_tdof_list[i]] = c2_bottom;
        }

        dc = new VisItDataCollection("data collection", &mesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3", phi3);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_Gummel_Solver()
    {
        delete phi1, phi2, phi3, c1, c2, phi3_n, c1_n, c2_n, dc;
        delete h1_space, h1_fec;
    }

    // 把下面的5个求解过程串联起来
    void Solve()
    {
#ifdef SELF_DEBUG
        cout << "alpha1: " << alpha1 << endl;
        cout << "alpha2: " << alpha2 << endl;
        cout << "alpha3: " << alpha3 << endl;
        cout << "alpha2 * alpha3: " << alpha2 * alpha3 << endl;
        cout << "(no units) phi top    Dirichlet boundary: " << phi_top << endl;
        cout << "(no units) phi bottom Dirichlet boundary: " << phi_bottom << endl;
        cout << "(no units) c1  top    Dirichlet boundary: " << c1_top << endl;
        cout << "(no units) c1  bottom Dirichlet boundary: " << c1_bottom << endl;
        cout << "(no units) c2  top    Dirichlet boundary: " << c2_top << endl;
        cout << "(no units) c2  bottom Dirichlet boundary: " << c2_bottom << endl;
#endif
        Solve_Singular();
        Solve_Harmonic();

        // -------------------- 开始 Gummel 迭代 --------------------
        cout << "\n\n---------------------- Begin Gummel iteration ----------------------\n" << endl;
        GridFunctionCoefficient phi3_n_coeff(phi3_n), c1_n_coeff(c1_n), c2_n_coeff(c2_n);
        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson(*c1_n, *c2_n);

            Vector diff(h1_space->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi3);
            diff -= (*phi3_n); // 不能把上述2步合并成1步: diff = (*phi3) - (*phi3_n)fff
            double tol = diff.Norml2() / phi3->Norml2(); // 相对误差
            (*phi3_n) = (*phi3);

            Solve_NP1(*phi3_n);
            (*c1_n) = (*c1);

            Solve_NP2(*phi3_n);
            (*c2_n) = (*c2);

            cout << "======> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "===> Gummel iteration converge!!!" << endl;
#ifdef SELF_VERBOSE
                cout << "l2 norm of phi3: " << phi3->Norml2() << endl;
                cout << "l2 norm of   c1: " << c1->Norml2() << endl;
                cout << "l2 norm of   c2: " << c2->Norml2() << '\n' << endl;
#endif
                break;
            }
#ifdef SELF_VERBOSE
            cout << "l2 norm of phi3: " << phi3->Norml2() << endl;
            cout << "l2 norm of   c1: " << c1->Norml2() << endl;
            cout << "l2 norm of   c2: " << c2->Norml2() << endl;
#endif
            iter++;
            cout << endl;
        }
        if (iter == Gummel_max_iters) cerr << "===> Gummel Not converge!!!" << endl;

#ifdef SELF_VERBOSE
        {
            (*phi3) += (*phi1); //把总的电势全部加到phi3上面
            (*phi3) += (*phi2);
            (*phi3) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi3", "phi3 (with units)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi3->SaveVTK(results, "phi", ref);
            c1  ->SaveVTK(results, "c1", ref);
            c2  ->SaveVTK(results, "c2", ref);
        }
#endif
    }

private:
    // 1.求解奇异电荷部分的电势
    void Solve_Singular()
    {
        ifstream phi1_txt_in(phi1_txt); // 从文件读取 phi1
//        if (phi1_txt_in.is_open())
        if (0) // just for test
        {
            phi1->Load(phi1_txt_in);
        }
        else // 文件不存在
        {
            phi1->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取

            ofstream phi1_txt_out(phi1_txt);
            phi1_txt_out.precision(14);
            phi1->Print(phi1_txt_out << phi1->Size() << '\n'); // 首先要写入向量的维数
            phi1_txt_out.close();
        }
#ifdef SELF_DEBUG
        /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
         * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
         */
        if (strcmp(pqr_file, "../data/1MAG.pqr") == 0)
        {
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            FiniteElementSpace h1_vec(&mesh, fsp.FEColl(), 3);
            GridFunction grad_phi1(&h1_vec);
            grad_phi1.ProjectCoefficient(gradG_coeff);
            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
        }
#endif
    }

    // 2.求解调和方程部分的电势
    void Solve_Harmonic()
    {
        BilinearForm blf(h1_space);
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        LinearForm lf(h1_space);
        Array<int> Gamma_m(mesh.bdr_attributes.Max());
        Gamma_m = 0;
        Gamma_m[Gamma_m_marker - 1] = 1;
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m); // Neumann bdc on Gamma_m, take negative below
        lf.Assemble();

        phi2->ProjectCoefficient(G_coeff);
        phi2->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1

        SparseMatrix A;
        Vector x, b;
        blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, A, x, b); //除了ess_tdof_list以外是0的Neumann边界
        for (int i=0; i<water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
        {
            A.EliminateRow(water_dofs[i], Matrix::DIAG_ONE);
#ifdef SELF_DEBUG
            assert(abs(b(water_dofs[i])) < 1E-10);
#endif
        }

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new MINRESSolver;
            solver->SetAbsTol(harmonic_atol);
            solver->SetRelTol(harmonic_rtol);
            solver->SetMaxIter(harmonic_maxiter);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(harmonic_printlvl);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf.RecoverFEMSolution(x, lf, *phi2);
#ifdef SELF_VERBOSE
        cout << "\nl2 norm of phi2: " << phi2->Norml2() << endl;
        if (solver->GetConverged() == 1)
            cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi2 solver: failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        /* Only for pqr file "../data/1MAG.pqr" and mesh file "../data/1MAG_2.msh", we do below tests.
           Only need pqr file (to compute singluar electrostatic potential phi1) and mesh file, we can compute phi2.
           Data is provided by Zhang Qianru */
        if (strcmp(pqr_file, "../data/1MAG.pqr") == 0 && strcmp(mesh_file, "../data/1MAG_2.msh") == 0)
        {
            for (int i=0; i<water_dofs.Size(); i++)
            {
                assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
            }
            for (int i=0; i<interface_ess_tdof_list.Size(); i++)
            {
                assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-10);
            }

            double L2norm = phi2->ComputeL2Error(zero);
            assert(abs(L2norm - 7.2139E+02) < 1); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi2 (no units)" << endl;
        }
#endif
        delete solver, smoother;
    }

    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson(GridFunction& c1_n_, GridFunction& c2_n_)
    {
        BilinearForm *blf(new BilinearForm(h1_space));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water_mark));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_protein_mark));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        LinearForm *lf(new LinearForm(h1_space)); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(&c1_n_), c2_n_coeff(&c2_n_);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        ProductCoefficient lf1(rhs1, mark_water_coeff);
        ProductCoefficient lf2(rhs2, mark_water_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf2));
        lf->Assemble();

        // Poisson方程的奇异项导出的interface部分
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        SelfDefined_LinearForm interface_term(h1_space);
        interface_term.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(h1_space, grad_phi1_plus_grad_phi2,
                                                                                          protein_marker, water_marker));
        interface_term.SelfDefined_Assemble();
        interface_term *= protein_rel_permittivity;
        (*lf) += interface_term; //界面项要移到方程的右端

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi3, *lf, A, x, b); // ess_tdof_list include: top, bottom

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new MINRESSolver;
            solver->SetAbsTol(phi3_atol);
            solver->SetRelTol(phi3_rtol);
            solver->SetMaxIter(phi3_maxiter);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(phi3_printlvl);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi3);
        (*phi3_n) *= relax_phi;
        (*phi3)   *= 1-relax_phi;
        (*phi3)   += (*phi3_n); // 利用松弛方法更新phi3
        (*phi3_n) /= relax_phi+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
//        cout << "            l2 norm of phi3: " << phi3->Norml2() << endl;
        if (solver->GetConverged() == 1)
            cout << "phi3 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi3 solver: failed to converged" << endl;
#endif

        delete blf, lf, solver, smoother;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1(GridFunction& phi3_n_)
    {
        LinearForm *lf(new LinearForm(h1_space)); //NP1方程的右端项
        *lf = 0.0;

        BilinearForm *blf(new BilinearForm(h1_space));
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi3_n_, &D1_prod_z1_water));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, A, x, b);
        for (int i=0; i<protein_dofs.Size(); ++i)
        {
            A.EliminateRow(protein_dofs[i], Matrix::DIAG_ONE);
#ifdef SELF_DEBUG
            assert(abs(b(protein_dofs[i])) < 1E-10);
#endif
        }

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(np1_atol);
            solver->SetRelTol(np1_rtol);
            solver->SetMaxIter(np1_maxiter);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(np1_printlvl);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);
        (*c1_n) *= relax_c1;
        (*c1)   *= 1-relax_c1;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax_c1; // 还原c1_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
//        cout << "            l2 norm of c1: " << c1->Norml2() << endl;
        if (solver->GetConverged() == 1)
            cout << "np1 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete lf, blf, solver, smoother;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2(GridFunction& phi3_n_)
    {
        LinearForm *lf(new LinearForm(h1_space)); //NP2方程的右端项
        *lf = 0.0;

        BilinearForm *blf(new BilinearForm(h1_space));
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi3_n_, &D2_prod_z2_water));
        blf->Assemble(0);
        blf->Finalize(0);

        SparseMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, A, x, b);
        for (int i=0; i<protein_dofs.Size(); ++i)
        {
            A.EliminateRow(protein_dofs[i], Matrix::DIAG_ONE);
#ifdef SELF_DEBUG
            assert(abs(b(protein_dofs[i])) < 1E-10);
#endif
        }

        IterativeSolver* solver;
        Solver* smoother;
        {
            smoother = new GSSmoother(A);
            solver = new GMRESSolver;
            solver->SetAbsTol(np2_atol);
            solver->SetRelTol(np2_rtol);
            solver->SetMaxIter(np2_maxiter);
            solver->SetOperator(A);
            solver->SetPreconditioner(*smoother);
            solver->SetPrintLevel(np2_printlvl);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);
        (*c2_n) *= relax_c2;
        (*c2)   *= 1-relax_c2;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax_c2+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
//        cout << "            l2 norm of c2: " << c2->Norml2() << endl;
        if (solver->GetConverged() == 1)
            cout << "np2 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete lf, blf, solver, smoother;
    }
};


class PNP_Gummel_Solver_par
{
private:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_space;
    /* 将电势分解成3部分: 奇异电荷部分phi1, 调和部分phi2, 其余部分phi3,
    * ref: Poisson–Nernst–Planck equations for simulating biomolecular diffusion–reaction processes I: Finite element solutions
    * */
    ParGridFunction *phi1, *phi2;
    ParGridFunction *phi3, *c1, *c2;       // FE 解
    ParGridFunction *phi3_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    // protein_dofs和water_dofs里面不包含interface_ess_tdof_list
    Array<int> interface_ess_tdof_list, ess_tdof_list, protein_dofs, water_dofs;
    Array<int> top_ess_tdof_list, bottom_ess_tdof_list;

    StopWatch chrono;
    int num_procs, myid;

public:
    PNP_Gummel_Solver_par(Mesh* mesh_) : mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

        h1_fec = new H1_FECollection(p_order, mesh->Dimension());
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        Array<int> ess_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        ess_bdr                    = 0;
        ess_bdr[top_marker - 1]    = 1;
        ess_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        Array<int> top_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        Array<int> bottom_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        Array<int> interface_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        interface_bdr = 0;
        interface_bdr[interface_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

        for (int i=0; i<h1_space->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_space->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                h1_space->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        protein_dofs.Sort(); protein_dofs.Unique();
        water_dofs.Sort(); water_dofs.Unique();
        for (int i=0; i<interface_ess_tdof_list.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }


        MPI_Comm_size(h1_space->GetComm(), &num_procs);
        MPI_Comm_rank(h1_space->GetComm(), &myid);

        phi1   = new ParGridFunction(h1_space);
        phi2   = new ParGridFunction(h1_space);
        phi3   = new ParGridFunction(h1_space);
        c1     = new ParGridFunction(h1_space);
        c2     = new ParGridFunction(h1_space);
        phi3_n = new ParGridFunction(h1_space);
        c1_n   = new ParGridFunction(h1_space);
        c2_n   = new ParGridFunction(h1_space);

        *phi3 = 0.0;
        *c1   = 0.0;
        *c2   = 0.0;
        for (int i=0; i<top_ess_tdof_list.Size(); ++i) // essential边界条件
        {
            (*phi3)  [top_ess_tdof_list[i]] = phi_top;
            (*c1)    [top_ess_tdof_list[i]] = c1_top;
            (*c2)    [top_ess_tdof_list[i]] = c2_top;
            (*phi3_n)[top_ess_tdof_list[i]] = phi_top;
            (*c1_n)  [top_ess_tdof_list[i]] = c1_top;
            (*c2_n)  [top_ess_tdof_list[i]] = c2_top;
        }
        for (int i=0; i<bottom_ess_tdof_list.Size(); ++i)
        {
            (*phi3)  [bottom_ess_tdof_list[i]] = phi_bottom;
            (*c1)    [bottom_ess_tdof_list[i]] = c1_bottom;
            (*c2)    [bottom_ess_tdof_list[i]] = c2_bottom;
            (*phi3_n)[bottom_ess_tdof_list[i]] = phi_bottom;
            (*c1_n)  [bottom_ess_tdof_list[i]] = c1_bottom;
            (*c2_n)  [bottom_ess_tdof_list[i]] = c2_bottom;
        }

        dc = new VisItDataCollection("data collection", mesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3", phi3);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_Gummel_Solver_par()
    {
        delete phi1, phi2, phi3, c1, c2, phi3_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve()
    {
#ifdef SELF_DEBUG
        cout << "alpha1: " << alpha1 << endl;
        cout << "alpha2: " << alpha2 << endl;
        cout << "alpha3: " << alpha3 << endl;
        cout << "alpha2 * alpha3: " << alpha2 * alpha3 << endl;
        cout << "(no units) phi top    Dirichlet boundary: " << phi_top << endl;
        cout << "(no units) phi bottom Dirichlet boundary: " << phi_bottom << endl;
        cout << "(no units) c1  top    Dirichlet boundary: " << c1_top << endl;
        cout << "(no units) c1  bottom Dirichlet boundary: " << c1_bottom << endl;
        cout << "(no units) c2  top    Dirichlet boundary: " << c2_top << endl;
        cout << "(no units) c2  bottom Dirichlet boundary: " << c2_bottom << endl;
#endif
        Solve_Singular();
        Solve_Harmonic();

        // -------------------- 开始 Gummel 迭代 --------------------
        cout << "\n\n---------------------- CG1, Gummel, protein, parallel ----------------------" << endl;
        GridFunctionCoefficient phi3_n_coeff(phi3_n), c1_n_coeff(c1_n), c2_n_coeff(c2_n);
        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson(*c1_n, *c2_n);

            Vector diff(h1_space->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi3);
            diff -= (*phi3_n); // 不能把上述2步合并成1步: diff = (*phi3) - (*phi3_n)fff
            double tol = diff.Norml2() / phi3->Norml2(); // 相对误差
            (*phi3_n) = (*phi3);

            Solve_NP1(*phi3_n);
            (*c1_n) = (*c1);

            Solve_NP2(*phi3_n);
            (*c2_n) = (*c2);

            cout << "======> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                break;
            }
#ifdef SELF_VERBOSE
            cout << "l2 norm of phi3: " << phi3->Norml2() << endl;
            cout << "l2 norm of   c1: " << c1->Norml2() << endl;
            cout << "l2 norm of   c2: " << c2->Norml2() << endl;
#endif
            iter++;
            cout << endl;
        }

        if (iter == Gummel_max_iters) cerr << "===> Gummel Not converge!!!" << endl;
        else {
            cout << "===> Gummel iteration converge!!!" << endl;
            cout << "l2 norm of phi1: " << phi1->Norml2() << endl;
            cout << "l2 norm of phi2: " << phi2->Norml2() << endl;
            cout << "l2 norm of phi3: " << phi3->Norml2() << endl;
            cout << "l2 norm of   c1: " << c1->Norml2() << endl;
            cout << "l2 norm of   c2: " << c2->Norml2() << '\n' << endl;
        }

#ifdef SELF_VERBOSE
        {
            (*phi3) += (*phi1); //把总的电势全部加到phi3上面
            (*phi3) += (*phi2);
            (*phi3) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi3", "phi3 (with units)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            ofstream results("phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh->PrintVTK(results, ref);
            phi3->SaveVTK(results, "phi", ref);
            c1  ->SaveVTK(results, "c1", ref);
            c2  ->SaveVTK(results, "c2", ref);
        }
#endif
    }

private:
    // 1.求解奇异电荷部分的电势
    void Solve_Singular()
    {
        ifstream phi1_txt_in(phi1_txt); // 从文件读取 phi1
//        if (phi1_txt_in.is_open())
        if (0) // just for test
        {
            phi1->Load(phi1_txt_in);
        }
        else // 文件不存在
        {
            phi1->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取

            ofstream phi1_txt_out(phi1_txt);
            phi1_txt_out.precision(14);
            phi1->Print(phi1_txt_out << phi1->Size() << '\n'); // 首先要写入向量的维数
            phi1_txt_out.close();
        }
#ifdef SELF_DEBUG
        /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
         * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
         */
        if (strcmp(pqr_file, "../data/1MAG.pqr") == 0)
        {
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            FiniteElementSpace h1_vec(&mesh, h1_space->FEColl(), 3);
            GridFunction grad_phi1(&h1_vec);
            grad_phi1.ProjectCoefficient(gradG_coeff);
            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
        }
#endif
    }

    // 2.求解调和方程部分的电势
    void Solve_Harmonic()
    {
        ParBilinearForm blf(h1_space);
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        ParLinearForm lf(h1_space);
        Array<int> Gamma_m(mesh->bdr_attributes.Max());
        Gamma_m = 0;
        Gamma_m[Gamma_m_marker - 1] = 1;
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m); // Neumann bdc on Gamma_m, take negative below
        lf.Assemble();

        phi2->ProjectCoefficient(G_coeff);
        phi2->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf.SetOperatorType(Operator::PETSC_MATAIJ);
        blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, *A, *x, *b); //除了ess_tdof_list以外是0的Neumann边界

        A->EliminateRows(water_dofs, 1.0); // fff自己修改了源码: 重载了这个函数
        for (int i=0; i<water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
        {
#ifdef SELF_DEBUG
            assert(abs((*b)(water_dofs[i])) < 1E-10);
#endif
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "harmonic_");
        solver->SetAbsTol(harmonic_atol);
        solver->SetRelTol(harmonic_rtol);
        solver->SetMaxIter(harmonic_maxiter);
        solver->SetPrintLevel(harmonic_printlvl);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf.RecoverFEMSolution(*x, lf, *phi2);

#ifdef SELF_VERBOSE
//        cout << "\nl2 norm of phi2: " << phi2->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi2 solver: failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        /* Only for pqr file "../data/1MAG.pqr" and mesh file "../data/1MAG_2.msh", we do below tests.
           Only need pqr file (to compute singluar electrostatic potential phi1) and mesh file, we can compute phi2.
           Data is provided by Zhang Qianru */
        if (strcmp(pqr_file, "../data/1MAG.pqr") == 0 && strcmp(mesh_file, "../data/1MAG_2.msh") == 0)
        {
            for (int i=0; i<water_dofs.Size(); i++)
            {
                assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
            }
            for (int i=0; i<interface_ess_tdof_list.Size(); i++)
            {
                assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-10);
            }

            double L2norm = phi2->ComputeL2Error(zero);
            assert(abs(L2norm - 7.2139E+02) < 1); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi2 (no units)" << endl;
        }
#endif
        delete solver, A, x, b;
    }

    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson(GridFunction& c1_n_, GridFunction& c2_n_)
    {
        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water_mark));
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_protein_mark));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf(new ParLinearForm(h1_space)); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(&c1_n_), c2_n_coeff(&c2_n_);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        ProductCoefficient lf1(rhs1, mark_water_coeff);
        ProductCoefficient lf2(rhs2, mark_water_coeff);
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf1));
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf2));
        lf->Assemble();

        // Poisson方程的奇异项导出的interface部分
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        SelfDefined_LinearForm interface_term(h1_space);
        interface_term.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(h1_space, grad_phi1_plus_grad_phi2,
                                                                                          protein_marker, water_marker));
        interface_term.SelfDefined_Assemble();
        interface_term *= protein_rel_permittivity;
        (*lf) += interface_term; //界面项要移到方程的右端

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *phi3, *lf, *A, *x, *b); // ess_tdof_list include: top, bottom

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi3_");
        solver->SetAbsTol(phi3_atol);
        solver->SetRelTol(phi3_rtol);
        solver->SetMaxIter(phi3_maxiter);
        solver->SetPrintLevel(phi3_printlvl);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *phi3);

        (*phi3_n) *= relax_phi;
        (*phi3)   *= 1-relax_phi;
        (*phi3)   += (*phi3_n); // 利用松弛方法更新phi3
        (*phi3_n) /= relax_phi+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
        //        cout << "            l2 norm of phi3: " << phi3->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "phi3 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi3 solver: failed to converged" << endl;
#endif

        delete blf, lf, solver, A, x, b;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1(GridFunction& phi3_n_)
    {
        ParLinearForm *lf(new ParLinearForm(h1_space)); //NP1方程的右端项
        *lf = 0.0;

        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi3_n_, &D1_prod_z1_water));
        blf->Assemble(0);
        blf->Finalize(0);

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        A->EliminateRows(protein_dofs, 1.0);
        for (int i=0; i<protein_dofs.Size(); ++i)
        {
#ifdef SELF_DEBUG
            assert(abs((*b)(protein_dofs[i])) < 1E-10);
#endif
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
        solver->SetAbsTol(np1_atol);
        solver->SetRelTol(np1_rtol);
        solver->SetMaxIter(np1_maxiter);
        solver->SetPrintLevel(np1_printlvl);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c1);

        (*c1_n) *= relax_c1;
        (*c1)   *= 1-relax_c1;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax_c1; // 还原c1_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
        //        cout << "            l2 norm of c1: " << c1->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np1 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete lf, blf, solver, A, x, b;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2(GridFunction& phi3_n_)
    {
        ParLinearForm *lf(new ParLinearForm(h1_space)); //NP2方程的右端项
        *lf = 0.0;

        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        blf->AddDomainIntegrator(new GradConvectionIntegrator(phi3_n_, &D2_prod_z2_water));
        blf->Assemble(0);
        blf->Finalize(0);

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        A->EliminateRows(protein_dofs, 1.0);
        for (int i=0; i<protein_dofs.Size(); ++i)
        {
#ifdef SELF_DEBUG
            assert(abs((*b)(protein_dofs[i])) < 1E-10);
#endif
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");
        solver->SetAbsTol(np2_atol);
        solver->SetRelTol(np2_rtol);
        solver->SetMaxIter(np2_maxiter);
        solver->SetPrintLevel(np2_printlvl);

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c2);

        (*c2_n) *= relax_c2;
        (*c2)   *= 1-relax_c2;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax_c2+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

#ifdef SELF_VERBOSE
        //        cout << "            l2 norm of c2: " << c2->Norml2() << endl;
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np2 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
        {
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }
#endif
        delete lf, blf, solver, A, x, b;
    }
};


class JacobianPrec_BlockPrec: public Solver
{
protected:
    FiniteElementSpace* fsp;
    Array<int>& block_offsets;

    SparseMatrix* phi3_stiff;
    BlockOperator* jacobian;
    IterativeSolver *phi3_stiff_solver, *c1_stiff_solver, *c2_stiff_solver;
    Solver *phi3_stiff_prec, *c1_stiff_prec,  *c2_stiff_prec;

    mutable StopWatch chrono;

public:
    JacobianPrec_BlockPrec(Array<int>& offsets, FiniteElementSpace* fsp_, SparseMatrix& phi3_stiff_)
            : block_offsets(offsets), phi3_stiff(&phi3_stiff_), fsp(fsp_)
    {
        int print_lvl = -1; // 一般不显示子块求解信息, 太多了

        GSSmoother* phi3_prec = new GSSmoother(*phi3_stiff);
        phi3_stiff_prec = phi3_prec;
        CGSolver* phi3_solver = new CGSolver();
        phi3_solver->SetRelTol(1e-8);
        phi3_solver->SetAbsTol(1e-10);
        phi3_solver->SetMaxIter(200);
        phi3_solver->SetPrintLevel(print_lvl);
        phi3_solver->SetOperator(*phi3_stiff); // phi3对应的刚度矩阵不随Newton迭代变化
        phi3_solver->SetPreconditioner(*phi3_stiff_prec);
        phi3_solver->iterative_mode = false;
        phi3_stiff_solver = phi3_solver;

        GSSmoother* c1_prec = new GSSmoother();
        c1_stiff_prec = c1_prec;
        GMRESSolver* c1_solver = new GMRESSolver();
        c1_solver->SetRelTol(1e-8);
        c1_solver->SetAbsTol(1e-10);
        c1_solver->SetMaxIter(200);
        c1_solver->SetPrintLevel(print_lvl);
        c1_solver->SetPreconditioner(*c1_stiff_prec);
        c1_solver->iterative_mode = false;
        c1_stiff_solver = c1_solver;

        GSSmoother* c2_prec = new GSSmoother();
        c2_stiff_prec = c2_prec;
        GMRESSolver* c2_solver = new GMRESSolver();
        c2_solver->SetRelTol(1e-8);
        c2_solver->SetAbsTol(1e-10);
        c2_solver->SetMaxIter(200);
        c2_solver->SetPrintLevel(print_lvl);
        c2_solver->SetPreconditioner(*c2_stiff_prec);
        c2_solver->iterative_mode = false;
        c2_stiff_solver = c2_solver;
    }
    virtual ~JacobianPrec_BlockPrec() {}

    virtual void Mult(const Vector& k, Vector& y) const
    {
        Vector phi3_in(k.GetData() + block_offsets[0], block_offsets[1] - block_offsets[0]);
        Vector   c1_in(k.GetData() + block_offsets[1], block_offsets[2] - block_offsets[1]);
        Vector   c2_in(k.GetData() + block_offsets[2], block_offsets[3] - block_offsets[2]);

        Vector phi3_out(y.GetData() + block_offsets[0], block_offsets[1] - block_offsets[0]);
        Vector   c1_out(y.GetData() + block_offsets[1], block_offsets[2] - block_offsets[1]);
        Vector   c2_out(y.GetData() + block_offsets[2], block_offsets[3] - block_offsets[2]);
#ifdef CLOSE
        {
            cout << "\nin JacobianPrec_BlockPrec::Mult()" << endl;
            cout << "phi3_in: " << phi3_in.Norml2() << endl;
            cout << "c1_in  : " << c1_in.Norml2() << endl;
            cout << "c2_in  : " << c2_in.Norml2() << endl;
        }
#endif
        chrono.Clear();
        chrono.Start();
        phi3_stiff_solver->Mult(phi3_in, phi3_out);
        chrono.Stop();
#ifdef SELF_VERBOSE
        //        if (phi3_stiff_solver->GetConverged())
//        {
//            cout << "       (JacobianPrec_BlockPrec) phi3_stiff_solver: Converged in " << phi3_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << phi3_stiff_solver->GetFinalNorm() << ", ";
//        }
//        else
//        {
//            cout << "       (JacobianPrec_BlockPrec) phi3_stiff_solver: Not Converge in " << phi3_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << phi3_stiff_solver->GetFinalNorm() << ", ";
//        }
//        cout << "took " << chrono.RealTime() << " s." << endl;
#endif

        chrono.Clear();
        chrono.Start();
        c1_stiff_solver->Mult(c1_in,   c1_out);
        chrono.Stop();
#ifdef SELF_VERBOSE
        //        if (c1_stiff_solver->GetConverged())
//        {
//            cout << "       (JacobianPrec_BlockPrec) c1_stiff_solver: Converged in " << c1_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << c1_stiff_solver->GetFinalNorm() << ", ";
//        }
//        else
//        {
//            cout << "       (JacobianPrec_BlockPrec) c1_stiff_solver: Not Converge in " << c1_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << c1_stiff_solver->GetFinalNorm() << ", ";
//        }
//        cout << "took " << chrono.RealTime() << " s." << endl;
#endif

        chrono.Clear();
        chrono.Start();
        c2_stiff_solver->Mult(c2_in,   c2_out);
        chrono.Stop();
#ifdef SELF_VERBOSE
        //        if (c2_stiff_solver->GetConverged())
//        {
//            cout << "       (JacobianPrec_BlockPrec) c2_stiff_solver: Converged in " << c2_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << c2_stiff_solver->GetFinalNorm() << ", ";
//        }
//        else
//        {
//            cout << "       (JacobianPrec_BlockPrec) c2_stiff_solver: Not Converge in " << c2_stiff_solver->GetNumIterations()
//                 << " iterations, residual norm: " << c2_stiff_solver->GetFinalNorm() << ", ";
//        }
//        cout << "took " << chrono.RealTime() << " s." << endl;
#endif
#ifdef CLOSE
        {
            cout << "phi3_out: " << phi3_out.Norml2() << endl;
            cout <<   "c1_out  : " << c1_out.Norml2() << endl;
            cout <<   "c2_out  : " << c2_out.Norml2() << endl;
//            MFEM_ABORT("in JacobianPrec_BlockPrec::Mult()");
        }
#endif
    }

    virtual void SetOperator(const Operator& op) //这个op就是Jacobian
    {
        jacobian = (BlockOperator *) &op;

//        phi3_stiff_solver->SetOperator(jacobian->GetBlock(0, 0));
        c1_stiff_solver  ->SetOperator(jacobian->GetBlock(1, 1));
        c2_stiff_solver  ->SetOperator(jacobian->GetBlock(2, 2));
    }
};
class PNP_Newton_Operator_ser: public Operator
{
protected:
    Mesh* mesh;
    FiniteElementSpace *h1_space;

    Array<int> block_offsets;
    BlockVector *u_k;       // next solution, current solution,
    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockMatrix *jac_k; // Jacobian at current solution
    Solver *jac_solver, *jac_prec;

    mutable SelfDefined_LinearForm *f, *f1, *f2;
    SelfDefined_LinearForm *g;
    mutable SparseMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable BilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;

    GridFunction *phi1, *phi2;

    NewtonSolver newton_solver;

    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list,
            interface_ess_tdof_list, water_dofs, protein_dofs;

    StopWatch chrono;

public:
    PNP_Newton_Operator_ser(Mesh* mesh_, int size_): Operator(size_), mesh(mesh_)
    {

        H1_FECollection h1_fec(p_order, mesh->Dimension());
        h1_space = new FiniteElementSpace(mesh, &h1_fec);

        Array<int> top_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        Array<int> bottom_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        Array<int> ess_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        ess_bdr                    = 0;
        ess_bdr[top_marker - 1]    = 1;
        ess_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        Array<int> interface_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        interface_bdr = 0;
        interface_bdr[interface_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

        Array<int> interface_dofs;
        for (int i=0; i<h1_space->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_space->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                h1_space->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        for (int i=0; i<mesh->GetNumFaces(); ++i)
        {
            FaceElementTransformations* tran = mesh->GetFaceElementTransformations(i);
            if (tran->Elem2No > 0) // interior facet
            {
                const Element* e1  = mesh->GetElement(tran->Elem1No);
                const Element* e2  = mesh->GetElement(tran->Elem2No);
                int attr1 = e1->GetAttribute();
                int attr2 = e2->GetAttribute();
                Array<int> fdofs;
                if (attr1 != attr2) // interface facet
                {
                    h1_space->GetFaceVDofs(i, fdofs);
                    interface_dofs.Append(fdofs);
                }
            }
        }
        protein_dofs.Sort(); protein_dofs.Unique();
        water_dofs.Sort(); water_dofs.Unique();
        interface_dofs.Sort(); interface_dofs.Unique();
        for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        phi1 = new GridFunction(h1_space);
        phi1->ProjectCoefficient(G_coeff);
#ifdef SELF_DEBUG
        {
            /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
            * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
            */
            assert(strcmp(pqr_file, "../data/1MAG.pqr") == 0);
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            FiniteElementSpace h1_vec(fsp->GetMesh(), fsp->FEColl(), 3);
            GridFunction grad_phi1(&h1_vec);
            grad_phi1.ProjectCoefficient(gradG_coeff);
            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
        }
#endif

        phi2 = new GridFunction(h1_space);
        {
            BilinearForm blf(h1_space);
            blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
            blf.Assemble(0);
            blf.Finalize(0);

            LinearForm lf(h1_space);
            Array<int> Gamma_m(h1_space->GetMesh()->bdr_attributes.Max());
            Gamma_m = 0;
            Gamma_m[Gamma_m_marker - 1] = 1;
            lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m); // Neumann bdc on Gamma_m, take negative below
            lf.Assemble();

            SparseMatrix A;
            Vector x, b;
            phi2->ProjectCoefficient(G_coeff);
            phi2->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1
            blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, A, x, b); //除了ess_tdof_list以外是0的Neumann边界
            for (int i=0; i<water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
            {
                A.EliminateRow(water_dofs[i], Matrix::DIAG_ONE);
#ifdef SELF_DEBUG
                assert(abs(b(water_dofs[i])) < 1E-10);
#endif
            }

            IterativeSolver* solver;
            Solver* smoother;
            {
                smoother = new GSSmoother(A);
                solver = new MINRESSolver;
                solver->SetAbsTol(harmonic_atol);
                solver->SetRelTol(harmonic_rtol);
                solver->SetMaxIter(harmonic_maxiter);
                solver->SetPrintLevel(harmonic_printlvl);
                solver->SetOperator(A);
                solver->SetPreconditioner(*smoother);
            }

            chrono.Clear();
            chrono.Start();
            solver->Mult(b, x);
            chrono.Stop();
            blf.RecoverFEMSolution(x, lf, *phi2);
            cout << "\nl2 norm of phi2: " << phi2->Norml2() << endl;
#ifdef SELF_VERBOSE
            if (solver->GetConverged() == 1)
                cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi2 solver: failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
            {
                /* Only for pqr file "../data/1MAG.pqr" and mesh file "../data/1MAG_2.msh", we do below tests.
                Only need pqr file (to compute singluar electrostatic potential phi1) and mesh file, we can compute phi2.
                Data is provided by Zhang Qianru */
                assert(strcmp(pqr_file, "../data/1MAG.pqr") == 0 &&
                       strcmp(mesh_file, "../data/1MAG_2.msh") == 0);
                for (int i=0; i<water_dofs.Size(); i++)
                {
                    assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
                }
                for (int i=0; i<interface_ess_tdof_list.Size(); i++)
                {
                    assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-10);
                }

                double L2norm = phi2->ComputeL2Error(zero);
                assert(abs(L2norm - 7.2139E+02) < 1); //数据由张倩如提供
                cout << "======> Test Pass: L2 norm of phi2 (no units)" << endl;
            }
#endif
        }

        block_offsets.SetSize(4); // number of variables + 1;
        block_offsets[0] = 0;
        block_offsets[1] = h1_space->GetVSize(); //GetVSize()返回的就是总的自由度个数,也就是线性方程组的维数
        block_offsets[2] = h1_space->GetVSize();
        block_offsets[3] = h1_space->GetVSize();
        block_offsets.PartialSum();

        rhs_k = new BlockVector(block_offsets);
        jac_k = new BlockMatrix(block_offsets);
        u_k   = new BlockVector(block_offsets); //必须满足essential边界条件
        GridFunction phi3_k(h1_space, u_k->GetData() + 0);
        GridFunction c1_k  (h1_space, u_k->GetData() + h1_space->GetVSize());
        GridFunction c2_k  (h1_space, u_k->GetData() + 2*h1_space->GetVSize());
        phi3_k = 0.0;
        c1_k   = 0.0;
        c2_k   = 0.0;
        // essential边界条件
        for (int i=0; i<top_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[top_ess_tdof_list[i]] = phi_top;
            (c1_k)  [top_ess_tdof_list[i]] =  c1_top;
            (c2_k)  [top_ess_tdof_list[i]] =  c2_top;
        }
        for (int i=0; i<bottom_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[bottom_ess_tdof_list[i]] = phi_bottom;
            (c1_k)  [bottom_ess_tdof_list[i]] =  c1_bottom;
            (c2_k)  [bottom_ess_tdof_list[i]] =  c2_bottom;
        }

        g  = new SelfDefined_LinearForm(h1_space);
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        g->AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(h1_space, grad_phi1_plus_grad_phi2, protein_marker, water_marker));
        g->SelfDefined_Assemble();
        (*g) *= protein_rel_permittivity;

        a11 = new BilinearForm(h1_space);
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water_mark));
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_protein_mark));
        a11->Assemble(0);
        a11->Finalize(0);
//        A11 = a11->SpMat();
        a11->SetDiagonalPolicy(Matrix::DIAG_ONE);
        a11->FormSystemMatrix(ess_tdof_list, A11);
        jac_k->SetBlock(0, 0, &A11);

        a12 = new BilinearForm(h1_space);
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K(neg, alpha2_prod_alpha3_prod_v_K);
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(0);
        a12->Finalize(0);
        A12 = a12->SpMat();
        for (int i=0; i<ess_tdof_list.Size(); ++i) A12.EliminateRow(ess_tdof_list[i], Matrix::DIAG_ZERO);
        for (int i=0; i<protein_dofs.Size(); ++i) A12.EliminateRow(protein_dofs[i], Matrix::DIAG_ZERO);
        jac_k->SetBlock(0, 1, &A12);

        a13 = new BilinearForm(h1_space);
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl(neg, alpha2_prod_alpha3_prod_v_Cl);
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(0);
        a13->Finalize(0);
        A13 = a13->SpMat();
        for (int i=0; i<ess_tdof_list.Size(); ++i) A13.EliminateRow(ess_tdof_list[i], Matrix::DIAG_ZERO);
        for (int i=0; i<protein_dofs.Size(); ++i) A13.EliminateRow(protein_dofs[i], Matrix::DIAG_ZERO);
        jac_k->SetBlock(0, 2, &A13);

        // construct solver and preconditioner
        JacobianPrec_BlockPrec* j_prec = new JacobianPrec_BlockPrec(block_offsets, h1_space, A11);
        jac_prec = j_prec;

        GMRESSolver* j_gmres = new GMRESSolver;
        j_gmres->SetRelTol(jacobi_rtol);
        j_gmres->SetAbsTol(jacobi_atol);
        j_gmres->SetMaxIter(jacobi_maxiter);
        j_gmres->SetPrintLevel(2);
        j_gmres->SetPreconditioner(*jac_prec);
        jac_solver = j_gmres;

        // Set the newton solve parameters
        newton_solver.iterative_mode = true;
        newton_solver.SetSolver(*jac_solver);
        newton_solver.SetOperator(*this);
        newton_solver.SetPrintLevel(newton_printlvl);
        newton_solver.SetRelTol(newton_rtol);
        newton_solver.SetAbsTol(newton_atol);
        newton_solver.SetMaxIter(newton_maxitr);
    }
    virtual ~PNP_Newton_Operator_ser()
    {
        delete f, f1, f2, g;
        delete a11, a12, a13, a21, a22, a31, a33;
        delete u_k, rhs_k, jac_k, phi1, phi2;
    }

    virtual void Mult(const Vector& x, Vector& y) const
    {
        cout << "\nin PNP_Newton_Solver::Mult()" << endl;
        GridFunction phi3_k(h1_space, x.GetData() + 0);
        GridFunction   c1_k(h1_space, x.GetData() + h1_space->GetVSize());
        GridFunction   c2_k(h1_space, x.GetData() + 2 * h1_space->GetVSize());
        cout << "in Mult(), l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "in Mult(), l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "in Mult(), l2 norm of   c2: " <<   c2_k.Norml2() << endl;

#ifdef SELF_DEBUG
        {
            // essential边界条件
            for (int i = 0; i < top_ess_tdof_list.Size(); ++i)
            {
                assert(abs((phi3_k)[top_ess_tdof_list[i]] - phi_top) < TOL);
                assert(abs((c1_k)  [top_ess_tdof_list[i]] - c1_top)  < TOL);
                assert(abs((c2_k)  [top_ess_tdof_list[i]] - c2_top)  < TOL);
            }
            for (int i = 0; i < bottom_ess_tdof_list.Size(); ++i)
            {
                assert(abs((phi3_k)[bottom_ess_tdof_list[i]] - phi_bottom) < TOL);
                assert(abs((c1_k)  [bottom_ess_tdof_list[i]] - c1_bottom)  < TOL);
                assert(abs((c2_k)  [bottom_ess_tdof_list[i]] - c2_bottom)  < TOL);
            }
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert( abs((c1_k)[protein_dofs[i]])  < TOL );
                assert( abs((c2_k)[protein_dofs[i]])  < TOL );
            }
        }
#endif

        rhs_k->Update(y.GetData(), block_offsets); // update residual

        f = new SelfDefined_LinearForm(h1_space);
        f->Update(h1_space, rhs_k->GetBlock(0), 0);
        GradientGridFunctionCoefficient grad_phi3_k(&phi3_k);
        ScalarVectorProductCoefficient epsilon_protein_prod_grad_phi3_k(epsilon_protein_mark, grad_phi3_k);
        ScalarVectorProductCoefficient epsilon_water_prod_grad_phi3_k(epsilon_water_mark, grad_phi3_k);
        f->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &epsilon_protein_prod_grad_phi3_k));
        f->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &epsilon_water_prod_grad_phi3_k));
        GridFunctionCoefficient c1_k_coeff(&c1_k);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff);
        GridFunctionCoefficient c2_k_coeff(&c2_k);
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff);
        SumCoefficient term(term1, term2);
        ProductCoefficient neg_term(neg, term);
        ProductCoefficient neg_term_water(neg_term, mark_water_coeff);
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term_water));
        f->SelfDefined_Assemble();
        (*f) -= (*g);
        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f)[ess_tdof_list[i]] = 0.0;

        f1 = new SelfDefined_LinearForm(h1_space);
        f1->Update(h1_space, rhs_k->GetBlock(1), 0);
        GradientGridFunctionCoefficient grad_c1_k(&c1_k);
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ScalarVectorProductCoefficient D1_prod_grad_c1_k(D1_water, grad_c1_k);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        ScalarVectorProductCoefficient D1_prod_v_K_prod_c1_k_prod_grad_phi3_k(D1_prod_z1_water_c1_k, grad_phi3_k);
        VectorSumCoefficient np1(D1_prod_grad_c1_k, D1_prod_v_K_prod_c1_k_prod_grad_phi3_k);
        f1->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &np1));
        f1->SelfDefined_Assemble();
        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f1)[ess_tdof_list[i]] = 0.0;

        f2 = new SelfDefined_LinearForm(h1_space);
        f2->Update(h1_space, rhs_k->GetBlock(2), 0);
        GradientGridFunctionCoefficient grad_c2_k(&c2_k);
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ScalarVectorProductCoefficient D2_prod_grad_c2_k(D2_water, grad_c2_k);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        ScalarVectorProductCoefficient D2_prod_v_Cl_prod_c2_k_prod_grad_phi3_k(D2_prod_z2_water_c2_k, grad_phi3_k);
        VectorSumCoefficient np2(D2_prod_grad_c2_k, D2_prod_v_Cl_prod_c2_k_prod_grad_phi3_k);
        f2->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &np2));
        f2->SelfDefined_Assemble();
        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f2)[ess_tdof_list[i]] = 0.0;

        cout.precision(14);
        cout << "after computing Residual, l2 norm of rhs_k(ser): " << rhs_k->Norml2() << endl;
        GridFunction phi3_k_(h1_space, y.GetData() + 0);
        GridFunction   c1_k_(h1_space, y.GetData() + h1_space->GetVSize());
        GridFunction   c2_k_(h1_space, y.GetData() + 2 * h1_space->GetVSize());
        cout << "in Mult(), l2 norm of phi3: " << phi3_k_.Norml2() << endl;
        cout << "in Mult(), l2 norm of   c1: " <<   c1_k_.Norml2() << endl;
        cout << "in Mult(), l2 norm of   c2: " <<   c2_k_.Norml2() << endl;

        ofstream rhs_k_file;
        rhs_k_file.open("./rhs_k_serial.txt");
        rhs_k->Print(rhs_k_file, 1);
        rhs_k_file.close();
    }

    virtual Operator &GetGradient(const Vector& x) const
    {
        cout << "in PNP_Newton_Solver::GetGradient()" << endl;
        GridFunction phi3_k(h1_space, x.GetData() + 0);
        GridFunction c1_k  (h1_space, x.GetData() + h1_space->GetVSize());
        GridFunction c2_k  (h1_space, x.GetData() + 2 * h1_space->GetVSize());
        cout << "in GetGradient(), l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "in GetGradient(), l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "in GetGradient(), l2 norm of   c2: " <<   c2_k.Norml2() << endl;

        a21 = new BilinearForm(h1_space);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        GridFunctionCoefficient c1_k_coeff(&c1_k);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_water_c1_k));
        a21->Assemble(0);
        a21->Finalize(0);
        A21 = a21->SpMat();
        for (int i=0; i<ess_tdof_list.Size(); ++i) A21.EliminateRow(ess_tdof_list[i], Matrix::DIAG_ZERO);
        for (int i=0; i<protein_dofs.Size(); ++i) A21.EliminateRow(protein_dofs[i], Matrix::DIAG_ZERO);

        a22 = new BilinearForm(h1_space);
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        a22->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        a22->AddDomainIntegrator(new GradConvectionIntegrator(phi3_k, &D1_prod_z1_water));
        a22->Assemble(0);
        a22->Finalize(0);
        a22->SetDiagonalPolicy(Matrix::DIAG_ONE);
        a22->FormSystemMatrix(ess_tdof_list, A22);
        for (int i=0; i<protein_dofs.Size(); ++i) A22.EliminateRow(protein_dofs[i], Matrix::DIAG_ONE);

        a31 = new BilinearForm(h1_space);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        GridFunctionCoefficient c2_k_coeff(&c2_k);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_water_c2_k));
        a31->Assemble(0);
        a31->Finalize(0);
        A31 = a31->SpMat();
        for (int i=0; i<ess_tdof_list.Size(); ++i) A31.EliminateRow(ess_tdof_list[i], Matrix::DIAG_ZERO);
        for (int i=0; i<protein_dofs.Size(); ++i) A31.EliminateRow(protein_dofs[i], Matrix::DIAG_ZERO);

        a33 = new BilinearForm(h1_space);
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        a33->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        a33->AddDomainIntegrator(new GradConvectionIntegrator(phi3_k, &D2_prod_z2_water));
        a33->Assemble(0);
        a33->Finalize(0);
        a33->SetDiagonalPolicy(Matrix::DIAG_ONE);
        a33->FormSystemMatrix(ess_tdof_list, A33);
        for (int i=0; i<protein_dofs.Size(); ++i) A33.EliminateRow(protein_dofs[i], Matrix::DIAG_ONE);

        jac_k->SetBlock(1, 0, &A21);
        jac_k->SetBlock(1, 1, &A22);
        jac_k->SetBlock(2, 0, &A31);
        jac_k->SetBlock(2, 2, &A33);
#ifdef CLOSE
        { // for test
            cout << "after Assemble() in GetGradient() in ser:\n";
            Vector temp(height/3), haha(height/3);
            for (int i=0; i<height/3; ++i) {
                haha[i] = i%10;
            }

            ofstream temp_file;

            temp_file.open("./A11_mult_phi3_k_par");
            A11.Mult(haha, temp);
            cout << "A11_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A12_mult_phi3_k_par");
            A12.Mult(haha, temp);
            cout << "A12_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A13_mult_phi3_k_par");
            A13.Mult(haha, temp);
            cout << "A13_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A21_mult_phi3_k_par");
            A21.Mult(haha, temp);
            cout << "A21_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A22_mult_phi3_k_par");
            A22.Mult(haha, temp);
            cout << "A22_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A31_mult_phi3_k_par");
            A31.Mult(haha, temp);
            cout << "A31_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A33_mult_phi3_k_par");
            A33.Mult(haha, temp);
            cout << "A33_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

//            MFEM_ABORT("save mesh done in ser");
        }
#endif
        return *jac_k;
    }

    void Solve() const
    {
        cout.precision(14);
        GridFunction phi3_k(h1_space, u_k->GetData() + 0);
        GridFunction   c1_k(h1_space, u_k->GetData() + h1_space->GetVSize());
        GridFunction   c2_k(h1_space, u_k->GetData() + 2 * h1_space->GetVSize());
        cout << "\nin Solve(), l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "in Solve(), l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "in Solve(), l2 norm of   c2: " <<   c2_k.Norml2() << endl;

        Vector zero;
        newton_solver.Mult(zero, *u_k);
        MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");

        cout << "l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "l2 norm of   c2: " <<   c2_k.Norml2() << endl;
    }
};
class PNP_Newton_Solver
{
protected:
    Mesh* mesh;
    FiniteElementSpace *h1_space;

    Array<int> block_offsets;
    BlockVector *u_k;       // next solution, current solution,
    Solver *jac_solver;

    mutable SparseMatrix A11;
    mutable BilinearForm *a11;

    GridFunction *phi1, *phi2;

    NewtonSolver newton_solver;

    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list,
            interface_ess_tdof_list, water_dofs, protein_dofs;

    StopWatch chrono;

public:
    PNP_Newton_Solver(Mesh* mesh_): mesh(mesh_)
    {

        H1_FECollection h1_fec(p_order, mesh->Dimension());
        h1_space = new FiniteElementSpace(mesh, &h1_fec);

        Array<int> top_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        Array<int> bottom_bdr(h1_space->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        block_offsets.SetSize(4); // number of variables + 1;
        block_offsets[0] = 0;
        block_offsets[1] = h1_space->GetVSize(); //GetVSize()返回的就是总的自由度个数,也就是线性方程组的维数
        block_offsets[2] = h1_space->GetVSize();
        block_offsets[3] = h1_space->GetVSize();
        block_offsets.PartialSum();

        u_k   = new BlockVector(block_offsets); //必须满足essential边界条件
        GridFunction phi3_k(h1_space, u_k->GetData() + 0);
        GridFunction c1_k  (h1_space, u_k->GetData() + h1_space->GetVSize());
        GridFunction c2_k  (h1_space, u_k->GetData() + 2*h1_space->GetVSize());
        phi3_k = 0.0;
        c1_k   = 0.0;
        c2_k   = 0.0;
        // essential边界条件
        for (int i=0; i<top_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[top_ess_tdof_list[i]] = phi_top;
            (c1_k)  [top_ess_tdof_list[i]] =  c1_top;
            (c2_k)  [top_ess_tdof_list[i]] =  c2_top;
        }
        for (int i=0; i<bottom_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[bottom_ess_tdof_list[i]] = phi_bottom;
            (c1_k)  [bottom_ess_tdof_list[i]] =  c1_bottom;
            (c2_k)  [bottom_ess_tdof_list[i]] =  c2_bottom;
        }

        a11 = new BilinearForm(h1_space);
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water_mark));
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_protein_mark));
        a11->Assemble(0);
        a11->Finalize(0);
//        A11 = a11->SpMat();
        a11->SetDiagonalPolicy(Matrix::DIAG_ONE);
        a11->FormSystemMatrix(ess_tdof_list, A11);

        // construct solver and preconditioner
        PNP_Newton_Operator_ser* op = new PNP_Newton_Operator_ser(mesh, h1_space->GetVSize()*3);

        JacobianPrec_BlockPrec* jac_prec = new JacobianPrec_BlockPrec(block_offsets, h1_space, A11);

        GMRESSolver* j_gmres = new GMRESSolver;
        j_gmres->SetRelTol(jacobi_rtol);
        j_gmres->SetAbsTol(jacobi_atol);
        j_gmres->SetMaxIter(jacobi_maxiter);
        j_gmres->SetPrintLevel(2);
        j_gmres->SetPreconditioner(*jac_prec);
        jac_solver = j_gmres;

        // Set the newton solve parameters
        newton_solver.iterative_mode = true;
        newton_solver.SetSolver(*jac_solver);
        newton_solver.SetOperator(*op);
        newton_solver.SetPrintLevel(newton_printlvl);
        newton_solver.SetRelTol(newton_rtol);
        newton_solver.SetAbsTol(newton_atol);
        newton_solver.SetMaxIter(newton_maxitr);
    }
    virtual ~PNP_Newton_Solver()
    {
        delete a11;
        delete u_k, phi1, phi2;
    }

    void Solve() const
    {
        cout.precision(14);
        GridFunction phi3_k(h1_space, u_k->GetData() + 0);
        GridFunction   c1_k(h1_space, u_k->GetData() + h1_space->GetVSize());
        GridFunction   c2_k(h1_space, u_k->GetData() + 2 * h1_space->GetVSize());
        cout << "\nin Solve(), l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "in Solve(), l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "in Solve(), l2 norm of   c2: " <<   c2_k.Norml2() << endl;

        Vector zero;
        newton_solver.Mult(zero, *u_k);
        MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");

        cout << "l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "l2 norm of   c2: " <<   c2_k.Norml2() << endl;
    }
};


class PNP_Newton_Operator_par;
class BlockPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    BlockPreconditionerSolver(const OperatorHandle& oh): Solver()
    {
        PetscErrorCode ierr;

        // Get the PetscParMatrix out of oh.
        PetscParMatrix *Jacobian_;
        oh.Get(Jacobian_);
        Mat Jacobian = *Jacobian_; // type cast to Petsc Mat

        // update base (Solver) class
        width = Jacobian_->Width();
        height = Jacobian_->Height();
        X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
        Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

        PetscInt M, N;
        ierr = MatNestGetSubMats(Jacobian, &N, &M, &sub); PCHKERRQ(sub[0][0], ierr); // get block matrices
        ierr = MatNestGetISs(Jacobian, index_set, NULL); PCHKERRQ(index_set, ierr); // get the index sets of the blocks
//        cout << "M: " << M << ", N: " << N << endl;
//        MatView(sub[0][0], PETSC_VIEWER_STDOUT_WORLD);
//        MatView(sub[1][1], PETSC_VIEWER_STDOUT_WORLD);
//        MatView(sub[2][2], PETSC_VIEWER_STDOUT_WORLD);
//        ISView(index_set[0],PETSC_VIEWER_STDOUT_WORLD);
//        ISView(index_set[1],PETSC_VIEWER_STDOUT_WORLD);
//        ISView(index_set[2],PETSC_VIEWER_STDOUT_WORLD);
//        Write_Mat_Matlab_txt("A11_.m", sub[0][0]);
//        Write_Mat_Matlab_txt("A22_.m", sub[1][1]);
//        Write_Mat_Matlab_txt("A33_.m", sub[2][2]);
#ifdef CLOSE
        {
            PetscScalar haha[height/3];
            PetscInt    id[height/3];
            for (int i=0; i<height/3; ++i) {
                haha[i] = i%10;
                id[i] = i;
            }
            Vec x,y;
            MatCreateVecs(sub[0][0], &x, &y);
            VecSetValues(x, height/3, id, haha, INSERT_VALUES);
            VecAssemblyBegin(x);
            VecAssemblyEnd(x);

            PetscScalar norm;

            MatMult(sub[0][0], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A11_haha: " << norm << endl;

            MatMult(sub[0][1], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A12_haha: " << norm << endl;

            MatMult(sub[0][2], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A13_haha: " << norm << endl;

            MatMult(sub[1][0], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A21_haha: " << norm << endl;

            MatMult(sub[1][1], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A22_haha: " << norm << endl;

            MatMult(sub[2][0], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A31_haha: " << norm << endl;

            MatMult(sub[2][2], x, y);
            VecNorm(y, NORM_2, &norm);
            cout << "A33_haha: " << norm << endl;
        }
#endif

        for (int i=0; i<3; ++i)
        {
            ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[i]); PCHKERRQ(kspblock[i], ierr);
            ierr = KSPSetOperators(kspblock[i], sub[i][i], sub[i][i]); PCHKERRQ(sub[i][i], ierr);
            if (i == 0)
                KSPAppendOptionsPrefix(kspblock[i], "sub_block1_");
            else if (i == 1)
                KSPAppendOptionsPrefix(kspblock[i], "sub_block2_");
            else if (i == 2)
                KSPAppendOptionsPrefix(kspblock[i], "sub_block3_");
            else MFEM_ABORT("Wrong block preconditioner solver!");
            KSPSetFromOptions(kspblock[i]);
            KSPSetUp(kspblock[i]);
        }
    }
    virtual ~BlockPreconditionerSolver()
    {
        for (int i=0; i<3; i++)
        {
            KSPDestroy(&kspblock[i]);
            //ISDestroy(&index_set[i]); no need to delete it
        }

        delete X;
        delete Y;
    }

    virtual void SetOperator(const Operator& op) { MFEM_ABORT("Not support!"); }

    virtual void Mult(const Vector& x, Vector& y) const
    {
        Vec blockx, blocky;
        Vec blockx0, blocky0;

        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
        Y->PlaceArray(y.GetData());
        // solve 3 equations
        for (int i=0; i<3; ++i)
        {
            VecGetSubVector(*X, index_set[i], &blockx);
            VecGetSubVector(*Y, index_set[i], &blocky);

            KSPSolve(kspblock[i], blockx, blocky);
#ifdef CLOSE
{
            PetscScalar normx, normy;
                VecNorm(blockx, NORM_2, &normx);
               VecNorm(blocky, NORM_2, &normy);
               cout << "norm x: " << normx
                    << ", norm y: " << normy << endl;
            }
#endif
            VecRestoreSubVector(*X, index_set[i], &blockx);
            VecRestoreSubVector(*Y, index_set[i], &blocky);
        }

        X->ResetArray();
        Y->ResetArray();
//        cout << "in BlockPreconditionerSolver::Mult(), l2 norm y after: " << y.Norml2() << endl;
//        MFEM_ABORT("in BlockPreconditionerSolver::Mult()");
    }
};
class PreconditionerFactory: public PetscPreconditionerFactory
{
private:
    const PNP_Newton_Operator_par& op; // op就是Nonlinear Operator(可用来计算Residual, Jacobian)

public:
    PreconditionerFactory(const PNP_Newton_Operator_par& op_, const string& name_): PetscPreconditionerFactory(name_), op(op_) {}
    virtual ~PreconditionerFactory() {}

    virtual Solver* NewPreconditioner(const OperatorHandle& oh) // oh就是当前Newton迭代步的Jacobian的句柄
    {
        return new BlockPreconditionerSolver(oh);
    }
};
class PNP_Newton_Operator_par: public Operator
{
protected:
    ParFiniteElementSpace *fsp;

    Array<int> block_offsets, block_trueoffsets;
    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution

    mutable SelfDefined_LinearForm *f, *f1, *f2;
    SelfDefined_LinearForm *g;
    mutable PetscParMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable ParBilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;

    ParGridFunction *phi1, *phi2;
    ParGridFunction *phi3_k, *c1_k, *c2_k;

    PetscNonlinearSolver* newton_solver;

    Array<int> ess_bdr, top_bdr, bottom_bdr, interface_bdr, Gamma_m_bdr;
    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list,
            interface_ess_tdof_list, water_dofs, protein_dofs, interface_dofs;

    StopWatch chrono;
    int num_procs, myid;
    Array<int> null_array;

public:
    PNP_Newton_Operator_par(ParFiniteElementSpace *fsp_, ParGridFunction* phi1_, ParGridFunction* phi2_)
    : Operator(fsp_->TrueVSize()*3), fsp(fsp_), phi1(phi1_), phi2(phi2_)
    {
        MPI_Comm_size(fsp->GetComm(), &num_procs);
        MPI_Comm_rank(fsp->GetComm(), &myid);

        ess_bdr.SetSize(fsp->GetMesh()->bdr_attributes.Max());
        ess_bdr                    = 0;
        ess_bdr[top_marker - 1]    = 1;
        ess_bdr[bottom_marker - 1] = 1;
        fsp->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        top_bdr.SetSize(fsp->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        fsp->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        bottom_bdr.SetSize(fsp->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        fsp->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        interface_bdr.SetSize(fsp->GetMesh()->bdr_attributes.Max());
        interface_bdr = 0;
        interface_bdr[interface_marker - 1] = 1;
        fsp->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

        Mesh* mesh = fsp->GetMesh();
        for (int i=0; i<fsp->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                fsp->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                fsp->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        for (int i=0; i<mesh->GetNumFaces(); ++i)
        {
            FaceElementTransformations* tran = mesh->GetFaceElementTransformations(i);
            if (tran->Elem2No > 0) // interior facet
            {
                const Element* e1  = mesh->GetElement(tran->Elem1No);
                const Element* e2  = mesh->GetElement(tran->Elem2No);
                int attr1 = e1->GetAttribute();
                int attr2 = e2->GetAttribute();
                Array<int> fdofs;
                if (attr1 != attr2) // interface facet
                {
                    fsp->GetFaceVDofs(i, fdofs);
                    interface_dofs.Append(fdofs);
                }

            }
        }
        protein_dofs.Sort(); protein_dofs.Unique();
        water_dofs.Sort(); water_dofs.Unique();
        interface_dofs.Sort(); interface_dofs.Unique();
        for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        block_offsets.SetSize(4); // number of variables + 1;
        block_offsets[0] = 0;
        block_offsets[1] = fsp->GetVSize(); //GetVSize()返回的就是总的自由度个数,也就是线性方程组的维数
        block_offsets[2] = fsp->GetVSize();
        block_offsets[3] = fsp->GetVSize();
        block_offsets.PartialSum();

        block_trueoffsets.SetSize(4); // number of variables + 1;
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = fsp->GetTrueVSize();
        block_trueoffsets[2] = fsp->GetTrueVSize();
        block_trueoffsets[3] = fsp->GetTrueVSize();
        block_trueoffsets.PartialSum();

        rhs_k = new BlockVector(block_trueoffsets); // not block_offsets !!!
        jac_k = new BlockOperator(block_trueoffsets);
        phi3_k= new ParGridFunction(fsp);
        c1_k  = new ParGridFunction(fsp);
        c2_k  = new ParGridFunction(fsp);

        f  = new SelfDefined_LinearForm(fsp);
        f1  = new SelfDefined_LinearForm(fsp);
        f2  = new SelfDefined_LinearForm(fsp);
        a21 = new ParBilinearForm(fsp);
        a22 = new ParBilinearForm(fsp);
        a31 = new ParBilinearForm(fsp);
        a33 = new ParBilinearForm(fsp);

        g  = new SelfDefined_LinearForm(fsp);
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        g->AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(fsp, grad_phi1_plus_grad_phi2, protein_marker, water_marker));
        g->SelfDefined_Assemble();
        (*g) *= protein_rel_permittivity;

        a11 = new ParBilinearForm(fsp);
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water_mark));
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_protein_mark));
        a11->Assemble(0);
        a11->Finalize(0);
        a11->SetOperatorType(Operator::PETSC_MATAIJ);
        A11 = *a11->ParallelAssemble();
        a11->FormSystemMatrix(ess_tdof_list, A11);

        a12 = new ParBilinearForm(fsp);
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K(neg, alpha2_prod_alpha3_prod_v_K);
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(0);
        a12->Finalize(0);
        a12->SetOperatorType(Operator::PETSC_MATAIJ);
        a12->FormSystemMatrix(null_array, A12);
        A12.EliminateRows(ess_tdof_list, 0.0);
        A12.EliminateRows(protein_dofs, 0.0);

        a13 = new ParBilinearForm(fsp);
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl(neg, alpha2_prod_alpha3_prod_v_Cl);
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(0);
        a13->Finalize(0);
        a13->SetOperatorType(Operator::PETSC_MATAIJ);
        a13->FormSystemMatrix(null_array, A13);
        A13.EliminateRows(ess_tdof_list, 0.0);
        A13.EliminateRows(protein_dofs, 0.0);
    }
    virtual ~PNP_Newton_Operator_par()
    {
        delete f, f1, f2, g;
        delete a11, a12, a13, a21, a22, a31, a33;
        delete rhs_k, jac_k;
        delete newton_solver;
    }

    virtual void Mult(const Vector& x, Vector& y) const
    {
//        cout << "\nin PNP_Newton_Operator::Mult(), l2 norm of x: " << x.Norml2() << endl;
//        cout << "l2 norm of y: " << y.Norml2() << endl;
        int sc = height / 3;
        Vector& x_ = const_cast<Vector&>(x);
        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
        cout << "in Mult(), l2 norm of phi3: " << phi3_k->Norml2() << endl;
        cout << "in Mult(), l2 norm of   c1: " <<   c1_k->Norml2() << endl;
        cout << "in Mult(), l2 norm of   c2: " <<   c2_k->Norml2() << endl;

#ifdef SELF_DEBUG
        {
            // essential边界条件
            for (int i = 0; i < top_ess_tdof_list.Size(); ++i)
            {
                assert(abs((phi3_k)[top_ess_tdof_list[i]] - phi_top) < TOL);
                assert(abs((c1_k)  [top_ess_tdof_list[i]] - c1_top)  < TOL);
                assert(abs((c2_k)  [top_ess_tdof_list[i]] - c2_top)  < TOL);
            }
            for (int i = 0; i < bottom_ess_tdof_list.Size(); ++i)
            {
                assert(abs((phi3_k)[bottom_ess_tdof_list[i]] - phi_bottom) < TOL);
                assert(abs((c1_k)  [bottom_ess_tdof_list[i]] - c1_bottom)  < TOL);
                assert(abs((c2_k)  [bottom_ess_tdof_list[i]] - c2_bottom)  < TOL);
            }
            for (int i=0; i<protein_dofs.Size(); ++i)
            {
                assert( abs((c1_k)[protein_dofs[i]])  < TOL );
                assert( abs((c2_k)[protein_dofs[i]])  < TOL );
            }
        }
#endif

        rhs_k->Update(y.GetData(), block_trueoffsets); // update residual
        Vector y1(y.GetData() +   0, sc);
        Vector y2(y.GetData() +  sc, sc);
        Vector y3(y.GetData() +2*sc, sc);

        delete f;
        f = new SelfDefined_LinearForm(fsp);
        f->Update(fsp, rhs_k->GetBlock(0), 0);
        GradientGridFunctionCoefficient grad_phi3_k(phi3_k);
        ScalarVectorProductCoefficient epsilon_protein_prod_grad_phi3_k(epsilon_protein_mark, grad_phi3_k);
        ScalarVectorProductCoefficient epsilon_water_prod_grad_phi3_k(epsilon_water_mark, grad_phi3_k);
        f->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &epsilon_protein_prod_grad_phi3_k));
        f->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &epsilon_water_prod_grad_phi3_k));
        GridFunctionCoefficient c1_k_coeff(c1_k);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff);
        GridFunctionCoefficient c2_k_coeff(c2_k);
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff);
        SumCoefficient term(term1, term2);
        ProductCoefficient neg_term(neg, term);
        ProductCoefficient neg_term_water(neg_term, mark_water_coeff);
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term_water));
        f->SelfDefined_Assemble();
        (*f) -= (*g);
        f->SetSubVector(ess_tdof_list, 0.0);
//        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f)[ess_tdof_list[i]] = 0.0;

        delete f1;
        f1 = new SelfDefined_LinearForm(fsp);
        f1->Update(fsp, rhs_k->GetBlock(1), 0);
        GradientGridFunctionCoefficient grad_c1_k(c1_k);
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ScalarVectorProductCoefficient D1_prod_grad_c1_k(D1_water, grad_c1_k);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        ScalarVectorProductCoefficient D1_prod_v_K_prod_c1_k_prod_grad_phi3_k(D1_prod_z1_water_c1_k, grad_phi3_k);
        VectorSumCoefficient np1(D1_prod_grad_c1_k, D1_prod_v_K_prod_c1_k_prod_grad_phi3_k);
        f1->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &np1));
        f1->SelfDefined_Assemble();
        f1->SetSubVector(ess_tdof_list, 0.0);
//        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f1)[ess_tdof_list[i]] = 0.0;

        delete f2;
        f2 = new SelfDefined_LinearForm(fsp);
        f2->Update(fsp, rhs_k->GetBlock(2), 0);
        GradientGridFunctionCoefficient grad_c2_k(c2_k);
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ScalarVectorProductCoefficient D2_prod_grad_c2_k(D2_water, grad_c2_k);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        ScalarVectorProductCoefficient D2_prod_v_Cl_prod_c2_k_prod_grad_phi3_k(D2_prod_z2_water_c2_k, grad_phi3_k);
        VectorSumCoefficient np2(D2_prod_grad_c2_k, D2_prod_v_Cl_prod_c2_k_prod_grad_phi3_k);
        f2->AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&one, &np2));
        f2->SelfDefined_Assemble();
        f2->SetSubVector(ess_tdof_list, 0.0);
//        for (int i=0; i<ess_tdof_list.Size(); ++i) (*f2)[ess_tdof_list[i]] = 0.0;
        y1 = (*f);
        y2 = (*f1);
        y3 = (*f2);

//        cout.precision(14);
//        cout << "after computing Residual, l2 norm of rhs_k(par): " << rhs_k->Norml2() << endl;
//        phi3_k->MakeTRef(fsp, y, 0);
//        c1_k->MakeTRef(fsp, y, sc);
//        c2_k->MakeTRef(fsp, y, 2*sc);
//        phi3_k->SetFromTrueVector();
//        c1_k->SetFromTrueVector();
//        c2_k->SetFromTrueVector();
//        cout << "in Mult(), l2 norm of phi3: " << phi3_k->Norml2() << endl;
//        cout << "in Mult(), l2 norm of   c1: " <<   c1_k->Norml2() << endl;
//        cout << "in Mult(), l2 norm of   c2: " <<   c2_k->Norml2() << endl;
//
//        ofstream rhs_k_file;
//        rhs_k_file.open("./rhs_k_par.txt");
//        rhs_k->Print(rhs_k_file, 1);
//        rhs_k_file.close();
    }

    virtual Operator &GetGradient(const Vector& x) const
    {
        cout << "in PNP_Newton_Operator::GetGradient()" << endl;
        int sc = height / 3;
        Vector& x_ = const_cast<Vector&>(x);
        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
//        cout << "in GetGradient(), l2 norm of phi3: " << phi3_k->Norml2() << endl;
//        cout << "in GetGradient(), l2 norm of   c1: " <<   c1_k->Norml2() << endl;
//        cout << "in GetGradient(), l2 norm of   c2: " <<   c2_k->Norml2() << endl;

        delete a21;
        a21 = new ParBilinearForm(fsp);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        GridFunctionCoefficient c1_k_coeff(c1_k);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_water_c1_k));
        a21->Assemble(0);
        a21->Finalize(0);
        a21->SetOperatorType(Operator::PETSC_MATAIJ);
        a21->FormSystemMatrix(null_array, A21);
        A21.EliminateRows(ess_tdof_list, 0.0);
        A21.EliminateRows(protein_dofs, 0.0);

        delete a22;
        a22 = new ParBilinearForm(fsp);
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        a22->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D1_prod_z1_water));
        a22->Assemble(0);
        a22->Finalize(0);
        a22->SetOperatorType(Operator::PETSC_MATAIJ);
        a22->FormSystemMatrix(ess_tdof_list, A22);
        A22.EliminateRows(protein_dofs, 1.0);

        delete a31;
        a31 = new ParBilinearForm(fsp);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        GridFunctionCoefficient c2_k_coeff(c2_k);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_water_c2_k));
        a31->Assemble(0);
        a31->Finalize(0);
        a31->SetOperatorType(Operator::PETSC_MATAIJ);
        a31->FormSystemMatrix(null_array, A31);
        A31.EliminateRows(ess_tdof_list, 0.0);
        A31.EliminateRows(protein_dofs, 0.0);

        delete a33;
        a33 = new ParBilinearForm(fsp);
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        a33->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D2_prod_z2_water));
        a33->Assemble(0);
        a33->Finalize(0);
        a33->SetOperatorType(Operator::PETSC_MATAIJ);
        a33->FormSystemMatrix(ess_tdof_list, A33);
        A33.EliminateRows(protein_dofs, 1.0);

        jac_k = new BlockOperator(block_trueoffsets);
        jac_k->SetBlock(0, 0, &A11);
        jac_k->SetBlock(0, 1, &A12);
        jac_k->SetBlock(0, 2, &A13);
        jac_k->SetBlock(1, 0, &A21);
        jac_k->SetBlock(1, 1, &A22);
        jac_k->SetBlock(2, 0, &A31);
        jac_k->SetBlock(2, 2, &A33);
#ifdef CLOSE
        { // for test
            cout << "after Assemble() in GetGradient() in par:\n";
            Vector temp(height/3), haha(height/3);
            for (int i=0; i<height/3; ++i) {
                haha[i] = i%10;
            }

            ofstream temp_file;

            temp_file.open("./A11_mult_phi3_k_par");
            A11.Mult(haha, temp);
            cout << "A11_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A12_mult_phi3_k_par");
            A12.Mult(haha, temp);
            cout << "A12_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A13_mult_phi3_k_par");
            A13.Mult(haha, temp);
            cout << "A13_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A21_mult_phi3_k_par");
            A21.Mult(haha, temp);
            cout << "A21_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A22_mult_phi3_k_par");
            A22.Mult(haha, temp);
            cout << "A22_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A31_mult_phi3_k_par");
            A31.Mult(haha, temp);
            cout << "A31_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

            temp_file.open("./A33_mult_phi3_k_par");
            A33.Mult(haha, temp);
            cout << "A33_temp norm: " << temp.Norml2() << endl;
            temp.Print(temp_file, 1);
            temp_file.close();

//            MFEM_ABORT("save mesh done in par");
        }
#endif
        return *jac_k;
    }
};
class PNP_Newton_Solver_par
{
protected:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_space;
    PNP_Newton_Operator_par* op;
    PetscPreconditionerFactory *jac_factory;
    PetscNonlinearSolver* newton_solver;

    Array<int> block_trueoffsets, top_bdr, bottom_bdr, interface_bdr, Gamma_m_bdr, top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list;
    Array<int> protein_dofs, water_dofs, interface_dofs;
    BlockVector* u_k;
    ParGridFunction phi3_k, c1_k, c2_k;
    ParGridFunction *phi1, *phi2;

    StopWatch chrono;

public:
    PNP_Newton_Solver_par(Mesh* mesh_): mesh(mesh_)
    {
        int mesh_dim = mesh->Dimension(); //网格的维数:1D,2D,3D
        for (int i=0; i<refine_times; ++i) mesh->UniformRefinement();

        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

        h1_fec = new H1_FECollection(p_order, mesh_dim);
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        top_bdr.SetSize(h1_space->GetMesh()->bdr_attributes.Max());
        top_bdr                 = 0;
        top_bdr[top_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

        bottom_bdr.SetSize(h1_space->GetMesh()->bdr_attributes.Max());
        bottom_bdr = 0;
        bottom_bdr[bottom_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

        interface_bdr.SetSize(h1_space->GetMesh()->bdr_attributes.Max());
        interface_bdr = 0;
        interface_bdr[interface_marker - 1] = 1;
        h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

        Gamma_m_bdr.SetSize(h1_space->GetMesh()->bdr_attributes.Max());
        Gamma_m_bdr                     = 0;
        Gamma_m_bdr[Gamma_m_marker - 1] = 1;

        block_trueoffsets.SetSize(4);
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = h1_space->GetTrueVSize();
        block_trueoffsets[2] = h1_space->GetTrueVSize();
        block_trueoffsets[3] = h1_space->GetTrueVSize();
        block_trueoffsets.PartialSum();

        // MakeTRef(), SetTrueVector(), SetFromTrueVector() 三者要配套使用ffffffffff
        u_k = new BlockVector(block_trueoffsets); //必须满足essential边界条件
        phi3_k.MakeTRef(h1_space, *u_k, block_trueoffsets[0]);
        c1_k  .MakeTRef(h1_space, *u_k, block_trueoffsets[1]);
        c2_k  .MakeTRef(h1_space, *u_k, block_trueoffsets[2]);
        phi3_k = 0.0;
        c1_k = 0.0;
        c2_k = 0.0;
        for (int i=0; i<top_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[top_ess_tdof_list[i]] = phi_top;
            (c1_k)  [top_ess_tdof_list[i]] =  c1_top;
            (c2_k)  [top_ess_tdof_list[i]] =  c2_top;
        }
        for (int i=0; i<bottom_ess_tdof_list.Size(); ++i)
        {
            (phi3_k)[bottom_ess_tdof_list[i]] = phi_bottom;
            (c1_k)  [bottom_ess_tdof_list[i]] =  c1_bottom;
            (c2_k)  [bottom_ess_tdof_list[i]] =  c2_bottom;
        }
        phi3_k.SetTrueVector();
        phi3_k.SetFromTrueVector();
        c1_k.SetTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetTrueVector();
        c2_k.SetFromTrueVector();

        phi1 = new ParGridFunction(h1_space);
        phi1->ProjectCoefficient(G_coeff);
        cout << "l2 norm of phi1: " << phi1->Norml2() << endl;
#ifdef SELF_DEBUG
        {
            /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
            * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
            */
            assert(strcmp(pqr_file, "../data/1MAG.pqr") == 0);
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            FiniteElementSpace h1_vec(fsp->GetMesh(), fsp->FEColl(), 3);
            GridFunction grad_phi1(&h1_vec);
            grad_phi1.ProjectCoefficient(gradG_coeff);
            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
        }
#endif

        Mesh* mesh = h1_space->GetMesh();
        for (int i=0; i<h1_space->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_space->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                h1_space->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        for (int i=0; i<mesh->GetNumFaces(); ++i)
        {
            FaceElementTransformations* tran = mesh->GetFaceElementTransformations(i);
            if (tran->Elem2No > 0) // interior facet
            {
                const Element* e1  = mesh->GetElement(tran->Elem1No);
                const Element* e2  = mesh->GetElement(tran->Elem2No);
                int attr1 = e1->GetAttribute();
                int attr2 = e2->GetAttribute();
                Array<int> fdofs;
                if (attr1 != attr2) // interface facet
                {
                    h1_space->GetFaceVDofs(i, fdofs);
                    interface_dofs.Append(fdofs);
                }

            }
        }
        protein_dofs.Sort(); protein_dofs.Unique();
        water_dofs.Sort(); water_dofs.Unique();
        interface_dofs.Sort(); interface_dofs.Unique();
        for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        phi2 = new ParGridFunction(h1_space);
        {
            ParBilinearForm blf(h1_space);
            blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
            blf.Assemble(0);
            blf.Finalize(0);

            ParLinearForm lf(h1_space);
            lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m_bdr); // Neumann bdc on Gamma_m
            lf.Assemble();

            PetscParMatrix *A = new PetscParMatrix();
            PetscParVector *x = new PetscParVector(h1_space);
            PetscParVector *b = new PetscParVector(h1_space);
            phi2->ProjectCoefficient(G_coeff);
            phi2->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1
            phi2->SetTrueVector();
            blf.SetOperatorType(Operator::PETSC_MATAIJ);
            blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, *A, *x, *b); //除了ess_tdof_list以外是0的Neumann边界
            A->EliminateRows(water_dofs, 1.0);
            for (int i=0; i<water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
            {
                assert(abs((*b)(water_dofs[i])) < 1E-10);
            }

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "harmonic_");
            solver->SetAbsTol(harmonic_atol);
            solver->SetRelTol(harmonic_rtol);
            solver->SetMaxIter(harmonic_maxiter);
            solver->SetPrintLevel(harmonic_printlvl);

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();
            blf.RecoverFEMSolution(*x, lf, *phi2);

            for (int i=0; i<interface_ess_tdof_list.Size(); i++)
            {
                assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-8);
            }
            for (int i=0; i<water_dofs.Size(); i++)
            {
                assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
            }

#ifdef SELF_VERBOSE
            if (solver->GetConverged() == 1)
                cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi2 solver: failed to converged" << endl;
#endif
#ifdef SELF_DEBUG
            {
                /* Only for pqr file "../data/1MAG.pqr" and mesh file "../data/1MAG_2.msh", we do below tests.
                Only need pqr file (to compute singluar electrostatic potential phi1) and mesh file, we can compute phi2.
                Data is provided by Zhang Qianru */
                assert(strcmp(pqr_file, "../data/1MAG.pqr") == 0 &&
                       strcmp(mesh_file, "../data/1MAG_2.msh") == 0);
                for (int i=0; i<water_dofs.Size(); i++)
                {
                    assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
                }
                for (int i=0; i<interface_ess_tdof_list.Size(); i++)
                {
                    assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-10);
                }

                double L2norm = phi2->ComputeL2Error(zero);
                assert(abs(L2norm - 7.2139E+02) < 1); //数据由张倩如提供
                cout << "======> Test Pass: L2 norm of phi2 (no units)" << endl;
            }
#endif
            delete A, x, b, solver;
        }
        cout << "l2 norm of phi2: " << phi2->Norml2() << endl;

        op = new PNP_Newton_Operator_par(h1_space, phi1, phi2);

        // Set the newton solve parameters
        jac_factory   = new PreconditionerFactory(*op, "Block Preconditioner");
        newton_solver = new PetscNonlinearSolver(h1_space->GetComm(), *op, "newton_");
        newton_solver->iterative_mode = true;
        newton_solver->SetAbsTol(newton_atol);
        newton_solver->SetRelTol(newton_rtol);
        newton_solver->SetMaxIter(newton_maxitr);
        newton_solver->SetPrintLevel(newton_printlvl);
        newton_solver->SetPreconditionerFactory(jac_factory);
    }
    virtual ~PNP_Newton_Solver_par()
    {
        delete newton_solver, op, jac_factory, u_k, mesh, pmesh;
    }

    void Solve()
    {
        cout << "---------------------- CG1, Newton, protein, parallel ----------------------" << endl;
        Vector zero;
        cout << "u_k l2 norm: " << u_k->Norml2() << endl;
        newton_solver->Mult(zero, *u_k); // u_k must be a true vector

        phi3_k.MakeTRef(h1_space, *u_k, block_trueoffsets[0]);
        c1_k  .MakeTRef(h1_space, *u_k, block_trueoffsets[1]);
        c2_k  .MakeTRef(h1_space, *u_k, block_trueoffsets[2]);
        phi3_k.SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();
        cout << "l2 norm of phi3: " << phi3_k.Norml2() << endl;
        cout << "l2 norm of   c1: " <<   c1_k.Norml2() << endl;
        cout << "l2 norm of   c2: " <<   c2_k.Norml2() << endl;
    }
};


#endif
