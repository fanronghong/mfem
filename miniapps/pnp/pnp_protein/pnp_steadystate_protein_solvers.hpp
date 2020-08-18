#ifndef _PNP_GUMMEL_SOLVER_HPP_
#define _PNP_GUMMEL_SOLVER_HPP_

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
#include "./pnp_protein_preconditioners.hpp"
using namespace std;
using namespace mfem;


class PNP_Gummel_CG_Solver_par
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
    Array<int> protein_dofs, water_dofs;
    Array<int> ess_bdr, top_bdr, bottom_bdr, interface_bdr, Gamma_m;
    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list;

    StopWatch chrono;
    int num_procs, myid;
    std::vector< Array<double> > Peclet;

public:
    PNP_Gummel_CG_Solver_par(Mesh* mesh_) : mesh(mesh_)
    {
        pmesh    = new ParMesh(MPI_COMM_WORLD, *mesh);
        h1_fec   = new H1_FECollection(p_order, mesh->Dimension());
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

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

        {
            int size = pmesh->bdr_attributes.Max();

            ess_bdr.SetSize(size);
            ess_bdr                    = 0;
            ess_bdr[top_marker - 1]    = 1;
            ess_bdr[bottom_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

            top_bdr.SetSize(size);
            top_bdr                 = 0;
            top_bdr[top_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

            bottom_bdr.SetSize(size);
            bottom_bdr                    = 0;
            bottom_bdr[bottom_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

            interface_bdr.SetSize(size);
            interface_bdr                       = 0;
            interface_bdr[interface_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

            Gamma_m.SetSize(size);
            Gamma_m                     = 0;
            Gamma_m[Gamma_m_marker - 1] = 1;
        }

        for (int i=0; i<h1_space->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_space->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            }
            else
            {
                assert(attr == water_marker);
                h1_space->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        protein_dofs.Sort();
        protein_dofs.Unique();
        water_dofs.Sort();
        water_dofs.Unique();
        for (int i=0; i<interface_ess_tdof_list.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        *phi3 = 0.0;
        *c1   = 0.0;
        *c2   = 0.0;
        *phi3_n = 0.0;
        *c1_n   = 0.0;
        *c2_n   = 0.0;
        // essential边界条件
        phi3->ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
        c1  ->ProjectBdrCoefficient( c1_D_top_coeff, top_bdr);
        c2  ->ProjectBdrCoefficient( c2_D_top_coeff, top_bdr);
        phi3->ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
        c1  ->ProjectBdrCoefficient( c1_D_bottom_coeff, bottom_bdr);
        c2  ->ProjectBdrCoefficient( c2_D_bottom_coeff, bottom_bdr);
        phi3->SetTrueVector();
        c1  ->SetTrueVector();
        c2  ->SetTrueVector();
        phi3->SetFromTrueVector();
        c1  ->SetFromTrueVector();
        c2  ->SetFromTrueVector();
        phi3_n->ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
        c1_n  ->ProjectBdrCoefficient( c1_D_top_coeff, top_bdr);
        c2_n  ->ProjectBdrCoefficient( c2_D_top_coeff, top_bdr);
        phi3_n->ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
        c1_n  ->ProjectBdrCoefficient( c1_D_bottom_coeff, bottom_bdr);
        c2_n  ->ProjectBdrCoefficient( c2_D_bottom_coeff, bottom_bdr);
        phi3_n->SetTrueVector();
        c1_n  ->SetTrueVector();
        c2_n  ->SetTrueVector();
        phi3_n->SetFromTrueVector();
        c1_n  ->SetFromTrueVector();
        c2_n  ->SetFromTrueVector();
        cout << "After set bdc, L2 norm of phi3: " << phi3_n->ComputeL2Error(zero) << endl;
        cout << "After set bdc, L2 norm of   c1: " << c1_n->ComputeL2Error(zero) << endl;
        cout << "After set bdc, L2 norm of   c2: " << c2_n->ComputeL2Error(zero) << endl;

        dc = new VisItDataCollection("data collection", mesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3", phi3);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);
    }
    ~PNP_Gummel_CG_Solver_par()
    {
        delete phi1, phi2, phi3, c1, c2, phi3_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve()
    {
        Solve_Singular();
        Solve_Harmonic();

        cout << "\n------> Gummel, CG" << p_order << ", Stabilization: " << AdvecStable << ". protein, parallel"
             << ", petsc option file: " << options_src
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << '\n' << endl;
        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson();

            Vector diff(h1_space->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi3);
            diff -= (*phi3_n); // 不能把上述2步合并成1步: diff = (*phi3) - (*phi3_n)fff
            double tol = diff.Norml2() / phi3->Norml2(); // 相对误差
            (*phi3_n) = (*phi3);

            if (strcmp(AdvecStable, "none") == 0)      Solve_NP1();
            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP1_EAFE();
            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP1_SUPG();
            else MFEM_ABORT("Not support stabilization.");
            (*c1_n) = (*c1);

            if (strcmp(AdvecStable, "none") == 0)      Solve_NP2();
            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP2_EAFE();
            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP2_SUPG();
            else MFEM_ABORT("Not support stabilization.");
            (*c2_n) = (*c2);

            if (verbose) {
                cout << "L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
                cout << "L2 norm of   c1: " << c1->ComputeL2Error(zero) << endl;
                cout << "L2 norm of   c2: " << c2->ComputeL2Error(zero) << endl;
            }

            cout << "======> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
                break;

            iter++;
            cout << endl;
        }

        if (iter == Gummel_max_iters) MFEM_ABORT("===> Gummel Not converge!!!");

        cout << "===> Gummel iteration converge!!!" << endl;
        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c1: " << c1->ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c2: " << c2->ComputeL2Error(zero) << '\n' << endl;

        if (visualize)
        {
//            (*phi3) += (*phi1); //把总的电势全部加到phi3上面
//            (*phi3) += (*phi2);
            (*phi3) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi3", "phi3 (with units)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            cout << "save output: gummel_cg_phi_c1_c2.vtk" << endl;
            ofstream results("gummel_cg_phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh->PrintVTK(results, ref);
            phi3->SaveVTK(results, "phi", ref);
            c1  ->SaveVTK(results, "c1", ref);
            c2  ->SaveVTK(results, "c2", ref);
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(Epsilon, *phi3, error);
            ComputeLocalConservation(D_K_, *c1, v_K_coeff, *phi3, error1);
            ComputeLocalConservation(D_Cl_, *c2, v_Cl_coeff, *phi3, error2);

            string mesh_temp(mesh_file);
            mesh_temp.erase(mesh_temp.find(".msh"), 4);
            mesh_temp.erase(mesh_temp.find("./"), 2);
            string name = "_ref" + to_string(refine_times) + "_" + string(Linearize) + "_"  + string(Discretize) + "_"  + mesh_temp;
            string title1 = "c1_conserv" + name;
            string title2 = "c2_conserv" + name;

            ofstream file1(title1), file2(title2);
            if (file1.is_open() && file2.is_open())
            {
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }

        if (show_peclet)
        {
            string mesh_temp(mesh_file);
            mesh_temp.erase(mesh_temp.find(".msh"), 4);
            mesh_temp.erase(mesh_temp.find("./"), 2);

            string name = "_ref" + to_string(refine_times) + "_" + mesh_temp + "_" + string(Discretize) + "_" + string(Linearize);
            string title1  = "c1_Peclet" + name;
            string title2  = "c2_Peclet" + name;

            for (int i=0; i<Peclet.size(); ++i)
            {
                ofstream file1(title1 + to_string(i)), file2(title2 + to_string(i));
                if (file1.is_open() && file2.is_open())
                {
                    Peclet[i].Print(file1, 1);
                    Peclet[i+1].Print(file2, 1);
                    i++;
                }
                else MFEM_ABORT("Peclet quantities not save!");

                file1.close();
                file2.close();
            }
        }
    }

private:
    // 1.求解奇异电荷部分的电势
    void Solve_Singular()
    {
        phi1->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取
        phi1->SetTrueVector();
        phi1->SetFromTrueVector();
        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;

        if (self_debug && strcmp(pqr_file, "./1MAG.pqr") == 0 && strcmp(mesh_file, "./1MAG_2.msh") == 0)
        {
            /* Only need a pqr file, we can compute singular electrostatic potential phi1, no need for mesh file.
             * Here for pqr file "../data/1MAG.pqr", we do a simple test for phi1. Data is provided by Zhang Qianru.
             */
            Vector zero_(3);
            zero_ = 0.0;
            VectorConstantCoefficient zero_vec(zero_);

            double L2norm = phi1->ComputeL2Error(zero);
            assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of phi1 (no units)" << endl;

            H1_FECollection temp_fec(1, mesh->Dimension());
            FiniteElementSpace temp_fes(mesh, h1_fec, 3);
            GridFunction grad_phi1(&temp_fes);
            grad_phi1.ProjectCoefficient(gradG_coeff);
            double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
            assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供
            cout << "======> Test Pass: L2 norm of grad(phi1) (no units)" << endl;
        }

    }

    // 2.求解调和方程部分的电势
    void Solve_Harmonic()
    {
        ParBilinearForm blf(h1_space);
        // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m: protein domain
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        ParLinearForm lf(h1_space);
        // -<grad(G).n, psi2>_{\Gamma_M}, G is phi1
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m);
        lf.Assemble();

        phi2->ProjectCoefficient(G_coeff);
        phi2->Neg(); // 在interface(\Gamma)上是Dirichlet边界: -phi1

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf.SetOperatorType(Operator::PETSC_MATAIJ);
        blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, *A, *x, *b);

        A->EliminateRows(water_dofs, 1.0); // ffff自己修改了源码: 重载了这个函数
        if (self_debug)
        {   // 确保只在水中(不包括蛋白质和interface)的自由度为0
            for (int i = 0; i < water_dofs.Size(); i++)
                assert(abs((*b)(water_dofs[i])) < 1E-10);
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi2_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf.RecoverFEMSolution(*x, lf, *phi2);
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;

        if (verbose) {
            cout << "\nL2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;
            if (solver->GetConverged() == 1 && myid == 0)
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

        delete solver, A, x, b;
    }

    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson()
    {
        GridFunctionCoefficient c1_n_coeff(c1_n), c2_n_coeff(c2_n);

        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        // epsilon (grad(phi3), grad(psi3))_{\Oemga}
        blf->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf(new ParLinearForm(h1_space)); //Poisson方程的右端项
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        ProductCoefficient lf1(rhs1, mark_water_coeff);
        ProductCoefficient lf2(rhs2, mark_water_coeff);
        // (alpha2 alpha3 z1 c1^k, psi3)_{\Omega_s}
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf1));
        // (alpha2 alpha3 z2 c2^k, psi3)_{\Omega_s}
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf2));
        // - epsilon_m <grad(phi1 + phi2).n, psi3>_{\Gamma}, interface integrator
        lf->AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator(&neg_epsilon_protein, &grad_phi1_plus_grad_phi2, mesh, protein_marker, water_marker));
        // omit 0 Neumann bdc on \Gamma_N and \Gamma_M
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *phi3, *lf, *A, *x, *b); // ess_tdof_list include: top, bottom

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi3_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *phi3);

        (*phi3_n) *= relax;
        (*phi3)   *= 1-relax;
        (*phi3)   += (*phi3_n); // 利用松弛方法更新phi3
        (*phi3_n) /= relax+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

        if (verbose)
        {
            cout << "            L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
            if (solver->GetConverged() == 1 && myid == 0)
                cout << "phi3 solver: successfully converged by iterating " << solver->GetNumIterations()
                     << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi3 solver: failed to converged" << endl;
        }

        delete blf, lf, solver, A, x, b;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1()
    {
        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        // D1 (grad(c1), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (c1 grad(phi3^k), grad(v1))_{\Omega_s}
        GradConvectionIntegrator* integ = new GradConvectionIntegrator(*phi3_n, &D1_prod_z1_water);
        blf->AddDomainIntegrator(integ);
        blf->Assemble(0);
        blf->Finalize(0);

        Peclet.push_back(integ->local_peclet);
        delete integ;

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (1) // 去掉蛋白区域的自由度
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np1_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c1: " << c1->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np1  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np1  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c1);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c1_n) *= relax;
        (*c1)   *= 1-relax;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
    void Solve_NP1_EAFE()
    {
        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        // D1 (grad(c1), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (c1 grad(phi3^k), grad(v1))_{\Omega_s}
        GradConvectionIntegrator* integ = new GradConvectionIntegrator(*phi3_n, &D1_prod_z1_water);
        blf->AddDomainIntegrator(integ);
        blf->Assemble(0);
        blf->Finalize(0);

        delete integ;

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        GradientGridFunctionCoefficient grad_phi3_n(phi3_n);
        ScalarVectorProductCoefficient adv(D1_prod_z1_water, grad_phi3_n); // advection, diffusion=D1
        SparseMatrix& _A = blf->SpMat();
        EAFE_Modify(*mesh, _A, D1_water, adv);
        blf->EliminateVDofs(ess_tdof_list, *c1, *lf);

        PetscParMatrix *A = new PetscParMatrix(&_A, Operator::PETSC_MATAIJ);
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);

        if (1)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np1_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c1: " << c1->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np1  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np1  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c1);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c1_n) *= relax;
        (*c1)   *= 1-relax;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
    void Solve_NP1_SUPG()
    {
        GradientGridFunctionCoefficient grad_phi3_n(phi3_n);
        ScalarVectorProductCoefficient adv(D1_prod_z1_water, grad_phi3_n); // advection, diffusion=D1

        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D1_water(D_K_, mark_water_coeff);
        ProductCoefficient D1_prod_z1_water(D_K_prod_v_K, mark_water_coeff);
        // D1 (grad(c1), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (c1 grad(phi3^k), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_n, &D1_prod_z1_water));
        // tau_k (adv . grad(c1), adv . grad(v1))
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&D1_water, one, adv, zero, zero, *mesh));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (0)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np1_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c1: " << c1->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np1  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np1  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c1);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c1_n) *= relax;
        (*c1)   *= 1-relax;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2()
    {
        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        // D2 (grad(c2), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (c2 grad(phi3^k), grad(v2))_{\Omega_s}
        GradConvectionIntegrator* integ = new GradConvectionIntegrator(*phi3_n, &D2_prod_z2_water);
        blf->AddDomainIntegrator(integ);
        blf->Assemble(0);
        blf->Finalize(0);

        Peclet.push_back(integ->local_peclet);
        delete integ;

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (1) // 去掉蛋白区域的自由度，形成一个较小的代数系统
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A * x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np2_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c2: " << c2->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np2  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np2  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c2);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c2_n) *= relax;
        (*c2)   *= 1-relax;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
    void Solve_NP2_EAFE()
    {
        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        // D2 (grad(c2), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (c2 grad(phi3^k), grad(v2))_{\Omega_s}
        GradConvectionIntegrator* integ = new GradConvectionIntegrator(*phi3_n, &D2_prod_z2_water);
        blf->AddDomainIntegrator(integ);
        blf->Assemble(0);
        blf->Finalize(0);

        delete integ;

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        GradientGridFunctionCoefficient grad_phi3_n(phi3_n);
        ScalarVectorProductCoefficient adv(D2_prod_z2_water, grad_phi3_n); // advection, diffusion=D1
        SparseMatrix& _A = blf->SpMat();
        EAFE_Modify(*mesh, _A, D2_water, adv);
        blf->EliminateVDofs(ess_tdof_list, *c2, *lf);

        PetscParMatrix *A = new PetscParMatrix(&_A, Operator::PETSC_MATAIJ);
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);

        if (1)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A * x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np2_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c2: " << c2->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np2  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np2  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c2);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c2_n) *= relax;
        (*c2)   *= 1-relax;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
    void Solve_NP2_SUPG()
    {
        GradientGridFunctionCoefficient grad_phi3_n(phi3_n);
        ScalarVectorProductCoefficient adv(D2_prod_z2_water, grad_phi3_n); // advection, diffusion=D2

        ParBilinearForm *blf(new ParBilinearForm(h1_space));
        ProductCoefficient D2_water(D_Cl_, mark_water_coeff);
        ProductCoefficient D2_prod_z2_water(D_Cl_prod_v_Cl, mark_water_coeff);
        // D2 (grad(c2), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (c2 grad(phi3^k), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_n, &D2_prod_z2_water));
        // tau_k (adv . grad(c2), adv . grad(v2))
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&D2_water, one, adv, zero, zero, *mesh));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf(new ParLinearForm(h1_space));
        // omit zero Neumann bdc
        *lf = 0.0;

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_space);
        PetscParVector *b = new PetscParVector(h1_space);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (1)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A); // form linear system: A * x = b
            Vec vec = Vec(*b), sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // extract subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb); // VecGetSubVector, VecRestoreSubVector
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np2_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            A->EliminateRows(protein_dofs, 1.0);

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c2: " << c2->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np2  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np2  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c2);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }

        (*c2_n) *= relax;
        (*c2)   *= 1-relax;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
};


class PNP_Gummel_DG_Solver_par
{
private:
    Mesh* mesh;
    ParMesh* pmesh;
    DG_FECollection* fec;
    ParFiniteElementSpace* fes;

    /* 将电势分解成3部分: 奇异电荷部分phi1, 调和部分phi2, 其余部分phi3,
    * ref: Poisson–Nernst–Planck equations for simulating biomolecular diffusion–reaction processes I: Finite element solutions
    * */
    ParGridFunction *phi1, *phi2, *phi2_;
    ParGridFunction *phi3, *c1, *c2;       // FE 解
    ParGridFunction *phi3_n, *c1_n, *c2_n; // Gummel迭代解

    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_fes;
    ParGridFunction *phi3_e, *c1_e, *c2_e;

    VisItDataCollection* dc;
    // protein_dofs和water_dofs里面不包含interface_ess_tdof_list
    Array<int> protein_dofs, water_dofs;
    Array<int> ess_bdr, top_bdr, bottom_bdr, interface_bdr, Gamma_m;
    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list;

    StopWatch chrono;
    int num_procs, myid;
    Array<int> null_array;

public:
    PNP_Gummel_DG_Solver_par(Mesh* mesh_) : mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        fec   = new DG_FECollection(p_order, mesh->Dimension());
        fes   = new ParFiniteElementSpace(pmesh, fec);

        MPI_Comm_size(fes->GetComm(), &num_procs);
        MPI_Comm_rank(fes->GetComm(), &myid);

        phi1   = new ParGridFunction(fes);
        phi2   = new ParGridFunction(fes);
        phi3   = new ParGridFunction(fes);
        c1     = new ParGridFunction(fes);
        c2     = new ParGridFunction(fes);
        phi3_n = new ParGridFunction(fes);
        c1_n   = new ParGridFunction(fes);
        c2_n   = new ParGridFunction(fes);

        {
            int size = pmesh->bdr_attributes.Max();

            ess_bdr.SetSize(size);
            ess_bdr                    = 0;
            ess_bdr[top_marker - 1]    = 1;
            ess_bdr[bottom_marker - 1] = 1;
//            fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // error, https://github.com/mfem/mfem/issues/1048

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

        for (int i=0; i<fes->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                fes->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            }
            else
            {
                assert(attr == water_marker);
                fes->GetElementDofs(i,dofs);
                water_dofs.Append(dofs);
            }
        }
        { // 因为是DG, 所以不可能有重复的dof, 可以把下面的内容注释掉
            protein_dofs.Sort();
            protein_dofs.Unique();
            water_dofs.Sort();
            water_dofs.Unique();
            for (int i=0; i<interface_ess_tdof_list.Size(); i++) // 去掉protein和water中的interface上的dofs
            {
                protein_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
                water_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
            }
        }

        *phi3 = 0.0;
        *c1   = 0.0;
        *c2   = 0.0;
        *phi3_n = 0.0;
        *c1_n   = 0.0;
        *c2_n   = 0.0;
        { // set essential bdc
            H1_FECollection h1_fec(p_order, mesh->Dimension());
            ParFiniteElementSpace h1_fes(pmesh, &h1_fec);

            ParGridFunction phi3_D_h1(&h1_fes), c1_D_h1(&h1_fes), c2_D_h1(&h1_fes);
            phi3_D_h1 = 0.0;
            c1_D_h1   = 0.0;
            c2_D_h1   = 0.0;

            phi3_D_h1.ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
            phi3_D_h1.SetTrueVector();
            phi3_D_h1.ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
            phi3_D_h1.SetTrueVector();

            phi3->ProjectGridFunction(phi3_D_h1);
            phi3->SetTrueVector();
            phi3_n->ProjectGridFunction(phi3_D_h1);
            phi3_n->SetTrueVector();

            c1_D_h1.ProjectBdrCoefficient(c1_D_top_coeff, top_bdr);
            c1_D_h1.SetTrueVector();
            c1_D_h1.ProjectBdrCoefficient(c1_D_bottom_coeff, bottom_bdr);
            c1_D_h1.SetTrueVector();
            c1->ProjectGridFunction(c1_D_h1);
            c1->SetTrueVector();
            c1_n->ProjectGridFunction(c1_D_h1);
            c1_n->SetTrueVector();

            c2_D_h1.ProjectBdrCoefficient(c2_D_top_coeff, top_bdr);
            c2_D_h1.SetTrueVector();
            c2_D_h1.ProjectBdrCoefficient(c2_D_bottom_coeff, bottom_bdr);
            c2_D_h1.SetTrueVector();
            c2->ProjectGridFunction(c2_D_h1);
            c2->SetTrueVector();
            c2_n->ProjectGridFunction(c2_D_h1);
            c2_n->SetTrueVector();

            cout << "After set bdc, L2 norm of phi3: " << phi3_n->ComputeL2Error(zero) << endl;
            cout << "After set bdc, L2 norm of   c1: " << c1_n->ComputeL2Error(zero) << endl;
            cout << "After set bdc, L2 norm of   c2: " << c2_n->ComputeL2Error(zero) << endl;
        }

        dc = new VisItDataCollection("data collection", mesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3", phi3);
        dc->RegisterField("c1",   c1);
        dc->RegisterField("c2",   c2);

//        Visualize(*dc, "phi3", "phi3");
//        Visualize(*dc, "c1", "c1");
//        Visualize(*dc, "c2", "c2");
//        MFEM_ABORT("stop for visualizing");
    }
    ~PNP_Gummel_DG_Solver_par()
    {
        delete phi1, phi2, phi3, c1, c2, phi3_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve()
    {
        Solve_Singular();
        Solve_Harmonic();

        cout << "\n------> Gummel, DG" << p_order << ", protein, parallel"
             << ", petsc option file: " << options_src
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << '\n' << endl;
        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_Poisson();

            Vector diff(fes->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi3);
            diff -= (*phi3_n); // 不能把上述2步合并成1步: diff = (*phi3) - (*phi3_n)fff
            double tol = diff.Norml2() / phi3->Norml2(); // 相对误差
            (*phi3_n) = (*phi3);

            Solve_NP1();
            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            if (verbose) {
                cout << "L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
                cout << "L2 norm of   c1: " << c1->ComputeL2Error(zero) << endl;
                cout << "L2 norm of   c2: " << c2->ComputeL2Error(zero) << endl;
            }

            cout << "======> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
                break;

            iter++;
            cout << endl;
        }

        if (iter == Gummel_max_iters) MFEM_ABORT("===> Gummel Not converge!!!");

        cout << "===> Gummel iteration converge!!!" << endl;
        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c1: " << c1->ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c2: " << c2->ComputeL2Error(zero) << '\n' << endl;

        if (visualize)
        {
//            (*phi3) += (*phi1); //把总的电势全部加到phi3上面
//            (*phi3) += (*phi2);
            (*phi3) /= alpha1;
            (*c1)   /= alpha3;
            (*c2)   /= alpha3;
            Visualize(*dc, "phi3", "phi3 (with units)");
            Visualize(*dc, "c1", "c1 (with units)");
            Visualize(*dc, "c2", "c2 (with units)");

            cout << "save output: gummel_cg_phi_c1_c2.vtk" << endl;
            ofstream results("gummel_cg_phi_c1_c2.vtk");
            results.precision(14);
            int ref = 0;
            mesh->PrintVTK(results, ref);
            phi3->SaveVTK(results, "phi", ref);
            c1  ->SaveVTK(results, "c1", ref);
            c2  ->SaveVTK(results, "c2", ref);
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(Epsilon, *phi3, error);
            ComputeLocalConservation(D_K_, *c1, v_K_coeff, *phi3, error1);
            ComputeLocalConservation(D_Cl_, *c2, v_Cl_coeff, *phi3, error2);

            string mesh_temp(mesh_file);
            mesh_temp.erase(mesh_temp.find(".msh"), 4);
            mesh_temp.erase(mesh_temp.find("./"), 2);
            string name = "_ref" + to_string(refine_times) + "_" + string(Linearize) + "_"  + string(Discretize) + "_"  + mesh_temp;
            string title1 = "c1_conserv" + name;
            string title2 = "c2_conserv" + name;

            ofstream file1(title1), file2(title2);
            if (file1.is_open() && file2.is_open())
            {
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }
    }

private:
    // 1.求解奇异电荷部分的电势
    void Solve_Singular()
    {
        phi1->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取
        phi1->SetTrueVector();
        phi1->SetFromTrueVector();

        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;
    }

    // 2.求解调和方程部分的电势
    void Solve_Harmonic()
    {
        h1_fec = new H1_FECollection(p_order, mesh->Dimension());
        h1_fes = new ParFiniteElementSpace(pmesh, h1_fec);

        phi3_e = new ParGridFunction(h1_fes);
        c1_e   = new ParGridFunction(h1_fes);
        c2_e   = new ParGridFunction(h1_fes);

        int size = pmesh->bdr_attributes.Max();
        Array<int> ess_tdof_list_, interface_ess_tdof_list_;

        ess_bdr.SetSize(size);
        ess_bdr                    = 0;
        ess_bdr[top_marker - 1]    = 1;
        ess_bdr[bottom_marker - 1] = 1;
        h1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);

        interface_bdr.SetSize(size);
        interface_bdr                       = 0;
        interface_bdr[interface_marker - 1] = 1;
        h1_fes->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list_);

        Array<int> protein_dofs_, water_dofs_;
        for (int i=0; i<h1_fes->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                h1_fes->GetElementDofs(i, dofs);
                protein_dofs_.Append(dofs);
            }
            else
            {
                assert(attr == water_marker);
                h1_fes->GetElementDofs(i,dofs);
                water_dofs_.Append(dofs);
            }
        }
        protein_dofs_.Sort();
        protein_dofs_.Unique();
        water_dofs_.Sort();
        water_dofs_.Unique();
        for (int i=0; i<interface_ess_tdof_list.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs_.DeleteFirst(interface_ess_tdof_list[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        ParBilinearForm blf(h1_fes);
        // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m: protein domain
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        ParLinearForm lf(h1_fes);
        // -<grad(G).n, psi2>_{\Gamma_M}, G is phi1
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m);
        lf.Assemble();

        ParGridFunction* phi2_ = new ParGridFunction(h1_fes);
        phi2_->ProjectCoefficient(G_coeff);
        phi2_->Neg(); // 在interface(\Gamma)上是Dirichlet边界: -phi1

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(h1_fes);
        PetscParVector *b = new PetscParVector(h1_fes);
        blf.SetOperatorType(Operator::PETSC_MATAIJ);
        blf.FormLinearSystem(interface_ess_tdof_list_, *phi2_, lf, *A, *x, *b);

        A->EliminateRows(water_dofs_, 1.0); // ffff自己修改了源码: 重载了这个函数
        if (self_debug)
        {   // 确保只在水中(不包括蛋白质和interface)的自由度为0
            for (int i = 0; i < water_dofs.Size(); i++)
                assert(abs((*b)(water_dofs[i])) < 1E-10);
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi2_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf.RecoverFEMSolution(*x, lf, *phi2_);

        if (verbose) {
            cout << "\nL2 norm of phi2: " << phi2_->ComputeL2Error(zero) << endl;
            if (solver->GetConverged() == 1 && myid == 0)
                cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations()
                     << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi2 solver: failed to converged" << endl;
        }

        phi2->ProjectGridFunction(*phi2_); // project from h1 space to dg1
        phi2->SetTrueVector();
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;
        delete solver, A, x, b;
    }

    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson()
    {
//        phi3_n->ProjectGridFunction(*phi3_e); // verify code
//        c1_n  ->ProjectGridFunction(*c1_e); // verify code
//        c2_n  ->ProjectGridFunction(*c2_e); // verify code
//        phi3_n->SetTrueVector();
//        c1_n->SetTrueVector();
//        c2_n->SetTrueVector();
//        cout << "L2 norm phi3_n (project from file): " << phi3_n->ComputeL2Error(zero) << endl;
//        cout << "L2 norm   c1_n (project from file): " << c1_n->ComputeL2Error(zero) << endl;
//        cout << "L2 norm   c2_n (project from file): " << c2_n->ComputeL2Error(zero) << endl;
//        MFEM_ABORT("stop for verify project from file");

        GridFunctionCoefficient c1_n_coeff(c1_n), c2_n_coeff(c2_n);
        double kappa = 20;

        ParBilinearForm *blf(new ParBilinearForm(fes));
        // Epsilon (grad(phi3), grad(psi3))_{\Omega}. Epsilon=epsilon_m in \Omega_m, Epsilon=epsilon_s in \Omega_s
        blf->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));
        // - <{Epsilon grad(phi3)}, [psi3]> + sigma <[phi3], {Epsilon grad(psi3)}> + kappa <{h^{-1} Epsilon}>
        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(Epsilon, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(Epsilon, sigma, kappa), ess_bdr);
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf(new ParLinearForm(fes)); //Poisson方程的右端项
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        VectorSumCoefficient grad_phi1_plus_grad_phi2(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        ProductCoefficient lf1(rhs1, mark_water_coeff);
        ProductCoefficient lf2(rhs2, mark_water_coeff);
        // (alpha2 alpha3 z1 c1^k, psi3)_{\Omega_s}
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf1));
        // (alpha2 alpha3 z2 c2^k, psi3)_{\Omega_s}
        lf->AddDomainIntegrator(new DomainLFIntegrator(lf2));
        // epsilon_m <grad(phi1 + phi2).n, {psi3}>_{\Gamma}, interface integrator, see below another way to define interface integrate
        lf->AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator1(&epsilon_protein_mark, &grad_phi1_plus_grad_phi2, mesh, protein_marker, water_marker));
        // sigma <phi3_D, (Epsilon grad(psi3).n)> + kappa <{h^{-1} Epsilon} phi3_D, psi3>. phi3_D includes phi_D_top and phi_D_bottom
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_top_coeff, epsilon_water, sigma, kappa), top_bdr);
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_bottom_coeff, epsilon_water, sigma, kappa), bottom_bdr);
        // omit 0 Neumann bdc on \Gamma_N and \Gamma_M
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fes);
        PetscParVector *b = new PetscParVector(fes);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(null_array, *phi3, *lf, *A, *x, *b); // ess_tdof_list include: top, bottom

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi3_");

        if (0)
        {
            ParBilinearForm* prec = new ParBilinearForm(fes);
            // Epsilon (grad(phi3), grad(psi3))_{\Omega}. Epsilon=epsilon_m in \Omega_m, Epsilon=epsilon_s in \Omega_s
            prec->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));
            // - <{Epsilon grad(phi3)}, [psi3]> + sigma <[phi3], {Epsilon grad(psi3)}> + kappa <{h^{-1} Epsilon}>
            prec->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(Epsilon, sigma, kappa));
            prec->AddBdrFaceIntegrator(new DGDiffusionIntegrator(Epsilon, sigma, kappa), ess_bdr);
            prec->Assemble();

            PetscParMatrix PC;
            prec->SetOperatorType(Operator::PETSC_MATAIJ);
            prec->FormSystemMatrix(ess_tdof_list, PC);

            PetscLinearSolver* pc = new PetscLinearSolver(PC, "phi3SPDPC_");
            solver->SetPreconditioner(*pc);
        }

        {
//            phi3_n->ProjectGridFunction(*phi3_e); // verify code
//            phi3_n->SetTrueVector();
//            Vector temp(*b);
//            A->Mult(1.0, *phi3_n, -1.0, temp);
//            cout << "l2 norm of ||A phi3_e - b|| / ||b||: " << temp.Norml2() / b->Norml2() << endl;
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        {
//            Vector temp(*b);
//            A->Mult(1.0, *x, -1.0, temp);
//            cout << "l2 norm of ||A x - b|| / ||b||: " << temp.Norml2() / b->Norml2() << endl;
        }
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *phi3);

        (*phi3_n) *= relax;
        (*phi3)   *= 1-relax;
        (*phi3)   += (*phi3_n); // 利用松弛方法更新phi3
        (*phi3_n) /= relax+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

        if (verbose) {
            cout << "            L2 norm of phi3: " << phi3->ComputeL2Error(zero) << endl;
            if (solver->GetConverged() == 1 && myid == 0)
                cout << "phi3 solver: successfully converged by iterating " << solver->GetNumIterations()
                     << " times, taking " << chrono.RealTime() << " s." << endl;
            else if (solver->GetConverged() != 1)
                cerr << "phi3 solver: failed to converged" << endl;
        }
        delete blf, lf, solver, A, x, b;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1()
    {
        ParBilinearForm *blf(new ParBilinearForm(fes));
        // D1 (grad(c1), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (c1 grad(phi3^k), grad(v1))_{\Omega_s}
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_n, &D1_prod_z1_water));
        // - <{D1 grad(c1)}, [v1]> + sigma <[c1], {D1 grad(v1)}> + kappa <{h^{-1} D1} [c1], [v1]> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D1_water, sigma, kappa, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D1_water, sigma, kappa, mesh, water_marker), ess_bdr);
        // - <{D1 z1 c1 grad(phi^k)}, [v1]> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(D1_prod_z1_water, *phi3_n, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(D1_prod_z1_water, *phi3_n, mesh, water_marker), ess_bdr);
        // sigma <[c1], {D1 z1 v1 grad(phi3^k}> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, *phi3_n, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, *phi3_n, mesh, water_marker), ess_bdr);
        blf->Assemble(1);
        blf->Finalize(1);

        ParLinearForm *lf(new ParLinearForm(fes));
        // sigma <c1_D, D1 grad(v1).n> + kappa <{h^{-1} D1} c1_D, v1>
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_D_top_coeff, D_K_, sigma, kappa), top_bdr);
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_D_bottom_coeff, D_K_, sigma, kappa), bottom_bdr);
        // sigma <c1_D, D1 z1 v1 grad(phi3^k)>
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_D_top_coeff, phi3_n), top_bdr);
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_D_bottom_coeff, phi3_n), bottom_bdr);
        // omit zero Neumann bdc
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fes);
        PetscParVector *b = new PetscParVector(fes);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(null_array, *c1, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (1)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A);
            Vec vec = Vec(*b);
            Vec sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb);
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np1_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");
            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c1: " << c1->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np1  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np1  solver: failed to converged" << endl;
            }

            delete solver;
        }

        blf->RecoverFEMSolution(*x, *lf, *c1);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c1)[protein_dofs[i]]) < 1E-10);
            }
        }

//        (*c1_n) *= relax_c1;
//        (*c1)   *= 1-relax_c1;
//        (*c1)   += (*c1_n); // 利用松弛方法更新c1
//        (*c1_n) /= relax_c1; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2()
    {
        ParBilinearForm *blf(new ParBilinearForm(fes));
        // D2 (grad(c2), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (c2 grad(phi3^k), grad(v2))_{\Omega_s}
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_n, &D2_prod_z2_water));
        // - <{D2 grad(c2)}, [v2]> + sigma <[c2], {D2 grad(v2)}> + kappa <{h^{-1} D2} [c2], [v2]> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D2_water, sigma, kappa, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D2_water, sigma, kappa, mesh, water_marker), ess_bdr);
        // - <{D2 z2 c2 grad(phi^k)}, [v2]> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(D2_prod_z2_water, *phi3_n, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(D2_prod_z2_water, *phi3_n, mesh, water_marker), ess_bdr);
        // sigma <[c2], {D2 z2 v2 grad(phi3^k}> on \mathcal(E)_h^{0,s} \cupp \mathcal(E)_h^D
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi3_n, mesh, water_marker));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi3_n, mesh, water_marker), ess_bdr);
        blf->Assemble(1);
        blf->Finalize(1);

        ParLinearForm *lf(new ParLinearForm(fes));
        // sigma <c2_D, D2 grad(v2).n> + kappa <{h^{-1} D2} c2_D, v2>
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_D_top_coeff, D_Cl_, sigma, kappa), top_bdr);
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_D_bottom_coeff, D_Cl_, sigma, kappa), bottom_bdr);
        // sigma <c2_D, D2 z2 v2 grad(phi3^k)>
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_D_top_coeff, phi3_n), top_bdr);
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_D_bottom_coeff, phi3_n), bottom_bdr);
        // omit zero Neumann bdc
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fes);
        PetscParVector *b = new PetscParVector(fes);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(null_array, *c2, *lf, *A, *x, *b);

        if (self_debug) {
            for (int i = 0; i < protein_dofs.Size(); ++i) {
                assert(abs((*b)(protein_dofs[i])) < 1E-10);
            }
        }

        if (1)
        {
            Array<int> need_dofs;
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort();

            Mat mat = Mat(*A);
            Vec vec = Vec(*b);
            Vec sol = Vec(*x);
            PetscInt size = PetscInt(need_dofs.Size());

            PetscInt* indices;
            PetscMalloc1(size, &indices);
            for (int i=0; i<size; ++i) indices[i] = need_dofs[i];

            IS is;
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            // subsystem subA * subx = subb
            Mat subA;
            Vec subx, subb;
            MatCreateSubMatrix(*A, is, is, MAT_INITIAL_MATRIX, &subA);
            VecGetSubVector(vec, is, &subb);
            VecGetSubVector(sol, is, &subx);

            KSP ksp;
            KSPCreate(MPI_COMM_WORLD, &ksp);
            KSPSetOptionsPrefix(ksp, "np2_");
            KSPSetOperators(ksp, subA, subA);
            KSPSetFromOptions(ksp);
            KSPSolve(ksp, subb, subx);

            VecRestoreSubVector(vec, is, &subb);
            VecRestoreSubVector(sol, is, &subx);

            MatDestroy(&mat);
            VecDestroy(&vec);
            VecDestroy(&sol);
            ISDestroy(&is);
            KSPDestroy(&ksp);
        }
        else
        {
            PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");
            solver->iterative_mode = true;

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();

            if (verbose) {
                cout << "            L2 norm of c2: " << c2->ComputeL2Error(zero) << endl;
                if (solver->GetConverged() == 1 && myid == 0)
                    cout << "np2  solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "np2  solver: failed to converged" << endl;
            }

            delete solver;
        }
        blf->RecoverFEMSolution(*x, *lf, *c2);

        if (self_debug) {
            for (int i=0; i<protein_dofs.Size(); ++i) {
                assert(abs((*c2)[protein_dofs[i]]) < 1E-10);
            }
        }

//        (*c2_n) *= relax_c2;
//        (*c2)   *= 1-relax_c2;
//        (*c2)   += (*c2_n); // 利用松弛方法更新c2
//        (*c2_n) /= relax_c2+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, A, x, b;
    }
};



class PNP_Newton_CG_Operator_par: public Operator
{
protected:
    ParFiniteElementSpace *fsp;

    Array<int> block_trueoffsets;
    IS is, long_is;

    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution
    PetscNonlinearSolver* newton_solver;

    mutable ParLinearForm *f, *f1, *f2, *g;
    mutable ParBilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;
    mutable PetscParMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable PetscParMatrix *A12_, *A13_, *A21_, *A22_, *A31_, *A33_;

    ParGridFunction *phi1, *phi2;
    VectorCoefficient* grad_phi1_plus_grad_phi2;
    ParGridFunction *phi3_k, *c1_k, *c2_k;

    Array<int> need_dofs, all_dofs;
    Array<int> ess_bdr, top_bdr, bottom_bdr, interface_bdr, Gamma_m_bdr;
    Array<int> ess_tdof_list, top_ess_tdof_list, bottom_ess_tdof_list,
            interface_ess_tdof_list, water_dofs, protein_dofs, interface_dofs;

    StopWatch chrono;
    int num_procs, myid;
    Array<int> null_array;

public:
    PNP_Newton_CG_Operator_par(ParFiniteElementSpace *fsp_, ParGridFunction* phi1_, ParGridFunction* phi2_)
            : fsp(fsp_), phi1(phi1_), phi2(phi2_)
    {
        MPI_Comm_size(fsp->GetComm(), &num_procs);
        MPI_Comm_rank(fsp->GetComm(), &myid);

        {
            int size = fsp->GetMesh()->bdr_attributes.Max();

            ess_bdr.SetSize(size);
            ess_bdr = 0;
            ess_bdr[top_marker - 1] = 1;
            ess_bdr[bottom_marker - 1] = 1;
            fsp->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

            top_bdr.SetSize(size);
            top_bdr = 0;
            top_bdr[top_marker - 1] = 1;
            fsp->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

            bottom_bdr.SetSize(size);
            bottom_bdr = 0;
            bottom_bdr[bottom_marker - 1] = 1;
            fsp->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

            interface_bdr.SetSize(size);
            interface_bdr = 0;
            interface_bdr[interface_marker - 1] = 1;
            fsp->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);
        }

        Mesh *mesh = fsp->GetMesh();
        for (int i = 0; i < fsp->GetNE(); ++i) {
            Element *el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker) {
                fsp->GetElementDofs(i, dofs);
                protein_dofs.Append(dofs);
            } else {
                assert(attr == water_marker);
                fsp->GetElementDofs(i, dofs);
                water_dofs.Append(dofs);
            }
        }
        for (int i = 0; i < mesh->GetNumFaces(); ++i) {
            FaceElementTransformations *tran = mesh->GetFaceElementTransformations(i);
            if (tran->Elem2No > 0) // interior facet
            {
                const Element *e1 = mesh->GetElement(tran->Elem1No);
                const Element *e2 = mesh->GetElement(tran->Elem2No);
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
        protein_dofs  .Sort();   protein_dofs.Unique();
        water_dofs    .Sort();     water_dofs.Unique();
        interface_dofs.Sort(); interface_dofs.Unique();
        for (int i = 0; i < interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        {
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_ess_tdof_list);
            need_dofs.Sort(); // c_i在溶剂和界面区域的自由度

            all_dofs.SetSize(fsp->GetTrueVSize()); // phi_3在整个区域的自由度
            for (int i = 0; i < fsp->GetTrueVSize(); ++i)
                all_dofs[i] = i;

            PetscInt *indices; // only include water dofs and interface dofs
            PetscInt size = PetscInt(need_dofs.Size());
            PetscMalloc1(size, &indices);
            for (int i = 0; i < need_dofs.Size(); ++i) indices[i] = need_dofs[i];
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            PetscInt *long_indices; // include all dofs
            PetscInt long_size = PetscInt(all_dofs.Size());
            PetscMalloc1(long_size, &long_indices);
            for (int i = 0; i < all_dofs.Size(); ++i) long_indices[i] = all_dofs[i];
            ISCreateGeneral(MPI_COMM_WORLD, long_size, long_indices, PETSC_COPY_VALUES, &long_is);
            PetscFree(long_indices);
        }

        block_trueoffsets.SetSize(4); // number of variables + 1;
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = fsp->GetTrueVSize();
        block_trueoffsets[2] = need_dofs.Size();
        block_trueoffsets[3] = need_dofs.Size();
        block_trueoffsets.PartialSum();
        assert(fsp->GetTrueVSize() == (protein_dofs.Size() + water_dofs.Size() + interface_dofs.Size()));

        height = width = fsp->GetTrueVSize() + 2 * (water_dofs.Size() + interface_dofs.Size());

        rhs_k = new BlockVector(block_trueoffsets); // not block_offsets !!!
        jac_k = new BlockOperator(block_trueoffsets);
        phi3_k= new ParGridFunction(fsp);
        c1_k  = new ParGridFunction(fsp);
        c2_k  = new ParGridFunction(fsp);

        f   = new ParLinearForm(fsp);
        f1  = new ParLinearForm(fsp);
        f2  = new ParLinearForm(fsp);
        a21 = new ParBilinearForm(fsp);
        a22 = new ParBilinearForm(fsp);
        a31 = new ParBilinearForm(fsp);
        a33 = new ParBilinearForm(fsp);

        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        grad_phi1_plus_grad_phi2 = new VectorSumCoefficient(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)

        g  = new ParLinearForm(fsp);
        // epsilon_m <grad(phi1 + phi2).n, psi3>_{\Gamma}
        g->AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator(&epsilon_protein, grad_phi1_plus_grad_phi2, mesh, protein_marker, water_marker));
        g->Assemble();

        a11 = new ParBilinearForm(fsp);
        // epsilon (grad(dphi3), grad(psi3))_{\Omega}
        a11->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));
        a11->Assemble();
        a11->Finalize();
        a11->SetOperatorType(Operator::PETSC_MATAIJ);
        a11->FormSystemMatrix(ess_tdof_list, A11);

        a12 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z1 (dc1, psi3)_{\Omega_s}
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K_water));
        a12->Assemble();
        a12->Finalize();
        a12->SetOperatorType(Operator::PETSC_MATAIJ);
        a12->FormSystemMatrix(null_array, A12);
        A12_ = new PetscParMatrix(A12, all_dofs, need_dofs);

        a13 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z2 (dc2, psi3)_{\Omega_s}
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl_water));
        a13->Assemble();
        a13->Finalize();
        a13->SetOperatorType(Operator::PETSC_MATAIJ);
        a13->FormSystemMatrix(null_array, A13);
        A13_ = new PetscParMatrix(A13, all_dofs, need_dofs);
    }
    virtual ~PNP_Newton_CG_Operator_par()
    {
        delete f, f1, f2, g;
        delete a11, a12, a13, a21, a22, a31, a33;
        delete rhs_k, jac_k;
        delete newton_solver;
    }

    virtual void Mult(const Vector& x, Vector& y) const
    {
//        cout << "\nin PNP_Newton_Operator::Mult(), l2 norm of x: " << x.Norml2() << endl;
//        cout << "x size: " << x.Size() << ", y size: " << y.Size() << endl;

        int sc = fsp->GetTrueVSize();
        Vector x_(sc * 3);
        x_ = 0.0;
        for (int i=0; i<sc; ++i)               x_[i]                   = x[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = x[sc + i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = x[sc + need_dofs.Size() + i];

        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
        cout << "After set bdc (in Newton::Mult()), L2 norm of phi3: " << phi3_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::Mult()), L2 norm of   c1: " <<   c1_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::Mult()), L2 norm of   c2: " <<   c2_k->ComputeL2Error(zero) << endl;

        GridFunctionCoefficient c1_k_coeff(c1_k), c2_k_coeff(c2_k);

//        rhs_k->Update(y.GetData(), block_trueoffsets); // update residual
        Vector y1(y.GetData() + 0, sc);
        Vector y2(y.GetData() + sc, need_dofs.Size());
        Vector y3(y.GetData() + sc + need_dofs.Size(), need_dofs.Size());
        cout << "1. l2 norm of y: " << y.Norml2() << endl;

        delete f;
        f = new ParLinearForm(fsp);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff);
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff);
        SumCoefficient term(term1, term2);
        ProductCoefficient neg_term(neg, term);
        ProductCoefficient neg_term_water(neg_term, mark_water_coeff);
        // - alpha2 alpha3 (z1 c1^k + z2 c2^k, psi3)_{\Omega_s}
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term_water));
        // epsilon_m (grad(phi3^k), grad(psi3))_{\Omega_m}
        f->AddDomainIntegrator(new GradConvectionIntegrator2(&epsilon_protein_mark, phi3_k));
        // epsilon_s (grad(phi3^k), grad(psi3))_{\Omega_s}
        f->AddDomainIntegrator(new GradConvectionIntegrator2(&epsilon_water_mark, phi3_k));
        // omit zero Neumann bdc
        f->Assemble();
        (*f) += (*g); // add interface integrate
        f->SetSubVector(ess_tdof_list, 0.0);
        y1 = (*f);
        cout << "2. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y1: " << y1.Norml2() << endl;

        delete f1;
        f1 = new ParLinearForm(fsp);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        // D1 (grad(c1^k), grad(v1))_{\Omega_s}
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_water, c1_k));
        // D1 (z1 c1^k grad(phi3^k), grad(v1))_{\Omega_s}
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_prod_z1_water_c1_k, phi3_k));
        f1->Assemble();
        f1->SetSubVector(ess_tdof_list, 0.0);
        for (int i=0; i<need_dofs.Size(); ++i) y2[i] = (*f1)[need_dofs[i]];
        cout << "3. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y2: " << y2.Norml2() << endl;

        delete f2;
        f2 = new ParLinearForm(fsp);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        // D2 (grad(c2^k), grad(v2))_{\Omega_s}
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_water, c2_k));
        // D2 (z2 c2^k grad(phi3^k), grad(v2))_{\Omega_s}
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_prod_z2_water_c2_k, phi3_k));
        f2->Assemble();
        f2->SetSubVector(ess_tdof_list, 0.0);
        for (int i=0; i<need_dofs.Size(); ++i) y3[i] = (*f2)[need_dofs[i]];
        cout << "4. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y3: " << y3.Norml2() << endl;

    }

    virtual Operator &GetGradient(const Vector& x) const
    {
//        cout << "in PNP_Newton_Operator::GetGradient()" << endl;
//        cout << "x size: " << x.Size() << endl;

        int sc = fsp->GetTrueVSize();
        Vector x_(sc * 3);
        x_ = 0.0;
        for (int i=0; i<sc; ++i)               x_[i]                   = x[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = x[sc + i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = x[sc + need_dofs.Size() + i];

        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of phi3: " << phi3_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of   c1: " <<   c1_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of   c2: " <<   c2_k->ComputeL2Error(zero) << endl;

        GridFunctionCoefficient c1_k_coeff(c1_k), c2_k_coeff(c2_k);

        delete a21;
        a21 = new ParBilinearForm(fsp);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        // D1 z1 c1^k (grad(dphi3), grad(v1))_{\Omega_s}
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_water_c1_k));
        a21->Assemble(0);
        a21->Finalize(0);
        a21->SetOperatorType(Operator::PETSC_MATAIJ);
        a21->FormSystemMatrix(null_array, A21);
        A21.EliminateRows(ess_tdof_list, 0.0);
        A21_ = new PetscParMatrix(A21, need_dofs, all_dofs);

        delete a22;
        a22 = new ParBilinearForm(fsp);
        // D1 (grad(dc1), grad(v1))_{\Omega_s}
        a22->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // D1 z1 (dc1 grad(phi3^k), grad(v1))_{\Omega_s}
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D1_prod_z1_water));
        a22->Assemble(0);
        a22->Finalize(0);
        a22->SetOperatorType(Operator::PETSC_MATAIJ);
        a22->FormSystemMatrix(ess_tdof_list, A22);
        A22.EliminateRows(protein_dofs, 1.0);
        A22_ = new PetscParMatrix(A22, need_dofs, need_dofs);

        delete a31;
        a31 = new ParBilinearForm(fsp);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        // D2 z2 c2^k (grad(dphi3), grad(v2))_{\Omega_s}
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_water_c2_k));
        a31->Assemble(0);
        a31->Finalize(0);
        a31->SetOperatorType(Operator::PETSC_MATAIJ);
        a31->FormSystemMatrix(null_array, A31);
        A31.EliminateRows(ess_tdof_list, 0.0);
        A31_ = new PetscParMatrix(A31, need_dofs, all_dofs);

        delete a33;
        a33 = new ParBilinearForm(fsp);
        // D2 (grad(dc2), grad(v2))_{\Omega_s}
        a33->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // D2 z2 (dc2 grad(phi3^k), grad(v2))_{\Omega_ss}
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D2_prod_z2_water));
        a33->Assemble(0);
        a33->Finalize(0);
        a33->SetOperatorType(Operator::PETSC_MATAIJ);
        a33->FormSystemMatrix(ess_tdof_list, A33);
        A33.EliminateRows(protein_dofs, 1.0);
        A33_ = new PetscParMatrix(A33, need_dofs, need_dofs);

        jac_k = new BlockOperator(block_trueoffsets);
        jac_k->SetBlock(0, 0, &A11);
        jac_k->SetBlock(0, 1, A12_);
        jac_k->SetBlock(0, 2, A13_);
        jac_k->SetBlock(1, 0, A21_);
        jac_k->SetBlock(1, 1, A22_);
        jac_k->SetBlock(2, 0, A31_);
        jac_k->SetBlock(2, 2, A33_);
        return *jac_k;
    }
};
class PNP_Newton_CG_Solver_par
{
protected:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_space;

    PNP_Newton_CG_Operator_par* op;
    PetscPreconditionerFactory *jac_factory;
    PetscNonlinearSolver* newton_solver;

    Array<int> block_trueoffsets, top_bdr, bottom_bdr, interface_bdr, Gamma_m_bdr, top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list;
    Array<int> protein_dofs, water_dofs, interface_dofs, need_dofs;
    BlockVector* u_k;
    ParGridFunction phi3_k, c1_k, c2_k;
    ParGridFunction *phi1, *phi2;

    SNES snes;
    map<string, Array<double>> out1;
    map<string, double> out2;
    Array<double> linear_iter;
    double linearize_iter, total_time, ndofs, linear_avg_iter;
    PetscInt *its=0, num_its=100;
    PetscReal *residual_norms=0;

    StopWatch chrono;
    VisItDataCollection* dc;

public:
    PNP_Newton_CG_Solver_par(Mesh* mesh_): mesh(mesh_)
    {
        int mesh_dim = mesh->Dimension(); //网格的维数:1D,2D,3D
        pmesh    = new ParMesh(MPI_COMM_WORLD, *mesh);
        h1_fec   = new H1_FECollection(p_order, mesh_dim);
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        {
            int size = h1_space->GetMesh()->bdr_attributes.Max();

            top_bdr.SetSize(size);
            top_bdr                 = 0;
            top_bdr[top_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(top_bdr, top_ess_tdof_list);

            bottom_bdr.SetSize(size);
            bottom_bdr                    = 0;
            bottom_bdr[bottom_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(bottom_bdr, bottom_ess_tdof_list);

            interface_bdr.SetSize(size);
            interface_bdr                       = 0;
            interface_bdr[interface_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);

            Gamma_m_bdr.SetSize(size);
            Gamma_m_bdr                     = 0;
            Gamma_m_bdr[Gamma_m_marker - 1] = 1;
        }

        // extract protein dofs, water dofs and interface dofs
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
        protein_dofs.Sort();   protein_dofs.Unique();
        water_dofs.Sort();     water_dofs.Unique();
        interface_dofs.Sort(); interface_dofs.Unique();
        for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
        {
            protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
            water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
        }

        // combine water dofs and interface dofs to use in variable concentration.
        need_dofs.Append(water_dofs);
        need_dofs.Append(interface_ess_tdof_list);
        need_dofs.Sort();

        block_trueoffsets.SetSize(4);
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = h1_space->GetTrueVSize();
        block_trueoffsets[2] = need_dofs.Size();
        block_trueoffsets[3] = need_dofs.Size();
        block_trueoffsets.PartialSum();

        int sc = h1_space->GetTrueVSize();
        Vector x_(sc * 3); // 解向量在所有自由度上的值组成的一个长向量：浓度变量在蛋白中为0的部分被保留
        x_ = 0.0;
        phi3_k.MakeTRef(h1_space, x_, 0);
        c1_k  .MakeTRef(h1_space, x_, sc);
        c2_k  .MakeTRef(h1_space, x_, 2*sc);
        phi3_k = 0.0;
        c1_k   = 0.0;
        c2_k   = 0.0;
        phi3_k.ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
        phi3_k.ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
        c1_k.ProjectBdrCoefficient(c1_D_top_coeff, top_bdr);
        c1_k.ProjectBdrCoefficient(c1_D_bottom_coeff, bottom_bdr);
        c2_k.ProjectBdrCoefficient(c2_D_top_coeff, top_bdr);
        c2_k.ProjectBdrCoefficient(c2_D_bottom_coeff, bottom_bdr);
        phi3_k.SetTrueVector();
        c1_k.SetTrueVector();
        c2_k.SetTrueVector();
        phi3_k.SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();
        cout << "After set bdc (before Solve()), L2 norm of phi3: " << phi3_k.ComputeL2Error(zero) << endl;
        cout << "After set bdc (before Solve()), L2 norm of   c1: " <<   c1_k.ComputeL2Error(zero) << endl;
        cout << "After set bdc (before Solve()), L2 norm of   c2: " <<   c2_k.ComputeL2Error(zero) << endl;

        u_k = new BlockVector(block_trueoffsets); // 解向量只在所在的非0的自由度上的取值：浓度变量在蛋白中的为0的部分被去掉
        *u_k = 0.0;
        for (int i=0; i<sc; ++i)               u_k->GetBlock(0)[i] = x_[i];
        for (int i=0; i<need_dofs.Size(); ++i) u_k->GetBlock(1)[i] = x_[sc + need_dofs[i]];
        for (int i=0; i<need_dofs.Size(); ++i) u_k->GetBlock(2)[i] = x_[2*sc + need_dofs[i]];

        phi1 = new ParGridFunction(h1_space);
        phi1->ProjectCoefficient(G_coeff);
        phi1->SetTrueVector();
        phi1->SetFromTrueVector();
        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;

        phi2 = new ParGridFunction(h1_space);
        {
            ParBilinearForm blf(h1_space);
            // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m is protein domain
            blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
            blf.Assemble(0);
            blf.Finalize(0);

            ParLinearForm lf(h1_space);
            // -<grad(G).n, psi2>_{\Gamma_M}
            lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m_bdr); // Neumann bdc on Gamma_m
            lf.Assemble();

            phi2->ProjectCoefficient(G_coeff);
            phi2->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1

            PetscParMatrix *A = new PetscParMatrix();
            PetscParVector *x = new PetscParVector(h1_space);
            PetscParVector *b = new PetscParVector(h1_space);
            blf.SetOperatorType(Operator::PETSC_MATAIJ);
            blf.FormLinearSystem(interface_ess_tdof_list, *phi2, lf, *A, *x, *b); //除了ess_tdof_list以外是0的Neumann边界

            A->EliminateRows(water_dofs, 1.0);
            if (self_debug) {
               for (int i = 0; i < water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
                   assert(abs((*b)(water_dofs[i])) < 1E-10);
           }

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi2_");

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();
            blf.RecoverFEMSolution(*x, lf, *phi2);

            if (self_debug) {
                for (int i=0; i<interface_ess_tdof_list.Size(); i++)
                    assert(abs((*phi2)[interface_ess_tdof_list[i]] + (*phi1)[interface_ess_tdof_list[i]]) < 1E-8);
                for (int i=0; i<water_dofs.Size(); i++)
                    assert(abs((*phi2)[water_dofs[i]]) < 1E-10);
            }

            if (verbose) {
                if (solver->GetConverged() == 1)
                    cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "phi2 solver: failed to converged" << endl;
            }

            delete A, x, b, solver;
        }
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;

        dc = new VisItDataCollection("data collection", pmesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3_k", &phi3_k);
        dc->RegisterField("c1_k",   &c1_k);
        dc->RegisterField("c2_k",   &c2_k);
        Visualize(*dc, "phi1", "phi1");
        Visualize(*dc, "phi2", "phi2");
        Visualize(*dc, "phi3_k", "phi3_k");
        Visualize(*dc, "c1_k", "c1_k");
        Visualize(*dc, "c2_k", "c2_k");

        op = new PNP_Newton_CG_Operator_par(h1_space, phi1, phi2);

        // Set the newton solve parameters
        jac_factory   = new PreconditionerFactory(*op, prec_type);
        newton_solver = new PetscNonlinearSolver(h1_space->GetComm(), *op, "newton_");
        newton_solver->iterative_mode = true;
        newton_solver->SetPreconditionerFactory(jac_factory);

        snes = SNES(*newton_solver);
        PetscMalloc(num_its * sizeof(PetscInt), &its);
        PetscMalloc(num_its * sizeof(PetscReal), &residual_norms);
        SNESSetConvergenceHistory(snes, residual_norms, its, num_its, PETSC_TRUE);
    }
    virtual ~PNP_Newton_CG_Solver_par()
    {
        delete newton_solver, op, jac_factory, u_k, mesh, pmesh;
    }

    void Solve()
    {
        cout << "\nNewton, CG" << p_order << ", protein, parallel"
             << ", preconditioner: " << prec_type << ", petsc option file: " << options_src
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        Vector x_; // convert u_k to x_: 相当于extension
        int sc = h1_space->GetTrueVSize();
        x_.SetSize(sc * 3);
        x_ = 0.0;
        for (int i=0; i<sc; ++i)               x_[i]                   = u_k->GetBlock(0)[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = u_k->GetBlock(1)[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = u_k->GetBlock(2)[i];
        phi3_k.MakeTRef(h1_space, x_, 0);
        c1_k  .MakeTRef(h1_space, x_, sc);
        c2_k  .MakeTRef(h1_space, x_, 2*sc);
        phi3_k.SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();
        cout << "After set bdc (in Solve()), L2 norm of phi3: " << phi3_k.ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Solve()), L2 norm of   c1: " <<   c1_k.ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Solve()), L2 norm of   c2: " <<   c2_k.ComputeL2Error(zero) << endl;

        Vector zero_vec;
        cout << "initial u_k l2 norm: " << u_k->Norml2() << endl;
        chrono.Start();
        newton_solver->Mult(zero_vec, *u_k); // u_k must be a true vector
        chrono.Stop();

        cout << "\nNewton, CG" << p_order << ", protein, parallel"
             << ", preconditioner: " << prec_type << ", petsc option file: " << options_src
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        linearize_iter = newton_solver->GetNumIterations();
        total_time = chrono.RealTime();
        ndofs = u_k->Size();
        out2["linearize_iter"] = linearize_iter;
        out2["total_time"] = total_time;
        out2["ndofs"] = ndofs;
        SNESGetConvergenceHistory(snes, &residual_norms, &its, &num_its);
//        for (int i=0; i<num_its; ++i)
//            cout << residual_norms[i] << endl;
        for (int i=1; i<num_its; ++i)
            linear_iter.Append(its[i]);
        out1["linear_iter"] = linear_iter;
        linear_avg_iter = round(linear_iter.Sum() / linear_iter.Size());
        out2["linear_avg_iter"] = linear_avg_iter;

        map<string, Array<double>>::iterator it1;
        for (it1=out1.begin(); it1!=out1.end(); ++it1)
            (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
        map<string, double>::iterator it2;
        for (it2=out2.begin(); it2!=out2.end(); ++it2)
            cout << (*it2).first << ": " << (*it2).second << endl;

        for (int i=0; i<sc; ++i)               x_[i]                   = u_k->GetBlock(0)[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = u_k->GetBlock(1)[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = u_k->GetBlock(2)[i];
        phi3_k.MakeTRef(h1_space, x_, 0);
        c1_k  .MakeTRef(h1_space, x_, sc);
        c2_k  .MakeTRef(h1_space, x_, 2*sc);
        phi3_k.SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();
        cout << "L2 norm of phi1(after newton->Mult()): " << phi1->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi2(after newton->Mult()): " << phi2->ComputeL2Error(zero) << endl;
        cout << "L2 norm of phi3(after newton->Mult()): " << phi3_k.ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c1(after newton->Mult()): " <<   c1_k.ComputeL2Error(zero) << endl;
        cout << "L2 norm of   c2(after newton->Mult()): " <<   c2_k.ComputeL2Error(zero) << endl;

        if (visualize)
        {
            (phi3_k) /= alpha1;
            (c1_k)  /= alpha3;
            (c2_k)  /= alpha3;
            phi3_k.SetTrueVector();
            c1_k.SetTrueVector();
            c2_k.SetTrueVector();

            Visualize(*dc, "phi3_k", "phi3_k");
            Visualize(*dc, "c1_k", "c1_k");
            Visualize(*dc, "c2_k", "c2_k");
            cout << "solution vector size on mesh: phi3, " << phi3_k.Size()
                 << "; c1, " << c1_k.Size() << "; c2, " << c2_k.Size() << endl;
            ofstream results("phi3_c1_c2_CG_Newton.vtk");
            results.precision(14);
            int ref = 0;
            pmesh->PrintVTK(results, ref);
            phi3_k.SaveVTK(results, "phi3_k", ref);
            c1_k.SaveVTK(results, "c1_k", ref);
            c2_k.SaveVTK(results, "c2_k", ref);

            (phi3_k) *= (alpha1);
            (c1_k)  *= (alpha3);
            (c2_k)  *= (alpha3);
            phi3_k.SetTrueVector();
            c1_k.SetTrueVector();
            c2_k.SetTrueVector();
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(Epsilon, phi3_k, error);
            ComputeLocalConservation(D_K_, c1_k, v_K_coeff, phi3_k, error1);
            ComputeLocalConservation(D_Cl_, c2_k, v_Cl_coeff, phi3_k, error2);

            string mesh_temp(mesh_file);
            mesh_temp.erase(mesh_temp.find(".msh"), 4);
            mesh_temp.erase(mesh_temp.find("./"), 2);
            string name = "_ref" + to_string(refine_times) + "_" + string(Linearize) + "_"  + string(Discretize) + "_"  + mesh_temp;
            string title1 = "c1_conserv" + name;
            string title2 = "c2_conserv" + name;

            ofstream file1(title1), file2(title2);
            if (file1.is_open() && file2.is_open())
            {
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }
    }
};


class PNP_Newton_DG_Operator_par: public Operator
{
protected:
    Mesh* mesh;
    ParFiniteElementSpace *fsp; // fsp is DG space

    Array<int> block_trueoffsets;
    IS is, long_is;
    Array<int> need_dofs, all_dofs;

    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution
    PetscNonlinearSolver* newton_solver;

    mutable ParLinearForm *f, *f1, *f2, *g, *g1, *g2;
    mutable PetscParMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable PetscParMatrix *A12__, *A13__, *A21__, *A22__, *A31__, *A33__;
    mutable ParBilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;

    ParGridFunction *phi1, *phi2;
    VectorCoefficient* grad_phi1_plus_grad_phi2;
    ParGridFunction *phi3_k, *c1_k, *c2_k;

    Array<int> null_array, Dirichlet_attr;
    Array<int> water_dofs, protein_dofs, interface_dofs;

    StopWatch chrono;
    int num_procs, myid;
    VisItDataCollection* dc;

public:
    PNP_Newton_DG_Operator_par(ParFiniteElementSpace *fsp_, ParGridFunction *phi1_, ParGridFunction *phi2_)
        : fsp(fsp_), phi1(phi1_), phi2(phi2_)
    {
        MPI_Comm_size(fsp->GetComm(), &num_procs);
        MPI_Comm_rank(fsp->GetComm(), &myid);

        mesh = fsp->GetMesh();

        // 对于DG格式不会有重复的自由度
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
        assert(fsp->GetTrueVSize() == protein_dofs.Size() + water_dofs.Size() + interface_dofs.Size());

        Dirichlet_attr.SetSize(mesh->bdr_attributes.Max());
        Dirichlet_attr = 0;
        Dirichlet_attr[top_marker - 1]    = 1;
        Dirichlet_attr[bottom_marker - 1] = 1;

        {
            need_dofs.Append(water_dofs);
            need_dofs.Append(interface_dofs);
            need_dofs.Sort(); // c_i在溶剂和界面区域的自由度

            all_dofs.SetSize(fsp->GetTrueVSize()); // phi_3在整个区域的自由度
            for (int i = 0; i < fsp->GetTrueVSize(); ++i)
                all_dofs[i] = i;

            PetscInt *indices; // only include water dofs and interface dofs
            PetscInt size = PetscInt(need_dofs.Size());
            PetscMalloc1(size, &indices);
            for (int i = 0; i < need_dofs.Size(); ++i) indices[i] = need_dofs[i];
            ISCreateGeneral(MPI_COMM_WORLD, size, indices, PETSC_COPY_VALUES, &is);
            PetscFree(indices);

            PetscInt *long_indices; // include all dofs
            PetscInt long_size = PetscInt(all_dofs.Size());
            PetscMalloc1(long_size, &long_indices);
            for (int i = 0; i < all_dofs.Size(); ++i) long_indices[i] = all_dofs[i];
            ISCreateGeneral(MPI_COMM_WORLD, long_size, long_indices, PETSC_COPY_VALUES, &long_is);
            PetscFree(long_indices);
        }

        height = width = fsp->GetTrueVSize() + 2 * need_dofs.Size();
        block_trueoffsets.SetSize(4); // number of variables + 1;
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = fsp->GetTrueVSize();
        block_trueoffsets[2] = need_dofs.Size();
        block_trueoffsets[3] = need_dofs.Size();
        block_trueoffsets.PartialSum();

        rhs_k = new BlockVector(block_trueoffsets); // not block_offsets !!!
        jac_k = new BlockOperator(block_trueoffsets);
        phi3_k= new ParGridFunction(fsp);
        c1_k  = new ParGridFunction(fsp);
        c2_k  = new ParGridFunction(fsp);

        a21 = new ParBilinearForm(fsp);
        a22 = new ParBilinearForm(fsp);
        a31 = new ParBilinearForm(fsp);
        a33 = new ParBilinearForm(fsp);

        a11 = new ParBilinearForm(fsp);
        // epsilon (grad(dphi3), grad(psi3))
        a11->AddDomainIntegrator(new DiffusionIntegrator(Epsilon));
        // - <{epsilon grad(dphi3)}, [psi3]> + sigma <[dphi3], {epsilon grad(psi3)}> + kappa <{h^{-1} epsilon} [dphi3], [psi3]>
        a11->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(Epsilon, 0.0, 0.0));
        a11->AddBdrFaceIntegrator(new DGDiffusionIntegrator(Epsilon, 0.0, 0.0), Dirichlet_attr);
        if (0) {
            a11->AddInteriorFaceIntegrator(new DGDiffusionSymmetryIntegrator(sigma));
            a11->AddBdrFaceIntegrator(new DGDiffusionSymmetryIntegrator(sigma));
        }
        if (0) {
            a11->AddInteriorFaceIntegrator(new DGDiffusionPenaltyIntegrator(kappa));
            a11->AddBdrFaceIntegrator(new DGDiffusionPenaltyIntegrator(kappa));
        }
        a11->Assemble();
        a11->Finalize();
        a11->SetOperatorType(Operator::PETSC_MATAIJ);
        a11->FormSystemMatrix(null_array, A11);

        a12 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z1 (dc1, psi3)_{\Omega_s}
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K_water));
        if (0) {
            // kappa <h^{-1} [dc1], [psi3]>_{\Omega_s}, 对ci的连续性的惩罚，可以去掉，对应地右端项也要去掉
            a12->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker));
            a12->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker), Dirichlet_attr);
        }
        a12->Assemble();
        a12->Finalize();
        a12->SetOperatorType(Operator::PETSC_MATAIJ);
        a12->FormSystemMatrix(null_array, A12);
        Mat A12_;
        MatCreateSubMatrix(A12, long_is, is, MAT_INITIAL_MATRIX, &A12_);
        A12__ = new PetscParMatrix(A12_, true);

        a13 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z2 (dc2, psi3)_{\Omega_s}
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl_water));
        if (0) {
            // kappa <h^{-1} [dc2], [psi3]>_{\Omega_s}
            a13->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker)); // fff
            a13->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker), Dirichlet_attr);
        }
        a13->Assemble();
        a13->Finalize();
        a13->SetOperatorType(Operator::PETSC_MATAIJ);
        a13->FormSystemMatrix(null_array, A13);
        Mat A13_;
        MatCreateSubMatrix(A13, long_is, is, MAT_INITIAL_MATRIX, &A13_);
        A13__ = new PetscParMatrix(A13_, true);

        f  = new ParLinearForm(fsp);
        f1 = new ParLinearForm(fsp);
        f2 = new ParLinearForm(fsp);
        GradientGridFunctionCoefficient grad_phi1(phi1), grad_phi2(phi2);
        grad_phi1_plus_grad_phi2 = new VectorSumCoefficient(grad_phi1, grad_phi2); //就是 grad(phi1 + phi2)

        g = new ParLinearForm(fsp);
        if (0) { // sigma <phi3_D, (epsilon grad(psi3)).n>
            g->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_coeff, epsilon_water, sigma, 0.0), Dirichlet_attr);
        }
        if (0) { //  + kappa <h^{-1} epsilon phi3_D, psi3>
            g->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_coeff, epsilon_water, 0.0, kappa), Dirichlet_attr);
        }
        if (0) {
            // kappa <h^{-1} c1_D, psi3>
            g->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c1_D_coeff), Dirichlet_attr);
            // kappa <h^{-1} c2_D, psi3>
            g->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c2_D_coeff), Dirichlet_attr);
        }
        // epsilon_m <grad(phi1 + phi2).n, {psi3}>_{\Gamma}
        g->AddInteriorFaceIntegrator(new ProteinWaterInterfaceIntegrator1(&epsilon_protein, grad_phi1_plus_grad_phi2, mesh, protein_marker, water_marker));
        // omit 0 Neumann boundary condition
        g->Assemble();

        g1 = new ParLinearForm(fsp);
        if (0) { // kappa <{h^{-1}} D1 c1_D, v1>
            g1->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_D_coeff, D_K_, 0.0, kappa), Dirichlet_attr);
        }
        if (0) { // sigma <c1_D, D1 grad(v1).n>
            g1->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_D_coeff, D_K_, sigma, 0.0), Dirichlet_attr);
        }
        if (0) {
            // kappa <{h^{-1}} phi3_D, v1>
            g1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &phi_D_coeff), Dirichlet_attr);
        }
        // omit zero Neumann bdc
        g1->Assemble();
        *g1 = 0.0;

        g2 = new ParLinearForm(fsp);
        if (0) {
            // sigma <c2_D, D2 grad(v2).n> + kappa <{h^{-1}} D2 c2_D, v2>
            g2->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_D_coeff, D_Cl_, sigma, kappa), Dirichlet_attr);
        }
        if (0) {
            // kappa <{h^{-1}} phi3_D, v2>
            g2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &phi_D_coeff), Dirichlet_attr);
        }
        // omit zero Neumann bdc
        g2->Assemble();
        *g2 = 0.0;

        dc = new VisItDataCollection("data collection", mesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3_k", phi3_k);
        dc->RegisterField("c1_k",   c1_k);
        dc->RegisterField("c2_k",   c2_k);
    }
    virtual ~PNP_Newton_DG_Operator_par()
    {
        delete f, f1, f2;
        delete a11, a12, a13, a21, a22, a31, a33;
        delete rhs_k, jac_k;
        delete newton_solver;
    }

    virtual void Mult(const Vector& x, Vector& y) const
    {
        Array<int>& Dirichlet_attr_ = const_cast<Array<int>&>(Dirichlet_attr);

        int sc = fsp->GetTrueVSize();
        Vector x_(sc * 3);
        x_ = 0.0;
        for (int i=0; i<sc; ++i)               x_[i]                   = x[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = x[sc + i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = x[sc + need_dofs.Size() + i];

        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k  ->MakeTRef(fsp, x_, sc);
        c2_k  ->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k  ->SetFromTrueVector();
        c2_k  ->SetFromTrueVector();
        cout << "After set bdc (in Newton::Mult()), L2 norm of phi3: " << phi3_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::Mult()), L2 norm of   c1: " <<   c1_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::Mult()), L2 norm of   c2: " <<   c2_k->ComputeL2Error(zero) << endl;
        Visualize(*dc, "phi3_k", "phi3_k");
        Visualize(*dc, "c1_k", "c1_k");
        Visualize(*dc, "c2_k", "c2_k");

        GridFunctionCoefficient phi3_k_coeff(phi3_k), c1_k_coeff(c1_k), c2_k_coeff(c2_k);

        Vector y1(y.GetData() + 0, sc);
        Vector y2(y.GetData() + sc, need_dofs.Size());
        Vector y3(y.GetData() + sc + need_dofs.Size(), need_dofs.Size());
        cout << "1. l2 norm of y: " << y.Norml2() << endl;

        delete f;
        f = new ParLinearForm(fsp);
        f->Update(fsp, rhs_k->GetBlock(0), 0);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff); // alpha2 alpha3 z1 c1^k
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff); // alpha2 alpha3 z2 c2^k
        SumCoefficient term(term1, term2); // alpha2 alpha3 (z1 c1^k + z2 c2^k)
        ProductCoefficient neg_term(neg, term); // - alpha2 alpha3 (z1 c1^k + z2 c2^k)
        ProductCoefficient neg_term_water(neg_term, mark_water_coeff);
        // -alpha2 alpha3 (z1 c1^k + z2 c2^k, psi3)_{\Omega_s}
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term_water));
        // epsilon (grad(phi3^k), grad(psi3))_{\Omega}
        f->AddDomainIntegrator(new GradConvectionIntegrator2(&Epsilon, phi3_k));
        // -<{epsilon grad(phi3^k)}, [psi3]>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_Epsilon, phi3_k));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_Epsilon, phi3_k), Dirichlet_attr_);
        if (0) { // sigma <[phi3^k], {epsilon grad(psi3)}>
            f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_Epsilon, phi3_k_coeff));
            f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_Epsilon, phi3_k_coeff), Dirichlet_attr_);
        }
        if (0) { // kappa epsilon <h^{-1} [phi3^k], [psi3]>
            f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_Epsilon, &phi3_k_coeff));
            f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_Epsilon, &phi3_k_coeff), Dirichlet_attr_);
        }
        if (0) {
            // kappa <h^{-1} [c1^k], [psi3]>
            f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_coeff, &c1_k_coeff, mesh, water_marker));
            f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_coeff, &c1_k_coeff, mesh, water_marker), Dirichlet_attr_);
            // kappa <h^{-1} [c2^k], [psi3]>
            f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_coeff, &c2_D_coeff, mesh, water_marker));
            f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_coeff, &c2_D_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        f->Assemble();
        (*f) -= (*g);
        y1 = (*f);
        cout << "2. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y1: " << y1.Norml2() << endl;

        delete f1;
        f1 = new ParLinearForm(fsp);
        f1->Update(fsp, rhs_k->GetBlock(1), 0);
        ProductCoefficient D1_prod_z1_prod_c1_k_water(D1_prod_z1_water, c1_k_coeff);
        ProductCoefficient D1_prod_z1_prod_c1_k(D_K_prod_v_K, c1_k_coeff);
        ProductCoefficient kappa_prod_D1_prod_z1_prod_c1_k(kappa_coeff, D1_prod_z1_prod_c1_k);
        ProductCoefficient neg_D1_prod_z1_prod_c1_k(neg, D1_prod_z1_prod_c1_k);
        ProductCoefficient sigma_D1_prod_z1_prod_c1_k(sigma_coeff, D1_prod_z1_prod_c1_k);
        ProductCoefficient neg_sigma_D1_prod_z1_prod_c1_k(neg, sigma_D1_prod_z1_prod_c1_k);
        // D1 (grad(c1^k), grad(v1))_{\Omega_s}
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_water, c1_k));
        // - <{D1 grad(c1^k)}, [v1]>  所有位于水中的单元的边界积分
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D1, c1_k, mesh, water_marker));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D1, c1_k, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            // sigma <[c1^k], {D1 grad(v1)}>  所有位于水中的单元的边界积分
            f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D1, c1_k_coeff, mesh, water_marker));
            f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D1, c1_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) { // kappa D1 <h^{-1} [c1^k], [v1]>  所有位于水中的单元的边界积分
            f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_D1, &c1_k_coeff, mesh, water_marker));
            f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_D1, &c1_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        // D1 z1 c1^k (grad(phi3^k), grad(v1))_{\Omega_s}
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_prod_z1_prod_c1_k_water, phi3_k));
        // -<{D1 z1 c1^k grad(phi3^k)}, [v1]>  所有位于水中的单元的边界积分
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D1_prod_z1_prod_c1_k, phi3_k, mesh, water_marker));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D1_prod_z1_prod_c1_k, phi3_k, mesh, water_marker), Dirichlet_attr_);
        if (0) { // sigma <[phi3^k], {D1 z1 c1^k grad(v1)}>  所有位于水中的单元的边界积分
            f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D1_prod_z1_prod_c1_k, phi3_k_coeff, mesh, water_marker));
            f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D1_prod_z1_prod_c1_k, phi3_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) {
            // kappa <{h^{-1} D1 z1 c1^k}[phi3^k], [v1]>  所有位于水中的单元的边界积分
            f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_prod_D1_prod_z1_prod_c1_k, &phi3_k_coeff, mesh, water_marker));
            f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_prod_D1_prod_z1_prod_c1_k, &phi3_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) {
            // - sigma <phi_D, D1 z1 c1^k grad(v1).n> - kappa D1 z1 c1^k <h^{-1} phi_D, v1>
            f1->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_coeff, D1_prod_z1_prod_c1_k, -1.0*sigma, 0.0), Dirichlet_attr_);
        }
        f1->Assemble();
        (*f1) -= (*g1);
        for (int i=0; i<need_dofs.Size(); ++i) y2[i] = (*f1)[need_dofs[i]];
        cout << "3. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y2: " << y2.Norml2() << endl;

        delete f2;
        f2 = new ParLinearForm(fsp);
        f2->Update(fsp, rhs_k->GetBlock(2), 0);
        ProductCoefficient D2_prod_z2_prod_c2_k_water(D2_prod_z2_water, c2_k_coeff);
        ProductCoefficient D2_prod_z2_prod_c2_k(D_Cl_prod_v_Cl, c2_k_coeff);
        ProductCoefficient kappa_prod_D2_prod_z2_prod_c2_k(kappa_coeff, D2_prod_z2_prod_c2_k);
        ProductCoefficient neg_D2_prod_z2_prod_c2_k(neg, D2_prod_z2_prod_c2_k);
        ProductCoefficient sigma_D2_prod_z2_prod_c2_k(sigma_coeff, D2_prod_z2_prod_c2_k);
        ProductCoefficient neg_sigma_D2_prod_z2_prod_c2_k(neg, sigma_D2_prod_z2_prod_c2_k);
        // D2 (grad(c2^k), grad(v2))_{\Omega_s}
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_water, c2_k));
        // - <{D2 grad(c2^k)}, [v2]>  所有位于水中的单元的边界积分
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D2, c2_k, mesh, water_marker));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D2, c2_k, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            // sigma <[c2^k], {D2 grad(v2)}>  所有位于水中的单元的边界积分
            f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D2, c2_k_coeff, mesh, water_marker));
            f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D2, c2_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) { // kappa D2 <h^{-1} [c2^k], [v2]>  所有位于水中的单元的边界积分
            f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D2, &c2_k_coeff));
            f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D2, &c2_k_coeff), Dirichlet_attr_);
        }
        // D2 z2 c2^k (grad(phi3^k), grad(v2))_{\Omega_s}
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_prod_z2_prod_c2_k_water, phi3_k));
        // -<{D2 z2 c2^k grad(phi3^k)}, [v2]>  所有位于水中的单元的边界积分
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D2_prod_z2_prod_c2_k, phi3_k, mesh, water_marker));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5_1(&neg_D2_prod_z2_prod_c2_k, phi3_k, mesh, water_marker), Dirichlet_attr_);
        if (0) { // sigma <[phi3^k], {D2 z2 c2^k grad(v2)}>  所有位于水中的单元的边界积分
            f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D2_prod_z2_prod_c2_k, phi3_k_coeff, mesh, water_marker));
            f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7_1(sigma_D2_prod_z2_prod_c2_k, phi3_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0){
            // kappa <{h^{-1} D2 z2 c2^k}[phi3^k], [v2]>  所有位于水中的单元的边界积分
            f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_prod_D2_prod_z2_prod_c2_k, &phi3_k_coeff, mesh, water_marker));
            f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4_1(&kappa_prod_D2_prod_z2_prod_c2_k, &phi3_k_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) {
            // - sigma <phi_D, D2 z2 c2^k grad(v2).n> - kappa D2 z2 c2^k <h^{-1} phi_D, v2>
            f2->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_D_coeff, D2_prod_z2_prod_c2_k, -1.0*sigma, 0.0), Dirichlet_attr_);
        }
        f2->Assemble();
        (*f2) -= (*g2);
        for (int i=0; i<need_dofs.Size(); ++i) y3[i] = (*f2)[need_dofs[i]];
        cout << "4. l2 norm of y: " << y.Norml2() << endl;
        cout << "   l2 norm of y3: " << y3.Norml2() << endl;
    }

    virtual Operator &GetGradient(const Vector& x) const
    {
        Array<int>& Dirichlet_attr_ = const_cast<Array<int>&>(Dirichlet_attr);

        int sc = fsp->GetTrueVSize();
        Vector x_(sc * 3);
        x_ = 0.0;
        for (int i=0; i<sc; ++i)               x_[i]                   = x[i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]]   = x[sc + i];
        for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = x[sc + need_dofs.Size() + i];

        phi3_k->MakeTRef(fsp, x_, 0);
        c1_k  ->MakeTRef(fsp, x_, sc);
        c2_k  ->MakeTRef(fsp, x_, 2*sc);
        phi3_k->SetFromTrueVector();
        c1_k  ->SetFromTrueVector();
        c2_k  ->SetFromTrueVector();
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of phi3: " << phi3_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of   c1: " <<   c1_k->ComputeL2Error(zero) << endl;
        cout << "After set bdc (in Newton::GetGradient()), L2 norm of   c2: " <<   c2_k->ComputeL2Error(zero) << endl;
        Visualize(*dc, "phi3_k", "phi3_k");
        Visualize(*dc, "c1_k", "c1_k");
        Visualize(*dc, "c2_k", "c2_k");

        GridFunctionCoefficient c1_k_coeff(c1_k), c2_k_coeff(c2_k);

        delete a21;
        a21 = new ParBilinearForm(fsp);
        ProductCoefficient D1_prod_z1_water_c1_k(D1_prod_z1_water, c1_k_coeff);
        // D1 z1 c1^k (grad(dphi3), grad(v1))_{\Omega_s}
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_water_c1_k));
        // - <{D1 z1 c1^k grad(dphi3)}, [v1]> + sigma <[dphi3], {D1 z1 c1^k grad(v1)}> + kappa <{h^{-1} D1 z1 c1^k} [dphi3], [v1]>, 在水单元的边界积分
        a21->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D1_prod_z1_water_c1_k, 0.0, 0.0, mesh, water_marker));
        a21->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D1_prod_z1_water_c1_k, 0.0, 0.0, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            a21->AddInteriorFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D1_prod_z1_water_c1_k, sigma, mesh, water_marker));
            a21->AddBdrFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D1_prod_z1_water_c1_k, sigma, mesh, water_marker));
        }
        a21->Assemble();
        a21->Finalize();
        a21->SetOperatorType(Operator::PETSC_MATAIJ);
        a21->FormSystemMatrix(null_array, A21);
        Mat A21_;
        MatCreateSubMatrix(A21, is, long_is, MAT_INITIAL_MATRIX, &A21_);
        A21__ = new PetscParMatrix(A21_, true);

        delete a22;
        a22 = new ParBilinearForm(fsp);
        // D1 (grad(dc1), grad(v1))_{\Omega_s}
        a22->AddDomainIntegrator(new DiffusionIntegrator(D1_water));
        // - <{D1 grad(dc1)}, [v1]> , 在水单元的边界积分
        a22->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D1_water, 0.0, 0.0, mesh, water_marker));
        a22->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D1_water, 0.0, 0.0, mesh, water_marker), Dirichlet_attr_);
        if (0) { // + sigma <[dc1], {D1 grad(v1)}>, 在水单元的边界积分
            a22->AddInteriorFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D1_water, sigma, mesh, water_marker));
            a22->AddBdrFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D1_water, sigma, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) { // + kappa <{h^{-1} D1} [dc1], [v1]>, 在水单元的边界积分
            a22->AddInteriorFaceIntegrator(new selfDGDiffusionPenaltyIntegrator(&D1_water, kappa, mesh, water_marker));
            a22->AddBdrFaceIntegrator(new selfDGDiffusionPenaltyIntegrator(&D1_water, kappa, mesh, water_marker));
        }
        // (D1 z1 dc1 grad(phi3^k), grad(v1))_{\Omega_s}
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D1_prod_z1_water));
        // - <{D1 z1 dc1 grad(phi3^k)}, [v1]>
        a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D1_z1, *phi3_k, mesh, water_marker));
        a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D1_z1, *phi3_k, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            // sigma <[dc1], {D1 z1 v1 grad(phi3^k)}>
            a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D1_z1, *phi3_k, mesh, water_marker));
            a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D1_z1, *phi3_k, mesh, water_marker), Dirichlet_attr_);
            // kappa <h^{-1} [dc1], [v1]>
            a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker));
            a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        a22->Assemble();
        a22->Finalize();
        a22->SetOperatorType(Operator::PETSC_MATAIJ);
        a22->FormSystemMatrix(null_array, A22);
        Mat A22_;
        MatCreateSubMatrix(A22, is, is, MAT_INITIAL_MATRIX, &A22_);
        A22__ = new PetscParMatrix(A22_, true);

        delete a31;
        a31 = new ParBilinearForm(fsp);
        ProductCoefficient D2_prod_z2_water_c2_k(D2_prod_z2_water, c2_k_coeff);
        // D2 z2 c2^k (grad(dphi3), grad(v2))_{\Omega_s}
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_water_c2_k));
        // - <{D2 z2 c2^k grad(dphi3)}, [v2]> + sigma <[dphi3], {D2 z2 c2^k grad(v2)}> + kappa <{h^{-1} D2 z2 c2^k} [dphi3], [v2]>, 在水单元的边界积分
        a31->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D2_prod_z2_water_c2_k, 0.0, 0.0, mesh, water_marker));
        a31->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D2_prod_z2_water_c2_k, 0.0, 0.0, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            a31->AddInteriorFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D2_prod_z2_water_c2_k, sigma, mesh, water_marker));
            a31->AddBdrFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D2_prod_z2_water_c2_k, sigma, mesh, water_marker));
        }
        a31->Assemble();
        a31->Finalize();
        a31->SetOperatorType(Operator::PETSC_MATAIJ);
        a31->FormSystemMatrix(null_array, A31);
        Mat A31_;
        MatCreateSubMatrix(A31, is, long_is, MAT_INITIAL_MATRIX, &A31_);
        A31__ = new PetscParMatrix(A31_, true);

        delete a33;
        a33 = new ParBilinearForm(fsp);
        // D2 (grad(dc2), grad(v2))_{\Omega_s}
        a33->AddDomainIntegrator(new DiffusionIntegrator(D2_water));
        // - <{D2 grad(dc2)}, [v2]> , 在水单元的边界积分
        a33->AddInteriorFaceIntegrator(new selfDGDiffusionIntegrator(D2_water, 0.0, 0.0, mesh, water_marker));
        a33->AddBdrFaceIntegrator(new selfDGDiffusionIntegrator(D2_water, 0.0, 0.0, mesh, water_marker), Dirichlet_attr_);
        if (0) { // + sigma <[dc2], {D2 grad(v2)}>, 在水单元的边界积分
            a33->AddInteriorFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D2_water, sigma, mesh, water_marker));
            a33->AddBdrFaceIntegrator(new selfDGDiffusionSymmetryIntegrator(&D2_water, sigma, mesh, water_marker), Dirichlet_attr_);
        }
        if (0) { // + kappa <{h^{-1} D2} [dc2], [v2]>, 在水单元的边界积分
            a33->AddInteriorFaceIntegrator(new selfDGDiffusionPenaltyIntegrator(&D2_water, kappa, mesh, water_marker));
            a33->AddBdrFaceIntegrator(new selfDGDiffusionPenaltyIntegrator(&D2_water, kappa, mesh, water_marker), Dirichlet_attr_);
        }
        // (D2 z2 dc2 grad(phi3^k), grad(v2))_{\Omega_s}
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi3_k, &D2_prod_z2_water));
        // - <{D2 z2 dc2 grad(phi3^k)}, [v2]>
        a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D2_z2, *phi3_k, mesh, water_marker));
        a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D2_z2, *phi3_k, mesh, water_marker), Dirichlet_attr_);
        if (0) {
            // sigma <[dc2], {D2 z2 v2 grad(phi3^k)}>
            a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D2_z2, *phi3_k, mesh, water_marker));
            a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D2_z2, *phi3_k, mesh, water_marker), Dirichlet_attr_);
            // kappa <h^{-1} [dc2], [v2]>
            a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker));
            a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3_1(kappa_coeff, mesh, water_marker), Dirichlet_attr_);
        }
        a33->Assemble();
        a33->Finalize();
        a33->SetOperatorType(Operator::PETSC_MATAIJ);
        a33->FormSystemMatrix(null_array, A33);
        Mat A33_;
        MatCreateSubMatrix(A33, is, is, MAT_INITIAL_MATRIX, &A33_);
        A33__ = new PetscParMatrix(A33_, true);

        jac_k = new BlockOperator(block_trueoffsets);
        jac_k->SetBlock(0, 0, &A11);
        jac_k->SetBlock(0, 1, A12__);
        jac_k->SetBlock(0, 2, A13__);
        jac_k->SetBlock(1, 0, A21__);
        jac_k->SetBlock(1, 1, A22__);
        jac_k->SetBlock(2, 0, A31__);
        jac_k->SetBlock(2, 2, A33__);
        return *jac_k;
    }
};
class PNP_Newton_DG_Solver_par
{
private:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    DG_FECollection* dg_fec;
    ParFiniteElementSpace* dg_space;
    ParFiniteElementSpace* h1_space;

    PreconditionerFactory *jac_factory;
    PNP_Newton_DG_Operator_par* op;
    PetscNonlinearSolver* newton_solver;

    BlockVector* u_k;
    ParGridFunction phi3_k, c1_k, c2_k;
    ParGridFunction *phi1, *phi2;

    Array<int> block_trueoffsets;
    Array<int> protein_dofs_dg, water_dofs_dg, interface_dofs_dg, need_dofs;

    SNES snes;
    map<string, Array<double>> out1;
    map<string, double> out2;
    Array<double> linear_iter;
    double linearize_iter, total_time, ndofs, linear_avg_iter;
    PetscInt *its=0, num_its=100;
    PetscReal *residual_norms=0;
    StopWatch chrono;
    VisItDataCollection* dc;

public:
    PNP_Newton_DG_Solver_par(Mesh& mesh_): mesh(&mesh_)
    {
        int mesh_dim = mesh->Dimension(); //网格的维数:1D,2D,3D
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

        dg_fec = new DG_FECollection(p_order, mesh_dim);
        h1_fec = new H1_FECollection(p_order, mesh_dim);
        dg_space = new ParFiniteElementSpace(pmesh, dg_fec);
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        // extract protein dofs, water dofs and interface dofs, 对于DG，不会有重复的自由度
        for (int i=0; i<dg_space->GetNE(); ++i)
        {
            Element* el = mesh->GetElement(i);
            int attr = el->GetAttribute();
            Array<int> dofs;
            if (attr == protein_marker)
            {
                dg_space->GetElementDofs(i, dofs);
                protein_dofs_dg.Append(dofs);
            } else {
                assert(attr == water_marker);
                dg_space->GetElementDofs(i,dofs);
                water_dofs_dg.Append(dofs);
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
                    dg_space->GetFaceVDofs(i, fdofs);
                    interface_dofs_dg.Append(fdofs);
                }
            }
        }
        assert(dg_space->GetTrueVSize() == protein_dofs_dg.Size() + water_dofs_dg.Size() + interface_dofs_dg.Size());

        // combine water dofs and interface dofs to use in variable concentration.
        need_dofs.Append(water_dofs_dg);
        need_dofs.Append(interface_dofs_dg);
        need_dofs.Sort();

        block_trueoffsets.SetSize(4);
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = dg_space->GetTrueVSize();
        block_trueoffsets[2] = need_dofs.Size();
        block_trueoffsets[3] = need_dofs.Size();
        block_trueoffsets.PartialSum();

        u_k = new BlockVector(block_trueoffsets); // 必须满足essential边界条件
        *u_k = 0.0;
        { // set essential bdc
            int size = h1_space->GetMesh()->bdr_attributes.Max();

            Array<int> top_bdr(size), bottom_bdr(size);
            top_bdr = 0;
            top_bdr[top_marker - 1] = 1;
            bottom_bdr = 0;
            bottom_bdr[bottom_marker - 1] = 1;

            int sc = dg_space->GetTrueVSize();
            Vector x_(sc * 3); // 解向量在所有自由度上的值组成的一个长向量：浓度变量在蛋白中为0的部分被保留
            x_ = 0.0;
            phi3_k.MakeTRef(dg_space, x_, 0);
            c1_k  .MakeTRef(dg_space, x_, sc);
            c2_k  .MakeTRef(dg_space, x_, 2*sc);

            phi3_k = 0.0;
            c1_k   = 0.0;
            c2_k   = 0.0;

            ParGridFunction phi3_D_h1(h1_space), c1_D_h1(h1_space), c2_D_h1(h1_space);
            phi3_D_h1 = 0.0;
            c1_D_h1   = 0.0;
            c2_D_h1   = 0.0;

            phi3_D_h1.ProjectBdrCoefficient(phi_D_top_coeff, top_bdr);
            phi3_D_h1.SetTrueVector();
            phi3_D_h1.ProjectBdrCoefficient(phi_D_bottom_coeff, bottom_bdr);
            phi3_D_h1.SetTrueVector();

            phi3_k.ProjectGridFunction(phi3_D_h1);
            phi3_k.SetTrueVector();

            c1_D_h1.ProjectBdrCoefficient(c1_D_top_coeff, top_bdr);
            c1_D_h1.SetTrueVector();
            c1_D_h1.ProjectBdrCoefficient(c1_D_bottom_coeff, bottom_bdr);
            c1_D_h1.SetTrueVector();

            c1_k.ProjectGridFunction(c1_D_h1);
            c1_k.SetTrueVector();

            c2_D_h1.ProjectBdrCoefficient(c2_D_top_coeff, top_bdr);
            c2_D_h1.SetTrueVector();
            c2_D_h1.ProjectBdrCoefficient(c2_D_bottom_coeff, bottom_bdr);
            c2_D_h1.SetTrueVector();

            c2_k.ProjectGridFunction(c2_D_h1);
            c2_k.SetTrueVector();

            cout << "After set bdc, L2 norm of phi3: " << phi3_k.ComputeL2Error(zero) << endl;
            cout << "After set bdc, L2 norm of   c1: " <<   c1_k.ComputeL2Error(zero) << endl;
            cout << "After set bdc, L2 norm of   c2: " <<   c2_k.ComputeL2Error(zero) << endl;

            // 扩展 x_ 成 u_k
            for (int i=0; i<sc; ++i)               u_k->GetBlock(0)[i] = x_[i];
            for (int i=0; i<need_dofs.Size(); ++i) u_k->GetBlock(1)[i] = x_[sc + need_dofs[i]];
            for (int i=0; i<need_dofs.Size(); ++i) u_k->GetBlock(2)[i] = x_[2*sc + need_dofs[i]];

            cout << "l2 norm of u_k (initial after set bdc): " << u_k->Norml2() << endl;
        }

        phi1 = new ParGridFunction(dg_space);
        phi1->ProjectCoefficient(G_coeff);
        phi1->SetTrueVector();
        phi1->SetFromTrueVector();
        cout << "L2 norm of phi1: " << phi1->ComputeL2Error(zero) << endl;

        ParGridFunction* phi2_ = new ParGridFunction(h1_space);
        {
            Array<int> protein_dofs, water_dofs, interface_dofs;
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
            protein_dofs.Sort();
            protein_dofs.Unique();
            water_dofs.Sort();
            water_dofs.Unique();
            interface_dofs.Sort();
            interface_dofs.Unique();
            for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
            {
                protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
                water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
            }

            Array<int> Gamma_m_bdr(h1_space->GetMesh()->bdr_attributes.Max());
            Gamma_m_bdr = 0;
            Gamma_m_bdr[Gamma_m_marker - 1] = 0;

            ParBilinearForm blf(h1_space);
            // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m is protein domain
            blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
            blf.Assemble(0);
            blf.Finalize(0);

            ParLinearForm lf(h1_space);
            // -<grad(G).n, psi2>_{\Gamma_M}
            lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m_bdr); // Neumann bdc on Gamma_m
            lf.Assemble();

            phi2_->ProjectCoefficient(G_coeff);
            phi2_->Neg(); // 在interface \Gamma 上是Dirichlet边界: -phi1

            PetscParMatrix *A = new PetscParMatrix();
            PetscParVector *x = new PetscParVector(h1_space);
            PetscParVector *b = new PetscParVector(h1_space);
            blf.SetOperatorType(Operator::PETSC_MATAIJ);
            blf.FormLinearSystem(interface_dofs, *phi2_, lf, *A, *x, *b); //除了ess_tdof_list以外是0的Neumann边界

            A->EliminateRows(water_dofs, 1.0);
            if (self_debug) {
                for (int i = 0; i < water_dofs.Size(); i++) // 确保只在水中(不包括蛋白质和interface)的自由度为0
                    assert(abs((*b)(water_dofs[i])) < 1E-10);
            }

            PetscLinearSolver* solver = new PetscLinearSolver(*A, "phi2_");

            chrono.Clear();
            chrono.Start();
            solver->Mult(*b, *x);
            chrono.Stop();
            blf.RecoverFEMSolution(*x, lf, *phi2_);

            if (verbose) {
                if (solver->GetConverged() == 1)
                    cout << "phi2 solver: successfully converged by iterating " << solver->GetNumIterations()
                         << " times, taking " << chrono.RealTime() << " s." << endl;
                else if (solver->GetConverged() != 1)
                    cerr << "phi2 solver: failed to converged" << endl;
            }

            delete A, x, b, solver;
        }

        phi2 = new ParGridFunction(dg_space);
        phi2->ProjectGridFunction(*phi2_);
        cout << "L2 norm of phi2: " << phi2->ComputeL2Error(zero) << endl;

        dc = new VisItDataCollection("data collection", pmesh);
        dc->RegisterField("phi1", phi1);
        dc->RegisterField("phi2", phi2);
        dc->RegisterField("phi3_k", &phi3_k);
        dc->RegisterField("c1_k",   &c1_k);
        dc->RegisterField("c2_k",   &c2_k);

        op = new PNP_Newton_DG_Operator_par(dg_space, phi1, phi2);

        jac_factory = new PreconditionerFactory(*op, prec_type);

        newton_solver = new PetscNonlinearSolver(dg_space->GetComm(), *op, "newton_");
        newton_solver->iterative_mode = true;
        newton_solver->SetPreconditionerFactory(jac_factory);
        snes = SNES(*newton_solver);
        PetscMalloc(num_its * sizeof(PetscInt), &its);
        PetscMalloc(num_its * sizeof(PetscReal), &residual_norms);
        SNESSetConvergenceHistory(snes, residual_norms, its, num_its, PETSC_TRUE);
    }
    virtual ~PNP_Newton_DG_Solver_par()
    {
        delete newton_solver, op, jac_factory, u_k, mesh, pmesh;
        PetscFree(its);
    }

    void Solve()
    {
        cout.precision(14);
        cout << "\nNewton, DG" << p_order << ", protein, parallel"
             << ", prec: " << prec_type << ", petsc option file: " << options_src
             << ", sigma: " << sigma << ", kappa: " << kappa
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        if (0) {
            int sc = dg_space->GetTrueVSize();
            Vector x_(sc * 3); // 解向量在所有自由度上的值组成的一个长向量：浓度变量在蛋白中为0的部分被保留
            x_ = 0.0;
            for (int i=0; i<sc; ++i)               x_[i] = u_k->GetBlock(0)[i];
            for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]] = u_k->GetBlock(1)[i];
            for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = u_k->GetBlock(2)[i];

            phi3_k.MakeTRef(dg_space, x_, 0);
            c1_k  .MakeTRef(dg_space, x_, sc);
            c2_k  .MakeTRef(dg_space, x_, 2*sc);
            phi3_k.SetFromTrueVector();
            phi3_k.SetFromTrueVector();
            phi3_k.SetFromTrueVector();

            Visualize(*dc, "phi3_k", "phi3_k");
            Visualize(*dc, "c1_k", "c1_k");
            Visualize(*dc, "c2_k", "c2_k");
        }

        cout << "l2 norm of u_k (initial): " << u_k->Norml2() << endl;
        Vector zero_vec;
        chrono.Start();
        newton_solver->Mult(zero_vec, *u_k); // u_k must be a true vector
        chrono.Stop();
        linearize_iter = newton_solver->GetNumIterations();
        total_time = chrono.RealTime();
        ndofs = u_k->Size();
        out2["linearize_iter"] = linearize_iter;
        out2["total_time"] = total_time;
        out2["ndofs"] = ndofs;

        cout << "\nNewton, DG" << p_order << ", protein, parallel"
             << ", prec: " << prec_type << ", petsc option file: " << options_src
             << ", sigma: " << sigma << ", kappa: " << kappa
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        SNESGetConvergenceHistory(snes, &residual_norms, &its, &num_its);
//        for (int i=0; i<num_its; ++i)
//            cout << residual_norms[i] << endl;
        for (int i=1; i<num_its; ++i)
            linear_iter.Append(its[i]);
        out1["linear_iter"] = linear_iter;
        linear_avg_iter = round(linear_iter.Sum() / linear_iter.Size());
        out2["linear_avg_iter"] = linear_avg_iter;

        {
            int sc = dg_space->GetTrueVSize();
            Vector x_(sc * 3); // 解向量在所有自由度上的值组成的一个长向量：浓度变量在蛋白中为0的部分被保留
            x_ = 0.0;
            for (int i=0; i<sc; ++i)               x_[i] = u_k->GetBlock(0)[i];
            for (int i=0; i<need_dofs.Size(); ++i) x_[sc + need_dofs[i]] = u_k->GetBlock(1)[i];
            for (int i=0; i<need_dofs.Size(); ++i) x_[2*sc + need_dofs[i]] = u_k->GetBlock(2)[i];

            phi3_k.MakeTRef(dg_space, x_, 0);
            c1_k  .MakeTRef(dg_space, x_, sc);
            c2_k  .MakeTRef(dg_space, x_, 2*sc);
            phi3_k.SetFromTrueVector();
            phi3_k.SetFromTrueVector();
            phi3_k.SetFromTrueVector();
        }

        cout.precision(14);
        cout << "L2 norm of phi: " << phi3_k.ComputeL2Error(zero) << '\n'
             << "L2 norm of c1 : " << c1_k.ComputeL2Error(zero) << '\n'
             << "L2 norm of c2 : " << c2_k.ComputeL2Error(zero) << endl;

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(Epsilon, phi3_k, error);
            ComputeLocalConservation(D_K_, c1_k, v_K_coeff, phi3_k, error1);
            ComputeLocalConservation(D_Cl_, c2_k, v_Cl_coeff, phi3_k, error2);

            string mesh_temp(mesh_file);
            mesh_temp.erase(mesh_temp.find(".msh"), 4);
            mesh_temp.erase(mesh_temp.find("./"), 2);
            string name = "_ref" + to_string(refine_times) + "_" + string(Linearize) + "_"  + string(Discretize) + "_"  + mesh_temp;
            string title1 = "c1_conserv" + name;
            string title2 = "c2_conserv" + name;

            ofstream file1(title1), file2(title2);
            if (file1.is_open() && file2.is_open())
            {
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }

        map<string, Array<double>>::iterator it1;
        for (it1=out1.begin(); it1!=out1.end(); ++it1)
            (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
        map<string, double>::iterator it2;
        for (it2=out2.begin(); it2!=out2.end(); ++it2)
            cout << (*it2).first << ": " << (*it2).second << endl;
    }
};


#endif
