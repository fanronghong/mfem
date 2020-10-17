//
// Created by fan on 2020/10/17.
//

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


class PNP_Protein_Gummel_CG_TimeDependent: public TimeDependentOperator
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
    PNP_Protein_Gummel_CG_TimeDependent(Mesh* mesh_) : mesh(mesh_)
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
    ~PNP_Protein_Gummel_CG_TimeDependent()
    {
        delete phi1, phi2, phi3, c1, c2, phi3_n, c1_n, c2_n, dc;
    }

    // 把下面的5个求解过程串联起来
    void Solve()
    {
        Solve_Phi1();
        Solve_Phi2();

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
    void Solve_Phi1()
    {
        phi1_gf = new ParGridFunction(fes);
        phi1_gf->ProjectCoefficient(G_coeff); // phi1求解完成, 直接算比较慢, 也可以从文件读取

        cout << "L2 norm of phi1: " << phi1_gf->ComputeL2Error(zero) << endl;

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

        H1_FECollection* h1_fec = new H1_FECollection(p_order, pmesh->Dimension());
        ParFiniteElementSpace* h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        Array<int> interface_bdr, interface_ess_tdof_list, Gamma_m;
        {
            int size = pmesh->bdr_attributes.Max();

            Gamma_m.SetSize(size);
            Gamma_m                     = 0;
            Gamma_m[Gamma_m_marker - 1] = 1;

            interface_bdr.SetSize(size);
            interface_bdr                       = 0;
            interface_bdr[interface_marker - 1] = 1;
            h1_space->GetEssentialTrueDofs(interface_bdr, interface_ess_tdof_list);
        }

        ParBilinearForm blf(h1_space);
        // (grad(phi2), grad(psi2))_{\Omega_m}, \Omega_m: protein domain
        blf.AddDomainIntegrator(new DiffusionIntegrator(mark_protein_coeff));
        blf.Assemble(0);
        blf.Finalize(0);

        ParLinearForm lf(h1_space);
        // -<grad(G).n, psi2>_{\Gamma_M}, G is phi1
        lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_gradG_coeff), Gamma_m);
        lf.Assemble();

        phi2_gf = new ParGridFunction(fes);
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

        cout << "L2 norm of phi2: " << phi2_gf->ComputeL2Error(zero) << endl;

        if (verbose) {
            cout << "\nL2 norm of phi2: " << phi2_gf->ComputeL2Error(zero) << endl;
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
        delete h1_fec;
        delete h1_space;
        delete A;
        delete x;
        delete b;
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
    Array<int> true_offset, ess_bdr;
    ParaViewDataCollection* pd;
    int num_procs, rank;
    StopWatch chrono;

public:
    PNP_Protein_TimeDependent_Solver(ParMesh* pmesh_, int ode_solver_type): pmesh(pmesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        t = t_init;

        if (strcmp(Discretize, "cg") == 0)
        {
            fec = new H1_FECollection(p_order, pmesh->Dimension());
        }
        fes = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        ess_bdr = 1; // 对于H1空间, 设置所有边界都是essential的; 对DG空间, 边界条件都是weak的

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
        phi3_gf->ProjectCoefficient(phi_D_coeff);
        phi3_gf->SetTrueVector();
        phi3_gf->SetFromTrueVector();

        c1_gf->ProjectCoefficient(c1_D_coeff);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_gf->ProjectCoefficient(c2_D_coeff);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        if (strcmp(Linearize, "gummel") == 0)
        {
            if (strcmp(Discretize, "cg") == 0)
            {
                oper = new PNP_Protein_Gummel_CG_TimeDependent(true_vsize, true_offset, ess_bdr, fes, t);
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
            string paraview_title = string("PNP_Protein") + Discretize + "_" + Linearize + "_Time_Dependent";
            cout << paraview_title << endl;
            pd = new ParaViewDataCollection(paraview_title, pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
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
        if (paraview) delete pd;
    }

    void Solve()
    {

        cout << "good" << endl;
    }
};


#endif //MFEM_PNP_PROTEIN_TIMEDEPENDENT_SOLVERS_HPP
