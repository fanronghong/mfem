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


class PNP_CG_Gummel_Solver_par
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* fsp;
    ParGridFunction *phi, *c1, *c2;       // FE解.
    ParGridFunction *phi_n, *c1_n, *c2_n; // Gummel迭代解

    VisItDataCollection* dc;
    Array<int> ess_tdof_list, top_tdof_list, bottom_tdof_list; // 所有未知量都在整个区域边界满足Dirichlet
    Array<int> Neumann_attr, Dirichlet_attr;

    StopWatch chrono;
    int num_procs, myid;
    map<string, Array<double>> out1;
    map<string, double> out2;
    Array<double> poisson_iter, poisson_time, np1_iter, np1_time, np2_iter, np2_time;
    double poisson_avg_iter, poisson_avg_time,
            np1_avg_iter, np1_avg_time,
            np2_avg_iter, np2_avg_time,
            linearize_iter, total_time, ndofs;

public:
    PNP_CG_Gummel_Solver_par(Mesh& mesh_) : mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);

        fec = new H1_FECollection(p_order, mesh.Dimension());
        fsp = new ParFiniteElementSpace(pmesh, fec);

        phi_n = new ParGridFunction(fsp); *phi_n = 0.0; // Gummel 迭代当前解
        c1_n  = new ParGridFunction(fsp); *c1_n  = 0.0;
        c2_n  = new ParGridFunction(fsp); *c2_n  = 0.0;

        phi = new ParGridFunction(fsp); *phi = 0.0; // Gummel 迭代下一步解
        c1  = new ParGridFunction(fsp); *c1  = 0.0;
        c2  = new ParGridFunction(fsp); *c2  = 0.0;

        int bdr_size = fsp->GetMesh()->bdr_attributes.Max();
        Neumann_attr  .SetSize(bdr_size);
        Dirichlet_attr.SetSize(bdr_size);
        {
            Neumann_attr = 0;
//            Neumann_attr[front_attr - 1] = 1;
//            Neumann_attr[back_attr  - 1] = 1;
//            Neumann_attr[left_attr  - 1] = 1;
//            Neumann_attr[right_attr - 1] = 1;

            Dirichlet_attr = 1;
//            Dirichlet_attr[top_attr    - 1] = 1;
//            Dirichlet_attr[bottom_attr - 1] = 1;
        }
        fsp->GetEssentialTrueDofs(Dirichlet_attr, ess_tdof_list);

        // set Dirichlet boundary condition
        phi_n->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1_n ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2_n ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);

        phi->ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
        c1 ->ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
        c2 ->ProjectBdrCoefficient(c2_exact, Dirichlet_attr);

        phi_n->SetTrueVector();
        c1_n ->SetTrueVector();
        c2_n ->SetTrueVector();
        phi->SetTrueVector();
        c1 ->SetTrueVector();
        c2 ->SetTrueVector();

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
        cout << "\nGummel, CG" << p_order << ", box, parallel"
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;
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

            if (strcmp(AdvecStable, "none") == 0)      Solve_NP1();
//            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP1_EAFE();
//            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP1_SUPG();
//            else MFEM_ABORT("Not support stabilization.");
            (*c1_n) = (*c1);

            if (strcmp(AdvecStable, "none") == 0)      Solve_NP2();
//            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP2_EAFE();
//            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP2_SUPG();
            else MFEM_ABORT("Not support stabilization.");
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            if (tol < Gummel_rel_tol)
            {
                cout << "------> Gummel iteration converge: " << iter << " times." << endl;
                break;
            }
            if (tol < Gummel_rel_tol) break;
            iter++;
            cout << endl;
        }
        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

        out1["poisson_iter"] = poisson_iter;
        out1["poisson_time"] = poisson_time;
        out1["np1_iter"] = np1_iter;
        out1["np1_time"] = np1_time;
        out1["np2_iter"] = np2_iter;
        out1["np2_time"] = np2_time;

        linearize_iter = iter;
        total_time = poisson_time.Sum() + np1_time.Sum() + np2_time.Sum();
        ndofs = fsp->GetVSize() * 3;
        out2["linearize_iter"] = linearize_iter;
        out2["total_time"] = total_time;
        out2["ndofs"] = ndofs;
        poisson_avg_iter = (poisson_iter.Sum() / poisson_iter.Size());
        poisson_avg_time = poisson_time.Sum() / poisson_time.Size();
        out2["poisson_avg_iter"] = poisson_avg_iter;
        out2["poisson_avg_time"] = poisson_avg_time;
        np1_avg_iter     = (np1_iter.Sum() / np1_iter.Size());
        np1_avg_time     = np1_time.Sum() / np1_iter.Size();
        out2["np1_avg_iter"] = np1_avg_iter;
        out2["np1_avg_time"] = np1_avg_time;
        np2_avg_iter     = (np2_iter.Sum() / np2_iter.Size());
        np2_avg_time     = np2_time.Sum() / np2_iter.Size();
        out2["np2_avg_iter"] = np2_avg_iter;
        out2["np2_avg_time"] = np2_avg_time;

        cout.precision(14);
        double phiL2err = phi->ComputeL2Error(phi_exact);
        double c1L2err = c1->ComputeL2Error(c1_exact);
        double c2L2err = c2->ComputeL2Error(c2_exact);

        cout << "\n======>Box, " << Linearize << ", " << Discretize << p_order << ", refine " << refine_times << " for " << mesh_file << ", " << options_src << ", -rate: " << ComputeConvergenceRate << endl;
        cout << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
             << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
             << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;

        if (ComputeConvergenceRate)
        {
            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);

            double totle_size = 0.0;
            for (int i=0; i<mesh.GetNE(); i++)
                totle_size += mesh.GetElementSize(0, 1);

            meshsizes_.Append(totle_size / mesh.GetNE());
        }

        if (visualize)
        {
            (*phi) /= alpha1;
            (*c1)  /= alpha3;
            (*c2)  /= alpha3;
            Visualize(*dc, "phi", "phi_Gummel_CG");
            Visualize(*dc, "c1", "c1_Gummel_CG");
            Visualize(*dc, "c2", "c2_Gummel_CG");
            ofstream results("phi_c1_c2_Gummel_CG.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);
            (*phi) *= (alpha1);
            (*c1)  *= (alpha3);
            (*c2)  *= (alpha3);

//            phi->ProjectCoefficient(phi_exact);
//            c1 ->ProjectCoefficient(c1_exact);
//            c2 ->ProjectCoefficient(c2_exact);
//            phi->SetTrueVector();
//            c1 ->SetTrueVector();
//            c2 ->SetTrueVector();
//            Visualize(*dc, "phi", "phi_e1");
//            Visualize(*dc, "c1", "c1_e1");
//            Visualize(*dc, "c2", "c2_e1");
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(epsilon_water, *phi, error);
            ComputeLocalConservation(D_K_, *c1, v_K_coeff, *phi, error1);
            ComputeLocalConservation(D_Cl_, *c2, v_Cl_coeff, *phi, error2);

            ofstream file("./phi_local_conservation_CG_Gummel_box.txt"),
                     file1("./c1_local_conservation_CG_Gummel_box.txt"),
                     file2("./c2_local_conservation_CG_Gummel_box.txt");
            if (file.is_open() && file1.is_open() && file2.is_open())
            {
                error.Print(file, 1);
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

        cout << "approximate mesh scale h: " << pow(fsp->GetTrueVSize(), -1.0/3) << endl;
    }

    void Solve(BlockVector& vec, Array<int>& offsets, double initTol)
    {
        cout << "\n    Obtain nonlinear iteration initial value, Gummel, CG" << p_order << ", box, parallel"
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;
        int iter = 1;
        double tol = 1;
        while (tol > initTol)
        {
            Solve_Poisson();

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            Solve_NP1();
            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            iter++;
        }

        phi->SetTrueVector();
        c1 ->SetTrueVector();
        c2 ->SetTrueVector();

        vec.GetBlock(0) = phi->GetTrueVector();
        vec.GetBlock(1) = c1->GetTrueVector();
        vec.GetBlock(2) = c2->GetTrueVector();

        // 为了测试vec是否正确被赋值
//        phi_n->MakeRef(fsp, vec, offsets[0]);
//        c1_n ->MakeRef(fsp, vec, offsets[1]);
//        c2_n ->MakeRef(fsp, vec, offsets[2]);
//        phi_n->SetFromTrueVector();
//        c1_n ->SetFromTrueVector();
//        c2_n ->SetFromTrueVector();
    }

private:
    // 3.求解耦合的方程Poisson方程
    void Solve_Poisson()
    {
//        c1_n->ProjectCoefficient(c1_exact); // for test Poisson convergence rate
//        c2_n->ProjectCoefficient(c2_exact); // for test Poisson convergence rate

        GridFunctionCoefficient* c1_n_coeff = new GridFunctionCoefficient(c1_n);
        GridFunctionCoefficient* c2_n_coeff = new GridFunctionCoefficient(c2_n);

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        // epsilon_s (grad(phi), grad(psi))
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf = new ParLinearForm(fsp); //Poisson方程的右端项
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K , *c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, *c2_n_coeff);
        // alpha2 alpha3 z1 (c1^k, psi)
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        // alpha2 alpha3 z2 (c2^k, psi)
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
        // epsilon_s <grad(phi_e).n, psi>, phi_flux = -epsilon_s grad(phi_e)
        ScalarVectorProductCoefficient neg_J(neg, J);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J), Neumann_attr);
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);

        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "phi_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi);

#ifdef SELF_VERBOSE
        cout << "l2 norm of phi: " << phi->Norml2() << endl;
        if (solver->GetConverged() == 1)
            cout << "phi solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif

        poisson_iter.Append(solver->GetNumIterations());
        poisson_time.Append(chrono.RealTime());

//        cout << "L2 error norm of |phi_h - phi_e|: " << phi->ComputeL2Error(phi_exact) << endl;
//        MFEM_ABORT("Stop here for testing Poisson convergence rate in PNP_CG_Gummel_Solver_par!");

        (*phi_n) *= relax;
        (*phi)   *= 1-relax;
        (*phi)   += (*phi_n); // 利用松弛方法更新phi3
        (*phi_n) /= relax+TOL; // 还原phi3_n.避免松弛因子为0的情况造成除0

        delete blf;
        delete lf;
        delete solver;
        delete c1_n_coeff;
        delete c2_n_coeff;
    }

    // 4.求解耦合的方程NP1方程
    void Solve_NP1()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP1 convergence rate

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        // D1 (grad(c1), grad(v1))
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi^k), grad(v1))
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
        *lf = 0.0;
        // D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J1(neg, J1);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J1), Neumann_attr);
        // (f1, v1)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np1_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np1 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif

        np1_iter.Append(solver->GetNumIterations());
        np1_time.Append(chrono.RealTime());

//        cout << "L2 error norm of | c1_h - c1_e |: " << c1->ComputeL2Error(c1_exact) << endl;
//        MFEM_ABORT("Stop here for test NP1 convergence rate in PNP_CG_Gummel_Solver_par!");

        (*c1_n) *= relax;
        (*c1)   *= 1-relax;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver;
    }

    // 5.求解耦合的方程NP2方程
    void Solve_NP2()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP2 convergence rate

        ParBilinearForm *blf(new ParBilinearForm(fsp));
        // D2 (grad(c2), grad(v2))
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi^k), grad(v2))
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
        // D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J2(neg, J2);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J2), Neumann_attr);
        // (f2, v2)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, A, x, b);

        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np2_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1)
            cout << "np2 solver: Converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver: Not Converge, taking " << chrono.RealTime() << " s." << endl;
#endif

        np2_iter.Append(solver->GetNumIterations());
        np2_time.Append(chrono.RealTime());

//        cout << "L2 error norm of | c2_h - c2_e |: " << c2->ComputeL2Error(c2_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate in PNP_CG_Gummel_Solver_par!");

        (*c2_n) *= relax;
        (*c2)   *= 1-relax;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf, blf, solver;
    }
};


class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    ParBilinearForm *a11, *a12, *a13, *m1, *m2;
    HypreParMatrix *A, *B1, *B2, *M1, *M2;
    mutable ParBilinearForm *a22, *a33;
    mutable HypreParMatrix *A1, *A2;
    mutable ParLinearForm *l, *l1, *l2;
    mutable HypreParVector *b, *b1, *b2;
    mutable Vector z;

    PetscLinearSolver *A_solver, *M1_solver, *M2_solver;
    HypreParVector *phi;

    ParGridFunction *phi_gf, *c1_gf, *c2_gf;
    BlockVector* c1c2;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr, ess_tdof_list;

public:
    PNP_Box_Gummel_CG_TimeDependent(Mesh& mesh_): mesh(mesh_)
    {
        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的
        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        true_size = h1->TrueVSize();
        true_offset.SetSize(3); // 表示 c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_size;
        true_offset[2] = 2*true_size;

        t = 0.0;
        z.SetSize(true_size);

        /*  Poisson Equation:
                           div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
                   NP Equation:
                           dc_i / dt = div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i  */
        phi_gf = new ParGridFunction(h1);
        phi_exact.SetTime(t);
        phi_gf->ProjectCoefficient(phi_exact);
        phi = phi_gf->GetTrueDofs();

        c1_gf = new ParGridFunction(h1);
        c2_gf = new ParGridFunction(h1);

        c1c2 = new BlockVector(true_offset);
        c1_gf->MakeRef(h1, *c1c2, true_offset[0]);
        c2_gf->MakeRef(h1, *c1c2, true_offset[1]);

        c1_exact.SetTime(t);
        c1_gf->ProjectCoefficient(c1_exact);
        c1_gf->SetTrueVector();

        c2_exact.SetTime(t);
        c2_gf->ProjectCoefficient(c2_exact);
        c2_gf->SetTrueVector();

        c1_gf->SetFromTrueVector(); c2_gf->SetFromTrueVector();

        ConstructConstantData();
    }

    virtual ~PNP_Box_Gummel_CG_TimeDependent() {}

    Vector& GetInitialValue()
    {
        return *c1c2;
    }

    void ConstructConstantData()
    {
        a11 = new ParBilinearForm(h1);
        // (epsilon_s grad(phi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a11->Assemble();
        a11->Finalize();
        a11->FormSystemMatrix(ess_tdof_list, A);
        A_solver = new PetscLinearSolver(*A, false, "phi_");

        a12 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z1 c1, psi)
        a12->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));
        a12->Assemble();
        a12->Finalize();
        B1 = a12->ParallelAssemble();

        a13 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z2 c2, psi)
        a13->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble();
        a13->Finalize();
        B2 = a13->ParallelAssemble();

        m1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1->AddDomainIntegrator(new MassIntegrator);
        m1->Assemble();
        m1->Finalize();
        m1->FormSystemMatrix(ess_tdof_list, M1);
        M1_solver = new PetscLinearSolver(*M1, false, "np1_");

        m2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2->AddDomainIntegrator(new MassIntegrator);
        m2->Assemble();
        m2->Finalize();
        m2->FormSystemMatrix(ess_tdof_list, M2);
        M2_solver = new PetscLinearSolver(*M2, false, "np2_");
    }

    virtual void Mult(const Vector &c1c2, Vector &dc1c2_dt) const
    {
        Vector c1(c1c2.GetData() +         0, true_size);
        Vector c2(c1c2.GetData() + true_size, true_size);
        Vector dc1_dt(dc1c2_dt.GetData() +         0, true_size);
        Vector dc2_dt(dc1c2_dt.GetData() + true_size, true_size);

        // 首先求解Poisson方程
        f1_analytic.SetTime(t);
        l = new ParLinearForm(h1);
        l->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l->Assemble();
        b = l->ParallelAssemble();

        B1->Mult(1.0, c1, 1.0, *b); // B1 c1 + b -> b
        B2->Mult(1.0, c2, 1.0, *b); // B1 c1 + B2 c2 + b -> b
        b->SetSubVector(ess_tdof_list, 0.0); // 给定essential bdc
        A_solver->Mult(*b, *phi);

        phi_gf->SetFromTrueVector(); // 求解NP方程时要用到phi

        // 然后求解NP1方程
        a22 = new ParBilinearForm(h1);
        a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi_gf, &D_K_prod_v_K));
        a22->Assemble();
        a22->Finalize();
        a22->FormSystemMatrix(ess_tdof_list, A1);

        f1_analytic.SetTime(t);
        l1 = new ParLinearForm(h1);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        b1 = l1->ParallelAssemble();
        b1->SetSubVector(ess_tdof_list, 0.0);

        A1->Mult(1.0, c1, 1.0, *b1); // A1 c1 + b1 -> b1
        M1_solver->Mult(*b1, dc1_dt); // solve M1 dc1_dt = A1 c1 + b1


        // 然后求解NP2方程
        a33 = new ParBilinearForm(h1);
        a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi_gf, &D_Cl_prod_v_Cl));
        a33->Assemble();
        a33->Finalize();
        a33->FormSystemMatrix(ess_tdof_list, A2);

        f2_analytic.SetTime(t);
        l2 = new ParLinearForm(h1);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();
        b2 = l2->ParallelAssemble();
        b2->SetSubVector(ess_tdof_list, 0.0);

        A2->Mult(1.0, c2, 1.0, *b2); // A2 c2 + b2 -> b2
        M2_solver->Mult(*b2, dc2_dt); // solve M2 dc2_dt = A2 c2 + b2

    }

    virtual void ImplicitSolve(const double dt, const Vector &c1c2, Vector &dc1c2_dt)
    {

    }

};

template <class T>
class TimeDependent_Solver
{
private:
    double t; // 当前时间
    Vector& init_true_vector;
    ODESolver *ode_solver;

public:
    TimeDependent_Solver(T* oper, int ode_solver_type)
    {
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

        t = init_t;
        oper->SetTime(t);
        ode_solver->Init(*oper);
        init_true_vector = oper->GetInitialValue();
    }
    ~TimeDependent_Solver() { delete ode_solver; }

    void Solve()
    {
        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(dt, t_final - t);

            ode_solver->Step(init_true_vector, t, dt_real);

            last_step = (t >= t_final - 1e-8*dt);
        }
    }
};
#endif
