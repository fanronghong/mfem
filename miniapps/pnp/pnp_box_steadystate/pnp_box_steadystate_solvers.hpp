#ifndef _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_
#define _PNP_STEADYSTATE_BOX_GUMMEL_SOLVERS_HPP_

#include "pnp_box_steadystate.hpp"
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
MatrixFunctionCoefficient diffusion_tensor_K(3, DiffusionTensor_K);
VectorFunctionCoefficient advection_vector_K(3, adv1);
MatrixFunctionCoefficient diffusion_tensor_Cl(3, DiffusionTensor_Cl);
VectorFunctionCoefficient advection_vector_Cl(3, adv2);


class SPD_PreconditionerSolver: public Solver
{
private:
    KSP ksp;
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    SPD_PreconditionerSolver(const OperatorHandle& oh): Solver()
    {
        cout << "in SPD_PreconditionerSolver::SPD_PreconditionerSolver()" << endl;
        PetscErrorCode ierr;

        // Get the PetscParMatrix out of oh.
        PetscParMatrix *prec;
        oh.Get(prec);
        Mat pc = *prec; // type cast to Petsc Mat

        // update base (Solver) class
        width = prec->Width();
        height = prec->Height();
        X = new PetscParVector(PETSC_COMM_WORLD, *this, true, false);
        Y = new PetscParVector(PETSC_COMM_WORLD, *this, false, false);

        ierr = KSPCreate(MPI_COMM_WORLD, &ksp); PCHKERRQ(ksp, ierr);
        ierr = KSPSetOperators(ksp, pc, pc); PCHKERRQ(ksp, ierr);
        KSPAppendOptionsPrefix(ksp, "np1spdpc_");
        KSPSetFromOptions(ksp);
        KSPSetUp(ksp);

        // cout << "in BlockPreconditionerSolver::BlockPreconditionerSolver()" << endl;
    }
    virtual ~SPD_PreconditionerSolver()
    {
        KSPDestroy(&ksp);
        delete X;
        delete Y;
    }

    virtual void SetOperator(const Operator& op) { MFEM_ABORT("Not support!"); }

    virtual void Mult(const Vector& x, Vector& y) const
    {
        cout << "in SPD_PreconditionerSolver::Mult()" << endl;
        Vec blockx, blocky;
        Vec blockx0, blocky0;

        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
        Y->PlaceArray(y.GetData());

        KSPSolve(ksp, *X, *Y);

        X->ResetArray();
        Y->ResetArray();
    }
};
class SPD_PreconditionerFactory: public PetscPreconditionerFactory
{
private:
    const Operator& op;

public:
    SPD_PreconditionerFactory(const Operator& op_, const string& name_): PetscPreconditionerFactory(name_), op(op_)
    {
        cout << "in PreconditionerFactory() " << endl;
    }
    virtual ~SPD_PreconditionerFactory() {}

    virtual Solver* NewPreconditioner(const OperatorHandle& oh)
    {
        cout << "in SPD_PreconditionerFactory::NewPreconditioner()" << endl;
        return new SPD_PreconditionerSolver(oh);
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
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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
            Dirichlet_attr = 1;
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
        delete fsp;
        delete phi;
        delete c1;
        delete c2;
        delete phi_n;
        delete c1_n;
        delete c2_n;
        delete dc;
        delete fec;
    }

    // 把下面的5个求解过程串联起来
    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        if (myid == 0) {
            cout << '\n';
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times
                 << ", " << options_src << '\n'
                 << endl;
        }

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
            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP1_EAFE();
            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP1_SUPG();
            else MFEM_ABORT("Not support stabilization.");
            (*c1_n) = (*c1);

            if (strcmp(AdvecStable, "none") == 0)      Solve_NP2();
            else if (strcmp(AdvecStable, "eafe") == 0) Solve_NP2_EAFE();
            else if (strcmp(AdvecStable, "supg") == 0) Solve_NP2_SUPG();
            else MFEM_ABORT("Not support stabilization.");
            (*c2_n) = (*c2);

            if (myid == 0) {
                cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            }
            if (tol < Gummel_rel_tol)
            {
                if (myid == 0) {
                    cout << "------> Gummel iteration converge: " << iter << " times." << endl;
                }
                break;
            }
            iter++;

            if (myid == 0) {
                cout << endl;
            }
        }

        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

        cout.precision(14);
        double phiL2err = phi->ComputeL2Error(phi_exact);
        double c1L2err = c1->ComputeL2Error(c1_exact);
        double c2L2err = c2->ComputeL2Error(c2_exact);

        if (myid == 0) {
            cout << "\n======>Box, " << Linearize << ", " << Discretize << p_order << ", refine " << refine_times
                 << " for " << mesh_file << ", " << options_src << ", -rate: " << ComputeConvergenceRate << endl;
            cout << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
                 << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
                 << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
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
            np1_avg_iter = (np1_iter.Sum() / np1_iter.Size());
            np1_avg_time = np1_time.Sum() / np1_iter.Size();
            out2["np1_avg_iter"] = np1_avg_iter;
            out2["np1_avg_time"] = np1_avg_time;
            np2_avg_iter = (np2_iter.Sum() / np2_iter.Size());
            np2_avg_time = np2_time.Sum() / np2_iter.Size();
            out2["np2_avg_iter"] = np2_avg_iter;
            out2["np2_avg_time"] = np2_avg_time;

            map<string, Array<double>>::iterator it1;
            for (it1=out1.begin(); it1!=out1.end(); ++it1)
                (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
            map<string, double>::iterator it2;
            for (it2=out2.begin(); it2!=out2.end(); ++it2)
                cout << (*it2).first << ": " << (*it2).second << endl;
        }

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

        if (myid == 0) {
            cout << "------------------------------ All Good! -------------------------\n\n" << endl;
        }
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

        // 所有并行的离散部分全部在MFEM和Hypre之间完成，PETSc不参与；
        // PETSc只参与求解线性方程组，且在把矩阵向量传入PETSc的时候，
        // 不需要自己进行数据类型转换.
        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "phi_");

        MPI_Barrier(MPI_COMM_WORLD);
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
        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            poisson_iter.Append(solver->GetNumIterations());
            poisson_time.Append(chrono.RealTime());
        }

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
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np1_");

        MPI_Barrier(MPI_COMM_WORLD);
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

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            np1_iter.Append(solver->GetNumIterations());
            np1_time.Append(chrono.RealTime());
        }
//        cout << "L2 error norm of | c1_h - c1_e |: " << c1->ComputeL2Error(c1_exact) << endl;
//        MFEM_ABORT("Stop here for test NP1 convergence rate in PNP_CG_Gummel_Solver_par!");

        (*c1_n) *= relax;
        (*c1)   *= 1-relax;
        (*c1)   += (*c1_n); // 利用松弛方法更新c1
        (*c1_n) /= relax; // 还原c1_n.避免松弛因子为0的情况造成除0

        delete lf;
        delete blf;
        delete solver;
    }
    void Solve_NP1_EAFE()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP1 convergence rate

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        // D1 (grad(c1), grad(v1))
        blf->AddDomainIntegrator(new DiffusionIntegrator);
//        // D1 z1 (c1 grad(phi^k), grad(v1))
//        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
        *lf = 0.0;
//        // D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
//        ScalarVectorProductCoefficient neg_J1(neg, J1);
//        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J1), Neumann_attr);
        // (f1, v1)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->Assemble();

        SparseMatrix& A_sp = blf->SpMat();
        EAFE_Modify(mesh, A_sp, DiffusionTensor_K, adv1);
        blf->EliminateVDofs(ess_tdof_list, *c1, *lf);

        PetscParMatrix *A = new PetscParMatrix(&A_sp, Operator::PETSC_MATAIJ);
        PetscParVector *x = new PetscParVector(A->GetComm(), *lf);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");

        chrono.Clear();
        chrono.Start();
        solver->Mult(*lf, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c1);

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
    void Solve_NP1_SUPG()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP1 convergence rate

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        // D1 (grad(c1), grad(v1))
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi^k), grad(v1))
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&diffusion_tensor_K, neg, advection_vector_K, neg, div_Adv1, mesh));
        blf->AddDomainIntegrator(new MassIntegrator(neg_div_Adv1));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
        *lf = 0.0;
        // D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J1(neg, J1);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J1), Neumann_attr);
        // (f1, v1)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        lf->AddDomainIntegrator(new SUPG_LinearFormIntegrator(diffusion_tensor_K, advection_vector_K, one, f1_analytic, mesh));
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c1, *lf, *A, *x, *b);

        {
//            Mat A_mat = Mat(*A);
//            Write_Mat(A_mat, "./A_matlab.txt");
//            MFEM_ABORT("output matrix A.");
        }

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np1_");

        if (use_np1spd)
        {
            ParBilinearForm* p_blf = new ParBilinearForm(fsp);
            p_blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
            p_blf->Assemble(0);
            p_blf->Finalize(0);

            PetscParMatrix* P = new PetscParMatrix();
            p_blf->SetOperatorType(Operator::PETSC_MATAIJ);
            p_blf->FormSystemMatrix(ess_tdof_list, *P);

            PetscLinearSolver* pc = new PetscLinearSolver(*P, "np1spdpc_");
            PetscPreconditioner* pc_ = new PetscPreconditioner(*P, "np1spdpc_");
            solver->SetPreconditioner(*pc_);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c1);

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
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np2_");

        MPI_Barrier(MPI_COMM_WORLD);
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

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            np2_iter.Append(solver->GetNumIterations());
            np2_time.Append(chrono.RealTime());
        }

//        cout << "L2 error norm of | c2_h - c2_e |: " << c2->ComputeL2Error(c2_exact) << endl;
//        MFEM_ABORT("Stop here for test convergence rate in PNP_CG_Gummel_Solver_par!");

        (*c2_n) *= relax;
        (*c2)   *= 1-relax;
        (*c2)   += (*c2_n); // 利用松弛方法更新c2
        (*c2_n) /= relax+TOL; // 还原c2_n.避免松弛因子为0的情况造成除0

        delete lf;
        delete blf;
        delete solver;
    }
    void Solve_NP2_EAFE()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP2 convergence rate

        ParBilinearForm *blf(new ParBilinearForm(fsp));
        // D2 (grad(c2), grad(v2))
        blf->AddDomainIntegrator(new DiffusionIntegrator);
//        // D2 z2 (c2 grad(phi^k), grad(v2))
//        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
//        // D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
//        ScalarVectorProductCoefficient neg_J2(neg, J2);
//        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J2), Neumann_attr);
        // (f2, v2)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->Assemble();

        SparseMatrix& A_sp = blf->SpMat();
        EAFE_Modify(mesh, A_sp, DiffusionTensor_Cl, adv2);
        blf->EliminateVDofs(ess_tdof_list, *c2, *lf);

        PetscParMatrix *A = new PetscParMatrix(&A_sp, Operator::PETSC_MATAIJ);
        PetscParVector *x = new PetscParVector(A->GetComm(), *lf);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();
        solver->Mult(*lf, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c2);

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
    void Solve_NP2_SUPG()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test NP2 convergence rate

        ParBilinearForm *blf(new ParBilinearForm(fsp));
        // D2 (grad(c2), grad(v2))
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi^k), grad(v2))
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&diffusion_tensor_Cl, neg, advection_vector_Cl, neg, div_Adv2, mesh));
        blf->AddDomainIntegrator(new MassIntegrator(neg_div_Adv2));
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
        // D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J2(neg, J2);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J2), Neumann_attr);
        // (f2, v2)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        lf->AddDomainIntegrator(new SUPG_LinearFormIntegrator(diffusion_tensor_Cl, advection_vector_Cl, one, f2_analytic, mesh));
        lf->Assemble();

        PetscParMatrix *A = new PetscParMatrix();
        PetscParVector *x = new PetscParVector(fsp);
        PetscParVector *b = new PetscParVector(fsp);
        blf->SetOperatorType(Operator::PETSC_MATAIJ);
        blf->FormLinearSystem(ess_tdof_list, *c2, *lf, *A, *x, *b);

        PetscLinearSolver* solver = new PetscLinearSolver(*A, "np2_");

        if (use_np2spd)
        {
            ParBilinearForm* p = new ParBilinearForm(fsp);
            p->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
            p->Assemble(0);
            p->Finalize(0);

            PetscParMatrix* P = new PetscParMatrix();
            p->SetOperatorType(Operator::PETSC_MATAIJ);
            p->FormSystemMatrix(ess_tdof_list, *P);

            PetscLinearSolver* pc = new PetscLinearSolver(*P, "np2spdpc_");
            solver->SetPreconditioner(*pc);
        }

        chrono.Clear();
        chrono.Start();
        solver->Mult(*b, *x);
        chrono.Stop();
        blf->RecoverFEMSolution(*x, *lf, *c2);

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
    Array<int> Dirichlet, Neumann, ess_tdof_list;

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
    PNP_DG_Gummel_Solver_par(Mesh& mesh_): mesh(mesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new DG_FECollection(p_order, mesh.Dimension());
        fsp   = new ParFiniteElementSpace(pmesh, fec);

        phi   = new ParGridFunction(fsp);
        c1    = new ParGridFunction(fsp);
        c2    = new ParGridFunction(fsp);
        phi_n = new ParGridFunction(fsp);
        c1_n  = new ParGridFunction(fsp);
        c2_n  = new ParGridFunction(fsp);

        *phi   = 0.0; phi  ->SetTrueVector();
        *phi_n = 0.0; phi_n->SetTrueVector();
        *c1    = 0.0; c1   ->SetTrueVector();
        *c1_n  = 0.0; c1_n ->SetTrueVector();
        *c2    = 0.0; c2   ->SetTrueVector();
        *c2_n  = 0.0; c2_n ->SetTrueVector();

        int size = fsp->GetMesh()->bdr_attributes.Max();
        Dirichlet.SetSize(size);
        Neumann.SetSize(size);
        {
            Dirichlet = 1;
            Dirichlet[top_attr - 1]    = 1;
            Dirichlet[bottom_attr - 1] = 1;

            Neumann = 0;
//            Neumann[front_attr - 1] = 1;
//            Neumann[back_attr  - 1] = 1;
//            Neumann[left_attr  - 1] = 1;
//            Neumann[right_attr - 1] = 1;
        }

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
        if (myid == 0) {
            cout << '\n';
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times
                 << ", " << options_src << ", sigma: " << sigma << ", kappa: " << kappa
                 << endl;
        }

        int iter = 1;
        while (iter < Gummel_max_iters)
        {
            Solve_NP1();
            MPI_Barrier(MPI_COMM_WORLD);
            cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" << endl;
            Solve_Poisson();

            Vector diff(fsp->GetNDofs());
            diff = 0.0; // 必须初始化,否则下面的计算结果不对fff
            diff += (*phi);
            diff -= (*phi_n); // 不能把上述2步合并成1步: diff = (*phi) - (*phi_n)fff
            double tol = diff.Norml2() / phi->Norml2(); // 相对误差
            (*phi_n) = (*phi);

            (*c1_n) = (*c1);

            Solve_NP2();
            (*c2_n) = (*c2);

            if (myid == 0) {
                cout << "===> " << iter << "-th Gummel iteration, phi relative tolerance: " << tol << endl;
            }
            if (tol < Gummel_rel_tol)
            {
                if (myid == 0) {
                    cout << "------> Gummel iteration converge: " << iter << " times." << endl;
                }
                break;
            }

            iter++;

            if (myid == 0) {
                cout << endl;
            }
        }
        if (iter == Gummel_max_iters) MFEM_ABORT("------> Gummel iteration Failed!!!");

        cout.precision(14);
        double phiL2err = phi->ComputeL2Error(phi_exact);
        double c1L2err = c1->ComputeL2Error(c1_exact);
        double c2L2err = c2->ComputeL2Error(c2_exact);

        if (myid == 0) {
            cout << "\n======>Box, " << Linearize << ", " << Discretize << p_order << ", refine " << refine_times
                 << " for " << mesh_file << ", " << options_src << ", -rate: " << ComputeConvergenceRate << endl;
            cout << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
                 << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
                 << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
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
            np1_avg_iter = (np1_iter.Sum() / np1_iter.Size());
            np1_avg_time = np1_time.Sum() / np1_iter.Size();
            out2["np1_avg_iter"] = np1_avg_iter;
            out2["np1_avg_time"] = np1_avg_time;
            np2_avg_iter = (np2_iter.Sum() / np2_iter.Size());
            np2_avg_time = np2_time.Sum() / np2_iter.Size();
            out2["np2_avg_iter"] = np2_avg_iter;
            out2["np2_avg_time"] = np2_avg_time;

            map<string, Array<double>>::iterator it1;
            for (it1=out1.begin(); it1!=out1.end(); ++it1)
                (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
            map<string, double>::iterator it2;
            for (it2=out2.begin(); it2!=out2.end(); ++it2)
                cout << (*it2).first << ": " << (*it2).second << endl;
        }

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
            (*c1) /= alpha3;
            (*c2) /= alpha3;
            Visualize(*dc, "phi", "phi_Gummel_DG");
            Visualize(*dc, "c1", "c1_Gummel_DG");
            Visualize(*dc, "c2", "c2_Gummel_DG");
            ofstream results("phi_c1_c2_Gummel_DG.vtk");
            results.precision(14);
            int ref = 0;
            mesh.PrintVTK(results, ref);
            phi->SaveVTK(results, "phi", ref);
            c1->SaveVTK(results, "c1", ref);
            c2->SaveVTK(results, "c2", ref);
            (*phi) *= (alpha1);
            (*c1) *= (alpha3);
            (*c2) *= (alpha3);
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(epsilon_water, *phi, error);
            ComputeLocalConservation(D_K_, *c1, v_K_coeff, *phi, error1);
            ComputeLocalConservation(D_Cl_, *c2, v_Cl_coeff, *phi, error2);

            ofstream file("./phi_local_conservation_DG_Gummel_box.txt"),
                     file1("./c1_local_conservation_DG_Gummel_box.txt"),
                     file2("./c2_local_conservation_DG_Gummel_box.txt");
            if (file.is_open() && file1.is_open() && file2.is_open())
            {
                error.Print(file, 1);
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }

        if (myid == 0) {
            cout << "------------------------------ All Good! -------------------------\n\n" << endl;
        }
    }

    void Solve(BlockVector& vec, Array<int>& offsets, double initTol)
    {
        cout << "\n    Obtain nonlinear iteration initial value, Gummel, DG" << p_order << ", box, parallel"
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

//        cout << "l2 norm of phi: " << phi->Norml2() << endl;
//        cout << "l2 norm of  c1: " << c1->Norml2() << endl;
//        cout << "l2 norm of  c2: " << c2->Norml2() << endl;
        phi->SetTrueVector();
        c1 ->SetTrueVector();
        c2 ->SetTrueVector();
//        cout << "l2 norm of phi: " << phi->Norml2() << endl;
//        cout << "l2 norm of  c1: " << c1->Norml2() << endl;
//        cout << "l2 norm of  c2: " << c2->Norml2() << endl;

//        cout << "l2 norm of vec: " << vec.Norml2() << endl;
        vec.GetBlock(0) = phi->GetTrueVector();
        vec.GetBlock(1) = c1->GetTrueVector();
        vec.GetBlock(2) = c2->GetTrueVector();
//        cout << "l2 norm of vec: " << vec.Norml2() << endl;

        // 为了测试vec是否正确被赋值
//        phi_n->MakeRef(fsp, vec, offsets[0]);
//        c1_n ->MakeRef(fsp, vec, offsets[1]);
//        c2_n ->MakeRef(fsp, vec, offsets[2]);
//        phi_n->SetTrueVector();
//        c1_n ->SetTrueVector();
//        c2_n ->SetTrueVector();
//        phi_n->SetFromTrueVector();
//        c1_n ->SetFromTrueVector();
//        c2_n ->SetFromTrueVector();
//        cout << "l2 norm of phi: " <<  phi->Norml2() << endl;
//        cout << "l2 norm of  c1: " << c1_n->Norml2() << endl;
//        cout << "l2 norm of  c2: " << c2_n->Norml2() << endl;
    }

private:
    void Solve_Poisson()
    {
//        c1_n->ProjectCoefficient(c1_exact); // for test convergence rate
//        c2_n->ProjectCoefficient(c2_exact); // for test convergence rate

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        // epsilon_s (grad(phi), grad(psi))
        blf->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        // - <{epsilon_s grad(phi)}, [psi]> + sigma <[phi], {epsilon_s grad(psi)}> + kappa <{h^{-1} epsilon_s} [phi], [psi]>
        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa)); // 后面两个参数分别是对称化参数, 惩罚参数
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa), Dirichlet);
        blf->Assemble();
        blf->Finalize();

        // Poisson方程关于离子浓度的两项
        ParLinearForm *lf = new ParLinearForm(fsp); //Poisson方程的右端项
        GridFunctionCoefficient c1_n_coeff(c1_n), c2_n_coeff(c2_n);
        ProductCoefficient rhs1(alpha2_prod_alpha3_prod_v_K, c1_n_coeff);
        ProductCoefficient rhs2(alpha2_prod_alpha3_prod_v_Cl, c2_n_coeff);
        // alpha2 alpha3 z1 (c1^k, psi)
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs1));
        // alpha2 alpha3 z2 (c2^k, psi)
        lf->AddDomainIntegrator(new DomainLFIntegrator(rhs2));
        ScalarVectorProductCoefficient neg_J(neg, J);
        // epsilon_s <grad(phi_e).n, psi>, phi_flux = -epsilon_s grad(phi_e)
        // fff BoundaryNormalLFIntegrator for DG is not OK?
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J), Neumann);
        // sigma <phi_e, (epsilon_s grad(psi)).n)> + kappa <{h^{-1} Q} phi_e, psi>
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, sigma, kappa), Dirichlet); // 用真解构造Dirichlet边界条件
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);

        PetscLinearSolver* solver = new PetscLinearSolver(A, "phi_");

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *phi);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "phi solver: successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "phi solver: failed to converged" << endl;
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            poisson_iter.Append(solver->GetNumIterations());
            poisson_time.Append(chrono.RealTime());
        }

//        cout << "L2 error norm of |phi_h - phi_e|: " << phi->ComputeL2Error(phi_exact) << endl;
//        MFEM_ABORT("Stop here for testing Poisson convergence rate in PNP_DG_Gummel_Solver_par!");

        delete blf;
        delete lf;
        delete solver;
    }

    void Solve_NP1()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test convergence rate

        MPI_Barrier(MPI_COMM_WORLD);
cout << "111" << endl;
        ParBilinearForm *blf = new ParBilinearForm(fsp);
        ProductCoefficient neg_D_K_v_K(neg, D_K_prod_v_K);
        ProductCoefficient sigma_D_K_v_K(sigma_coeff, D_K_prod_v_K);
//        // D1 (grad(c1), grad(v1))
//        blf->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
//        // D1 z1 (c1 grad(phi^k), grad(v1))
//        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_K_prod_v_K));
//        // -<{D1 grad(c1).n}, [v1]> + sigma <[c1], {D1 grad(v1).n}> + kappa <{h^{-1} D1} [c1], [v1]>
//        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa));
//        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa), Dirichlet);
        // -D1 z1 <{c1 grad(phi^k).n}, [v1]>
//        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, *phi_n));
//        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_K_v_K, *phi_n), Dirichlet);
//        // sigma <[c1], {D1 z1 v1 grad(phi^k).n}>
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(one, *phi_n));
//        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_K_v_K, *phi_n), Dirichlet);
        int gdb_break = 1;
        while(gdb_break) {};
        blf->Assemble(0);
        blf->Finalize(0);
        MPI_Barrier(MPI_COMM_WORLD);
cout << "2222" << endl;

        ParLinearForm *lf = new ParLinearForm(fsp); //NP1方程的右端项
        // D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J1(neg, J1);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J1), Neumann);
        // (f1, v1)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        // sigma <c1_e, D1 grad(v1).n> + kappa <{h^{-1} D1} c1_e, v1>
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_exact, D_K_, sigma, kappa));
        // sigma D1 z1 <c1_e, v1 grad(phi^k).n>
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_K_v_K, &c1_exact, phi_n));
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np1_");

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c1);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np1 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np1 solver : failed to converged" << endl;
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            np1_iter.Append(solver->GetNumIterations());
            np1_time.Append(chrono.RealTime());
        }

        delete blf;
        delete lf;
        delete solver;
    }

    void Solve_NP2()
    {
//        phi_n->ProjectCoefficient(phi_exact); // test convergence rate

        ParBilinearForm *blf = new ParBilinearForm(fsp);
        ProductCoefficient sigma_D_Cl_v_Cl(sigma_coeff, D_Cl_prod_v_Cl);
        ProductCoefficient neg_D_Cl_v_Cl(neg, D_Cl_prod_v_Cl);
        // D2 (grad(c2), grad(v2))
        blf->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi^k), grad(v2))
        blf->AddDomainIntegrator(new GradConvectionIntegrator(*phi_n, &D_Cl_prod_v_Cl));
        // -<{D2 grad(c2).n}, [v2]> + sigma <[c2], {D2 grad(v2).n}> + kappa <{h^{-1} D2} [c2], [v2]>
        blf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa));
        blf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa), Dirichlet);
        // -D2 z2 <{c2 grad(phi^k).n}, [v2]>
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D_Cl_v_Cl, *phi_n), Dirichlet);
        // sigma D2 z2 <[c2], {v2 grad(phi^k).n}>
        blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi_n));
        blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D_Cl_v_Cl, *phi_n), Dirichlet);
        blf->Assemble(0);
        blf->Finalize(0);

        ParLinearForm *lf = new ParLinearForm(fsp); //NP2方程的右端项
        // D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J2(neg, J2);
        lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J2), Neumann);
        // (f2, v2)
        lf->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        // sigma <c2_e, D2 grad(v2).n> + kappa <{h^{-1} D2} c2_e, v2>
        lf->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_exact, D_Cl_, sigma, kappa));
        // sigma D2 z2 <c2_e, v2 grad(phi^k).n>
        lf->AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&sigma_D_Cl_v_Cl, &c2_exact, phi_n));
        lf->Assemble();

        HypreParMatrix A;
        Vector x, b;
        blf->FormLinearSystem(ess_tdof_list, *phi, *lf, A, x, b);
        PetscLinearSolver* solver = new PetscLinearSolver(A, false, "np2_");

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();
        solver->Mult(b, x);
        chrono.Stop();
        blf->RecoverFEMSolution(x, *lf, *c2);

#ifdef SELF_VERBOSE
        if (solver->GetConverged() == 1 && myid == 0)
            cout << "np2 solver : successfully converged by iterating " << solver->GetNumIterations() << " times, taking " << chrono.RealTime() << " s." << endl;
        else if (solver->GetConverged() != 1)
            cerr << "np2 solver : failed to converged" << endl;
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            np2_iter.Append(solver->GetNumIterations());
            np2_time.Append(chrono.RealTime());
        }

        delete blf;
        delete lf;
        delete solver;
    }
};


class PNP_CG_Newton_Operator_par;
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
//        cout << "in BlockPreconditionerSolver::BlockPreconditionerSolver()" << endl;
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

            VecRestoreSubVector(*X, index_set[i], &blockx);
            VecRestoreSubVector(*Y, index_set[i], &blocky);
        }

        X->ResetArray();
        Y->ResetArray();
    }
};
class PreconditionerFactory: public PetscPreconditionerFactory
{
private:
    const Operator& op; // op就是Nonlinear Operator(可用来计算Residual, Jacobian)

public:
    PreconditionerFactory(const Operator& op_, const string& name_): PetscPreconditionerFactory(name_), op(op_)
    {
//        cout << "in PreconditionerFactory() " << endl;
    }
    virtual ~PreconditionerFactory() {}

    virtual Solver* NewPreconditioner(const OperatorHandle& oh) // oh就是当前Newton迭代步的Jacobian的句柄
    {
//        cout << "in NewPreconditioner() " << endl;
        return new BlockPreconditionerSolver(oh);
    }
};


class PNP_CG_Newton_Operator_par: public Operator
{
protected:
    ParFiniteElementSpace *fsp;

    Array<int> block_offsets, block_trueoffsets;
    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution
    PetscNonlinearSolver* newton_solver;

    mutable ParLinearForm *f, *f1, *f2;
    mutable PetscParMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable ParBilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;
    ParGridFunction *phi, *c1_k, *c2_k;

    Array<int> Neumann_attr, Dirichlet_attr, ess_tdof_list;

    StopWatch chrono;
    int num_procs, myid;
    Array<int> null_array;

public:
    PNP_CG_Newton_Operator_par(ParFiniteElementSpace *fsp_): Operator(fsp_->TrueVSize()*3), fsp(fsp_)
    {
        MPI_Comm_size(fsp->GetComm(), &num_procs);
        MPI_Comm_rank(fsp->GetComm(), &myid);

        int bdr_size = fsp->GetMesh()->bdr_attributes.Max();
        {
            Neumann_attr.SetSize(bdr_size);
            Neumann_attr = 0;
//            Neumann_attr[left_attr - 1]  = 1;
//            Neumann_attr[right_attr - 1] = 1;
//            Neumann_attr[front_attr - 1] = 1;
//            Neumann_attr[back_attr - 1]  = 1;

            Dirichlet_attr.SetSize(bdr_size);
            Dirichlet_attr = 1;
            Dirichlet_attr[top_attr - 1]    = 1;
            Dirichlet_attr[bottom_attr - 1] = 1;

            fsp->GetEssentialTrueDofs(Dirichlet_attr, ess_tdof_list);
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
        phi   = new ParGridFunction(fsp);
        c1_k  = new ParGridFunction(fsp);
        c2_k  = new ParGridFunction(fsp);

        f  = new ParLinearForm(fsp);
        f1  = new ParLinearForm(fsp);
        f2  = new ParLinearForm(fsp);
        a21 = new ParBilinearForm(fsp);
        a22 = new ParBilinearForm(fsp);
        a31 = new ParBilinearForm(fsp);
        a33 = new ParBilinearForm(fsp);

        a11 = new ParBilinearForm(fsp);
        // epsilon_s (grad(dphi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a11->Assemble(0);
        a11->Finalize(0);
        a11->SetOperatorType(Operator::PETSC_MATAIJ);
        a11->FormSystemMatrix(ess_tdof_list, A11);

        a12 = new ParBilinearForm(fsp);
        // -alpha2 alpha3 z1 (dc1, psi)
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_K(neg, alpha2_prod_alpha3_prod_v_K);
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(0);
        a12->Finalize(0);
        a12->SetOperatorType(Operator::PETSC_MATAIJ);
        a12->FormSystemMatrix(null_array, A12);

        a13 = new ParBilinearForm(fsp);
        // -alpha2 alpha3 z2 (dc2, psi)
        ProductCoefficient neg_alpha2_prod_alpha3_prod_v_Cl(neg, alpha2_prod_alpha3_prod_v_Cl);
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(0);
        a13->Finalize(0);
        a13->SetOperatorType(Operator::PETSC_MATAIJ);
        a13->FormSystemMatrix(null_array, A13);
    }
    virtual ~PNP_CG_Newton_Operator_par()
    {
        delete f, f1, f2;
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
        Array<int>& Neumann_attr_ = const_cast<Array<int>&>(Neumann_attr);

        phi->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
//        cout << "in Mult(), l2 norm of phi: " <<  phi->Norml2() << endl;
//        cout << "in Mult(), l2 norm of  c1: " << c1_k->Norml2() << endl;
//        cout << "in Mult(), l2 norm of  c2: " << c2_k->Norml2() << endl;
//        cout << "in Mult(), l2 norm of Residual: " << rhs_k->Norml2() << endl;

        rhs_k->Update(y.GetData(), block_trueoffsets); // update residual
        Vector y1(y.GetData() +   0, sc);
        Vector y2(y.GetData() +  sc, sc);
        Vector y3(y.GetData() +2*sc, sc);

        delete f;
        f = new ParLinearForm(fsp);
        f->Update(fsp, rhs_k->GetBlock(0), 0);
        GridFunctionCoefficient c1_k_coeff(c1_k), c2_k_coeff(c2_k);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff);
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff);
        SumCoefficient term(term1, term2);
        ProductCoefficient neg_term(neg, term);
        // - alpha2 alpha3 (z1 c1^k + z2 c2^k, psi)
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term));
        // epsilon_s (grad(phi^k), grad(psi))
        f->AddDomainIntegrator(new GradConvectionIntegrator2(&epsilon_water, phi));
        // epsilon_s <grad(phi_e).n, psi>, phi_flux = J = -epsilon_s grad(phi_e)
        f->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(J), Neumann_attr_);
        f->Assemble();
        f->SetSubVector(ess_tdof_list, 0.0);

        delete f1;
        f1 = new ParLinearForm(fsp);
        f1->Update(fsp, rhs_k->GetBlock(1), 0);
        ProductCoefficient D1_prod_z1_prod_c1_k(D_K_prod_v_K, c1_k_coeff);
        // D1 (grad(c1^k), grad(v1))
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D_K_, c1_k));
        // D1 z1 c1^k (grad(phi^k), grad(psi))
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_prod_z1_prod_c1_k, phi));
        // -D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
        f1->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(J1), Neumann_attr_);
        // -(f1, v1)
        ProductCoefficient neg_f1(neg, f1_analytic);
        f1->AddDomainIntegrator(new DomainLFIntegrator(neg_f1));
        f1->Assemble();
        f1->SetSubVector(ess_tdof_list, 0.0);

        delete f2;
        f2 = new ParLinearForm(fsp);
        f2->Update(fsp, rhs_k->GetBlock(2), 0);
        ProductCoefficient D2_prod_z2_prod_c2_k(D_Cl_prod_v_Cl, c2_k_coeff);
        GradientGridFunctionCoefficient grad_c2_k(c2_k);
        // D2 (grad(c2^k), grad(v2))
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D_Cl_, c2_k));
        // D2 z2 c2^k (grad(phi^k), grad(v2))
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_prod_z2_prod_c2_k, phi));
        // -D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
        f2->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(J2), Neumann_attr_);
        // -(f2, v2)
        ProductCoefficient neg_f2(neg, f2_analytic);
        f2->AddDomainIntegrator(new DomainLFIntegrator(neg_f2));
        f2->Assemble();
        f2->SetSubVector(ess_tdof_list, 0.0);

//        cout << "in Mult(), l2 norm of Residual: " << rhs_k->Norml2() << endl;
    }

    virtual Operator &GetGradient(const Vector& x) const
    {
        int sc = height / 3;
        Vector& x_ = const_cast<Vector&>(x);

        phi->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();
//        cout << "in GetGradient(), l2 norm of phi: "  << phi->Norml2() << endl;
//        cout << "in GetGradient(), l2 norm of  c1: " <<   c1_k->Norml2() << endl;
//        cout << "in GetGradient(), l2 norm of  c2: " <<   c2_k->Norml2() << endl;

        delete a21;
        a21 = new ParBilinearForm(fsp);
        GridFunctionCoefficient c1_k_coeff(c1_k);
        ProductCoefficient D1_prod_z1_prod_c1_k(D_K_prod_v_K, c1_k_coeff);
        // D1 z1 c1^k (grad(dphi), grad(v1))
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_prod_c1_k));
        a21->Assemble(0);
        a21->Finalize(0);
        a21->SetOperatorType(Operator::PETSC_MATAIJ);
        a21->FormSystemMatrix(null_array, A21);
        A21.EliminateRows(ess_tdof_list, 0.0);

        delete a22;
        a22 = new ParBilinearForm(fsp);
        // D1 (grad(dc1), grad(v1))
        a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (dc1 grad(phi^k), grad(v1))
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi, &D_K_prod_v_K));
        a22->Assemble(0);
        a22->Finalize(0);
        a22->SetOperatorType(Operator::PETSC_MATAIJ);
        a22->FormSystemMatrix(ess_tdof_list, A22);

        delete a31;
        a31 = new ParBilinearForm(fsp);
        GridFunctionCoefficient c2_k_coeff(c2_k);
        ProductCoefficient D2_prod_z2_prod_c2_k(D_Cl_prod_v_Cl, c2_k_coeff);
        // D2 z2 c2^k (grad(dphi), grad(v2))
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_prod_c2_k));
        a31->Assemble(0);
        a31->Finalize(0);
        a31->SetOperatorType(Operator::PETSC_MATAIJ);
        a31->FormSystemMatrix(null_array, A31);
        A31.EliminateRows(ess_tdof_list, 0.0);

        delete a33;
        a33 = new ParBilinearForm(fsp);
        // D2 (grad(dc2), grad(v2))
        a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (dc2 grad(phi^k), grad(v2))
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi, &D_Cl_prod_v_Cl));
        a33->Assemble(0);
        a33->Finalize(0);
        a33->SetOperatorType(Operator::PETSC_MATAIJ);
        a33->FormSystemMatrix(ess_tdof_list, A33);

        jac_k = new BlockOperator(block_trueoffsets);
        jac_k->SetBlock(0, 0, &A11);
        jac_k->SetBlock(0, 1, &A12);
        jac_k->SetBlock(0, 2, &A13);
        jac_k->SetBlock(1, 0, &A21);
        jac_k->SetBlock(1, 1, &A22);
        jac_k->SetBlock(2, 0, &A31);
        jac_k->SetBlock(2, 2, &A33);
        return *jac_k;
    }
};
class PNP_CG_Newton_box_Solver_par
{
protected:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_space;
    PNP_CG_Newton_Operator_par* op;
    PetscPreconditionerFactory *jac_factory;
    PetscNonlinearSolver* newton_solver;

    Array<int> block_trueoffsets, Dirichlet_attr;
    BlockVector* u_k;
    ParGridFunction phi, c1_k, c2_k;

    StopWatch chrono;
    SNES snes;
    map<string, Array<double>> out1;
    map<string, double> out2;
    Array<double> linear_iter;
    double linearize_iter, total_time, ndofs, linear_avg_iter;
    PetscInt *its=0, num_its=100;
    PetscReal *residual_norms=0;

public:
    PNP_CG_Newton_box_Solver_par(Mesh* mesh_): mesh(mesh_)
    {
        int mesh_dim = mesh->Dimension(); //网格的维数:1D,2D,3D
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

        h1_fec = new H1_FECollection(p_order, mesh_dim);
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        block_trueoffsets.SetSize(4);
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = h1_space->GetTrueVSize();
        block_trueoffsets[2] = h1_space->GetTrueVSize();
        block_trueoffsets[3] = h1_space->GetTrueVSize();
        block_trueoffsets.PartialSum();

        int bdr_size = pmesh->bdr_attributes.Max();
        {
            Dirichlet_attr.SetSize(bdr_size);
            Dirichlet_attr = 1;
        }

        op = new PNP_CG_Newton_Operator_par(h1_space);

        jac_factory   = new PreconditionerFactory(*op, "Block Preconditioner");

        newton_solver = new PetscNonlinearSolver(h1_space->GetComm(), *op, "newton_");
        newton_solver->iterative_mode = true;
        newton_solver->SetMaxIter(max_newton);
        newton_solver->SetPreconditionerFactory(jac_factory);
        snes = SNES(*newton_solver);
        PetscMalloc(num_its * sizeof(PetscInt), &its);
        PetscMalloc(num_its * sizeof(PetscReal), &residual_norms);
        SNESSetConvergenceHistory(snes, residual_norms, its, num_its, PETSC_TRUE);
    }
    virtual ~PNP_CG_Newton_box_Solver_par()
    {
        delete newton_solver, op, jac_factory, u_k, mesh, pmesh;
        PetscFree(its);
        PetscFree(residual_norms);
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout << "\nNewton, CG" << p_order << ", box, parallel"
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        // 给Newton迭代赋初值，必须满足essential边界条件
        u_k = new BlockVector(block_trueoffsets);
        if (zero_initial)
        {
            // MakeTRef(), SetTrueVector(), SetFromTrueVector() 三者要配套使用ffffffffff
            phi .MakeTRef(h1_space, *u_k, block_trueoffsets[0]);
            c1_k.MakeTRef(h1_space, *u_k, block_trueoffsets[1]);
            c2_k.MakeTRef(h1_space, *u_k, block_trueoffsets[2]);
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl; // 为了证实SetTrueVector(), SetFromTrueVector()的作用
            phi = 0.0;
            c1_k = 0.0;
            c2_k = 0.0;
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
            phi .ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
            c1_k.ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
            c2_k.ProjectBdrCoefficient(c2_exact, Dirichlet_attr);
//            phi .ProjectCoefficient(phi_exact); // 测试真解和边界条件是否正确赋值
//            c1_k.ProjectCoefficient(c1_exact );
//            c2_k.ProjectCoefficient(c2_exact );

//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
            phi.SetTrueVector(); // 必须要
            c1_k.SetTrueVector();
            c2_k.SetTrueVector();
//            phi.SetFromTrueVector(); // 似乎可以不要
//            c1_k.SetFromTrueVector();
//            c2_k.SetFromTrueVector();
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
            {
//                VisItDataCollection* dc = new VisItDataCollection("data collection", mesh);
//                dc->RegisterField("phi", &phi);
//                dc->RegisterField("c1",  &c1_k);
//                dc->RegisterField("c2",  &c2_k);
//
//                (phi) /= alpha1;
//                (c1_k)/= alpha3;
//                (c2_k)/= alpha3;
//                Visualize(*dc, "phi", "phi_Newton_CG");
//                Visualize(*dc, "c1", "c1_Newton_CG");
//                Visualize(*dc, "c2", "c2_Newton_CG");
//                (phi)  *= (alpha1);
//                (c1_k) *= (alpha3);
//                (c2_k) *= (alpha3);
            }
        }
        else
        {
            PNP_CG_Gummel_Solver_par initial_solver(*mesh);
            initial_solver.Solve(*u_k, block_trueoffsets, initTol);

            // 为了测试u_k是否正确被赋值
//            phi .MakeTRef(h1_space, *u_k, block_trueoffsets[0]);
//            c1_k.MakeTRef(h1_space, *u_k, block_trueoffsets[1]);
//            c2_k.MakeTRef(h1_space, *u_k, block_trueoffsets[2]);
//            phi .SetFromTrueVector();
//            c1_k.SetFromTrueVector();
//            c2_k.SetFromTrueVector();
//            cout << "l2 norm of phi: " <<  phi.Norml2() << endl;
//            cout << "l2 norm of  c1: " << c1_k.Norml2() << endl;
//            cout << "l2 norm of  c2: " << c2_k.Norml2() << endl;
        }
        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;

        Vector zero_vec;
        chrono.Start();
        newton_solver->Mult(zero_vec, *u_k); // u_k must be a true vector
        chrono.Stop();
        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;

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
        linear_avg_iter = (linear_iter.Sum() / linear_iter.Size());
        out2["linear_avg_iter"] = linear_avg_iter;

        phi .MakeTRef(h1_space, *u_k, block_trueoffsets[0]);
        c1_k.MakeTRef(h1_space, *u_k, block_trueoffsets[1]);
        c2_k.MakeTRef(h1_space, *u_k, block_trueoffsets[2]);
        phi .SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();

        cout.precision(14);
        double phiL2err = phi.ComputeL2Error(phi_exact);
        double c1L2err = c1_k.ComputeL2Error(c1_exact);
        double c2L2err = c2_k.ComputeL2Error(c2_exact);

        cout << "\n======>Box, " << Linearize << ", " << Discretize << p_order << ", refine " << refine_times << " for " << mesh_file << ", " << options_src << ", -rate: " << ComputeConvergenceRate << ", -zero: " << zero_initial << endl;
        cout << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
             << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
             << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;

        if (ComputeConvergenceRate)
        {
            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);

            double totle_size = 0.0;
            for (int i=0; i<mesh->GetNE(); i++)
                totle_size += mesh->GetElementSize(0, 1);

            meshsizes_.Append(totle_size / mesh->GetNE());
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(epsilon_water, phi, error);
            ComputeLocalConservation(D_K_, c1_k, v_K_coeff, phi, error1);
            ComputeLocalConservation(D_Cl_, c2_k, v_Cl_coeff, phi, error2);

            ofstream file("./phi_local_conservation_CG_Newton_box.txt"),
                     file1("./c1_local_conservation_CG_Newton_box.txt"),
                     file2("./c2_local_conservation_CG_Newton_box.txt");
            if (file.is_open() && file1.is_open() && file2.is_open()) {
                error.Print(file, 1);
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }

        if (visualize)
        {
            VisItDataCollection* dc = new VisItDataCollection("data collection", mesh);
            dc->RegisterField("phi", &phi);
            dc->RegisterField("c1",  &c1_k);
            dc->RegisterField("c2",  &c2_k);

            (phi) /= alpha1;
            (c1_k)  /= alpha3;
            (c2_k)  /= alpha3;
            Visualize(*dc, "phi", "phi_Newton_CG");
            Visualize(*dc, "c1", "c1_Newton_CG");
            Visualize(*dc, "c2", "c2_Newton_CG");
            ofstream results("phi_c1_c2_Newton_CG.vtk");
            results.precision(14);
            int ref = 0;
            mesh->PrintVTK(results, ref);
            phi.SaveVTK(results, "phi", ref);
            c1_k.SaveVTK(results, "c1", ref);
            c2_k.SaveVTK(results, "c2", ref);
            (phi)  *= (alpha1);
            (c1_k) *= (alpha3);
            (c2_k) *= (alpha3);
        }

        map<string, Array<double>>::iterator it1;
        for (it1=out1.begin(); it1!=out1.end(); ++it1)
            (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
        map<string, double>::iterator it2;
        for (it2=out2.begin(); it2!=out2.end(); ++it2)
            cout << (*it2).first << ": " << (*it2).second << endl;

        cout << "approximate mesh scale h: " << pow(h1_space->GetTrueVSize(), -1.0/3) << endl;
    }
};


class PNP_DG_Newton_Operator_par: public Operator
{
protected:
    ParFiniteElementSpace *fsp;

    Array<int> block_trueoffsets;
    mutable BlockVector *rhs_k; // current rhs corresponding to the current solution
    mutable BlockOperator *jac_k; // Jacobian at current solution

    mutable ParLinearForm *f, *f1, *f2, *g, *g1, *g2;
    mutable PetscParMatrix A11, A12, A13, A21, A22, A31, A33;
    mutable ParBilinearForm *a11, *a12, *a13, *a21, *a22, *a31, *a33;

    ParGridFunction *phi, *c1_k, *c2_k;

    PetscNonlinearSolver* newton_solver;

    Array<int> ess_bdr, ess_tdof_list;
    Array<int> null_array, Dirichlet_attr, Neumann_attr;

    StopWatch chrono;
    int num_procs, myid;

public:
    PNP_DG_Newton_Operator_par(ParFiniteElementSpace *fsp_): Operator(fsp_->TrueVSize()*3), fsp(fsp_)
    {
        MPI_Comm_size(fsp->GetComm(), &num_procs);
        MPI_Comm_rank(fsp->GetComm(), &myid);

        int bdr_size = fsp->GetMesh()->bdr_attributes.Max();
        {
            Neumann_attr.SetSize(bdr_size);
            Neumann_attr = 0;
//            Neumann_attr[left_attr - 1]  = 1;
//            Neumann_attr[right_attr - 1] = 1;
//            Neumann_attr[front_attr - 1] = 1;
//            Neumann_attr[back_attr - 1]  = 1;

            Dirichlet_attr.SetSize(bdr_size);
            Dirichlet_attr = 1;
            Dirichlet_attr[top_attr - 1]    = 1;
            Dirichlet_attr[bottom_attr - 1] = 1;

            fsp->GetEssentialTrueDofs(Dirichlet_attr, ess_tdof_list);
        }

        block_trueoffsets.SetSize(4); // number of variables + 1;
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = fsp->GetTrueVSize();
        block_trueoffsets[2] = fsp->GetTrueVSize();
        block_trueoffsets[3] = fsp->GetTrueVSize();
        block_trueoffsets.PartialSum();

        rhs_k = new BlockVector(block_trueoffsets); // not block_offsets !!!
        jac_k = new BlockOperator(block_trueoffsets);
        phi   = new ParGridFunction(fsp);
        c1_k  = new ParGridFunction(fsp);
        c2_k  = new ParGridFunction(fsp);

        f  = new ParLinearForm(fsp);
        f1  = new ParLinearForm(fsp);
        f2  = new ParLinearForm(fsp);
        a21 = new ParBilinearForm(fsp);
        a22 = new ParBilinearForm(fsp);
        a31 = new ParBilinearForm(fsp);
        a33 = new ParBilinearForm(fsp);

        g = new ParLinearForm(fsp);
        // epsilon_s <grad(phi_e).n, psi>, phi_flux = -epsilon_s grad(phi_e)
        ScalarVectorProductCoefficient neg_J(neg, J);
        g->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J), Neumann_attr);
        // sigma <phi_D, (epsilon_s grad(psi)).n> + kappa <h^{-1} epsilon_s phi_D, psi>
        g->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, epsilon_water, sigma, kappa), Dirichlet_attr);
        // kappa * <h^{-1} [c1_D], [psi]>
        g->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c1_exact), Dirichlet_attr);
        // kappa * <h^{-1} [c2_D], [psi]>
        g->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c2_exact), Dirichlet_attr);
        g->Assemble();

        g1 = new ParLinearForm(fsp);
        // sigma <c1_D, D1 grad(v1).n> + kappa <h^{-1} D1 c1_D, v1>
        g1->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c1_exact, D_K_, sigma, kappa), Dirichlet_attr);
        // D1 <(grad(c1_e) + z1 c1_e grad(phi_e)) . n, v1>, c1_flux = J1 = -D1 (grad(c1_e) + z1 c1_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J1(neg, J1);
        g1->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J1), Neumann_attr);
        // (f1, v1)
        g1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        g1->Assemble();

        g2 = new ParLinearForm(fsp);
        // sigma <c2_D, D2 grad(v2).n> + kappa <h^{-1} D2 c2_D, v2>
        g2->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(c2_exact, D_Cl_, sigma, kappa), Dirichlet_attr);
        // D2 <(grad(c2_e) + z2 c2_e grad(phi_e)) . n, v2>, c2_flux = J2 = -D2 (grad(c2_e) + z2 c2_e grad(phi_e))
        ScalarVectorProductCoefficient neg_J2(neg, J2);
        g2->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(neg_J2), Neumann_attr);
        // (f2, v2)
        g2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        g2->Assemble();

        a11 = new ParBilinearForm(fsp);
        // epsilon_s (grad(dphi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        // - <{epsilon_s grad(dphi)}, [psi]> + sigma <[dphi], {epsilon_s grad(psi)}>
        //                                   + kappa <{h^{-1} epsilon_s} [dphi], [psi]>
        a11->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa));
        a11->AddBdrFaceIntegrator(new DGDiffusionIntegrator(epsilon_water, sigma, kappa), Dirichlet_attr);
        a11->Assemble(0);
        a11->Finalize(0);
        a11->SetOperatorType(Operator::PETSC_MATAIJ);
        a11->FormSystemMatrix(null_array, A11);

        a12 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z1 (dc1, psi)
        a12->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_K));
        // kappa <h^{-1} [dc1], [psi]>
        a12->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff));
        a12->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff), Dirichlet_attr);
        a12->Assemble(0);
        a12->Finalize(0);
        a12->SetOperatorType(Operator::PETSC_MATAIJ);
        a12->FormSystemMatrix(null_array, A12);

        a13 = new ParBilinearForm(fsp);
        // - alpha2 alpha3 z2 (dc2, psi)
        a13->AddDomainIntegrator(new MassIntegrator(neg_alpha2_prod_alpha3_prod_v_Cl));
        // kappa <h^{-1} [dc2], [psi]>
        a13->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff));
        a13->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff), Dirichlet_attr);
        a13->Assemble(0);
        a13->Finalize(0);
        a13->SetOperatorType(Operator::PETSC_MATAIJ);
        a13->FormSystemMatrix(null_array, A13);
    }
    virtual ~PNP_DG_Newton_Operator_par()
    {
        delete f, f1, f2;
        delete a11, a12, a13, a21, a22, a31, a33;
        delete rhs_k, jac_k;
        delete newton_solver;
    }

    virtual void Mult(const Vector& x, Vector& y) const
    {
//        cout << "\nin PNP_DG_Newton_Operator::Mult()" << endl;
        int sc = height / 3;
        Vector& x_ = const_cast<Vector&>(x);
        Array<int>& Dirichlet_attr_ = const_cast<Array<int>&>(Dirichlet_attr);

        phi ->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi ->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();

//        {
//            cout << "L2 error norm of |phi_h - phi_e|: " << phi->ComputeL2Error(phi_exact) << endl;
//            cout << "L2 error norm of   |c1_h - c1_e|: " << c1_k->ComputeL2Error(c1_exact) << endl;
//            cout << "L2 error norm of   |c2_h - c2_e|: " << c2_k->ComputeL2Error(c2_exact) << endl;
//
//            VisItDataCollection* dc = new VisItDataCollection("data collection", fsp->GetMesh());
//            dc->RegisterField("phi", phi);
//            dc->RegisterField("c1",  c1_k);
//            dc->RegisterField("c2",  c2_k);
//            (*phi)  /= alpha1;
//            (*c1_k) /= alpha3;
//            (*c2_k) /= alpha3;
//            Visualize(*dc, "phi", "phi_Newton_DG_");
//            Visualize(*dc, "c1", "c1_Newton_DG_");
//            Visualize(*dc, "c2", "c2_Newton_DG_");
//            (*phi)  *= (alpha1);
//            (*c1_k) *= (alpha3);
//            (*c2_k) *= (alpha3);
//        }

        GridFunctionCoefficient phi_coeff(phi), c1_k_coeff(c1_k), c2_k_coeff(c2_k);

        rhs_k->Update(y.GetData(), block_trueoffsets); // update residual

        delete f;
        f = new ParLinearForm(fsp);
        f->Update(fsp, rhs_k->GetBlock(0), 0);
        ProductCoefficient term1(alpha2_prod_alpha3_prod_v_K,  c1_k_coeff);
        ProductCoefficient term2(alpha2_prod_alpha3_prod_v_Cl, c2_k_coeff);
        SumCoefficient term(term1, term2);
        ProductCoefficient neg_term(neg, term);
        // epsilon_s (grad(phi^k), grad(psi))
        f->AddDomainIntegrator(new GradConvectionIntegrator2(&epsilon_water, phi));
        // -<{epsilon_s grad(phi^k)}, [psi]>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_epsilon_water, phi));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_epsilon_water, phi), Dirichlet_attr_);
        // -alpha2 alpha3 (z1 c1^k + z2 c2^k, psi)
        f->AddDomainIntegrator(new DomainLFIntegrator(neg_term));
        // sigma <[phi^k], {epsilon_s grad(psi)}>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_epsilon_water, phi_coeff));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_epsilon_water, phi_coeff), Dirichlet_attr_);
        // kappa epsilon_s <h^{-1} [phi^k], [psi]>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&epsilon_water_prod_kappa, &phi_coeff));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&epsilon_water_prod_kappa, &phi_coeff), Dirichlet_attr_);
        // kappa <h^{-1} [c1^k], [psi]>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c1_k_coeff));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c1_k_coeff), Dirichlet_attr_);
        // kappa <h^{-1} [c2^k], [psi]>
        f->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c2_exact));
        f->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_coeff, &c2_exact), Dirichlet_attr_);
        f->Assemble();
        (*f) -= (*g);

        delete f1;
        f1 = new ParLinearForm(fsp);
        f1->Update(fsp, rhs_k->GetBlock(1), 0);
        ProductCoefficient D1_prod_z1_prod_c1_k(D_K_prod_v_K, c1_k_coeff);
        ProductCoefficient kappa_prod_D1_prod_z1_prod_c1_k(kappa_coeff, D1_prod_z1_prod_c1_k);
        ProductCoefficient neg_D1_prod_z1_prod_c1_k(neg, D1_prod_z1_prod_c1_k);
        ProductCoefficient sigma_D1_prod_z1_prod_c1_k(sigma_coeff, D1_prod_z1_prod_c1_k);
        ProductCoefficient neg_sigma_D1_prod_z1_prod_c1_k(neg, sigma_D1_prod_z1_prod_c1_k);
        // D1 (grad(c1^k), grad(v1))
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D_K_, c1_k));
        // - <{D1 grad(c1^k)}, [v1]>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D1, c1_k));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D1, c1_k), Dirichlet_attr_);
        // sigma <[c1^k], {D1 grad(v1)}>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D1, c1_k_coeff));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D1, c1_k_coeff), Dirichlet_attr_);
        // kappa D1 <h^{-1} [c1^k], [v1]>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D1, &c1_k_coeff));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D1, &c1_k_coeff), Dirichlet_attr_);
        // D1 z1 c1^k (grad(phi^k), grad(v1))
        f1->AddDomainIntegrator(new GradConvectionIntegrator2(&D1_prod_z1_prod_c1_k, phi));
        // -<{D1 z1 c1^k grad(phi^k)}, [v1]>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D1_prod_z1_prod_c1_k, phi));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D1_prod_z1_prod_c1_k, phi), Dirichlet_attr_);
        // sigma <[phi^k], {D1 z1 c1^k grad(v1)}>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D1_prod_z1_prod_c1_k, phi_coeff));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D1_prod_z1_prod_c1_k, phi_coeff), Dirichlet_attr_);
        // kappa <{h^{-1} D1 z1 c1^k}[phi^k], [v1]>
        f1->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_prod_D1_prod_z1_prod_c1_k, &phi_coeff));
        f1->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_prod_D1_prod_z1_prod_c1_k, &phi_coeff), Dirichlet_attr_);
        // - sigma <phi_D, D1 z1 c1^k grad(v1).n> - kappa D1 z1 c1^k <h^{-1} phi_D, v1>
        f1->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, D1_prod_z1_prod_c1_k, -1.0*sigma, -1.0*kappa), Dirichlet_attr_);
        f1->Assemble();
        (*f1) -= (*g1);

        delete f2;
        f2 = new ParLinearForm(fsp);
        f2->Update(fsp, rhs_k->GetBlock(2), 0);
        ProductCoefficient D2_prod_z2_prod_c2_k(D_Cl_prod_v_Cl, c2_k_coeff);
        ProductCoefficient kappa_prod_D2_prod_z2_prod_c2_k(kappa_coeff, D2_prod_z2_prod_c2_k);
        ProductCoefficient neg_D2_prod_z2_prod_c2_k(neg, D2_prod_z2_prod_c2_k);
        ProductCoefficient sigma_D2_prod_z2_prod_c2_k(sigma_coeff, D2_prod_z2_prod_c2_k);
        ProductCoefficient neg_sigma_D2_prod_z2_prod_c2_k(neg, sigma_D2_prod_z2_prod_c2_k);
        // D2 (grad(c2^k), grad(v2))
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D_Cl_, c2_k));
        // -D2 <{grad(c2^k)}, [v2]>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D2, c2_k));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D2, c2_k), Dirichlet_attr_);
        // sigma <[c2^k], {D2 grad(v2)}>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D2, c2_k_coeff));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D2, c2_k_coeff), Dirichlet_attr_);
        // kappa D2 <h^{-1} [c2^k], [v2]>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D2, &c2_k_coeff));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_D2, &c2_k_coeff), Dirichlet_attr_);
        // D2 z2 c2^k (grad(phi^k), grad(v2))
        f2->AddDomainIntegrator(new GradConvectionIntegrator2(&D2_prod_z2_prod_c2_k, phi));
        // -<{D2 z2 c2^k grad(phi^k)}, [v2]>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D2_prod_z2_prod_c2_k, phi));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_5(&neg_D2_prod_z2_prod_c2_k, phi), Dirichlet_attr_);
        // sigma <[phi^k], {D2 z2 c2^k grad(v2)}>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D2_prod_z2_prod_c2_k, phi_coeff));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_7(sigma_D2_prod_z2_prod_c2_k, phi_coeff), Dirichlet_attr_);
        // kappa * <{h^{-1} D2 z2 c2^k} [phi^k], [v2]>
        f2->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_prod_D2_prod_z2_prod_c2_k, &phi_coeff));
        f2->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_4(&kappa_prod_D2_prod_z2_prod_c2_k, &phi_coeff), Dirichlet_attr_);
        // - sigma <phi_D, D2 z2 c2^k grad(v2).n> - kappa D2 z2 c2^k <h^{-1} phi_D, v2>
        f2->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(phi_exact, D2_prod_z2_prod_c2_k, -1.0*sigma, -1.0*kappa), Dirichlet_attr_);
        f2->Assemble();
        (*f2) -= (*g2);
//        cout << "in Mult(), l2 norm of Residual: " << rhs_k->Norml2() << endl;
    }

    virtual Operator &GetGradient(const Vector& x) const
    {
        int sc = height / 3;
        Vector& x_ = const_cast<Vector&>(x);
        Array<int>& Dirichlet_attr_ = const_cast<Array<int>&>(Dirichlet_attr);

        phi ->MakeTRef(fsp, x_, 0);
        c1_k->MakeTRef(fsp, x_, sc);
        c2_k->MakeTRef(fsp, x_, 2*sc);
        phi->SetFromTrueVector();
        c1_k->SetFromTrueVector();
        c2_k->SetFromTrueVector();

        delete a21;
        a21 = new ParBilinearForm(fsp);
        GridFunctionCoefficient c1_k_coeff(c1_k);
        ProductCoefficient D1_prod_z1_prod_c1_k(D_K_prod_v_K, c1_k_coeff);
        // D1 z1 c1^k (grad(dphi), grad(v1))
        a21->AddDomainIntegrator(new DiffusionIntegrator(D1_prod_z1_prod_c1_k));
        // - <{D1 z1 c1^k grad(dphi)}, [v1]> + sigma <[dphi], {D1 z1 c1^k grad(v1)}> + kappa <{h^{-1} D1 z1 c1^k} [dphi], [v1]>
        a21->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D1_prod_z1_prod_c1_k, sigma, kappa));
        a21->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D1_prod_z1_prod_c1_k, sigma, kappa), Dirichlet_attr_);
        a21->Assemble(0);
        a21->Finalize(0);
        a21->SetOperatorType(Operator::PETSC_MATAIJ);
        a21->FormSystemMatrix(null_array, A21);

        delete a22;
        a22 = new ParBilinearForm(fsp);
        // D1 (grad(dc1), grad(v1))
        a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // - <{D1 grad(dc1)}, [v1]> + sigma <[dc1], {D1 grad(v1)}> + kappa <{h^{-1} D1} [dc1], [v1]>
        a22->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa));
        a22->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_K_, sigma, kappa), Dirichlet_attr_);
        // (D1 z1 dc1 grad(phi^k), grad(v1))
        a22->AddDomainIntegrator(new GradConvectionIntegrator(*phi, &D_K_prod_v_K));
        // - <{D1 z1 dc1 grad(phi^k)}, [v1]>
        a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D1_z1, *phi));
        a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D1_z1, *phi), Dirichlet_attr_);
        // sigma <[dc1], {D1 z1 v1 grad(phi^k)}>
        a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D1_z1, *phi));
        a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D1_z1, *phi), Dirichlet_attr_);
        // kappa <h^{-1} [dc1], [v1]>
        a22->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff));
        a22->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff), Dirichlet_attr_);
        a22->Assemble(0);
        a22->Finalize(0);
        a22->SetOperatorType(Operator::PETSC_MATAIJ);
        a22->FormSystemMatrix(null_array, A22);

        delete a31;
        a31 = new ParBilinearForm(fsp);
        GridFunctionCoefficient c2_k_coeff(c2_k);
        ProductCoefficient D2_prod_z2_prod_c2_k(D_Cl_prod_v_Cl, c2_k_coeff);
        // D2 z2 c2^k (grad(dphi), grad(v2))
        a31->AddDomainIntegrator(new DiffusionIntegrator(D2_prod_z2_prod_c2_k));
        // - <{D2 z2 c2^k grad(dphi)}, [v2]> + sigma <[dphi], {D2 z2 c2^k grad(v2)}> + kappa <{h^{-1} D2 z2 c2^k} [dphi], [v2]>
        a31->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D2_prod_z2_prod_c2_k, sigma, kappa));
        a31->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D2_prod_z2_prod_c2_k, sigma, kappa), Dirichlet_attr_);
        a31->Assemble(0);
        a31->Finalize(0);
        a31->SetOperatorType(Operator::PETSC_MATAIJ);
        a31->FormSystemMatrix(null_array, A31);

        delete a33;
        a33 = new ParBilinearForm(fsp);
        // D2 (grad(dc2), grad(v2))
        a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // - <{D2 grad(dc2)}, [v2]> + sigma <[dc2], {D2 grad(v2)}> + kappa <{h^{-1} D2} [dc2], [v2]>
        a33->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa));
        a33->AddBdrFaceIntegrator(new DGDiffusionIntegrator(D_Cl_, sigma, kappa), Dirichlet_attr_);
        // (D2 z2 dc2 grad(phi^k), grad(v2))
        a33->AddDomainIntegrator(new GradConvectionIntegrator(*phi, &D_Cl_prod_v_Cl));
        // - <{D2 z2 dc2 grad(phi^k)}, [v2]>
        a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D2_z2, *phi));
        a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_1(neg_D2_z2, *phi), Dirichlet_attr_);
        // sigma <[dc2], {D2 z2 v2 grad(phi^k)}>
        a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D2_z2, *phi));
        a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_2(sigma_D2_z2, *phi), Dirichlet_attr_);
        // kappa <h^{-1} [dc2], [v2]>
        a33->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff));
        a33->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_3(kappa_coeff), Dirichlet_attr_);
        a33->Assemble(0);
        a33->Finalize(0);
        a33->SetOperatorType(Operator::PETSC_MATAIJ);
        a33->FormSystemMatrix(null_array, A33);

        jac_k = new BlockOperator(block_trueoffsets);
        jac_k->SetBlock(0, 0, &A11);
        jac_k->SetBlock(0, 1, &A12);
        jac_k->SetBlock(0, 2, &A13);
        jac_k->SetBlock(1, 0, &A21);
        jac_k->SetBlock(1, 1, &A22);
        jac_k->SetBlock(2, 0, &A31);
        jac_k->SetBlock(2, 2, &A33);

        return *jac_k;
    }
};
class PNP_DG_Newton_box_Solver_par
{
private:
    Mesh* mesh;
    ParMesh* pmesh;
    H1_FECollection* h1_fec;
    ParFiniteElementSpace* h1_space;
    DG_FECollection* dg_fec;
    ParFiniteElementSpace* dg_space;
    PNP_DG_Newton_Operator_par* op;
    PetscPreconditionerFactory *jac_factory;
    PetscNonlinearSolver* newton_solver;

    Array<int> block_trueoffsets, Dirichlet_attr;
    BlockVector* u_k;
    ParGridFunction phi, c1_k, c2_k;

    StopWatch chrono;
    SNES snes;
    map<string, Array<double>> out1;
    map<string, double> out2;
    Array<double> linear_iter;
    double linearize_iter, total_time, ndofs, linear_avg_iter;
    PetscInt *its=0, num_its=100;
    PetscReal *residual_norms=0;

public:
    PNP_DG_Newton_box_Solver_par(Mesh& mesh_): mesh(&mesh_)
    {
        int mesh_dim = mesh->Dimension(); //网格的维数:1D,2D,3D
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

        h1_fec = new H1_FECollection(p_order, mesh_dim);
        h1_space = new ParFiniteElementSpace(pmesh, h1_fec);

        dg_fec = new DG_FECollection(p_order, mesh_dim);
        dg_space = new ParFiniteElementSpace(pmesh, dg_fec);

        block_trueoffsets.SetSize(4);
        block_trueoffsets[0] = 0;
        block_trueoffsets[1] = dg_space->GetTrueVSize();
        block_trueoffsets[2] = dg_space->GetTrueVSize();
        block_trueoffsets[3] = dg_space->GetTrueVSize();
        block_trueoffsets.PartialSum();

        int bdr_size = pmesh->bdr_attributes.Max();
        {
            Dirichlet_attr.SetSize(bdr_size);
            Dirichlet_attr = 1;
        }

        op = new PNP_DG_Newton_Operator_par(dg_space);

        jac_factory = new PreconditionerFactory(*op, "Block Preconditioner");

        newton_solver = new PetscNonlinearSolver(dg_space->GetComm(), *op, "newton_");
        newton_solver->iterative_mode = true;
        newton_solver->SetMaxIter(max_newton);
        newton_solver->SetPreconditionerFactory(jac_factory);
        snes = SNES(*newton_solver);
        PetscMalloc(num_its * sizeof(PetscInt), &its);
        PetscMalloc(num_its * sizeof(PetscReal), &residual_norms);
        SNESSetConvergenceHistory(snes, residual_norms, its, num_its, PETSC_TRUE);
    }
    virtual ~PNP_DG_Newton_box_Solver_par()
    {
        delete newton_solver, op, jac_factory, u_k, mesh, pmesh;
        PetscFree(its);
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        cout.precision(14);
        cout << "\nNewton, DG" << p_order << ", box, parallel"
             << ", sigma: " << sigma << ", kappa: " << kappa
             << ", mesh: " << mesh_file << ", refine times: " << refine_times << endl;

        // 给定Newton迭代的初值，并使之满足边界条件
        u_k = new BlockVector(block_trueoffsets);
        if (zero_initial)
        {
            // MakeTRef(), SetTrueVector(), SetFromTrueVector() 三者要配套使用ffffffffff
            phi .MakeTRef(dg_space, *u_k, block_trueoffsets[0]);
            c1_k.MakeTRef(dg_space, *u_k, block_trueoffsets[1]);
            c2_k.MakeTRef(dg_space, *u_k, block_trueoffsets[2]);
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl; // 输出 nan
            phi = 0.0;
            c1_k = 0.0;
            c2_k = 0.0;
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl; // 输出 nan
            phi .SetTrueVector();
            c1_k.SetTrueVector();
            c2_k.SetTrueVector();
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl; // 输出 0.0
//        phi .SetFromTrueVector(); // 似乎可以不用
//        c1_k.SetFromTrueVector();
//        c2_k.SetFromTrueVector();
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl; // 输出 0.0
            // DG的GridFunction不能ProjectBdrCoefficient
//        phi .ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
//        c1_k.ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
//        c2_k.ProjectBdrCoefficient(c2_exact, Dirichlet_attr);
            {
                ParGridFunction phi_D_h1(h1_space), c1_D_h1(h1_space), c2_D_h1(h1_space);
                phi_D_h1 = 0.0; // 不能去掉，因为下面只是对boundary做投影，其他地方没有赋初值
                c1_D_h1  = 0.0;
                c2_D_h1  = 0.0;
                phi_D_h1.ProjectBdrCoefficient(phi_exact, Dirichlet_attr);
                c1_D_h1 .ProjectBdrCoefficient(c1_exact, Dirichlet_attr);
                c2_D_h1 .ProjectBdrCoefficient(c2_exact, Dirichlet_attr);
//            phi_D_h1.ProjectCoefficient(phi_exact); // for test code
//            c1_D_h1 .ProjectCoefficient(c1_exact );
//            c2_D_h1 .ProjectCoefficient(c2_exact );

                phi .ProjectGridFunction(phi_D_h1);
                c1_k.ProjectGridFunction(c1_D_h1);
                c2_k.ProjectGridFunction(c2_D_h1);
            }

//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
            phi .SetTrueVector(); // 必须用
            c1_k.SetTrueVector();
            c2_k.SetTrueVector();
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
//        phi .SetFromTrueVector(); // 似乎可以不用
//        c1_k.SetFromTrueVector();
//        c2_k.SetFromTrueVector();
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;
        }
        else
        {
            PNP_DG_Gummel_Solver_par initial_solver(*mesh);
            initial_solver.Solve(*u_k, block_trueoffsets, initTol);

            // 为了测试u_k是否正确被赋值
            phi .MakeTRef(dg_space, *u_k, block_trueoffsets[0]);
            c1_k.MakeTRef(dg_space, *u_k, block_trueoffsets[1]);
            c2_k.MakeTRef(dg_space, *u_k, block_trueoffsets[2]);
            phi .SetFromTrueVector();
            c1_k.SetFromTrueVector();
            c2_k.SetFromTrueVector();
//            cout << "l2 norm of phi: " <<  phi.Norml2() << endl;
//            cout << "l2 norm of  c1: " << c1_k.Norml2() << endl;
//            cout << "l2 norm of  c2: " << c2_k.Norml2() << endl;
        }
//        cout << "l2 norm of u_k: " << u_k->Norml2() << endl;

        {
//            phi .MakeTRef(dg_space, *u_k, block_trueoffsets[0]);
//            c1_k.MakeTRef(dg_space, *u_k, block_trueoffsets[1]);
//            c2_k.MakeTRef(dg_space, *u_k, block_trueoffsets[2]);
//            phi .SetFromTrueVector();
//            c1_k.SetFromTrueVector();
//            c2_k.SetFromTrueVector();
//
//            VisItDataCollection* dc = new VisItDataCollection("data collection", mesh);
//            dc->RegisterField("phi", &phi);
//            dc->RegisterField("c1",  &c1_k);
//            dc->RegisterField("c2",  &c2_k);
//            (phi)  /= alpha1;
//            (c1_k) /= alpha3;
//            (c2_k) /= alpha3;
//            Visualize(*dc, "phi", "phi_Newton_DG_init");
//            Visualize(*dc, "c1", "c1_Newton_DG_init");
//            Visualize(*dc, "c2", "c2_Newton_DG_init");
//            (phi)  *= (alpha1);
//            (c1_k) *= (alpha3);
//            (c2_k) *= (alpha3);
        }

        Vector zero_vec;
        zero_vec = 0.0;
        chrono.Start();
        newton_solver->Mult(zero_vec, *u_k); // u_k must be a true vector
        chrono.Stop();

        {
//            phi .MakeTRef(dg_space, *u_k, block_trueoffsets[0]);
//            c1_k.MakeTRef(dg_space, *u_k, block_trueoffsets[1]);
//            c2_k.MakeTRef(dg_space, *u_k, block_trueoffsets[2]);
//            phi .SetFromTrueVector();
//            c1_k.SetFromTrueVector();
//            c2_k.SetFromTrueVector();
//
//            VisItDataCollection* dc = new VisItDataCollection("data collection", mesh);
//            dc->RegisterField("phi", &phi);
//            dc->RegisterField("c1",  &c1_k);
//            dc->RegisterField("c2",  &c2_k);
//            (phi)  /= alpha1;
//            (c1_k) /= alpha3;
//            (c2_k) /= alpha3;
//            Visualize(*dc, "phi", "phi_Newton_DG_final");
//            Visualize(*dc, "c1", "c1_Newton_DG_final");
//            Visualize(*dc, "c2", "c2_Newton_DG_final");
//            (phi)  *= (alpha1);
//            (c1_k) *= (alpha3);
//            (c2_k) *= (alpha3);
        }

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
        linear_avg_iter = (linear_iter.Sum() / linear_iter.Size());
        out2["linear_avg_iter"] = linear_avg_iter;

        phi .MakeTRef(dg_space, *u_k, block_trueoffsets[0]);
        c1_k.MakeTRef(dg_space, *u_k, block_trueoffsets[1]);
        c2_k.MakeTRef(dg_space, *u_k, block_trueoffsets[2]);
        phi .SetFromTrueVector();
        c1_k.SetFromTrueVector();
        c2_k.SetFromTrueVector();

        cout.precision(14);
        double phiL2err = phi.ComputeL2Error(phi_exact);
        double c1L2err = c1_k.ComputeL2Error(c1_exact);
        double c2L2err = c2_k.ComputeL2Error(c2_exact);

        cout << "\n======>Box, " << Linearize << ", " << Discretize << p_order << ", refine " << refine_times << " for " << mesh_file << ", " << options_src << ", -rate: " << ComputeConvergenceRate << ", -zero: " << zero_initial << endl;
        cout << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
             << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
             << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;

        if (ComputeConvergenceRate)
        {
            phiL2errornorms_.Append(phiL2err);
            c1L2errornorms_.Append(c1L2err);
            c2L2errornorms_.Append(c2L2err);

            double totle_size = 0.0;
            for (int i=0; i<mesh->GetNE(); i++)
                totle_size += mesh->GetElementSize(0, 1);

            meshsizes_.Append(totle_size / mesh->GetNE());
        }

        if (local_conservation)
        {
            Vector error, error1, error2;
            ComputeLocalConservation(epsilon_water, phi, error);
            ComputeLocalConservation(D_K_, c1_k, v_K_coeff, phi, error1);
            ComputeLocalConservation(D_Cl_, c2_k, v_Cl_coeff, phi, error2);

            ofstream file("./phi_local_conservation_DG_Gummel_box.txt"),
                    file1("./c1_local_conservation_DG_Gummel_box.txt"),
                    file2("./c2_local_conservation_DG_Gummel_box.txt");
            if (file.is_open() && file1.is_open() && file2.is_open())
            {
                error.Print(file, 1);
                error1.Print(file1, 1);
                error2.Print(file2, 1);
            } else {
                MFEM_ABORT("local conservation quantities not save!");
            }
        }

        if (visualize)
        {
            VisItDataCollection* dc = new VisItDataCollection("data collection", mesh);
            dc->RegisterField("phi", &phi);
            dc->RegisterField("c1",  &c1_k);
            dc->RegisterField("c2",  &c2_k);

            (phi)  /= alpha1;
            (c1_k) /= alpha3;
            (c2_k) /= alpha3;
            Visualize(*dc, "phi", "phi_Newton_DG");
            Visualize(*dc, "c1", "c1_Newton_DG");
            Visualize(*dc, "c2", "c2_Newton_DG");
            ofstream results("phi_c1_c2_Newton_DG.vtk");
            results.precision(14);
            int ref = 0;
            mesh->PrintVTK(results, ref);
            phi.SaveVTK(results, "phi", ref);
            c1_k.SaveVTK(results, "c1", ref);
            c2_k.SaveVTK(results, "c2", ref);
            (phi)  *= (alpha1);
            (c1_k) *= (alpha3);
            (c2_k) *= (alpha3);
        }

        map<string, Array<double>>::iterator it1;
        for (it1=out1.begin(); it1!=out1.end(); ++it1)
            (*it1).second.Print(cout << (*it1).first << ": ", (*it1).second.Size());
        map<string, double>::iterator it2;
        for (it2=out2.begin(); it2!=out2.end(); ++it2)
            cout << (*it2).first << ": " << (*it2).second << endl;

        cout << "approximate mesh scale h: " << pow(dg_space->GetTrueVSize(), -1.0/3) << endl;
    }
};

#endif
