/*
 * 参考文献
 * [1] a monotone finite element scheme for convection-diffusion equations
 *
 *  -\nabla\cdot(\alpha \nabla u + \beta u) = f, in \Omega,
 *                                        u = u_D, on \partial\Omega
 */
#include <string>
#include <iostream>

#include "mfem.hpp"
#include "adv_diff_3D.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/SUPG_Integrator.hpp"
#include "../utils/EAFE_ModifyStiffnessMatrix.hpp"

using namespace std;
using namespace mfem;


void EAFE_advec_diffu(Mesh& mesh, Array<double>& L2norms, Array<double>& meshsizes)
{
    int dim = mesh.Dimension();
    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    if (mesh.bdr_attributes.Size()) {
        ess_bdr = 1;
        h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm lf(&h1_space);
    lf.AddDomainIntegrator(new DomainLFIntegrator(analytic_rhs_));
    lf.Assemble();

    BilinearForm blf(&h1_space);
    //EAFE基本的刚度矩阵只需要Poisson方程,后面修改的就是这个Poisson方程的刚度矩阵,参考[1],(3.24)式
    blf.AddDomainIntegrator(new DiffusionIntegrator); // (grad(u), grad(v))
    blf.Assemble(0);
    blf.Finalize(0);

    GridFunction uh(&h1_space);
    uh.ProjectCoefficient(analytic_solution_);//使得uh满足边界条件,必须

    SparseMatrix& A = blf.SpMat();
    Vector &b=lf;

    EAFE_Modify(mesh, A, DiffusionTensor, AdvectionVector);

    blf.EliminateVDofs(ess_tdof_list, uh, lf);

    if (EAFE_Only_Dump_data)
    {
        cout << "number of mesh nodes: \n" << mesh.GetNV() << endl;

        GridFunction u_exact(&h1_space);
        u_exact.ProjectCoefficient(analytic_solution_);
//        WriteVector("u_exact_eafe.txt", u_exact);
//        WriteCSR("A_eafe.txt", A);
//        WriteVector("b_eafe.txt", b);
        BinaryWriteVector("u_exact_eafe.bin", u_exact);
        BinaryWriteCSR("A_eafe.bin", A);
        BinaryWriteVector("b_eafe.bin", b);
        return;
    }

    GMRESSolver solver;
    solver.SetOperator(A);
    solver.SetAbsTol(gmres_atol);
    solver.SetRelTol(gmres_rtol);
    solver.SetPrintLevel(gmres_printlevel);
    solver.SetMaxIter(gmres_maxiter);
    Vector x(lf.Size());
    solver.Mult(b, x);
    if (!solver.GetConverged()) throw "GMRES solver not converged!";

    uh = x;
    {
        VisItDataCollection uh_dc("uh for EAFE", &mesh);
        uh_dc.RegisterField("value", &uh);
        Visualize(uh_dc, "value");
        Wx += offx;
    }
    double l2norm = uh.ComputeL2Error(analytic_solution_);
    L2norms.Append(l2norm);

    double totle_size = 0.0;
    for (int i=0; i<mesh.GetNE(); i++) {
        totle_size += mesh.GetElementSize(0, 1);
    }
    meshsizes.Append(totle_size / mesh.GetNE());
}


void SUPG_advec_diffu(Mesh& mesh, Array<double>& L2norms, Array<double>& meshsizes)
{
    int dim = mesh.Dimension();
    FiniteElementCollection *fec = new H1_FECollection(p_order, dim);
    FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);

    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    if (mesh.bdr_attributes.Size())
    {
        ess_bdr = 1; //标记所有的边界都为essential boundary
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm *lf = new LinearForm(fespace);
    lf->AddDomainIntegrator(new DomainLFIntegrator(analytic_rhs_)); // (f, v)
    lf->AddDomainIntegrator(new SUPG_LinearFormIntegrator(diffusion_tensor, advection_vector, one, analytic_rhs_, mesh));
    lf->Assemble();

    BilinearForm *blf = new BilinearForm(fespace);
    blf->AddDomainIntegrator(new DiffusionIntegrator(diffusion_tensor)); // (alpha grad(u), grad(v))
    blf->AddDomainIntegrator(new ConvectionIntegrator(advection_vector, -1.0)); // -(beta \cdot grad(u), v)
    blf->AddDomainIntegrator(new MassIntegrator(neg_div_advection_)); // (-div(beta) u, v)
    blf->AddDomainIntegrator(new SUPG_BilinearFormIntegrator(&diffusion_tensor, neg, advection_vector, neg, div_adv, mesh));
    blf->Assemble(0);
//    {
//        SparseMatrix temp(blf->SpMat());
//        temp.Print(cout << std::setprecision(3) << "3D SUPG stiffness matrix (before apply bdc)\n");
//        PrintSparsePattern(temp, "3D SUPG stiffness matrix");
//    }

    GridFunction uh(fespace);
//    uh.ProjectCoefficient(analytic_solution_);
    uh.ProjectBdrCoefficient(analytic_solution_, ess_bdr); //两种加边界条件的方式

    blf->EliminateEssentialBC(ess_bdr, uh, *lf);
    blf->Finalize(1);
    SparseMatrix &A = blf->SpMat();

    if (SUPG_Only_Dump_data)
    {
        cout << "number of mesh nodes: \n" << mesh.GetNV() << endl;

        GridFunction u_exact(fespace);
        u_exact.ProjectCoefficient(analytic_solution_);
        BinaryWriteVector("u_exact_supg.bin", u_exact);
        BinaryWriteCSR("A_supg.bin", A);
        BinaryWriteVector("b_supg.bin", *lf);
        return;
    }

    GMRESSolver solver;
    solver.SetOperator(A);
    solver.SetAbsTol(gmres_atol);
    solver.SetRelTol(gmres_rtol);
    solver.SetPrintLevel(gmres_printlevel);
    solver.SetMaxIter(gmres_maxiter);
    Vector x(lf->Size());
    solver.Mult(*lf, x);
    if (!solver.GetConverged()) MFEM_ABORT("GMRES solver not converged!");

    uh = x;
    double l2norm = uh.ComputeL2Error(analytic_solution_);
    L2norms.Append(l2norm);

    double totle_size = 0.0;
    for (int i=0; i<mesh.GetNE(); i++) {
        totle_size += mesh.GetElementSize(0, 1);
    }
    meshsizes.Append(totle_size / mesh.GetNE());

    // 11. Free the used memory.
    delete blf;
    delete lf;
    delete fespace;
    delete fec;
}


void advec_diffu(Mesh& mesh, Array<double>& L2norms, Array<double>& meshsizes)
{
    int dim = mesh.Dimension();

    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> ess_tdof_list;
    if (mesh.bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;
        h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    LinearForm lf(&h1_space);
    lf.AddDomainIntegrator(new DomainLFIntegrator(analytic_rhs_)); // (f, v)
    lf.Assemble();

    BilinearForm blf(&h1_space);
    blf.AddDomainIntegrator(new DiffusionIntegrator(diffusion_tensor)); // (alpha * grad(u), grad(v))
    blf.AddDomainIntegrator(new ConvectionIntegrator(advection_vector, -1.0)); // -(beta * grad(u), v)
    blf.AddDomainIntegrator(new MassIntegrator(neg_div_advection_)); // (-div(beta)u, v). 扩散速度不是divergence free的
    blf.Assemble();

    GridFunction uh(&h1_space);
    uh.ProjectCoefficient(analytic_solution_);

    SparseMatrix A;
    Vector x, b;
    blf.FormLinearSystem(ess_tdof_list, uh, lf, A, x, b);

    GMRESSolver solver;
    solver.SetOperator(A);
    solver.SetAbsTol(gmres_atol);
    solver.SetRelTol(gmres_rtol);
    solver.SetPrintLevel(gmres_printlevel);
    solver.SetMaxIter(gmres_maxiter);
    solver.Mult(b, x);
    if (!solver.GetConverged()) throw("GMRES solver not converged!");

    uh = x;
    {
        VisItDataCollection uh_dc("uh for FEM", &mesh);
        uh_dc.RegisterField("value", &uh);
        Visualize(uh_dc, "value");
        Wx += offx;
    }
    double l2norm = uh.ComputeL2Error(analytic_solution_);
    L2norms.Append(l2norm);

    double totle_size = 0.0;
    for (int i=0; i<mesh.GetNE(); i++) {
        totle_size += mesh.GetElementSize(0, 1);
    }
    meshsizes.Append(totle_size / mesh.GetNE());
}


int main(int args, char **argv)
{
    if (Run_SUPG)
    {
        Mesh mesh(mesh_file, 1, 1);

        Array<double> L2norms;
        Array<double> meshsizes;
        for (int i=0; i<refine_times; i++)
        {
            mesh.UniformRefinement();
            SUPG_advec_diffu(mesh, L2norms, meshsizes);
        }
//        SUPG_advec_diffu(mesh, L2norms, meshsizes);

        Array<double> rates = compute_convergence(L2norms, meshsizes);
        rates.Print(cout << "SUPG convergence rate: \n", rates.Size());
    }

    if (Run_EAFE)
    {
        Mesh mesh(mesh_file, 1, 1);

        Array<double> L2norms;
        Array<double> meshsizes;
        for (int i=0; i<refine_times; i++)
        {
            mesh.UniformRefinement();
            EAFE_advec_diffu(mesh, L2norms, meshsizes);
        }
//        EAFE_advec_diffu(mesh, L2norms, meshsizes);

        Array<double> rates = compute_convergence(L2norms, meshsizes);
        rates.Print(cout << "EAFE convergence rate: \n", rates.Size());
    }

    if (Run_FEM)
    {
        Mesh mesh(mesh_file, 1, 1);

        Array<double> L2norms;
        Array<double> meshsizes;
        for (int i=0; i<refine_times; i++)
        {
            mesh.UniformRefinement();
            advec_diffu(mesh, L2norms, meshsizes);
        }
//        advec_diffu(mesh, L2norms, meshsizes);

        Array<double> rates = compute_convergence(L2norms, meshsizes);
        rates.Print(cout << "FEM convergence rate: \n", rates.Size());
    }

    return 0;
}