/* - \nabla\cdot(diff \nabla u + adv u) = f
 *                                    u = 0
 *  \Omega = [0, 1] \times [0, 1]
 *  u_exact = cos(pi*x/2) * cos(pi*y/2) * sin(pi*x/2) * sin(pi*y/2)
 *  f =
 * */
#include <iostream>
#include "mfem.hpp"
#include "../utils/GradConvection_Integrator.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
using namespace std;
using namespace mfem;

const double diff = 100.0;
void Adv(const Vector& x, Vector& y)
{
    y[0] = 1.0;
    y[1] = 1.0;
}
double u_exact(const Vector& x)
{
    return cos(M_PI * x[0] / 2) * cos(M_PI * x[1] / 2) * sin(M_PI * x[0] / 2) * sin(M_PI * x[1] / 2);
}
double f_exact(const Vector& x)
{
    return 1.570796326795*pow(sin(1.570796326795*x[0]), 2)*sin(1.570796326795*x[1])*cos(1.570796326795*x[1]) + 1.570796326795*sin(1.570796326795*x[0])*pow(sin(1.570796326795*x[1]), 2)*cos(1.570796326795*x[0]) + 1973.92088021813*sin(1.570796326795*x[0])*sin(1.570796326795*x[1])*cos(1.570796326795*x[0])*cos(1.570796326795*x[1]) - 1.570796326795*sin(1.570796326795*x[0])*cos(1.570796326795*x[0])*pow(cos(1.570796326795*x[1]), 2) - 1.570796326795*sin(1.570796326795*x[1])*pow(cos(1.570796326795*x[0]), 2)*cos(1.570796326795*x[1]);
}

int main(int argc, char *argv[])
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MFEMInitializePetsc(NULL, NULL, "./adv_diff_petsc.opts", NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int p_order = 1;
    Array<int> ess_tdof_list;
    FunctionCoefficient u_e(u_exact);

    Mesh mesh(10, 10, Element::TRIANGLE, true, 1.0, 1.0);
    int dim = mesh.Dimension();
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
    for (int i=0; i<0; ++i) pmesh->UniformRefinement();

    DG_FECollection dg_fec(p_order, dim);
    ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, &dg_fec);

    ParBilinearForm* blf = new ParBilinearForm(fes);
    ConstantCoefficient one(1.0);
    ConstantCoefficient neg(-1.0);
    ConstantCoefficient diff_coeff(diff);
    ProductCoefficient neg_diff(neg, diff_coeff);
    VectorFunctionCoefficient adv_coeff(dim, Adv);
    // diff (grad(u), grad(v))
    blf->AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
    // (adv u, grad(v))
    blf->AddDomainIntegrator(new GradConvectionIntegrator(&adv_coeff, &one));
    // <{diff grad(u)}, [v]>
    blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_6(&neg_diff));
    blf->AddBdrFaceIntegrator(new DGSelfTraceIntegrator_6(&neg_diff));
    // <{adv u}, [v]>
    blf->AddInteriorFaceIntegrator(new DGSelfTraceIntegrator_8(adv_coeff));
    blf->Assemble();

    ParLinearForm* lf = new ParLinearForm(fes);
    FunctionCoefficient f_coeff(f_exact);
    // (f, v)
    lf->AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
    lf->Assemble();

    PetscParMatrix *A = new PetscParMatrix();
    PetscParVector *x = new PetscParVector(fes);
    PetscParVector *b = new PetscParVector(fes);
    ParGridFunction* u = new ParGridFunction(fes);
    blf->SetOperatorType(Operator::PETSC_MATAIJ);
    blf->FormLinearSystem(ess_tdof_list, *u, *lf, *A, *x, *b);

    PetscLinearSolver* solver = new PetscLinearSolver(*A, "advdiff_");
    solver->Mult(*b, *x);
    blf->RecoverFEMSolution(*x, *lf, *u);

    cout << "L2 errornorm of |u_e - u_h|: " << u->ComputeL2Error(u_e) << endl;

    MFEMFinalizePetsc();
    MPI_Finalize();
}