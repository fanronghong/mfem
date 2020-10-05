/// ref: https://github.com/mfem/mfem/issues/142
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include "mfem.hpp"

using namespace mfem;

double input_function(const Vector &x, const double t) {
    if (x(0) == 1.0) {
        return cos(2.0 * 3.14 * t);
    } else {
        return 0.0;
    }
}

class TDHeatOperator : public TimeDependentOperator {
public:
    TDHeatOperator(Mesh &mesh, int order);

    virtual ~TDHeatOperator();

    virtual void Mult(const Vector &x, Vector &y) const;

    virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

    void UpdateEqs();

    void PrintSolverStats();

    void Recover();

    GridFunction &GetTempGF() { return *u_; };

private:
    // Basis function order
    int order_;

    Mesh *mesh_;

    Array<int> ess_tdof_list_;

    // Finite element collection
    H1_FECollection *h1_fecoll_;

    // Continuous space for u
    FiniteElementSpace *h1_fespace_;

    // Laplacian operator
    BilinearForm *div_grad_; // 生成刚度矩阵 K_: (grad(u), grad(v))

    // H1 Mass operator
    BilinearForm *h1_mass_; // 生成刚度矩阵 M_: (u, v)

    FunctionCoefficient *input_coef_;

    // RHS
    LinearForm *rhs_; // (0.0, v)

    // Temperature
    GridFunction *u_;

    SparseMatrix M_;
    SparseMatrix K_;
    SparseMatrix *T_;  // T = M + dt K

    Vector b_;
    Vector X_;

    // Krylov solver for inverting the mass matrix M
    CGSolver Minv_;

    // Preconditioner for the mass matrix M
    GSSmoother M_prec_;

    // Implicit solver for T = M + dt K
    CGSolver Tinv_;

    // Preconditioner for the implicit solver
    GSSmoother T_prec_;

    mutable Vector z_;
};

TDHeatOperator::TDHeatOperator(Mesh &mesh, int order)
        : TimeDependentOperator(),
          order_(order),
          mesh_(&mesh),
          h1_fespace_(nullptr),
          u_(nullptr)
{
    h1_fecoll_ = new H1_FECollection(order_, mesh_->Dimension());
    h1_fespace_ = new FiniteElementSpace(mesh_, h1_fecoll_);

    this->SetTime(0.0);
    this->height = h1_fespace_->GetTrueVSize();
    this->width = this->height;
    z_.SetSize(this->height);

    // Define Dirichlet boundary conditions. Left side (attribute 0) is Neumann
    // Right side (attribute 1) is Dirichlet
    Array<int> ess_bdr(mesh_->bdr_attributes.Max());
    ess_bdr[0] = 0;
    ess_bdr[1] = 1;
    h1_fespace_->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);

    input_coef_ = new FunctionCoefficient(input_function);

    // Setup mass matrix
    h1_mass_ = new BilinearForm(h1_fespace_);
    h1_mass_->AddDomainIntegrator(new MassIntegrator());
    h1_mass_->Assemble();
    h1_mass_->FormSystemMatrix(ess_tdof_list_, M_);

    // Setup laplace operator
    div_grad_ = new BilinearForm(h1_fespace_);
    div_grad_->AddDomainIntegrator(new DiffusionIntegrator());
    div_grad_->Assemble();

    // Setup rhs
    ConstantCoefficient zero_coef(0.0);
    rhs_ = new LinearForm(h1_fespace_);
    rhs_->AddDomainIntegrator(new DomainLFIntegrator(zero_coef));
    rhs_->Assemble();

    u_ = new GridFunction(h1_fespace_);
    *u_ = 0.0;

    Minv_.SetPrintLevel(0);
    Minv_.iterative_mode = false;
    Minv_.SetMaxIter(200);
    Minv_.SetRelTol(1e-12);
    Minv_.SetAbsTol(0.0);
    Minv_.SetPreconditioner(M_prec_);
    Minv_.SetOperator(M_);

    Tinv_.SetPrintLevel(0);
    Tinv_.iterative_mode = false;
    Tinv_.SetMaxIter(200);
    Tinv_.SetRelTol(1e-12);
    Tinv_.SetAbsTol(0.0);
    Tinv_.SetPreconditioner(T_prec_);

    T_ = nullptr;

    UpdateEqs();
}

TDHeatOperator::~TDHeatOperator() {
    delete h1_fespace_;
    delete h1_fecoll_;
    delete h1_mass_;
    delete div_grad_;
    delete rhs_;
    delete input_coef_;
    delete u_;
    if (T_) {
        delete T_;
    }
}

void TDHeatOperator::UpdateEqs() {
    Array<int> ess_bdr(mesh_->bdr_attributes.Max());
    ess_bdr[0] = 0;
    ess_bdr[1] = 1;

    input_coef_->SetTime(this->GetTime());
    u_->ProjectBdrCoefficient(*input_coef_, ess_bdr);

    std::cout << "Forming linear system with size: " << K_.Height() << std::endl;
//    b_.SetSize(u_->Size());
    div_grad_->FormLinearSystem(ess_tdof_list_, *u_, *rhs_, K_, X_, b_, true);
}

void TDHeatOperator::PrintSolverStats() {
    std::cout << "Iterations: " << Tinv_.GetNumIterations() << std::endl;
    std::cout << "Final Norm: " << Tinv_.GetFinalNorm() << std::endl;
}

void TDHeatOperator::Recover() { div_grad_->RecoverFEMSolution(X_, b_, *u_); }

void TDHeatOperator::Mult(const Vector &u, Vector &du_dt) const {
    // du_dt = M^{-1} (-K x + b)
    K_.Mult(u, z_);
    z_.Neg();
    z_ += b_;
    Minv_.Mult(z_, du_dt);
}

void TDHeatOperator::ImplicitSolve(const double dt, const Vector &u,
                                   Vector &du_dt) {
    // Solve the equation for du_dt
    // T = M + dt K
    if (!T_) {
        T_ = Add(1.0, M_, dt, K_);
        Tinv_.SetOperator(*T_);
    }
    K_.Mult(u, z_);
    z_.Neg();
    z_ += b_;
    Tinv_.Mult(z_, du_dt);
}

int main(int argc, char *argv[]) {
    bool visualization = true;
    int order = 1;
    int num_domainp = 20;
    double dt = 1e-2;
    double t_end = 2.0;

    // Create 1D mesh
    auto mesh = new Mesh(num_domainp, 1.0);

    // Create H1 finite element space
    auto fec = new H1_FECollection(order, mesh->Dimension());
    auto fes = new FiniteElementSpace(mesh, fec);
    fes->GetEssentialTrueDofs()

    TDHeatOperator tdheat_op(*mesh, 1);
    GridFunction u = tdheat_op.GetTempGF();

    socketstream sol_sock;
    if (visualization) {
        char vishost[] = "localhost";
        int visport = 19916;
        sol_sock.open(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << u;
        sol_sock << "pause\n";
        sol_sock << std::flush;
    }

    ODESolver *ode_solver = new BackwardEulerSolver;
    // ODESolver *ode_solver = new SDIRK33Solver;
    // ODESolver *ode_solver = new RK4Solver;
    ode_solver->Init(tdheat_op);
    double t = 0.0;
    int ti = 0;

    while (t <= t_end)
    {
        std::cout << "Cycle: " << ti << " Current time: " << tdheat_op.GetTime()
                  << std::endl;
        ode_solver->Step(u, t, dt);
        tdheat_op.PrintSolverStats();
        tdheat_op.Recover();
        if (visualization) {
            sol_sock << "solution\n" << *mesh << u << std::flush;
        }
        tdheat_op.UpdateEqs();
        std::cout << std::endl;
    }

    delete ode_solver;
    delete mesh;
    delete fec;
}