/* Modified from ex16p.cpp
 * ref: https://github.com/mfem/mfem/issues/1811
 * usage:
 *       Explicit: mpirun -np 1 example -visit -rs 0 -rp 0 -s 11 -dt 0.001 -o 1
 *       Implicit: mpirun -np 1 example -visit -rs 0 -rp 0 -s 3 -dt 0.001 -o 1
 *
 * Description:  This example solves a time dependent nonlinear heat equation
 *      problem of the form du/dt = C(u), with a non-linear diffusion
 *      operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
 * */
#include <fstream>
#include <iostream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator {
protected:
    ParFiniteElementSpace& fespace;
    Array<int> ess_tdof_list;  // this list remains empty for pure Neumann b.c.

    ParBilinearForm* M;
    ParBilinearForm* K;

    HypreParMatrix Mmat;
    HypreParMatrix Kmat;
    HypreParMatrix* T;  // T = M + dt K
    double current_dt;

    CGSolver M_solver;     // Krylov solver for inverting the mass matrix M
    HypreSmoother M_prec;  // Preconditioner for the mass matrix M

    CGSolver T_solver;     // Implicit solver for T = M + dt K
    HypreSmoother T_prec;  // Preconditioner for the implicit solver

    double alpha, kappa;

    mutable Vector z;  // auxiliary vector

public:
    ConductionOperator(ParFiniteElementSpace& f, double alpha, double kappa,
                       const Vector& u, const Array<int>& ess_tdof);

    virtual void Mult(const Vector& u, Vector& du_dt) const;
    /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector& u, Vector& k);

    /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
    void SetParameters(const Vector& u);

    virtual ~ConductionOperator();
};

int main(int argc, char* argv[]) {
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    int ser_ref_levels = 2;
    int par_ref_levels = 1;
    int order = 2;
    int ode_solver_type = 11;
    double t_final = 0.5;
    double dt = 1.0e-2;
    double alpha = 1.0e-2;
    double kappa = 0.5;
    bool visualization = true;
    bool visit = false;
    int vis_steps = 5;
    bool adios2 = false;

    int precision = 8;
    cout.precision(precision);

    OptionsParser args(argc, argv);
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Order (degree) of the finite elements.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                   "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                   "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
    args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step.");
    args.AddOption(&alpha, "-a", "--alpha", "Alpha coefficient.");
    args.AddOption(&kappa, "-k", "--kappa", "Kappa coefficient offset.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.l1lnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                   "--no-adios2-streams", "Save data using adios2 streams.");
    args.Parse();
    if (!args.Good()) {
        args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }

    if (myid == 0) {
        args.PrintOptions(cout);
    }

    // ================================================================= //
    // 3. Use inline-quad.mesh file for example.
    // Boundary elements are tagged as:
    // Bottom = 1
    // Right = 2
    // Top = 3
    // Left = 4
    const char* mesh_file = "../../../data/inline-quad.mesh";
    auto mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    // ================================================================= //

    // 4. Define the ODE solver used for time integration. Several implicit
    //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    //    explicit Runge-Kutta methods are available.
    ODESolver* ode_solver;
    switch (ode_solver_type) {
        // Implicit L-stable methods
        case 1:
            ode_solver = new BackwardEulerSolver;
            break;
        case 2:
            ode_solver = new SDIRK23Solver(2);
            break;
        case 3:
            ode_solver = new SDIRK33Solver;
            break;
            // Explicit methods
        case 11:
            ode_solver = new ForwardEulerSolver;
            break;
        case 12:
            ode_solver = new RK2Solver(0.5);
            break;  // midpoint method
        case 13:
            ode_solver = new RK3SSPSolver;
            break;
        case 14:
            ode_solver = new RK4Solver;
            break;
        case 15:
            ode_solver = new GeneralizedAlphaSolver(0.5);
            break;
            // Implicit A-stable methods (not L-stable)
        case 22:
            ode_solver = new ImplicitMidpointSolver;
            break;
        case 23:
            ode_solver = new SDIRK23Solver;
            break;
        case 24:
            ode_solver = new SDIRK34Solver;
            break;
        default:
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
            delete mesh;
            return 3;
    }

    // 5. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++) {
        mesh->UniformRefinement();
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++) {
        pmesh->UniformRefinement();
    }

    // 7. Define the vector finite element space representing the current and the
    //    initial temperature, u_ref.
    H1_FECollection fe_coll(order, dim);
    ParFiniteElementSpace fespace(pmesh, &fe_coll);

    int fe_size = fespace.GlobalTrueVSize();
    if (myid == 0) {
        cout << "Number of temperature unknowns: " << fe_size << endl;
    }

    ParGridFunction u_gf(&fespace);

    // ================================================================= //
    // 8. Set u initially to zero.
    ConstantCoefficient zero(0.0);
    u_gf.ProjectCoefficient(zero);
    Vector u;
    u_gf.GetTrueDofs(u);
    // ================================================================= //

    // ================================================================= //
    // 8.b Mark Essential true DOFs for bottom and top walls (will treat
    // as inhomogeneous Dirichlet). Fill boundary of u to be 10.0.
    mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    mfem::Array<int> ess_tdof_list;
    ess_bdr = 0;
    ess_bdr[1 - 1] = 1;  // bottom wall for inline-quad.mesh
    ess_bdr[3 - 1] = 1;  // top wall for inline-quad.mesh
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    ConstantCoefficient ten(10.0);
    u_gf.ProjectBdrCoefficient(ten, ess_bdr);
    u_gf.GetTrueDofs(u);

    // ================================================================= //

    // 9. Initialize the conduction operator and the VisIt visualization.
    ConductionOperator oper(fespace, alpha, kappa, u, ess_tdof_list);

    u_gf.SetFromTrueDofs(u);
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "ex16-mesh." << setfill('0') << setw(6) << myid;
        sol_name << "ex16-init." << setfill('0') << setw(6) << myid;
        ofstream omesh(mesh_name.str().c_str());
        omesh.precision(precision);
        pmesh->Print(omesh);
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        u_gf.Save(osol);
    }

    VisItDataCollection visit_dc("Example16-Parallel", pmesh);
    visit_dc.RegisterField("temperature", &u_gf);
    if (visit) {
        visit_dc.SetCycle(0);
        visit_dc.SetTime(0.0);
        visit_dc.Save();
    }

    // Optionally output a BP (binary pack) file using ADIOS2. This can be
    // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
    ADIOS2DataCollection* adios2_dc = NULL;
  if (adios2) {
    std::string postfix(mesh_file);
    postfix.erase(0, std::string("../data/").size());
    postfix += "_o" + std::to_string(order);
    postfix += "_solver" + std::to_string(ode_solver_type);
    const std::string collection_name = "ex16-p-" + postfix + ".bp";

    adios2_dc =
        new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
    adios2_dc->SetParameter("SubStreams", std::to_string(num_procs / 2));
    adios2_dc->RegisterField("temperature", &u_gf);
    adios2_dc->SetCycle(0);
    adios2_dc->SetTime(0.0);
    adios2_dc->Save();
  }
#endif

    socketstream sout;
    if (visualization) {
        char vishost[] = "localhost";
        int visport = 19916;
        sout.open(vishost, visport);
        sout << "parallel " << num_procs << " " << myid << endl;
        int good = sout.good(), all_good;
        MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
        if (!all_good) {
            sout.close();
            visualization = false;
            if (myid == 0) {
                cout << "Unable to connect to GLVis server at " << vishost << ':'
                     << visport << endl;
                cout << "GLVis visualization disabled.\n";
            }
        } else {
            sout.precision(precision);
            sout << "solution\n" << *pmesh << u_gf;
            sout << "pause\n";
            sout << flush;
            if (myid == 0) {
                cout << "GLVis visualization paused."
                     << " Press space (in the GLVis window) to resume it.\n";
            }
        }
    }

    // 10. Perform time-integration (looping over the time iterations, ti, with a
    //     time-step dt).
    ode_solver->Init(oper);
    double t = 0.0;

    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
        if (t + dt >= t_final - dt / 2) {
            last_step = true;
        }

        ode_solver->Step(u, t, dt);

        if (last_step || (ti % vis_steps) == 0) {
            if (myid == 0) {
                cout << "step " << ti << ", t = " << t << endl;
            }

            u_gf.SetFromTrueDofs(u);
            if (visualization) {
                sout << "parallel " << num_procs << " " << myid << "\n";
                sout << "solution\n" << *pmesh << u_gf << flush;
            }

            if (visit) {
                visit_dc.SetCycle(ti);
                visit_dc.SetTime(t);
                visit_dc.Save();
            }

#ifdef MFEM_USE_ADIOS2
            if (adios2) {
        adios2_dc->SetCycle(ti);
        adios2_dc->SetTime(t);
        adios2_dc->Save();
      }
#endif
        }
        oper.SetParameters(u);
    }

    double norm = u_gf.ComputeL2Error(zero);
    if (myid == 0) {
        cout << "L2 norm of u: " << norm << endl;
    }

#ifdef MFEM_USE_ADIOS2
    if (adios2) {
    delete adios2_dc;
  }
#endif

    // 11. Save the final solution in parallel. This output can be viewed later
    //     using GLVis: "glvis -np <np> -m ex16-mesh -g ex16-final".
    {
        ostringstream sol_name;
        sol_name << "ex16-final." << setfill('0') << setw(6) << myid;
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        u_gf.Save(osol);
    }

    // 12. Free the used memory.
    delete ode_solver;
    delete pmesh;

    MPI_Finalize();

    return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace& f, double al,
                                       double kap, const Vector& u,
                                       const Array<int>& ess_tdof)
        : TimeDependentOperator(f.GetTrueVSize(), 0.0),
          fespace(f),
          ess_tdof_list(ess_tdof),
          M(NULL),
          K(NULL),
          T(NULL),
          current_dt(0.0),
          M_solver(f.GetComm()),
          T_solver(f.GetComm()),
          z(height) {
    const double rel_tol = 1e-8;

    M = new ParBilinearForm(&fespace);
    M->AddDomainIntegrator(new MassIntegrator());
    M->Assemble(0);  // keep sparsity pattern of M and K the same
    M->FormSystemMatrix(ess_tdof_list, Mmat);

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(rel_tol);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(Mmat);

    alpha = al;
    kappa = kap;

    T_solver.iterative_mode = false;
    T_solver.SetRelTol(rel_tol);
    T_solver.SetAbsTol(0.0);
    T_solver.SetMaxIter(100);
    T_solver.SetPrintLevel(0);
    T_solver.SetPreconditioner(T_prec);

    SetParameters(u);
}

// Inhomogeneous Dirichlet on bottom/top walls, inhomogeneous Neumann on
// left/right wall.
// *** Serial and parallel produce very different results ***
void ConductionOperator::Mult(const Vector& u, Vector& du_dt) const {
    // Compute:
    //    du_dt = M^{-1}*-K(u)
    // for du_dt
    ParGridFunction tmp_u(&fespace), tmp_du_dt(&fespace);
    ParLinearForm tmp_z(&fespace);
    tmp_u.SetFromTrueDofs(u);
    tmp_du_dt.SetFromTrueDofs(du_dt);
    K->Mult(tmp_u, tmp_z);  // <----- Segfaults in Mult routine
    tmp_z.Neg();            // z = -z

    // OperatorHandle K_op;
    // K->FormSystemMatrix(ess_tdof_list, K_op);
    // K_op->Mult(tmp_u, tmp_z);  // <--- Works in serial, mismatch in
    //                            // matrix/vector sizes when parallel
    // tmp_z.Neg();               // z = -z

    // Construct and set inhomogeneous Neumann condition
    ParLinearForm neumann(&fespace);
    ConstantCoefficient ten(10.0);
    Array<int> neumann_bdr(fespace.GetParMesh()->bdr_attributes.Max());
    neumann_bdr = 0;
    neumann_bdr[1] = 1;  // Right wall for inline-quad.mesh
    neumann_bdr[3] = 1;  // Left wal for inline-quad.mesh
    neumann.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(ten),
                                  neumann_bdr);
    neumann.Assemble();

    // Add inhomogeneous Neumann to RHS
    tmp_z.Add(1.0, neumann);

    OperatorHandle A;
    Vector X, B;
    K->FormLinearSystem(ess_tdof_list, tmp_u, tmp_z, A, X, B);

    M_solver.Mult(B, du_dt);
    M->RecoverFEMSolution(du_dt, tmp_z, tmp_du_dt);
    tmp_du_dt.GetTrueDofs(du_dt);
    du_dt.SetSubVector(ess_tdof_list, 0.0);
}

// Inhomogeneous Dirichlet on bottom/top walls, homogeneous Neumann on
// left/right wall.
// *** Serial and parallel produce very different results ***
void ConductionOperator::ImplicitSolve(const double dt, const Vector& u,
                                       Vector& du_dt) {
    // Solve the equation:
    //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
    // for du_dt
    if (!T) {
        K->Finalize();
        Kmat = *K->ParallelAssemble();
        T = Add(1.0, Mmat, dt, Kmat);
        current_dt = dt;
        T_solver.SetOperator(*T);
    }
    MFEM_VERIFY(dt == current_dt, "");  // SDIRK methods use the same dt

    ParGridFunction tmp_u(&fespace), tmp_du_dt(&fespace);
    ParLinearForm tmp_z(&fespace);

    tmp_u.SetFromTrueDofs(u);
    tmp_du_dt.SetFromTrueDofs(du_dt);

    K->Mult(tmp_u, tmp_z);
    tmp_z.Neg();

    // Construct and set inhomogeneous Neumann condition
    ParLinearForm neumann(&fespace);
    ConstantCoefficient ten(10.0);
    Array<int> neumann_bdr(fespace.GetParMesh()->bdr_attributes.Max());
    neumann_bdr = 0;
    neumann_bdr[1] = 1;  // Right wall for inline-quad.mesh
    neumann_bdr[3] = 1;  // Left wal for inline-quad.mesh
    neumann.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(ten),
                                  neumann_bdr);
    neumann.Assemble();

    // Add inhomogeneous Neumann to RHS
    tmp_z.Add(1.0, neumann);

    OperatorHandle A;
    Vector X, B;
    K->FormLinearSystem(ess_tdof_list, tmp_u, tmp_z, A, X, B);

    T_solver.Mult(B, du_dt);
    du_dt.SetSubVector(ess_tdof_list, 0.0);
}

void ConductionOperator::SetParameters(const Vector& u) {
    ParGridFunction u_alpha_gf(&fespace);
    u_alpha_gf.SetFromTrueDofs(u);
    for (int i = 0; i < u_alpha_gf.Size(); i++) {
        u_alpha_gf(i) = kappa + alpha * u_alpha_gf(i);
    }

    delete K;
    K = new ParBilinearForm(&fespace);

    GridFunctionCoefficient u_coeff(&u_alpha_gf);

    K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
    K->Assemble(0);  // keep sparsity pattern of M and K the same
//    K->FormSystemMatrix(ess_tdof_list, Kmat);
    delete T;
    T = NULL;  // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator() {
    delete T;
    delete M;
    delete K;
}