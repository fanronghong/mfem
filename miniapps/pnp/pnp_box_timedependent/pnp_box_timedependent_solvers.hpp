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


/* Poisson Equation:
 *     div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
 * NP Equation:
 *     dc_i / dt = div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * */
class PNP_Box_Gummel_CG_TimeDependent: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* h1;

    mutable ParBilinearForm *a0, *a1, *a2, *b1, *b2, *m0;
    HypreParMatrix *A0, *A1, *A2, *M0;
    Vector *X0, *B0, *X1, *B1, *X2, *B2;
    PetscLinearSolver *A_solver, *M1_solver, *M2_solver;

    int vsize;
    Array<int> &offsets, &ess_tdof_list, &ess_bdr;
    int num_procs, myid;

public:
    PNP_Box_Gummel_CG_TimeDependent(Array<int>& offset, Array<int>& ess_bdr_, Array<int>& ess_list,
                                    ParFiniteElementSpace* fsp, double time)
        : TimeDependentOperator(offset.Last(), time), vsize(fsp->GetVSize()),
          offsets(offset), ess_bdr(ess_bdr_), ess_tdof_list(ess_list), h1(fsp),
          a0(NULL), a1(NULL), a2(NULL), b1(NULL), b2(NULL), m0(NULL),
          A0(NULL), A1(NULL), A2(NULL), B1(NULL), B2(NULL), M0(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        A0 = new HypreParMatrix;
        M0 = new HypreParMatrix;
        X0 = new Vector;
        X1 = new Vector;
        X2 = new Vector;
        B0 = new Vector;
        B1 = new Vector;
        B2 = new Vector;

    }
    virtual ~PNP_Box_Gummel_CG_TimeDependent()
    {
        delete a0;
        delete a1;
        delete a2;
        delete b1;
        delete b2;
        delete m0;

        delete A0;
        delete M0;
        delete X0;
        delete X1;
        delete X2;
        delete B0;
        delete B1;
        delete B2;
    }

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const {}

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        dphic1c2_dt = 0.0;

        // 下面就可以通过修改 phic1c2_ 从而达到修改 phic1c2 的目的
        Vector* phic1c2_ = (Vector*) &phic1c2;
        ParGridFunction phi_gf, c1_gf, c2_gf;
        phi_gf.MakeRef(h1, *phic1c2_, offsets[0]);
        c1_gf .MakeRef(h1, *phic1c2_, offsets[1]);
        c2_gf .MakeRef(h1, *phic1c2_, offsets[2]);

        // 开始Gummel线性化(即Gummel迭代), Gummel迭代的初值为0
        ParGridFunction phi_Gummel_gf(h1), c1_Gummel_gf(h1), c2_Gummel_gf(h1),
                        phi_Gummel_gf_old(h1), c1_Gummel_gf_old(h1), c2_Gummel_gf_old(h1);
        phi_Gummel_gf = 0.0;
        c1_Gummel_gf  = 0.0;
        c2_Gummel_gf  = 0.0;
        phi_exact.SetTime(t);
        c1_exact .SetTime(t);
        c2_exact .SetTime(t);
        phi_Gummel_gf.ProjectBdrCoefficient(phi_exact, ess_bdr); // 需要满足下一时刻的边界条件
        c1_Gummel_gf .ProjectBdrCoefficient( c1_exact, ess_bdr);
        c2_Gummel_gf .ProjectBdrCoefficient( c2_exact, ess_bdr);
        phi_Gummel_gf_old = phi_Gummel_gf;
        c1_Gummel_gf_old  = c1_Gummel_gf;
        c2_Gummel_gf_old  = c2_Gummel_gf;

        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // ----------------------- 求解Poisson -----------------------
            ParLinearForm l0(h1);
            // alpha2 alpha3 (z1 c1 + z2 c2, psi)
            GridFunctionCoefficient c1_Gummel_coeff(&c1_Gummel_gf_old);
            GridFunctionCoefficient c2_Gummel_coeff(&c2_Gummel_gf_old);
            ProductCoefficient alpha2_alpha3_z1_c1(alpha2_prod_alpha3_prod_v_K,  c1_Gummel_coeff);
            ProductCoefficient alpha2_alpha3_z2_c2(alpha2_prod_alpha3_prod_v_Cl, c2_Gummel_coeff);
            l0.AddDomainIntegrator(new DomainLFIntegrator(alpha2_alpha3_z1_c1));
            l0.AddDomainIntegrator(new DomainLFIntegrator(alpha2_alpha3_z2_c2));
            // (f, psi)
            f0_analytic.SetTime(t);
            l0.AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
            l0.Assemble();

            this->builda0();
            a0->FormLinearSystem(ess_tdof_list, phi_Gummel_gf, l0, *A0, *X0, *B0);
            PetscLinearSolver poisson(*A0, false, "phi_");
            poisson.Mult(*B0, *X0);
            a0->RecoverFEMSolution(*X0, l0, phi_Gummel_gf); // 得到下一个Gummel迭代步的解
            // 使用松弛方法
            phi_Gummel_gf *= (1 - relax);
            phi_Gummel_gf.Add(relax, phi_Gummel_gf_old);

            double L2err1 = phi_Gummel_gf_old.ComputeL2Error(zero);
            double L2err2 = phi_Gummel_gf.ComputeL2Error(zero);
            double diff = abs(L2err1 - L2err2) / L2err2;
            if (diff < Gummel_rel_tol) { // Gummel迭代停止
                last_gummel_step = true;
            }
            if (myid == 0 && verbose >= 2) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << diff << endl;
            }
            phi_Gummel_gf_old = phi_Gummel_gf;

            // ----------------------- 求解NP1 -----------------------
            ParLinearForm l1(h1);
            // dt (f1, v1)
            f1_analytic.SetTime(t);
            l1.AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            l1.Assemble();
            l1 *= dt;

            // l1 = (f1, v1) + (c1^n, v1), c1^n为上一个时间步的解
            this->buildM0();
            m0->AddMult(c1_gf, l1); // c1_gf表示上一个时间步的解
            // l1 = (f1, v1) + (c1^n, v1) + dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1))
            this->bulidA1(D_K_, phi_Gummel_gf, v_K_coeff, t_stepsize);
            a1->AddMult(c1_Gummel_gf, l1);

            this->buildM0();
            m0->FormLinearSystem(ess_tdof_list, c1_Gummel_gf, l1, *M0, *X1, *B1);
            PetscLinearSolver np1(*M0, false, "np1_");
            if (abs(t - 0.06) < 10E-8 && gummel_step == 29) {
                int fff=1;
                while(fff == 1) {}
            }
            np1.Mult(*B1, *X1);
            m0->RecoverFEMSolution(*X1, l1, c1_Gummel_gf);
            // 使用松弛方法
            c1_Gummel_gf *= (1 - relax);
            c1_Gummel_gf.Add(relax, c1_Gummel_gf_old);
            c1_Gummel_gf_old = c1_Gummel_gf;


            // ----------------------- 求解NP2 -----------------------
            ParLinearForm l2(h1);
            // dt (f2, v2)
            f2_analytic.SetTime(t);
            l2.AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
            l2.Assemble();
            l2 *= dt;

            // l2 = (f2, v2) + (c2^n, v2), c2^n为上一个时间步的解
            this->buildM0();
            m0->AddMult(c2_gf, l2); // c2_gf表示上一个时间步的解
            // l2 = (f2, v2) + (c2^n, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2))
            this->bulidA2(D_Cl_, phi_Gummel_gf, v_Cl_coeff, t_stepsize);
            a2->AddMult(c2_Gummel_gf, l2);

            this->buildM0();
            m0->FormLinearSystem(ess_tdof_list, c2_Gummel_gf, l2, *M0, *X2, *B2);
            PetscLinearSolver np2(*M0, false, "np2_");
            np2.Mult(*B2, *X2);
            m0->RecoverFEMSolution(*X2, l2, c2_Gummel_gf);
            // 使用松弛方法
            c2_Gummel_gf *= (1 - relax);
            c2_Gummel_gf.Add(relax, c2_Gummel_gf_old);
            c2_Gummel_gf_old = c2_Gummel_gf;
        }

        phi_gf = phi_Gummel_gf; // 更新下一个时间步的解
        c1_gf = c1_Gummel_gf;
        c2_gf = c2_Gummel_gf;
        {
            phi_exact.SetTime(t);
            c1_exact.SetTime(t);
            c2_exact.SetTime(t);
            double phiL2errornorm = phi_gf.ComputeL2Error(phi_exact);
            double  c1L2errornorm =  c1_gf.ComputeL2Error(c1_exact);
            double  c2L2errornorm =  c2_gf.ComputeL2Error(c2_exact);
            if (myid == 0 && verbose >= 1) {
                cout << "current time: " << t << '\n'
                     << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << '\n' << endl;
            }
        }
    }

    void builda0() const
    {
        // epsilon_s (grad(phi), grad(psi))
        if (a0 != NULL) { delete a0; }
        a0 = new ParBilinearForm(h1);
        a0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a0->Assemble(skip_zero_entries);
    }
    void bulidA1(Coefficient& D1, ParGridFunction& phi, Coefficient& z1, double dt) const
    {
        // dt D1 ( grad(c1) + z1 c1 grad(phi), grad(v1) )
        ProductCoefficient dt_D1(dt, D1);
        ProductCoefficient dt_D1_z1(dt_D1, z1);

        if (a1 != NULL) { delete a1; }
        a1 = new ParBilinearForm(h1);
        a1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        a1->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D1_z1));
        a1->Assemble(skip_zero_entries);
    }
    void bulidA2(Coefficient& D2, ParGridFunction& phi, Coefficient& z2, double dt) const
    {
        // dt D2 ( grad(c2) + z2 c2 grad(phi), grad(v2) )
        ProductCoefficient dt_D2(dt, D2);
        ProductCoefficient dt_D2_z2(dt_D2, z2);

        if (a2 != NULL) { delete a2; }
        a2 = new ParBilinearForm(h1);
        a2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        a2->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D2_z2));
        a2->Assemble(skip_zero_entries);
    }
    void buildb1(Coefficient& alpha2_alpha3_z1) const
    {
        // alpha2_alpha3_z1 (c1, psi)
        if (b1 != NULL) { delete b1; }
        b1 = new ParBilinearForm(h1);
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_alpha3_z1));
        b1->Assemble(skip_zero_entries);
    }
    void buildb2(Coefficient& alpha2_alpha3_z2) const
    {
        // alpha2_alpha3_z2 (c2, psi)
        if (b2 != NULL) { delete b2; }
        b2 = new ParBilinearForm(h1);
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_alpha3_z2));
        b2->Assemble(skip_zero_entries);
    }
    void buildM0() const
    {
        if (m0 != NULL) { delete m0; }
        m0 = new ParBilinearForm(h1);
        m0->AddDomainIntegrator(new MassIntegrator);
        m0->Assemble(skip_zero_entries);
    }
};
class PNP_Box_Gummel_CG_TimeDependent_Solver
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    int vsize; // 有限元空间维数
    double t; // 当前时间

    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    PNP_Box_Gummel_CG_TimeDependent* oper;
    ODESolver *ode_solver;

    Array<int> offsets, ess_bdr, ess_tdof_list;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_Solver(Mesh& mesh_, int ode_solver_type): mesh(mesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的
        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        vsize = h1->GetVSize();
        offsets.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        offsets[0] = 0;
        offsets[1] = vsize;
        offsets[2] = vsize * 2;
        offsets[3] = vsize * 3;

        phic1c2 = new BlockVector(offsets);
        *phic1c2 = 0.0;

        phi_gf  = new ParGridFunction(h1);
        c1_gf   = new ParGridFunction(h1);
        c2_gf   = new ParGridFunction(h1);
        phi_gf->MakeRef(h1, *phic1c2, offsets[0]);
        c1_gf ->MakeRef(h1, *phic1c2, offsets[1]);
        c2_gf ->MakeRef(h1, *phic1c2, offsets[2]);

        // 设定初值
        {
            t = t_init;

            phi_exact.SetTime(t);
            phi_gf->ProjectCoefficient(phi_exact);

            c1_exact.SetTime(t);
            c1_gf->ProjectCoefficient(c1_exact);

            c2_exact.SetTime(t);
            c2_gf->ProjectCoefficient(c2_exact);
        }

        {
            double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
            double  c1L2errornorm =  c1_gf->ComputeL2Error(c1_exact);
            double  c2L2errornorm =  c2_gf->ComputeL2Error(c2_exact);
            if (myid == 0) {
                cout << "After setting initial conditions, t = " << t << '\n'
                     << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
            }
        }

        oper = new PNP_Box_Gummel_CG_TimeDependent(offsets, ess_bdr, ess_tdof_list, h1, t);

        switch (ode_solver_type)
        {
            // Implicit L-stable methods
            case 1:  ode_solver = new BackwardEulerSolver; break;
                // Explicit methods
            case 11: ode_solver = new ForwardEulerSolver; break;
            default:
            MFEM_ABORT("Not support ODE solver.");
        }

        oper->SetTime(t);
        ode_solver->Init(*oper);

        if (paraview)
        {
            pd = new ParaViewDataCollection("PNP_CG_Gummel_Time_Dependent", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);
        }
    }
    ~PNP_Box_Gummel_CG_TimeDependent_Solver()
    {
        delete pmesh;
        delete fec;
        delete h1;

        delete phic1c2;
        delete phi_gf;
        delete c1_gf;
        delete c2_gf;

        delete oper;
        delete ode_solver;
        if (paraview) delete pd;
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        if (myid == 0) {
            cout << '\n';
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times << '\n'
                 << "petsc options: " << options_src << '\n'
                 << ((ode_type == 1) ? ("Backward Euler") : (ode_type == 11 ? "Forward Euler" \
                                                                       : "wrong type")) << ", " << "time step: " << t_stepsize
                 << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();

        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(t_stepsize, t_final - t);

            ode_solver->Step(*phic1c2, t, dt_real); // 进过这一步之后phic1c2和t都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);
//            int fff=1;
//            while (fff==1) {}

            if (paraview)
            {
                pd->SetCycle(ti); // 第 i 个时间步
                pd->SetTime(t); // 第i个时间步所表示的时间
                pd->Save();
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Stop();

        {
            // 计算误差范数只能是在所有进程上都运行，输出误差范数可以只在root进程
            double phiL2err = phi_gf->ComputeL2Error(phi_exact);
            double c1L2err = c1_gf->ComputeL2Error(c1_exact);
            double c2L2err = c2_gf->ComputeL2Error(c2_exact);

            if (myid == 0 && verbose >= 1) {
                cout << "ODE solver taking " << chrono.RealTime() << " s." << endl;
                cout.precision(14);
                cout << "At final time: " << t << '\n'
                     << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
                     << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
                     << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;
            }

            if (ComputeConvergenceRate)
            {
                double totle_size = 0.0;
                for (int i = 0; i < mesh.GetNE(); i++)
                    totle_size += mesh.GetElementSize(0, 1);
                meshsizes_.Append(totle_size / mesh.GetNE());

                phiL2errornorms_.Append(phiL2err);
                c1L2errornorms_.Append(c1L2err);
                c2L2errornorms_.Append(c2L2err);
            }
        }
    }
};


class PNP_Box_Gummel_CG_TimeDependent1: public TimeDependentOperator
{
private:
    ParFiniteElementSpace* h1;

    mutable ParBilinearForm *a0, *a1, *a2, *b1, *b2, *m, *m1_dta1, *m2_dta2;
    mutable HypreParMatrix *A0, *M, *M1_dtA1, *M2_dtA2;

    Vector *temp_x0, *temp_b0, *temp_x1, *temp_b1, *temp_x2, *temp_b2;
    int true_size;
    mutable Array<int> true_offset, ess_bdr, ess_tdof_list;
    int num_procs, myid;

public:
    PNP_Box_Gummel_CG_TimeDependent1(int truesize, Array<int>& offset, Array<int>& ess_bdr_,
                                    ParFiniteElementSpace* fsp, double time)
            : TimeDependentOperator(3*truesize, time), true_size(truesize), true_offset(offset),
              ess_bdr(ess_bdr_), h1(fsp),
              b1(NULL), b2(NULL), a0(NULL), a1(NULL), m(NULL), a2(NULL), m1_dta1(NULL), m2_dta2(NULL)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        A0 = new HypreParMatrix;
        M = new HypreParMatrix;
        M1_dtA1 = new HypreParMatrix;
        M2_dtA2 = new HypreParMatrix;

        temp_x0 = new Vector;
        temp_b0 = new Vector;
        temp_x1 = new Vector;
        temp_b1 = new Vector;
        temp_x2 = new Vector;
        temp_b2 = new Vector;

    }
    virtual ~PNP_Box_Gummel_CG_TimeDependent1()
    {
        delete m;
        delete m1_dta1;
        delete m2_dta2;
        delete a0;
        delete a1;
        delete a2;
        delete b1;
        delete b2;

        delete temp_x0;
        delete temp_b0;
        delete temp_x1;
        delete temp_b1;
        delete temp_x2;
        delete temp_b2;
    }

    // (ci, vi), i=1,2
    void buildm() const
    {
        if (m != NULL) { delete m; }
        
        m = new ParBilinearForm(h1);
        m->AddDomainIntegrator(new MassIntegrator);
        
        m->Assemble(skip_zero_entries);
    }
    
    // (c1, v1) + dt D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given dt and phi
    void buildm1_dta1(double dt, ParGridFunction& phi) const
    {
        if (m1_dta1 != NULL) { delete m1_dta1; }

        ProductCoefficient dt_D1(dt, D_K_);
        ProductCoefficient dt_D1_z1(dt_D1, v_K_coeff);

        m1_dta1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1_dta1->AddDomainIntegrator(new MassIntegrator);
        // dt D1 (grad(c1), grad(v1))
        m1_dta1->AddDomainIntegrator(new DiffusionIntegrator(dt_D1));
        // dt D1 z1 (c1 grad(phi), grad(v1))
        m1_dta1->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D1_z1));

        m1_dta1->Assemble(skip_zero_entries);
    }

    // (c2, v2) + dt D2 (grad(c2) + z2 c2 grad(phi), grad(v2)), given dt and phi
    void buildm2_dta2(double dt, ParGridFunction& phi) const
    {
        if (m2_dta2 != NULL) { delete m2_dta2; }

        ProductCoefficient dt_D2(dt, D_Cl_);
        ProductCoefficient dt_D2_z2(dt_D2, v_Cl_coeff);

        m2_dta2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2_dta2->AddDomainIntegrator(new MassIntegrator);
        // dt D2 (grad(c2), grad(v2))
        m2_dta2->AddDomainIntegrator(new DiffusionIntegrator(dt_D2));
        // dt D2 z2 (c2 grad(phi), grad(v2))
        m2_dta2->AddDomainIntegrator(new GradConvectionIntegrator(phi, &dt_D2_z2));

        m2_dta2->Assemble(skip_zero_entries);
    }

    // epsilon_s (grad(phi), grad(psi))
    void builda0() const
    {
        if (a0 != NULL) { delete a0; }

        a0 = new ParBilinearForm(h1);
        a0->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));

        a0->Assemble(skip_zero_entries);
    }

    // D1 (grad(c1) + z1 c1 grad(phi), grad(v1)), given phi
    void builda1(ParGridFunction& phi) const
    {
        if (a1 != NULL) { delete a1; }

        a1 = new ParBilinearForm(h1);
        // D1 (grad(c1), grad(v1))
        a1->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
        // D1 z1 (c1 grad(phi), grad(v1))
        a1->AddDomainIntegrator(new GradConvectionIntegrator(phi, &D_K_prod_v_K));

        a1->Assemble(skip_zero_entries);
    }

    // D2 ( grad(c2) + z2 c2 grad(phi), grad(v2) )
    void builda2(ParGridFunction& phi) const
    {
        if (a2 != NULL) { delete a2; }

        a2 = new ParBilinearForm(h1);
        // D2 (grad(c2), grad(v2))
        a2->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
        // D2 z2 (c2 grad(phi), grad(v2))
        a2->AddDomainIntegrator(new GradConvectionIntegrator(phi, &D_Cl_prod_v_Cl));

        a2->Assemble(skip_zero_entries);
    }

    // alpha2 alpha3 z1 (c1, psi)
    void buildb1() const
    {
        if (b1 != NULL) { delete b1; }

        b1 = new ParBilinearForm(h1);
        // alpha2 alpha3 z1 (c1, psi)
        b1->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));

        b1->Assemble(skip_zero_entries);
        b1->Finalize(skip_zero_entries);
    }

    // alpha2 alpha3 z2 (c2, psi)
    void buildb2() const
    {
        if (b2 != NULL) { delete b2; }

        b2 = new ParBilinearForm(h1);
        // alpha2 alpha3 z2 (c2, psi)
        b2->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));

        b2->Assemble(skip_zero_entries);
        b2->Finalize(skip_zero_entries);
    }

    virtual void Mult(const Vector &phic1c2, Vector &dphic1c2_dt) const
    {
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(h1, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(h1, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(h1, *phic1c2_ptr, true_offset[2]);
        
        ParGridFunction dc1dt, dc2dt;
        dc1dt.MakeTRef(h1, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(h1, dphic1c2_dt, true_offset[2]);
        
        ParGridFunction temp_phi(h1); // 保留初始的phi, 后面要用(old_phi在Poisson solver中会被改变)
        temp_phi = old_phi;
        
        // 求解 Poisson
        ParLinearForm *l0 = new ParLinearForm(h1);
        // (f0, psi)
        f0_analytic.SetTime(t);
        l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
        l0->Assemble();

        buildb1();
        b1->AddMult(old_c1, *l0, 1.0); // l0 = l0 + b1 c1
        buildb2();
        b2->AddMult(old_c2, *l0, 1.0); // l0 = (l0 + b1 c1) + b2 c2

        builda0();
        phi_exact.SetTime(t);
        old_phi.ProjectBdrCoefficient(phi_exact, ess_bdr);
        a0->FormLinearSystem(ess_tdof_list, old_phi, *l0, *A0, *temp_x0, *temp_b0);

        PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi_");
        poisson_solver->Mult(*temp_b0, *temp_x0);
        a0->RecoverFEMSolution(*temp_x0, *l0, old_phi); // 更新old_phi
        delete l0;
        delete poisson_solver;

        // 求解 NP1
        ParLinearForm *l1 = new ParLinearForm(h1);
        f1_analytic.SetTime(t);
        l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
        l1->Assemble();
        
        builda1(temp_phi);
        a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1
        
        buildm();
        dc1dt_exact.SetTime(t);
        dc1dt.ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
        m->FormLinearSystem(ess_tdof_list, dc1dt, *l1, *M, *temp_x1, *temp_b1);

        PetscLinearSolver* np1_solver = new PetscLinearSolver(*M, false, "np1_");
        np1_solver->Mult(*temp_b1, *temp_x1);
        m->RecoverFEMSolution(*temp_x1, *l1, dc1dt); // 更新 dc1dt
        delete l1;
        delete np1_solver;

        // 求解 NP2
        ParLinearForm *l2 = new ParLinearForm(h1);
        f2_analytic.SetTime(t);
        l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
        l2->Assemble();

        builda2(temp_phi);
        a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2

        buildm();
        dc2dt_exact.SetTime(t);
        dc2dt.ProjectBdrCoefficient(dc2dt_exact, ess_bdr);
        m->FormLinearSystem(ess_tdof_list, dc2dt, *l2, *M, *temp_x2, *temp_b2);

        PetscLinearSolver* np2_solver = new PetscLinearSolver(*M, false, "np2_");
        np2_solver->Mult(*temp_b2, *temp_x2);
        m->RecoverFEMSolution(*temp_x2, *l2, dc2dt); // 更新 dc2dt
        delete l2;
        delete np2_solver;
    }

    virtual void ImplicitSolve(const double dt, const Vector &phic1c2, Vector &dphic1c2_dt)
    {
        dphic1c2_dt = 0.0;

        Vector* phic1c2_ptr = (Vector*) &phic1c2;
        ParGridFunction old_phi, old_c1, old_c2;
        // 后面更新 old_phi 的同时也会更新 phic1c2_ptr, 从而更新 phic1c2
        old_phi.MakeTRef(h1, *phic1c2_ptr, true_offset[0]);
        old_c1 .MakeTRef(h1, *phic1c2_ptr, true_offset[1]);
        old_c2 .MakeTRef(h1, *phic1c2_ptr, true_offset[2]);
        old_phi.SetFromTrueVector();

        ParGridFunction dc1dt, dc2dt;
        dc1dt.MakeTRef(h1, dphic1c2_dt, true_offset[1]);
        dc2dt.MakeTRef(h1, dphic1c2_dt, true_offset[2]);

        ParGridFunction phi_Gummel(h1), dc1dt_Gummel(h1), dc2dt_Gummel(h1);
        phi_Gummel = 0.0; dc1dt_Gummel = 0.0; dc2dt_Gummel = 0.0;
        phi_exact.SetTime(t);
        dc1dt_exact.SetTime(t);
        dc2dt_exact.SetTime(t);
        old_phi.ProjectBdrCoefficient(phi_exact, ess_bdr);
        phi_Gummel.ProjectBdrCoefficient(phi_exact, ess_bdr);
        dc1dt.ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
        dc1dt_Gummel.ProjectBdrCoefficient(dc1dt_exact, ess_bdr);
        dc2dt.ProjectBdrCoefficient(dc2dt_exact, ess_bdr);
        dc2dt_Gummel.ProjectBdrCoefficient(dc2dt_exact, ess_bdr);

        Vector diff(true_size);
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // 求解 Poisson
            ParLinearForm *l0 = new ParLinearForm(h1);
            // (f0, psi)
            f0_analytic.SetTime(t);
            l0->AddDomainIntegrator(new DomainLFIntegrator(f0_analytic));
            l0->Assemble();

            buildb1();
            buildb2();
            b1->AddMult(old_c1, *l0, 1.0);    // l0 = l0 + b1 c1
            b1->AddMult(dc1dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + dt b1 dc1dt
            b2->AddMult(old_c2, *l0, 1.0);    // l0 = l0 + b1 c1 + dt b1 dc1dt + b2 c2
            b2->AddMult(dc2dt_Gummel, *l0, dt);  // l0 = l0 + b1 c1 + dt b1 dc1dt + b2 c2 + dt b2 dc2dt

            builda0();
            a0->FormLinearSystem(ess_tdof_list, phi_Gummel, *l0, *A0, *temp_x0, *temp_b0);

            PetscLinearSolver* poisson_solver = new PetscLinearSolver(*A0, false, "phi_");
            poisson_solver->Mult(*temp_b0, *temp_x0);
            a0->RecoverFEMSolution(*temp_x0, *l0, phi_Gummel); // 更新old_phi
            delete l0;
            delete poisson_solver;

            diff = 0.0;
            diff += phi_Gummel;
            diff -= old_phi;
            double tol = diff.Norml2() / phi_Gummel.Norml2();
            old_phi = phi_Gummel; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
            if (myid == 0) {
                cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
            }
            if (tol < Gummel_rel_tol) { // Gummel迭代停止
                last_gummel_step = true;
            }

            // 求解 NP1
            ParLinearForm *l1 = new ParLinearForm(h1);
            f1_analytic.SetTime(t);
            l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
            l1->Assemble();

            builda1(phi_Gummel);
            a1->AddMult(old_c1, *l1, -1.0); // l1 = l1 - a1 c1

            buildm1_dta1(dt, phi_Gummel);
            m1_dta1->FormLinearSystem(ess_tdof_list, dc1dt_Gummel, *l1, *M1_dtA1, *temp_x1, *temp_b1);

            PetscLinearSolver* np1_solver = new PetscLinearSolver(*M1_dtA1, false, "np1_");
            np1_solver->Mult(*temp_b1, *temp_x1);
            m1_dta1->RecoverFEMSolution(*temp_x1, *l1, dc1dt_Gummel); // 更新 dc1dt
            delete l1;
            delete np1_solver;

            // 求解 NP1
            ParLinearForm *l2 = new ParLinearForm(h1);
            f2_analytic.SetTime(t);
            l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
            l2->Assemble();

            builda2(phi_Gummel);
            a2->AddMult(old_c2, *l2, -1.0); // l2 = l2 - a2 c2

            buildm2_dta2(dt, phi_Gummel);
            m2_dta2->FormLinearSystem(ess_tdof_list, dc2dt_Gummel, *l2, *M2_dtA2, *temp_x2, *temp_b2);

            PetscLinearSolver* np2_solver = new PetscLinearSolver(*M2_dtA2, false, "np2_");
            np2_solver->Mult(*temp_b2, *temp_x2);
            m2_dta2->RecoverFEMSolution(*temp_x2, *l2, dc2dt_Gummel); // 更新 dc2dt
            delete l2;
            delete np2_solver;
        }
    }
};
class PNP_Box_Gummel_CG_TimeDependent_Solver1
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    PNP_Box_Gummel_CG_TimeDependent1* oper;
    double t; // 当前时间
    Vector init_value;
    ODESolver *ode_solver;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_Solver1(Mesh& mesh_, int ode_solver_type): mesh(mesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        t = t_init;

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的

        phi_gf = new ParGridFunction(h1); *phi_gf = 0.0;
        c1_gf  = new ParGridFunction(h1); *c1_gf  = 0.0;
        c2_gf  = new ParGridFunction(h1); *c2_gf  = 0.0;

        true_size = h1->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_size;
        true_offset[2] = true_size * 2;
        true_offset[3] = true_size * 3;

        phic1c2 = new BlockVector(true_offset); *phic1c2 = 0.0;
        phi_gf->MakeTRef(h1, *phic1c2, true_offset[0]);
        c1_gf ->MakeTRef(h1, *phic1c2, true_offset[1]);
        c2_gf ->MakeTRef(h1, *phic1c2, true_offset[2]);

        // 设定初值
        phi_exact.SetTime(t);
        phi_gf->ProjectCoefficient(phi_exact);
        phi_gf->SetTrueVector();
        phi_gf->SetFromTrueVector();

        c1_exact.SetTime(t);
        c1_gf->ProjectCoefficient(c1_exact);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_exact.SetTime(t);
        c2_gf->ProjectCoefficient(c2_exact);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        oper = new PNP_Box_Gummel_CG_TimeDependent1(true_size, true_offset, ess_bdr, h1, t);

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
            pd = new ParaViewDataCollection("PNP_CG_Gummel_Time_Dependent", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);
        }
    }
    ~PNP_Box_Gummel_CG_TimeDependent_Solver1()
    {
        delete pmesh;
        delete fec;
        delete h1;
        delete phi_gf;
        delete c1_gf;
        delete c2_gf;
        delete phic1c2;
        delete oper;
        delete ode_solver;
        if (paraview) delete pd;
    }

    void Solve(Array<double>& phiL2errornorms_, Array<double>& c1L2errornorms_,
               Array<double>& c2L2errornorms_, Array<double>& meshsizes_)
    {
        if (myid == 0) {
            cout << '\n';
            cout << Discretize << p_order << ", " << Linearize << ", " << mesh_file << ", refine times: " << refine_times << '\n'
                 << ", " << options_src << '\n'
                 << ((ode_type == 1) ? ("backward Euler") : (ode_type == 11 ? "forward Euler" \
                                                                       : "wrong type")) << ", " << "time step: " << t_stepsize
                 << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Clear();
        chrono.Start();

        bool last_step = false;
        for (int ti=1; !last_step; ti++)
        {
            double dt_real = min(t_stepsize, t_final - t);

            ode_solver->Step(*phic1c2, t, dt_real); // 进过这一步之后phic1c2和t都被更新了

            last_step = (t >= t_final - 1e-8*t_stepsize);

            phi_gf->SetFromTrueVector();
            c1_gf->SetFromTrueVector();
            c2_gf->SetFromTrueVector();

            if (paraview)
            {
                pd->SetCycle(ti); // 第 i 个时间步
                pd->SetTime(t); // 第i个时间步所表示的时间
                pd->Save();
            }
            if (verbose)
            {
                phi_exact.SetTime(t);
                c1_exact.SetTime(t);
                c2_exact.SetTime(t);

                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm = c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm = c2_gf->ComputeL2Error(c2_exact);
                if (myid == 0) {
                    cout << "\nTime: " << t << '\n'
                         << "phi L2 errornorm: " << phiL2errornorm << '\n'
                         << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                         << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        chrono.Stop();

        {
            phi_exact.SetTime(t);
            phi_gf->SetFromTrueVector();

            c1_exact.SetTime(t);
            c1_gf->SetFromTrueVector();

            c2_exact.SetTime(t);
            c2_gf->SetFromTrueVector();

            // 计算误差范数只能是在所有进程上都运行，输出误差范数可以只在root进程
            double phiL2err = phi_gf->ComputeL2Error(phi_exact);
            double c1L2err = c1_gf->ComputeL2Error(c1_exact);
            double c2L2err = c2_gf->ComputeL2Error(c2_exact);

            if (myid == 0) {
                cout << "ODE solver taking " << chrono.RealTime() << " s." << endl;
                cout.precision(14);
                cout << "At final time: " << t << '\n'
                     << "L2 errornorm of |phi_h - phi_e|: " << phiL2err << ", \n"
                     << "L2 errornorm of | c1_h - c1_e |: " << c1L2err << ", \n"
                     << "L2 errornorm of | c2_h - c2_e |: " << c2L2err << endl;
            }

            if (ComputeConvergenceRate)
            {
                double totle_size = 0.0;
                for (int i = 0; i < mesh.GetNE(); i++)
                    totle_size += mesh.GetElementSize(0, 1);
                meshsizes_.Append(totle_size / mesh.GetNE());

                phiL2errornorms_.Append(phiL2err);
                c1L2errornorms_.Append(c1L2err);
                c2L2errornorms_.Append(c2L2err);
            }
        }
    }
};






/* Poisson Equation:
 *     div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
 * NP Equation:
 *     dc_i / dt = div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
 * 使用Gummel线性化方法: 给定 phi^n, c1^n, c2^n, 求 phi^n+1, c1^n+1, c2^n+1
 *     A phi^n+1 = B1 c1^n + B2 c2^n + b
 *     M1 c1^n+1 = M1 c1^n + dt A1 c1^n + dt b1
 *     M2 c2^n+1 = M2 c2^n + dt A2 c2^n + dt b2
 * */
class PNP_Box_Gummel_CG_TimeDependent_ForwardEuler
{
private:
    Mesh& mesh;
    ParMesh* pmesh;
    H1_FECollection* fec;
    ParFiniteElementSpace* h1;

    ParBilinearForm *a11, *a12, *a13, *m1, *m2;
    HypreParMatrix *A, *B1, *B2, *M1, *M2, *M1_no_bc, *M2_no_bc;
    BlockVector* phic1c2;
    ParGridFunction *phi_gf, *c1_gf, *c2_gf;

    double current_t;
    mutable Vector z;
    mutable HypreParVector *b, *b1, *b2;
    mutable HypreParMatrix *A1, *A2;
    PetscLinearSolver *A_solver, *M1_solver, *M2_solver;

    int true_size; // 有限元空间维数
    Array<int> true_offset, ess_bdr, ess_tdof_list;
    socketstream vis_ostream;
    ParaViewDataCollection* pd;
    int num_procs, myid;
    StopWatch chrono;

public:
    PNP_Box_Gummel_CG_TimeDependent_ForwardEuler(Mesh& mesh_): mesh(mesh_)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);

        pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
        fec   = new H1_FECollection(p_order, mesh.Dimension());
        h1    = new ParFiniteElementSpace(pmesh, fec);

        ess_bdr.SetSize(mesh.bdr_attributes.Max());
        ess_bdr = 1; // 设置所有边界都是essential的
        h1->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

        a11 = new ParBilinearForm(h1);
        // (epsilon_s grad(phi), grad(psi))
        a11->AddDomainIntegrator(new DiffusionIntegrator(epsilon_water));
        a11->Assemble(skip_zero_entries);
        a11->Finalize(skip_zero_entries);
        // 这里暂时不形成最终的刚度矩阵, 后面在形成的时候同时设定essential bc

        a12 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z1 c1, psi)
        a12->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_K));
        a12->Assemble(skip_zero_entries);
        a12->Finalize(skip_zero_entries);
        B1 = a12->ParallelAssemble();

        a13 = new ParBilinearForm(h1);
        // (alpha2 alpha3 z2 c2, psi)
        a13->AddDomainIntegrator(new MassIntegrator(alpha2_prod_alpha3_prod_v_Cl));
        a13->Assemble(skip_zero_entries);
        a13->Finalize(skip_zero_entries);
        B2 = a13->ParallelAssemble();

        m1 = new ParBilinearForm(h1);
        // (c1, v1)
        m1->AddDomainIntegrator(new MassIntegrator);
        m1->Assemble(skip_zero_entries); // keep sparsity pattern of A1 and M1 the same
        m1->Finalize(skip_zero_entries);
        M1_no_bc = m1->ParallelAssemble();

        m2 = new ParBilinearForm(h1);
        // (c2, v2)
        m2->AddDomainIntegrator(new MassIntegrator);
        m2->Assemble(skip_zero_entries); // keep sparsity pattern of A2 and M2 the same
        m2->Finalize(skip_zero_entries);
        M2_no_bc = m2->ParallelAssemble();

        A  = new HypreParMatrix();
        M1 = new HypreParMatrix();
        M2 = new HypreParMatrix();

        phi_gf = new ParGridFunction(h1); *phi_gf = 0.0;
        c1_gf  = new ParGridFunction(h1); *c1_gf  = 0.0;
        c2_gf  = new ParGridFunction(h1); *c2_gf  = 0.0;

        true_size = h1->TrueVSize();
        true_offset.SetSize(3 + 1); // 表示 phi, c1，c2的TrueVector
        true_offset[0] = 0;
        true_offset[1] = true_size;
        true_offset[2] = true_size * 2;
        true_offset[3] = true_size * 3;

        phic1c2  = new BlockVector(true_offset); *phic1c2 = 0.0;
        phi_gf->MakeTRef(h1, *phic1c2, true_offset[0]);
        c1_gf ->MakeTRef(h1, *phic1c2, true_offset[1]);
        c2_gf ->MakeTRef(h1, *phic1c2, true_offset[2]);

        // 设定初始条件，同时满足了边界条件
        current_t = t_init;
        phi_exact.SetTime(current_t);
        phi_gf->ProjectCoefficient(phi_exact);
        phi_gf->SetTrueVector();
        phi_gf->SetFromTrueVector();

        c1_exact.SetTime(current_t);
        c1_gf->ProjectCoefficient(c1_exact);
        c1_gf->SetTrueVector();
        c1_gf->SetFromTrueVector();

        c2_exact.SetTime(current_t);
        c2_gf->ProjectCoefficient(c2_exact);
        c2_gf->SetTrueVector();
        c2_gf->SetFromTrueVector();

        {
            double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
            double  c1L2errornorm =  c1_gf->ComputeL2Error(c1_exact);
            double  c2L2errornorm =  c2_gf->ComputeL2Error(c2_exact);
            if (myid == 0) {
                cout << "After set initial conditions: \n"
                     << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
            }
        }

        if (paraview)
        {
            pd = new ParaViewDataCollection("PNP_Box_Gummel_CG_TimeDependent_ForwardEuler", pmesh);
            pd->SetPrefixPath("Paraview");
            pd->SetLevelsOfDetail(p_order);
            pd->SetDataFormat(VTKFormat::BINARY32);
            pd->SetHighOrderOutput(true);
            pd->RegisterField("phi", phi_gf);
            pd->RegisterField("c1",   c1_gf);
            pd->RegisterField("c2",   c2_gf);

            pd->SetCycle(0);
            pd->SetTime(current_t);
            pd->Save();
        }
    }
    ~PNP_Box_Gummel_CG_TimeDependent_ForwardEuler()
    {
        delete pmesh;
        delete fec;
        delete h1;

        delete a11;
        delete a12;
        delete B1;
        delete a13;
        delete B2;
        delete m1;
        delete M1_no_bc;
        delete m2;
        delete M2_no_bc;

        delete A;
        delete M1;
        delete M2;

        delete phi_gf;
        delete c1_gf;
        delete c2_gf;
        delete phic1c2;

        if (paraview) delete pd;

        delete b;
        delete A1;
        delete b1;
        delete A2;
        delete b2;

        delete A_solver;
        delete M1_solver;
        delete M2_solver;
    }

    void Step()
    {
        // 获得已知的上一时刻的解的TrueVector
        Vector phi(phic1c2->GetData() + 0*true_size, true_size);
        Vector c1 (phic1c2->GetData() + 1*true_size, true_size);
        Vector c2 (phic1c2->GetData() + 2*true_size, true_size);

        // new_*_gf 用在Gummel迭代中
        ParGridFunction new_phi_gf(h1), phi_Gummel(h1), new_c1_gf(h1), new_c2_gf(h1);
        new_phi_gf = 0.0;
        new_c1_gf  = 0.0;
        new_c2_gf  = 0.0;
        phi_exact.SetTime(current_t);
        c1_exact .SetTime(current_t);
        c2_exact .SetTime(current_t);
        new_phi_gf.ProjectBdrCoefficient(phi_exact, ess_bdr); // 需要满足下一时刻的边界条件
        new_c1_gf .ProjectBdrCoefficient( c1_exact, ess_bdr);
        new_c2_gf .ProjectBdrCoefficient( c2_exact, ess_bdr);

        const Operator &R = *h1->GetRestrictionMatrix(); // Operator要不要换成SparseMatrixfff
        const Operator &P = *h1->GetProlongationMatrix();
        f1_analytic.SetTime(current_t);
        f2_analytic.SetTime(current_t);

        Vector diff(h1->GetVSize());
        bool last_gummel_step = false;
        for (int gummel_step=1; !last_gummel_step; ++gummel_step)
        {
            // new_* 就是对应 new_*_gf 的TrueVector, 用在Gummel迭代中
            new_phi_gf.SetTrueVector();
            new_c1_gf .SetTrueVector();
            new_c2_gf .SetTrueVector();
            HypreParVector* new_phi = new_phi_gf.GetTrueDofs();
            HypreParVector* new_c1  = new_c1_gf.GetTrueDofs();
            HypreParVector* new_c2  = new_c2_gf.GetTrueDofs();
            phi_Gummel = new_phi_gf;

            {
            /// ------------------------------ 求解Poisson方程 ------------------------------
                // 下一时刻的Poisson方程的右端项
                ParLinearForm *l = new ParLinearForm(h1);
                l->Assemble();
                delete b;
                b = l->ParallelAssemble(); // 一定要自己delete b

                // 在求解器求解的外面所使用的Vector，Matrix全部是Hypre类型的，在给PETSc的Krylov求解器传入参数时也是传入的Hypre类型的(因为求解器内部会将Hypre的矩阵和向量转化为PETSc的类型)
                B1->Mult(1.0, *new_c1, 1.0, *b); // B1 c1^k + b -> b. k表示Gummel迭代次数, 下同
                B2->Mult(1.0, *new_c2, 1.0, *b); // B1 c1^k + B2 c2^k + b -> b

                R.MultTranspose(*b, *l);
                a11->FormLinearSystem(ess_tdof_list, new_phi_gf, *l, *A, *new_phi, *b);
                A_solver = new PetscLinearSolver(*A, false, "phi_");
                A_solver->Mult(*b, *new_phi);

                new_phi_gf = *new_phi; // 更新phi
                delete l;

                diff = 0.0;
                diff += new_phi_gf;
                diff -= phi_Gummel;
                double tol = diff.Norml2() / new_phi_gf.Norml2();
                phi_Gummel = new_phi_gf; // 算完本次Gummel迭代的tol就可以更新phi_Gummel
                if (myid == 0) {
                    cout << "Gummel step: " << gummel_step << ", Relative Tol: " << tol << endl;
                }
                if (tol < Gummel_rel_tol) { // Gummel迭代停止
                    last_gummel_step = true;
                }
            }

            {
            /// ------------------------------ 求解NP1方程 ------------------------------
                ParBilinearForm *a22 = new ParBilinearForm(h1);
                // D1 (grad(c1^n), grad(v1))
                a22->AddDomainIntegrator(new DiffusionIntegrator(D_K_));
                // D1 z1 c1^n (grad(phi^n+1), grad(v1)), 这里phi^n+1就是new_phi
                a22->AddDomainIntegrator(new GradConvectionIntegrator(new_phi_gf, &D_K_prod_v_K));
                a22->Assemble(skip_zero_entries);
                a22->Finalize(skip_zero_entries);
                A1 = a22->ParallelAssemble(); // 一定要自己delete A1

                ParLinearForm *l1 = new ParLinearForm(h1);
                // (f1, v1)
                l1->AddDomainIntegrator(new DomainLFIntegrator(f1_analytic));
                l1->Assemble();
                b1 = l1->ParallelAssemble(); // 一定要自己delete b1

//                A1->Mult(t_stepsize, *new_c1, t_stepsize, *b1); // dt (A1 c1^k + b1) -> b1
                M1_no_bc->Mult(1.0, c1, t_stepsize, *b1); // M1 c1 + dt b1 -> b1

                R.MultTranspose(*b1, *l1);
                m1->FormLinearSystem(ess_tdof_list, new_c1_gf, *l1, *M1, *new_c1, *b1); // goon
                M1_solver = new PetscLinearSolver(*M1, false, "np1_");
                M1_solver->Mult(*b1, *new_c1);

                new_c1_gf = *new_c1;
                delete a22;
                delete l1;
            }
            {
            /// ------------------------------ 求解NP2方程 ------------------------------
                ParBilinearForm *a33 = new ParBilinearForm(h1);
                // D2 (grad(c2^n), grad(v2))
                a33->AddDomainIntegrator(new DiffusionIntegrator(D_Cl_));
                // D2 z2 c2^n (grad(phi^n+1), grad(v2)), 这里phi^n+1就是new_phi
                a33->AddDomainIntegrator(new GradConvectionIntegrator(new_phi_gf, &D_Cl_prod_v_Cl));
                a33->Assemble(skip_zero_entries);
                a33->Finalize(skip_zero_entries);
                A2 = a33->ParallelAssemble(); // 一定要自己delete A2

                ParLinearForm *l2 = new ParLinearForm(h1);
                // (f2, v2)
                l2->AddDomainIntegrator(new DomainLFIntegrator(f2_analytic));
                l2->Assemble();
                b2 = l2->ParallelAssemble(); // 一定要自己delete b2

//                A2->Mult(t_stepsize, *new_c2, t_stepsize, *b2); // dt (A2 c2^k + b2) -> b2
                M2_no_bc->Mult(1.0, c2, t_stepsize, *b2); // M2 c2 + dt b2 -> b2

                R.MultTranspose(*b2, *l2);
                m2->FormLinearSystem(ess_tdof_list, new_c2_gf, *l2, *M2, *new_c2, *b2);
                M2_solver = new PetscLinearSolver(*M2, false, "np2_");
                M2_solver->Mult(*b2, *new_c2);

                new_c2_gf = *new_c2;
                delete a33;
                delete l2;
            }
        }

        {
            phi_exact.SetTime(current_t);
            c1_exact.SetTime(current_t);
            c2_exact.SetTime(current_t);
            double phiL2errornorm = new_phi_gf.ComputeL2Error(phi_exact);
            double  c1L2errornorm =  new_c1_gf.ComputeL2Error(c1_exact);
            double  c2L2errornorm =  new_c2_gf.ComputeL2Error(c2_exact);
            if (myid == 0) {
                cout << "phi L2 errornorm: " << phiL2errornorm << '\n'
                     << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                     << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
            }
        }
    }

    void Solve()
    {
        bool last_step = false;
        for (int step=1; !last_step; ++step)
        {
            current_t += t_stepsize; // 更新时间：即将求的下一个时刻的物理量
            if (current_t + t_stepsize >= t_final - t_stepsize/2) last_step = true;

            Step(); // 更新解向量TrueVector

            phi_gf->SetFromTrueVector();
            c1_gf ->SetFromTrueVector();
            c2_gf ->SetFromTrueVector();
            {
                phi_exact.SetTime(current_t);
                c1_exact.SetTime(current_t);
                c2_exact.SetTime(current_t);
                double phiL2errornorm = phi_gf->ComputeL2Error(phi_exact);
                double  c1L2errornorm =  c1_gf->ComputeL2Error(c1_exact);
                double  c2L2errornorm =  c2_gf->ComputeL2Error(c2_exact);
                if (myid == 0) {
                    cout << "\nStep: " << step << '\n'
                         << "phi L2 errornorm: " << phiL2errornorm << '\n'
                         << " c1 L2 errornorm: " <<  c1L2errornorm << '\n'
                         << " c2 L2 errornorm: " <<  c2L2errornorm << endl;
                }
            }

            if (paraview)
            {
                pd->SetCycle(step);
                pd->SetTime(current_t);
                pd->Save();
            }
        }
    }
};
#endif
