// 利用 SLEPc求解一个简单的特征值问题
#include "slepceps.h"
#include "petscksp.h"
#include "petscpc.h"
#include "../utils/petsc_utils.hpp"

int main(int argc,char **argv)
{
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

    Mat A, P;
    Read_Mat(A, "../pnp_box/A_matlab.txt");
//    Read_CSRMat_txt("../pnp_box/A_matlab.txt", A);
//    Read_CSRMat_txt("/home/fan/Desktop/B82.txt", P);
//    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    PetscInt dim;
    MatGetSize(A, &dim, NULL);

    Vec xr,xi;
    ierr = MatCreateVecs(A,NULL,&xr); CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&xi); CHKERRQ(ierr);

    // create eigensolver
    EPS eps;
    ierr = EPSCreate(PETSC_COMM_WORLD,&eps); CHKERRQ(ierr);
//    ierr = EPSSetOperators(eps,A,P); CHKERRQ(ierr); // Ax = kPx
    ierr = EPSSetOperators(eps,A,NULL); CHKERRQ(ierr); // Ax = kx
    ierr = EPSSetType(eps, EPSKRYLOVSCHUR); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRQ(ierr);

    ST st;
    ierr = EPSGetST(eps,&st);CHKERRQ(ierr);

    // select intereseted eigenvalues
    PetscInt nev=dim;

    switch (3)
    {
        case 0:
            ierr = EPSSetTarget(eps,1);CHKERRQ(ierr);
            ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
            // Use shift-and-invert to avoid solving linear systems with a singular B in case nulldim>0
            ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
            ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
            break;
        case 1:
            ierr = EPSSetWhichEigenpairs(eps,EPS_ALL);CHKERRQ(ierr);
            ierr = EPSSetInterval(eps,1.1,1.3);CHKERRQ(ierr);
//            ierr = EPSSetInterval(eps, -1.2, -0.9);CHKERRQ(ierr);
//            ierr = EPSSetInterval(eps, 0.9, 1.2);CHKERRQ(ierr);
            break;
        case 2:
            ierr = EPSSetDimensions(eps,nev,3*nev,PETSC_DECIDE);CHKERRQ(ierr);
            break;
        case 3:
            ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
//            ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
//            ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);
//            ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE);CHKERRQ(ierr);
            break;
    }

    // set options for eigensolver
    KSP ksp;
    ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
    PC pc;
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCILU);CHKERRQ(ierr);

    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    if (0) { // 额外的信息(可选)
        PetscInt its;
        ierr = EPSGetIterationNumber(eps,&its); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its); CHKERRQ(ierr);
        EPSType type;
        ierr = EPSGetType(eps,&type); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type); CHKERRQ(ierr);
        PetscInt nev;
        ierr = EPSGetDimensions(eps,&nev,NULL,NULL); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev); CHKERRQ(ierr);
        PetscInt maxit;
        PetscReal tol;
        ierr = EPSGetTolerances(eps,&tol,&maxit); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit); CHKERRQ(ierr);
    }

    PetscInt nconv;
    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv); CHKERRQ(ierr);
    if (nconv>0)
    {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                           "           k          ||Ax-kx||/||kx||\n"
                           "   ----------------- ------------------\n"); CHKERRQ(ierr);
        for (int i=0; i<nconv; i++)
        {
            PetscScalar kr,ki;
            ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi); CHKERRQ(ierr);
            PetscReal error;
            ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error); CHKERRQ(ierr);

            PetscReal re,im;
            re = kr;
            im = ki;
            if (fabs(im) > 1E-10)
            {
                ierr = PetscPrintf(PETSC_COMM_WORLD, " %9f%+9fi %12g\n",(double)re,(double)im,(double)error); CHKERRQ(ierr);
            }
            else
            {
                ierr = PetscPrintf(PETSC_COMM_WORLD, "   %12f       %12g\n",(double)re,(double)error); CHKERRQ(ierr);
            }
        }
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
    }

    // Free work space
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = EPSDestroy(&eps); CHKERRQ(ierr);
    ierr = VecDestroy(&xr); CHKERRQ(ierr);
    ierr = VecDestroy(&xi); CHKERRQ(ierr);
    ierr = SlepcFinalize();
    return ierr;
}


