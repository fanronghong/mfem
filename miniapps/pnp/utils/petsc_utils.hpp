#ifndef LEARN_MFEM_PETSC_UTILS_HPP
#define LEARN_MFEM_PETSC_UTILS_HPP

#include <iostream>
#include <fstream>
#include "petscksp.h"


/* Error handling
// Prints PETSc's stacktrace and then calls MFEM_ABORT
// We cannot use PETSc's CHKERRQ since it returns a PetscErrorCode */
#define PCHKERRQ(obj,err) do {                                                   \
     if ((err))                                                                  \
     {                                                                           \
        PetscError(PetscObjectComm((PetscObject)(obj)),__LINE__,_MFEM_FUNC_NAME, \
                   __FILE__,(err),PETSC_ERROR_REPEAT,NULL);                      \
        MFEM_ABORT("Error in PETSc. See stacktrace above.");                     \
     }                                                                           \
    } while(0);



// ------------------------------ 读写各种格式的矩阵,向量 ------------------------------
// 从二进制文件中(只包含一个矩阵)读取一个矩阵
PetscErrorCode Read_Mat(Mat& mat, char *filename)
{
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-show_read_data", &flg, NULL);
    flg = PETSC_TRUE;
    if (flg) {printf("======> Reading Matrix from file: %s\n", filename); }

    PetscErrorCode ierr;
    PetscViewer viewer;

    MatCreate(PETSC_COMM_WORLD, &mat);
    MatSetType(mat, MATAIJ); /* MATSEQAIJ, MATMPIAIJ, MATAIJ */

    MPI_Barrier(PETSC_COMM_WORLD); /* All processors wait until test matrix has been dumped */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
    ierr = MatLoad(mat, viewer); CHKERRQ(ierr);
//    MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    if (flg) {printf("------> Complete Reading\n"); }
    return 0;
}
// 将一个矩阵(只能是单个矩阵)写入到一个二进制文件中
PetscErrorCode Write_Mat(const Mat& mat, char *filename)
{
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-show_write_data", &flg, NULL);
    flg = PETSC_TRUE;
    if (flg) {printf("======> Writing Matrix from file: %s\n", filename); }

    PetscErrorCode ierr;
    PetscViewer viewer;

    MPI_Barrier(PETSC_COMM_WORLD); /* All processors wait until test matrix has been dumped */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
    ierr = MatView(mat, viewer); CHKERRQ(ierr);
//    MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    if (flg) {printf("------> Complete Writing\n");}
    return 0;
}
// 从一个二进制文件中(只包含一个向量)读取一个向量
PetscErrorCode Read_Vec(Vec vec, char *filename, PetscViewer viewer, PetscErrorCode ierr)
{
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-main_show_read_write_data", &flg, NULL);
    if (flg) {printf("======> Reading Vector from file: %s\n", filename); }

    MPI_Barrier(PETSC_COMM_WORLD); /* All processors wait until test matrix has been dumped */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
    ierr = VecLoad(vec, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    if (flg) {printf("------> Complete Reading\n");}
    return 0;
}
// 讲一个向量(只能是单个向量)写入到一个二进制文件中
PetscErrorCode Write_Vec(Vec vec, const char *filename, PetscViewer viewer, PetscErrorCode ierr)
{
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-main_show_read_write_data", &flg, NULL);
    if (flg) {printf("======> Writing Vector from file: %s\n", filename); }

    MPI_Barrier(PETSC_COMM_WORLD); /* All processors wait until test matrix has been dumped */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = VecView(vec, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    if (flg) {printf("-----> Complete Writing\n");}
    return 0;
}

void Read_CSRMat_txt(const char* filename, Mat& mat)
{
//    std::string path = getcwd(NULL, 0);
//    std::cout << "==> getcwd(NULL, 0): " << path << std::endl;
    std::cout << "\n===> Begin Reading CSR matrix from " << filename << std::endl;
    std::ifstream tomemory(filename);
    tomemory.precision(16);

    std::string line;

    std::getline(tomemory, line); //不读取每行的换行符
    PetscInt N = std::stol(line); //I(即row offsets)的维数

    PetscInt* I=0;
    PetscMalloc1(N, &I);
    for (int i=0; i<N; i++)
    {
        std::getline(tomemory, line); //不读取每行的换行符
        I[i] = std::stol(line);
    }

    PetscInt size = N - 1; //矩阵的维数
    PetscInt* nnz_each_row=0;
    PetscMalloc1(size, &nnz_each_row);
    for (int i=0; i<size; i++)
    {
        nnz_each_row[i] = I[i+1] - I[i];
    }

    PetscInt nnz = I[N-1] - I[0];
    PetscInt* J=0;
    PetscScalar* Vals=0;

    PetscMalloc1(nnz, &J);
    for (int i=0; i<nnz; i++)
    {
        std::getline(tomemory, line); //不读取每行的换行符
        J[i] = std::stol(line);
    }

    PetscMalloc1(nnz, &Vals);
    for (int i=0; i<nnz; i++)
    {
        std::getline(tomemory, line);
        Vals[i] = std::stod(line);
    }
    tomemory.close();

    MatCreate(PETSC_COMM_WORLD, &mat);
    MatSetType(mat, MATAIJ); /* MATSEQAIJ, MATMPIAIJ, MATAIJ */
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, size, size);
    MatSetFromOptions(mat);
    MatSeqAIJSetPreallocation(mat, 0, nnz_each_row); //加速Assemble过程
    MatSetUp(mat);

    int nnz_per_row = 0;
    int idx = 0;
    for (int i=0; i<size; i++)
    {
        nnz_per_row = I[i+1] - I[i];
        MatSetValues(mat, 1, &i, nnz_per_row, &J[idx], &Vals[idx], INSERT_VALUES);
        idx += nnz_per_row;
    }

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    std::cout << "     Matrix dimension: " << size << " x " << size << std::endl;
    std::cout << "===> End   Reading CSR matrix" << std::endl;
}
void Read_Vec_txt(const char* filename, Vec& vec)
{
    std::cout << "\n===> Begin Reading Vector from " << filename << std::endl;
    std::ifstream tomemory(filename);
    tomemory.precision(16);

    PetscInt nrows;
    tomemory >> nrows;

    VecCreate(PETSC_COMM_WORLD, &vec);
    VecSetType(vec, VECSEQ);
    VecSetSizes(vec, PETSC_DECIDE, nrows);
    VecSetFromOptions(vec);
    VecSetUp(vec);

    PetscScalar tmp;
    for (int i=0; i<nrows; i++)
    {
        tomemory >> tmp;
        VecSetValues(vec, 1, &i, &tmp, INSERT_VALUES);
    }
    tomemory.close();

    VecAssemblyBegin(vec);
    VecAssemblyEnd(vec);

    std::cout << "     Vector dimension: " << nrows << std::endl;
    std::cout << "===> End Reading Vector" << std::endl;
}
void Test_Read_CSRMat_txt_and_Read_Vec_txt()
{
    const char* file = "../../data/test_A.txt";
    Mat A;
    Read_CSRMat_txt(file, A);
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    Vec b;
    Read_Vec_txt("../../data/test_b.txt", b);
    VecView(b, PETSC_VIEWER_STDOUT_WORLD);

}

void Write_Mat_Matlab_txt(const char* filename, const Mat& mat)
{
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(mat, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}
void Write_Vec_Matlab_txt(const char* filename, const Vec& vec)
{
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    VecView(vec, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}
void Read_Vec_Matlab_txt(const char* filename, Vec& vec)
{
    VecCreate(PETSC_COMM_WORLD, &vec);
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    VecLoad(vec, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}
void Write_Mat_Matlab_bin(const char* filename, const Mat& mat)
{
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_BINARY_MATLAB);
    MatView(mat, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}
void Write_Vec_Matlab_bin(const char* filename, const Vec& vec)
{
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_BINARY_MATLAB);
    VecView(vec, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
}


// ------------------------------ User-defined KSP monitor ------------------------------
// 用户自定义的KSP迭代过程的monitor 以及释放内存空间
struct UserKSPMonitorContext
{
    PetscInt step; //迭代step步输出1次
    Vec rhs;
    PetscReal rhs_l2;
};
PetscErrorCode UserKSPMonitor(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx)
{
    UserKSPMonitorContext *user_ctx = (UserKSPMonitorContext *)ctx;
    if (it % (user_ctx->step) == 0)
    {
        Vec resid; //真实的残差向量
        KSPBuildResidual(ksp, NULL, NULL, &resid);

        PetscReal truenorm, relativenorm;
        VecNorm(resid, NORM_2, &truenorm);
        relativenorm = truenorm /(user_ctx->rhs_l2);

        printf("%.3d-th iteration: Relative Residual=%.5e; True Residual=%.5e\n", it, relativenorm, truenorm);
    }
    return 0;
}
PetscErrorCode UserKSPMonitorDestroy(void **ctx)
{
    UserKSPMonitorContext *dummy = (UserKSPMonitorContext *) *ctx;
//    VecDestroy(&dummy->rhs); //这一步要注意: 有可能在main函数中会对向量rhs释放
    PetscFree(dummy);
    return 0;
}


// ------------------------------ 用户自定义的KSP迭代过程的收敛性测试 ------------------------------
struct UserKSPConvergeTestContext
{
    Vec rhs;
    PetscReal rhs_l2;
};
PetscErrorCode UserKSPConvergeTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
    UserKSPConvergeTestContext *user_ctx = (UserKSPConvergeTestContext *)ctx;

    PetscReal tol; // true norm: ||b - Ax||_2; relative norm: ||b - Ax||_2 / ||b||_2.
    PetscOptionsGetReal(NULL, NULL, "-main_ksp_user_true_l2_tol", &tol, NULL);

    Vec resid; //真实的残差向量
    KSPBuildResidual(ksp, NULL, NULL, &resid);

    PetscReal truenorm, relativenorm;
    VecNorm(resid, NORM_2, &truenorm);
    relativenorm = truenorm /(user_ctx->rhs_l2);

    if(truenorm < tol)
    {
        printf("\n======> In User-Defind Convergence Test(), %d-th iteration:\n", it);
        printf("======> Relative residual l2-norm %g\n", relativenorm);
        printf("======> True     residual l2-norm %g(<= %g, Stop)\n\n", truenorm, tol);
        *reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
    }
    return 0;
}
PetscErrorCode UserKSPConvergeTestDestroy(void *ctx)
{
    UserKSPMonitorContext *dummy = (UserKSPMonitorContext *) ctx;
//    VecDestroy(&dummy->rhs); //注意在main函数中有可能会释放rhs
    PetscFree(dummy);
    return 0;
}


// PNP 方程中使用 SPD 部分做预条件子
typedef struct {
    /*
    给定 x, 求解 y
    SPD * y = x
    */
    KSP spd_inv_ksp; //求解SPD预条件子的逆
} PNP_SPD_UserPCContext;
PetscErrorCode PNP_SPD_UserPCCreate(PNP_SPD_UserPCContext **shell)
{
    PetscErrorCode ierr;

    PNP_SPD_UserPCContext *newcontext;
    ierr = PetscNew(&newcontext);CHKERRQ(ierr);

    newcontext->spd_inv_ksp = 0;
    *shell = newcontext;
    return 0;
}
PetscErrorCode PNP_SPD_UserPCSetUp(PC pc, Mat spd)
{
    PetscErrorCode ierr;
    PetscBool flg=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-spd_ksp_verbose", &flg, NULL); CHKERRQ(ierr);
    if (flg)
    {
        printf("===> Starting Setting Up: SPD Preconditioner for PNP equations\n");
    }

    KSP spd_inv_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &spd_inv_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(spd_inv_ksp, "spd_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(spd_inv_ksp, spd, spd); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(spd_inv_ksp); CHKERRQ(ierr);

    PNP_SPD_UserPCContext* shell;
//    PetscNew(&shell); //不需要分配内存空间
    ierr = PCShellGetContext(pc, (void**)&shell); CHKERRQ(ierr);

    shell->spd_inv_ksp = spd_inv_ksp;

    if (flg)
    {
        printf("===> Ending Setting Up.\n");
    }
    return 0;
}
PetscErrorCode PNP_SPD_UserPCApply(PC pc, Vec x, Vec y)
{   //给定x,求解y
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-spd_ksp_verbose", &flg, NULL);
    if (flg) {printf("\n---> solve SPD Preconditioner inverse.\n"); }

    PNP_SPD_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);

    KSPSolve(shell->spd_inv_ksp, x, y);
    return 0;
}
PetscErrorCode PNP_SPD_UserPCDestroy(PC pc)
{
    PNP_SPD_UserPCContext *shell;
    PCShellGetContext(pc, (void**)&shell);
    KSPDestroy(&shell->spd_inv_ksp);
    PetscFree(shell);
    return 0;
}
PetscErrorCode Use_PNP_SPD_Prec(PC& pc, Mat& spd)
{
    printf("\n===> SPD Preconditioner for PNP equation\n");
    PetscErrorCode ierr;

    PNP_SPD_UserPCContext* shell;

    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);

    ierr = PNP_SPD_UserPCCreate(&shell); CHKERRQ(ierr);

    ierr = PCShellSetApply(pc, PNP_SPD_UserPCApply); CHKERRQ(ierr);

    ierr = PCShellSetContext(pc, shell); CHKERRQ(ierr);

    ierr = PCShellSetDestroy(pc, PNP_SPD_UserPCDestroy); CHKERRQ(ierr);

    ierr = PCShellSetName(pc, "SPD_Preconditioner_For_PNP"); CHKERRQ(ierr);

    ierr = PNP_SPD_UserPCSetUp(pc, spd); CHKERRQ(ierr);

    return 0;
}


// Two-Grid Preconditioner
typedef struct {
    /*
    给定 x, 求解 y
    第一步: AH * y1 = Prolongation^t * x
    第二步: As * y2 = beta*x
    y = Prolongation*y1 + y2
    */
    Mat AH, As, Prolongation, Ah;
    PetscReal beta; //
    Vec sol1, sol2; //第一,二步的解
    KSP fine_solver, coarse_solver; // 第一,二步的求解器
} TwoGridPC_UserPCContext;
PetscErrorCode TwoGridPC_UserPCCreate(TwoGridPC_UserPCContext **shell)
{
    TwoGridPC_UserPCContext *newcontext;
    PetscNew(&newcontext);
    newcontext->AH = 0;
    newcontext->As = 0;
    newcontext->Prolongation = 0;
    newcontext->Ah = 0;
    newcontext->beta = 0;
    newcontext->sol1 = 0;
    newcontext->sol2 = 0;
    newcontext->fine_solver = 0;
    newcontext->coarse_solver = 0;
    *shell = newcontext;
    return 0;
}
PetscErrorCode TwoGridPC_UserPCSetUp(PC pc, Mat AH, Mat As, Mat Prolongation, Mat Ah)
{
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-main_show_where_setup_user_pc", &flg, NULL);
    if (flg) {printf("======> Starting Setting Up: User-Defined Preconditioner\n"); }

    PetscInt Nfine, Ncoarse;
    MatGetSize(Prolongation, &Nfine, &Ncoarse);

    Vec sol1;
    VecCreate(MPI_COMM_WORLD, &sol1);
    VecSetSizes(sol1, PETSC_DECIDE, Ncoarse);
    VecSetFromOptions(sol1);
    VecSetUp(sol1); //Sets up the internal vector data structures for the later use

    Vec sol2;
    VecCreate(MPI_COMM_WORLD, &sol2);
    VecSetSizes(sol2, PETSC_DECIDE, Nfine);
    VecSetFromOptions(sol2);
    VecSetUp(sol2);

    KSP coarse_solver;
    KSPCreate(PETSC_COMM_WORLD, &coarse_solver);
    KSPSetOptionsPrefix(coarse_solver, "coarsesolver_");
    KSPSetOperators(coarse_solver, AH, AH);
    KSPSetFromOptions(coarse_solver);

    KSP fine_solver;
    KSPCreate(PETSC_COMM_WORLD, &fine_solver);
    KSPSetOptionsPrefix(fine_solver, "finesolver_");
    // KSPSetOperators(fine_solver, As, As);
    KSPSetOperators(fine_solver, Ah, Ah);
    KSPSetFromOptions(fine_solver);

    TwoGridPC_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);
    shell->coarse_solver = coarse_solver;
    shell->fine_solver = fine_solver;
    // shell->Am = Am;
    shell->AH = AH;
    shell->As = As;
    shell->Prolongation = Prolongation;
    shell->sol1 = sol1;
    shell->sol2 = sol2;
    shell->Ah = Ah;
    PetscOptionsGetReal(NULL,NULL,"-beta",&(shell->beta),NULL);
    if (flg) {printf("======> Ending Setting Up: User-Defined Preconditioner\n"); }
    return 0;
}
PetscErrorCode TwoGridPC_UserPCApply(PC pc, Vec x, Vec y) //给定x,求解y
{
    TwoGridPC_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);

    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-main_show_where_apply_user_pc", &flg, NULL);
    if (flg) {printf("\n-------> Solve Coarse SubProblem.\n"); }

    Vec dummy1;
    VecDuplicate(shell->sol1, &dummy1);
    MatMultTranspose(shell->Prolongation, x, dummy1);
    KSPSolve(shell->coarse_solver, dummy1, shell->sol1);

    if (flg) {printf("=======> Solve Fine SubProblem.\n"); }
    Vec dummy2;
    VecDuplicate(shell->sol2, &dummy2);
    VecAXPBY(dummy2, 1.0/(shell->beta), 0.0, x); // dummy2 = x/beta
    KSPSolve(shell->fine_solver, dummy2, shell->sol2);

    Vec dummy3;
    VecDuplicate(shell->sol2, &dummy3);
    MatMult(shell->Prolongation, shell->sol1, dummy3);
    VecWAXPY(y, 1.0, dummy3, shell->sol2);
    return 0;
}
PetscErrorCode TwoGridPC_UserPCDestroy(PC pc)
{
    TwoGridPC_UserPCContext *shell;
    PCShellGetContext(pc, (void**)&shell);
    VecDestroy(&shell->sol1);
    VecDestroy(&shell->sol2);
    MatDestroy(&shell->AH);
    MatDestroy(&shell->As);
    MatDestroy(&shell->Prolongation);
    KSPDestroy(&shell->coarse_solver);
    KSPDestroy(&shell->fine_solver);
    PetscFree(shell);
    return 0;
}


// PNP 方程中使用 Two-Stage Preconditioner
typedef struct {
    /*
    给定 x, 求解 y
    TwoStage * y = x
    */
    KSP ts1_ksp, ts2_ksp; //求解two-stage预条件子的逆
    Mat jacobi, subjacobi;
    Vec resid, subresid;
    IS rows, cols;
    PetscInt size, subsize;
} PNP_TwoStage_UserPCContext;
PetscErrorCode PNP_TwoStage_UserPCCreate(PNP_TwoStage_UserPCContext **shell)
{
    PetscErrorCode ierr;

    PNP_TwoStage_UserPCContext *newcontext;
    ierr = PetscNew(&newcontext); CHKERRQ(ierr);

    newcontext->ts1_ksp = 0;
    newcontext->ts2_ksp = 0;
    newcontext->jacobi = 0;
    newcontext->subjacobi = 0;
    newcontext->resid = 0;
    newcontext->subresid = 0;
    newcontext->size = 0;
    newcontext->subsize = 0;
    *shell = newcontext;
    return 0;
}
PetscErrorCode PNP_TwoStage_UserPCSetUp(PC pc, Mat& jacobi, Vec& resid, const IS& rows, const IS& cols)
{
    PetscErrorCode ierr;
    PetscBool flag=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-twostage_verbose", NULL, &flag); CHKERRQ(ierr);
    if (flag) {
        printf("===> Starting Setting Up: Two-Stage Preconditioner for PNP equations\n");
    }

    PetscInt size, subsize;
    VecGetSize(resid, &size);
    subsize = size / 3; //3表示PDEs中有3个未知量, PNP方程中的电势, 2种离子的浓度

    Mat subjacobi;
    MatCreate(PETSC_COMM_WORLD, &subjacobi);
    MatSetSizes(subjacobi, PETSC_DECIDE, PETSC_DECIDE, subsize, subsize);
    MatSetFromOptions(subjacobi);
    MatSetUp(subjacobi);
    MatCreateSubMatrix(jacobi, rows, cols, MAT_INITIAL_MATRIX, &subjacobi);

    Vec subresid;
    VecCreate(MPI_COMM_WORLD, &subresid);
    VecSetSizes(subresid, PETSC_DECIDE, subsize);
    VecSetFromOptions(subresid);
    VecSetUp(subresid);
    VecGetSubVector(resid, rows, &subresid);
//    VecRestoreSubVector(resid, rows, &subresid); //ffferror, 为啥petsc的demo里面都有?

    KSP ts1_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ts1_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ts1_ksp, "ts1_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(ts1_ksp, subjacobi, subjacobi); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ts1_ksp); CHKERRQ(ierr);

    KSP ts2_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ts2_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ts2_ksp, "ts2_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(ts2_ksp, jacobi, jacobi); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ts2_ksp); CHKERRQ(ierr);

    PNP_TwoStage_UserPCContext* shell;
    ierr = PCShellGetContext(pc, (void**)&shell); CHKERRQ(ierr);

    shell->jacobi = jacobi;
    shell->subjacobi = subjacobi;
    shell->resid = resid;
    shell->subresid = subresid;
    shell->rows = rows;
    shell->cols = cols;
    shell->ts1_ksp = ts1_ksp;
    shell->ts2_ksp = ts2_ksp;
    shell->size = size;
    shell->subsize = subsize;
    if (flag) {
        printf("===> Ending Setting Up.\n");
    }
    return 0;
}
PetscErrorCode PNP_TwoStage_UserPCApply(PC pc, Vec x, Vec y)
{
    PetscErrorCode ierr;
    //给定x,求解y
    PetscBool flag=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-twostage_verbose", NULL, &flag);
    if (flag) {
        printf("\n---> solve Two-Stage Preconditioner inverse.\n");
    }

    PNP_TwoStage_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);

    Vec sol1; //two-stage中的first-stage中的解向量 P_hat
    VecDuplicate(shell->subresid, &sol1);

    KSPSolve(shell->ts1_ksp, shell->subresid, sol1);

    PetscInt size;
    VecGetSize(sol1, &size);
    PetscScalar *sol1_vals;
    PetscMalloc1(size, &sol1_vals);
    PetscInt *idx;
    PetscMalloc1(size, &idx);
    for (int i=0; i<size; i++) {
        idx[i] = i;
    }
    VecGetValues(sol1, size, idx, sol1_vals);

    Vec resid1; //向量 (P_hat, 0)
    VecDuplicate(shell->resid, &resid1);
    VecSet(resid1, 0);
    VecSetValues(resid1, size, idx, sol1_vals, INSERT_VALUES);
    VecScale(resid1, -1.0); //向量 -(P_hat, 0)

    Vec resid2; // r_tilde - J_tilde * (P_hat, 0) = r_hat
    VecDuplicate(shell->resid, &resid2);
    MatMultAdd(shell->jacobi, resid1, shell->resid, resid2);

    Vec sol2;
    VecDuplicate(shell->resid, &sol2);
    KSPSolve(shell->ts2_ksp, resid2, sol2);

    VecWAXPY(y, 1.0, resid1, sol2);
    return 0;
}
PetscErrorCode PNP_TwoStage_UserPCDestroy(PC pc)
{
    PNP_TwoStage_UserPCContext *shell;
    PCShellGetContext(pc, (void**)&shell);
    KSPDestroy(&shell->ts1_ksp);
    KSPDestroy(&shell->ts2_ksp);
    MatDestroy(&shell->subjacobi);
    VecDestroy(&shell->subresid);
    PetscFree(shell);
    return 0;
}
PetscErrorCode Use_PNP_TwoStage_Prec(PC& pc, Mat& jacobi, Vec& resid, const IS& rows, const IS& cols)
{
    //用Two-Stage Preconditioner求解 jacobi * y = resid
    //rows, cols分别表示要提取的子矩阵的行列索引
    //后面的几个函数里面的变量命名参考论文中的数学符号: Analytical decoupling techniques for fully implicit reservoir simulation
    printf("\n===> Two-Stage Preconditioner for PNP equation\n");
    PetscErrorCode ierr;

    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);

    PNP_TwoStage_UserPCContext* shell;
    ierr = PNP_TwoStage_UserPCCreate(&shell); CHKERRQ(ierr);

    ierr = PCShellSetApply(pc, PNP_TwoStage_UserPCApply); CHKERRQ(ierr);

    ierr = PCShellSetContext(pc, shell); CHKERRQ(ierr);

    ierr = PCShellSetDestroy(pc, PNP_TwoStage_UserPCDestroy); CHKERRQ(ierr);

    ierr = PCShellSetName(pc, "TwoStage_Preconditioner_For_PNP"); CHKERRQ(ierr);

    ierr = PNP_TwoStage_UserPCSetUp(pc, jacobi, resid, rows, cols); CHKERRQ(ierr);

    return 0;
}


// PNP 方程中使用 PCG-BMG(block multi-grid) Preconditioner
typedef struct {
    /*
    给定 x, 求解 y
    SPD * y = x
    */
    KSP poisson_ksp, np1_ksp, np2_ksp; //求解PNP方程中3个对角块的逆
    KSP smooth1_ksp, smooth2_ksp; //预条件前后的两次对整体PNP方程的磨光
    Mat A11, A22, A33; //PNP方程的3个对角块矩阵
    Mat A; //PNP方程的整个Jacobian
    Vec b; //PNP方程的Jacobian对应的右端项
    PetscInt size; //PNP方程的3个对角块矩阵的维数(3个一样)
    IS is1, is2, is3; //3个子矩阵相对于整体Jacobian的索引
} PNP_BMG_UserPCContext;
PetscErrorCode PNP_BMG_UserPCCreate(PNP_BMG_UserPCContext **shell)
{
    PetscErrorCode ierr;

    PNP_BMG_UserPCContext *newcontext;
    ierr = PetscNew(&newcontext);CHKERRQ(ierr);

    newcontext->poisson_ksp = 0;
    newcontext->np1_ksp = 0;
    newcontext->np2_ksp = 0;
    newcontext->smooth1_ksp = 0;
    newcontext->smooth2_ksp = 0;
    newcontext->A11 = 0;
    newcontext->A22 = 0;
    newcontext->A33 = 0;
    *shell = newcontext;
    return 0;
}
PetscErrorCode PNP_BMG_UserPCSetUp(PC pc, Mat& mat, Vec& b, const IS& is1, const IS& is2, const IS& is3, PetscInt size)
{
    PetscErrorCode ierr;
    PetscBool flg=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-pcgbmg_ksp_verbose", &flg, NULL); CHKERRQ(ierr);
    if (flg)
    {
        printf("===> Starting Setting Up: PCG-BMG Preconditioner for PNP equations\n");
    }

    KSP smooth1_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &smooth1_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(smooth1_ksp, "smooth1_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(smooth1_ksp, mat, mat); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(smooth1_ksp); CHKERRQ(ierr);

    Mat A11, A22, A33;
    ierr = MatCreate(PETSC_COMM_WORLD, &A11); CHKERRQ(ierr);
    ierr = MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, size, size); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A11); CHKERRQ(ierr);
    ierr = MatSetUp(A11); CHKERRQ(ierr);
    ierr = MatDuplicate(A11, MAT_DO_NOT_COPY_VALUES, &A22); CHKERRQ(ierr);
    ierr = MatDuplicate(A11, MAT_DO_NOT_COPY_VALUES, &A33); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is1, is1, MAT_INITIAL_MATRIX, &A11); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is2, is2, MAT_INITIAL_MATRIX, &A22); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is3, is3, MAT_INITIAL_MATRIX, &A33); CHKERRQ(ierr);

    KSP poisson_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &poisson_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(poisson_ksp, "poisson_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(poisson_ksp, A11, A11); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(poisson_ksp); CHKERRQ(ierr);

    KSP np1_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &np1_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(np1_ksp, "np1_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(np1_ksp, A22, A22); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(np1_ksp); CHKERRQ(ierr);

    KSP np2_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &np2_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(np2_ksp, "np2_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(np2_ksp, A33, A33); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(np2_ksp); CHKERRQ(ierr);

    KSP smooth2_ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &smooth2_ksp); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(smooth2_ksp, "smooth2_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(smooth2_ksp, mat, mat); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(smooth2_ksp); CHKERRQ(ierr);


    PNP_BMG_UserPCContext* shell;
    ierr = PCShellGetContext(pc, (void**)&shell); CHKERRQ(ierr);

    shell->smooth1_ksp = smooth1_ksp;
    shell->poisson_ksp = poisson_ksp;
    shell->np1_ksp = np1_ksp;
    shell->np2_ksp = np2_ksp;
    shell->smooth2_ksp = smooth2_ksp;
    shell->A11 = A11;
    shell->A22 = A22;
    shell->A33 = A33;
    shell->size = size;
    shell->A = mat;
    shell->b = b;
    shell->is1 = is1;
    shell->is2 = is2;
    shell->is3 = is3;

    if (flg)
    {
        printf("===> Ending Setting Up.\n");
    }
    return 0;
}
PetscErrorCode PNP_BMG_UserPCApply(PC pc, Vec x, Vec y)
{   //给定x,求解y
//    PetscScalar nm;
//    VecNorm(x, NORM_2, &nm);
//    PetscPrintf(MPI_COMM_WORLD, "x norm_2: %f\n", nm);

    PetscErrorCode ierr;
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-pcgbmg_ksp_verbose", &flg, NULL);
    if (flg) {printf("\n---> solve PCG-BMG Preconditioner inverse.\n"); }

    PNP_BMG_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);

    Vec smooth1_sol, smooth2_sol, poisson_sol_extend, np1_sol_extend, np2_sol_extend;
    VecDuplicate(x, &smooth1_sol);
    VecDuplicate(x, &smooth2_sol);
    VecDuplicate(x, &poisson_sol_extend); ierr = VecSet(poisson_sol_extend, 0); CHKERRQ(ierr);
    VecDuplicate(x, &np1_sol_extend);     ierr = VecSet(np1_sol_extend, 0); CHKERRQ(ierr);
    VecDuplicate(x, &np2_sol_extend);     ierr = VecSet(np2_sol_extend, 0); CHKERRQ(ierr);

    Vec poisson_sol, np1_sol, np2_sol;
    ierr = VecCreate(MPI_COMM_WORLD, &poisson_sol); CHKERRQ(ierr);
    ierr = VecSetSizes(poisson_sol, PETSC_DECIDE, shell->size); CHKERRQ(ierr);
    ierr = VecSetFromOptions(poisson_sol); CHKERRQ(ierr);
    ierr = VecSetUp(poisson_sol); CHKERRQ(ierr);
    ierr = VecDuplicate(poisson_sol, &np1_sol); CHKERRQ(ierr);
    ierr = VecDuplicate(poisson_sol, &np2_sol); CHKERRQ(ierr);


//    KSPSolve(shell->smooth1_ksp, x, smooth1_sol);

    Vec b1;
    VecDuplicate(poisson_sol, &b1);
    VecGetSubVector(x, shell->is1, &b1);
    KSPSolve(shell->poisson_ksp, b1, poisson_sol);

    Vec b2;
    VecDuplicate(np1_sol, &b2);
    VecGetSubVector(x, shell->is2, &b2);
    KSPSolve(shell->np1_ksp, b2, np1_sol);

    Vec b3;
    VecDuplicate(np2_sol, &b3);
    VecGetSubVector(x, shell->is3, &b3);
    KSPSolve(shell->np2_ksp, b3, np2_sol);

//    KSPSolve(shell->smooth2_ksp, x, smooth2_sol);

    PetscInt idx1[1], idx2[1], idx3[1];
    PetscScalar val1, val2, val3;
    for (int i=0; i<shell->size; i++) {
        idx1[0] = i;
        idx2[0] = i + 1*shell->size;
        idx3[0] = i + 2*shell->size;
        VecGetValues(poisson_sol, 1, &i, &val1);
        VecGetValues(np1_sol, 1, &i, &val2);
        VecGetValues(np2_sol, 1, &i, &val3);
        VecSetValue(y, idx1[0], val1, INSERT_VALUES);
        VecSetValue(y, idx2[0], val2, INSERT_VALUES);
        VecSetValue(y, idx3[0], val3, INSERT_VALUES);
    }

    return ierr;
}
PetscErrorCode PNP_BMG_UserPCDestroy(PC pc)
{
    PNP_BMG_UserPCContext *shell;
    PCShellGetContext(pc, (void**)&shell);
    KSPDestroy(&shell->smooth1_ksp);
    KSPDestroy(&shell->smooth2_ksp);
    KSPDestroy(&shell->poisson_ksp);
    KSPDestroy(&shell->np1_ksp);
    KSPDestroy(&shell->np2_ksp);
    MatDestroy(&shell->A11);
    MatDestroy(&shell->A22);
    MatDestroy(&shell->A33);
    PetscFree(shell);
    return 0;
}
PetscErrorCode Use_PNP_BMG_Prec(PC& pc, Mat& mat, Vec& b, int num_vars)
{
    printf("\n===> PCG-BMG Preconditioner for PNP equation\n");
    PetscErrorCode ierr;

    PNP_BMG_UserPCContext* shell;

    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);

    ierr = PNP_BMG_UserPCCreate(&shell); CHKERRQ(ierr);

    ierr = PCShellSetApply(pc, PNP_BMG_UserPCApply); CHKERRQ(ierr);

    ierr = PCShellSetContext(pc, shell); CHKERRQ(ierr);

    ierr = PCShellSetDestroy(pc, PNP_BMG_UserPCDestroy); CHKERRQ(ierr);

    ierr = PCShellSetName(pc, "PCG-BMG_Preconditioner_For_PNP"); CHKERRQ(ierr);

    PetscInt rowsize, colnsize, size; //size是PNP对角块矩阵的大小
    MatGetSize(mat, &rowsize, &colnsize);
    if (rowsize != colnsize) throw "Not a square-matrix";
    size = rowsize / num_vars;

    PetscInt *arr1, *arr2, *arr3;
    PetscMalloc1(size, &arr1);
    PetscMalloc1(size, &arr2);
    PetscMalloc1(size, &arr3);
    for (int i=0; i<size; i++) {
        arr1[i] = i;
        arr2[i] = i + size;
        arr3[i] = i + 2*size;
    }
    IS is1, is2, is3;
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr1, PETSC_COPY_VALUES, &is1);
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr2, PETSC_COPY_VALUES, &is2);
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr3, PETSC_COPY_VALUES, &is3);

    ierr = PNP_BMG_UserPCSetUp(pc, mat, b, is1, is2, is3, size); CHKERRQ(ierr);

    return 0;
}


// PNP 方程中使用 SPD_Lower Preconditioner
typedef struct {
    /*
     *使用PNP的Jacobian的对角块中的SPD部分, 加上严格下三角块部分做预条件子
     */
    KSP poisson_ksp, np1_diffusion_ksp, np2_diffusion_ksp; //求解PNP方程中3个对角块的逆
    Mat A11, A21, A22_diffusion, A31, A33_diffusion; //PNP方程的3个对角块矩阵对应的SPD部分,加上严格下三角块部分
    Mat A; //PNP方程的整个Jacobian
    Vec b; //PNP方程的Jacobian对应的右端项
    PetscInt size; //PNP方程的3个对角块矩阵的维数(3个一样)
    IS is1, is2, is3; //3个子矩阵相对于整体Jacobian的索引
} PNP_SPDLower_UserPCContext;
PetscErrorCode PNP_SPDLower_UserPCCreate(PNP_SPDLower_UserPCContext **shell)
{
    PetscErrorCode ierr;

    PNP_SPDLower_UserPCContext *newcontext;
    ierr = PetscNew(&newcontext);CHKERRQ(ierr);

    newcontext->poisson_ksp = 0;
    newcontext->np1_diffusion_ksp = 0;
    newcontext->np2_diffusion_ksp = 0;
    newcontext->A11 = 0;
    newcontext->A22_diffusion = 0;
    newcontext->A21 = 0;
    newcontext->A33_diffusion = 0;
    newcontext->A31 = 0;
    *shell = newcontext;
    return 0;
}
PetscErrorCode PNP_SPDLower_UserPCSetUp(PC pc, Mat& mat, Vec& b, const IS& is1, const IS& is2, const IS& is3, PetscInt size)
{
    PetscErrorCode ierr;
    PetscBool flg=PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-pcgbmg_ksp_verbose", &flg, NULL); CHKERRQ(ierr);
    if (flg)
    {
        printf("===> Starting Setting Up: PCG-BMG Preconditioner for PNP equations\n");
    }

    Mat A11, A22_diffusion, A33_diffusion, A21, A31;
    ierr = MatCreate(PETSC_COMM_WORLD, &A11); CHKERRQ(ierr);
    ierr = MatSetSizes(A11, PETSC_DECIDE, PETSC_DECIDE, size, size); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A11); CHKERRQ(ierr);
    ierr = MatSetUp(A11); CHKERRQ(ierr);

    ierr = MatDuplicate(A11, MAT_DO_NOT_COPY_VALUES, &A22_diffusion); CHKERRQ(ierr);
    ierr = MatDuplicate(A11, MAT_DO_NOT_COPY_VALUES, &A33_diffusion); CHKERRQ(ierr);

    ierr = MatCreateSubMatrix(mat, is1, is1, MAT_INITIAL_MATRIX, &A11); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is2, is2, MAT_INITIAL_MATRIX, &A22_diffusion); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is3, is3, MAT_INITIAL_MATRIX, &A33_diffusion); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is2, is1, MAT_INITIAL_MATRIX, &A21); CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(mat, is3, is1, MAT_INITIAL_MATRIX, &A31); CHKERRQ(ierr);

    {//for test
        PetscInt A33_size;
        MatGetSize(A33_diffusion, &A33_size, NULL);
        std::cout << A33_size << std::endl;
        PetscInt b_size;
        MatGetSize(mat, &b_size, NULL);
        std::cout << b_size << std::endl;
    }

    KSP poissonspd;
    ierr = KSPCreate(PETSC_COMM_WORLD, &poissonspd); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(poissonspd, "poissonspd_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(poissonspd, A11, A11); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(poissonspd); CHKERRQ(ierr);

    KSP np1_diffusion;
    ierr = KSPCreate(PETSC_COMM_WORLD, &np1_diffusion); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(np1_diffusion, "np1diffusion_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(np1_diffusion, A22_diffusion, A22_diffusion); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(np1_diffusion); CHKERRQ(ierr);

    KSP np2_diffusion;
    ierr = KSPCreate(PETSC_COMM_WORLD, &np2_diffusion); CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(np2_diffusion, "np2diffusion_"); CHKERRQ(ierr);
    ierr = KSPSetOperators(np2_diffusion, A33_diffusion, A33_diffusion); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(np2_diffusion); CHKERRQ(ierr);


    PNP_SPDLower_UserPCContext* shell;
    ierr = PCShellGetContext(pc, (void**)&shell); CHKERRQ(ierr);

    shell->poisson_ksp = poissonspd;
    shell->np1_diffusion_ksp = np1_diffusion;
    shell->np2_diffusion_ksp = np2_diffusion;
    shell->A11 = A11;
    shell->A22_diffusion = A22_diffusion;
    shell->A33_diffusion = A33_diffusion;
    shell->A21 = A21;
    shell->A31 = A31;
    shell->size = size;
    shell->A = mat;
    shell->b = b;
    shell->is1 = is1;
    shell->is2 = is2;
    shell->is3 = is3;

    if (flg)
    {
        printf("===> Ending Setting Up.\n");
    }
    return 0;
}
PetscErrorCode PNP_SPDLower_UserPCApply(PC pc, Vec x, Vec y)
{   //给定x,求解y
//    PetscScalar nm;
//    VecNorm(x, NORM_2, &nm);
//    PetscPrintf(MPI_COMM_WORLD, "x norm_2: %f\n", nm);

    PetscErrorCode ierr;
    PetscBool flg=PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-spd_lower_ksp_verbose", &flg, NULL);
    if (flg) {printf("\n---> solve SPD-Lower Preconditioner inverse.\n"); }

    PNP_SPDLower_UserPCContext *shell;
    PetscNew(&shell);
    PCShellGetContext(pc, (void**)&shell);

    Vec poisson_sol_extend, np1diffusion_sol_extend, np2diffusion_sol_extend;
    VecDuplicate(x, &poisson_sol_extend);          ierr = VecSet(poisson_sol_extend, 0);      CHKERRQ(ierr);
    VecDuplicate(x, &np1diffusion_sol_extend);     ierr = VecSet(np1diffusion_sol_extend, 0); CHKERRQ(ierr);
    VecDuplicate(x, &np2diffusion_sol_extend);     ierr = VecSet(np2diffusion_sol_extend, 0); CHKERRQ(ierr);

    Vec poisson_sol, np1_sol, np2_sol;
    ierr = VecCreate(MPI_COMM_WORLD, &poisson_sol); CHKERRQ(ierr);
    ierr = VecSetSizes(poisson_sol, PETSC_DECIDE, shell->size); CHKERRQ(ierr);
    ierr = VecSetFromOptions(poisson_sol); CHKERRQ(ierr);
    ierr = VecSetUp(poisson_sol); CHKERRQ(ierr);
    ierr = VecDuplicate(poisson_sol, &np1_sol); CHKERRQ(ierr);
    ierr = VecDuplicate(poisson_sol, &np2_sol); CHKERRQ(ierr);

    Vec b1;
    VecDuplicate(poisson_sol, &b1);
    VecGetSubVector(x, shell->is1, &b1);
    KSPSolve(shell->poisson_ksp, b1, poisson_sol);

    Vec b2, b2_tmp;
    VecDuplicate(np1_sol, &b2);
    VecDuplicate(np1_sol, &b2_tmp);
    VecGetSubVector(x, shell->is2, &b2); //得到x的子向量
    MatMult(shell->A21, poisson_sol, b2_tmp);
    VecAXPY(b2_tmp, -1.0, b2);
    VecScale(b2_tmp, -1.0);
    KSPSolve(shell->np1_diffusion_ksp, b2_tmp, np1_sol);

    Vec b3, b3_tmp;
    VecDuplicate(np2_sol, &b3);
    VecDuplicate(np2_sol, &b3_tmp);
    VecGetSubVector(x, shell->is3, &b3);
    MatMult(shell->A31, poisson_sol, b3_tmp);
    VecAXPY(b3_tmp, -1.0, b3);
    VecScale(b3_tmp, -1.0);
    KSPSolve(shell->np2_diffusion_ksp, b3_tmp, np2_sol);


    PetscInt idx1[1], idx2[1], idx3[1];
    PetscScalar val1, val2, val3;
    for (int i=0; i<shell->size; i++) {
        idx1[0] = i;
        idx2[0] = i + 1*shell->size;
        idx3[0] = i + 2*shell->size;
        VecGetValues(poisson_sol, 1, &i, &val1);
        VecGetValues(np1_sol, 1, &i, &val2);
        VecGetValues(np2_sol, 1, &i, &val3);
        VecSetValue(y, idx1[0], val1, INSERT_VALUES);
        VecSetValue(y, idx2[0], val2, INSERT_VALUES);
        VecSetValue(y, idx3[0], val3, INSERT_VALUES);
    }

    return ierr;
}
PetscErrorCode PNP_SPDLower_UserPCDestroy(PC pc)
{
    PNP_SPDLower_UserPCContext *shell;
    PCShellGetContext(pc, (void**)&shell);
    KSPDestroy(&shell->poisson_ksp);
    KSPDestroy(&shell->np1_diffusion_ksp);
    KSPDestroy(&shell->np2_diffusion_ksp);
    MatDestroy(&shell->A11);
    MatDestroy(&shell->A22_diffusion);
    MatDestroy(&shell->A21);
    MatDestroy(&shell->A33_diffusion);
    MatDestroy(&shell->A31);
    PetscFree(shell);
    return 0;
}
PetscErrorCode Use_SPDLower_BMG_Prec(PC& pc, Mat& mat, Vec& b, int num_vars)
{
    printf("\n===> SPD-Lower Preconditioner for PNP equation\n");
    PetscErrorCode ierr;

    PNP_SPDLower_UserPCContext* shell;

    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);

    ierr = PNP_SPDLower_UserPCCreate(&shell); CHKERRQ(ierr);

    ierr = PCShellSetApply(pc, PNP_SPDLower_UserPCApply); CHKERRQ(ierr);

    ierr = PCShellSetContext(pc, shell); CHKERRQ(ierr);

    ierr = PCShellSetDestroy(pc, PNP_SPDLower_UserPCDestroy); CHKERRQ(ierr);

    ierr = PCShellSetName(pc, "SPD-Lower_Preconditioner_For_PNP"); CHKERRQ(ierr);

    PetscInt rowsize, colnsize, size; //size是PNP对角块矩阵的大小
    MatGetSize(mat, &rowsize, &colnsize);
    if (rowsize != colnsize) throw "Not a square-matrix";
    size = rowsize / num_vars;

    PetscInt *arr1, *arr2, *arr3;
    PetscMalloc1(size, &arr1);
    PetscMalloc1(size, &arr2);
    PetscMalloc1(size, &arr3);
    for (int i=0; i<size; i++) {
        arr1[i] = i;
        arr2[i] = i + size;
        arr3[i] = i + 2*size;
    }
    IS is1, is2, is3;
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr1, PETSC_COPY_VALUES, &is1);
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr2, PETSC_COPY_VALUES, &is2);
    ISCreateGeneral(PETSC_COMM_WORLD, size, arr3, PETSC_COPY_VALUES, &is3);

    ierr = PNP_SPDLower_UserPCSetUp(pc, mat, b, is1, is2, is3, size); CHKERRQ(ierr);

    return 0;
}




// -------------------------------- 一些辅助性函数 ------------------------------------------
void MatrixTranspose(double A[], int n)
{
    double tmp = 0;
    for (int i=0; i<n; i++)
        for (int j=0; j<i; j++)
        {
            tmp = A[j + i*n];
            A[j + i*n] = A[i + j*n];
            A[i + j*n] = tmp;
        }

}
void Test_MatrixTranspose()
{
    double A[9]={1,2,3,4,5,6,7,8,9};
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            std::cout << A[i*3 + j] << ", ";
        }
        std::cout << std::endl;
    }

    MatrixTranspose(A, 3);
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            std::cout << A[i*3 + j] << ", ";
        }
        std::cout << std::endl;
    }

}
void SelfMatrixInverse(double A[], int n, double C[])
{
    //参考:
    //https://www.cnblogs.com/xiaoxi666/p/6421228.html
    //http://www.voidcn.com/article/p-xicwlhqh-vy.html
    int i,j,k,m=2*n;
    double mik,temp;
    double **a=new double*[n];
    double **B=new double*[n];

    MatrixTranspose(A, n);

    for(i=0;i<n;i++)
    {
        a[i]=new double[2*n];
        B[i]=new double[n];
    }

    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            if(i==j)
                B[i][j]=1.0;
            else
                B[i][j]=0.0;
        }
    }        //初始化B=I

    int iii= 0;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            a[i][j]=A[iii];  //复制A到a，避免改变A的值. 可以看出A是按行存储的
            iii++;
        }
    }

    for(i=0;i<n;i++)
        for(j=n;j<m;j++)
            a[i][j]=B[i][j-n];  //复制B到a，增广矩阵

    for(k=1;k<=n-1;k++)
    {
        for(i=k+1;i<=n;i++)
        {
            mik=a[i-1][k-1]/a[k-1][k-1];
            for(j=k+1;j<=m;j++)
            {
                a[i-1][j-1]-=mik*a[k-1][j-1];
            }
        }
    }        //顺序高斯消去法化左下角为零

    for(i=1;i<=n;i++)
    {
        temp=a[i-1][i-1];
        for(j=1;j<=m;j++)
        {
            a[i-1][j-1]/=temp;
        }
    }        //归一化

    for(k=n-1;k>=1;k--)
    {
        for(i=k;i>=1;i--)
        {
            mik=a[i-1][k];
            for(j=k+1;j<=m;j++)
            {
                a[i-1][j-1]-=mik*a[k][j-1];
            }
        }
    }        //逆序高斯消去法化增广矩阵左边为单位矩阵

    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            B[i][j]=a[i][j+n];  //取出求逆结果

    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            if(fabs(B[i][j])<0.0001)
                B[i][j]=0.0;

    int jjj = 0;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            C[jjj] = B[i][j];
            jjj++;
        }
    }

    MatrixTranspose(C, n);
    delete []a;
    delete []B;
}

void TestMatrixInverse()
{
    double A[9] = {2, 3, 4, 5, -2, 1, 1, 2, 3};
    double Ainv[9]; //-0.1*{-8, -1, 11, -14, 2, 18, 12, -1, -19}
    SelfMatrixInverse(A, 3, Ainv);
    for (int i=0; i<9; i++)
        std::cout << -10*Ainv[i] << ", ";
    std::cout << std::endl;

    double B[9] = {1, 5, 2, 0, 3, 10, 1, 2, 1};
    double Binv[9]; //1/27 * {-17, -1, 44, 10, -1, -10, -3, 3, 3}
    SelfMatrixInverse(B, 3, Binv);
    for (int i=0; i<9; i++)
        std::cout << 27*Binv[i] << ", ";
    std::cout << std::endl;

}

void FormPermutationMatrix(const Vec& map, Mat& perm)
{
    PetscInt size;
    VecGetSize(map, &size);

    PetscInt* I; //row offsets
    PetscMalloc1(size+1, &I);
    for (int i=0; i<size+1; i++)
    {
        I[i] = i;
    }

    PetscInt* J;
    PetscMalloc1(size, &J);
    PetscScalar tmp;
    for (int j=0; j<size; j++)
    {
        VecGetValues(map, 1, &j, &tmp);
        J[j] = (PetscInt)tmp;
    }

    PetscScalar* Vals;
    PetscMalloc1(size, &Vals);
    for (int i=0; i<size; i++)
    {
        Vals[i] = 1.0; //所有非0元都是1
    }

    MatCreate(PETSC_COMM_WORLD, &perm);
    MatSetType(perm, MATAIJ); /* MATSEQAIJ, MATMPIAIJ, MATAIJ */
    MatSetSizes(perm, PETSC_DECIDE, PETSC_DECIDE, size, size);
    MatSetFromOptions(perm);
    MatSeqAIJSetPreallocation(perm, 1, NULL); //permutation matrix 每行只有1个非0元素. 加速Assemble过程
    MatSetUp(perm);

    for (int i=0; i<size; i++)
    {
        MatSetValues(perm, 1, &i, 1, &J[i], &Vals[i], INSERT_VALUES);
    }

    MatAssemblyBegin(perm, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(perm, MAT_FINAL_ASSEMBLY);
}

void TestFormPermutationMatrix()
{
    Vec dof2vet;
    Read_Vec_txt("../dof_to_vertex_map.txt", dof2vet);
    Mat permut;
    FormPermutationMatrix(dof2vet, permut);

    PetscInt size;
    VecGetSize(dof2vet, &size);

    Vec dummy, result1, result2; //dummy元素为: 1,2,3,...,size; 通过下面计算, result1, result2二者应该都是0
    VecCreate(PETSC_COMM_WORLD, &dummy);
    VecSetType(dummy, VECSEQ);
    VecSetSizes(dummy, PETSC_DECIDE, size);
    VecSetFromOptions(dummy);
    VecSetUp(dummy);
    VecDuplicate(dummy, &result1);
    VecDuplicate(dummy, &result2);

    PetscScalar tmp;
    for (int i=0; i<size; i++)
    {
        tmp = -i;
        VecSetValues(dummy, 1, &i, &tmp, INSERT_VALUES);
    }
    VecAssemblyBegin(dummy);
    VecAssemblyEnd(dummy);

    //上面的permutation矩阵将(1,2,...,size)变成dof2vet
    MatMultAdd(permut, dummy, dof2vet, result1);
    MatMultTransposeAdd(permut, dof2vet, dummy, result2);
    VecView(result1, PETSC_VIEWER_STDOUT_WORLD);
    VecView(result2, PETSC_VIEWER_STDOUT_WORLD);
}

void ABFDecouple(const Mat& mat, const Mat& permutation, const PetscInt num_vars, Mat& Dinv, IS& is)
{
    PetscInt mat_size, submat_size;
    MatGetSize(mat, &mat_size, NULL);
    submat_size = mat_size / num_vars; //PDEs中未知的物理量个数

    PetscInt *array; //自由度的排序reorder
    PetscMalloc1(submat_size, &array);
    for (int i=0; i<submat_size; i++) {
        array[i] = i;
    }
    ISCreateGeneral(PETSC_COMM_WORLD, submat_size, array, PETSC_COPY_VALUES, &is);

    Mat D, Dtilde, Dtilde_inv; //不同自由度编号下的解耦矩阵及其逆
    MatCreate(PETSC_COMM_WORLD, &Dinv);
    MatSetType(Dinv, MATAIJ); /* MATSEQAIJ, MATMPIAIJ, MATAIJ */
    MatSetSizes(Dinv, PETSC_DECIDE, PETSC_DECIDE, mat_size, mat_size);
    MatSetFromOptions(Dinv);
    MatSeqAIJSetPreallocation(Dinv, num_vars, NULL); //加速Assemble过程
    MatSetUp(Dinv);
    MatDuplicate(Dinv, MAT_DO_NOT_COPY_VALUES, &D);
    MatDuplicate(Dinv, MAT_DO_NOT_COPY_VALUES, &Dtilde);
    MatDuplicate(Dinv, MAT_DO_NOT_COPY_VALUES, &Dtilde_inv);


    double submat[num_vars*num_vars], submat_inv[num_vars*num_vars]; //每一个3*3子块矩阵和它的逆, 按行存储
    PetscInt *array1, *array2; //自由度的两种不同排序方式对应的子矩阵的行列编号
    PetscMalloc1(2*num_vars, &array1);
    PetscMalloc1(2*num_vars, &array2);
    for (int i=0; i<submat_size; i++)
    {
        for (int j=0; j<num_vars; j++)
        {
            array1[j] = i + j*submat_size;           //要抽取的子矩阵的行编号
            array1[j+num_vars] = i + j*submat_size;  //要抽取的子矩阵的列编号
            array2[j] = i*num_vars + j;                       //得到的子矩阵要放入到reorder后的大矩阵的行编号
            array2[j+num_vars] = i*num_vars + j;              //得到的子矩阵要放入到reorder后的大矩阵的列编号
        }

        MatGetValues(mat, num_vars, &array1[0], num_vars, &array1[0+num_vars], submat); //得到子矩阵
//        for (int i=0; i<2*num_vars; i++) std::cout << array1[i] << ", "; std::cout << std::endl;
//        for (int i=0; i<2*num_vars; i++) std::cout << array2[i] << ", "; std::cout << std::endl;
//        for (int i=0; i<num_vars*num_vars; i++) std::cout << submat[i] << ", "; std::cout << std::endl;

        MatSetValues(D, num_vars, &array1[0], num_vars, &array1[0+num_vars], submat, INSERT_VALUES);

        MatSetValues(Dtilde, num_vars, &array2[0], num_vars, &array2[0+num_vars], submat, INSERT_VALUES);

        SelfMatrixInverse(submat, num_vars, submat_inv);
        MatSetValues(Dtilde_inv, num_vars, &array2[0], num_vars, &array2[0+num_vars], submat_inv, INSERT_VALUES);
    }
    MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Dtilde, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Dtilde, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Dtilde_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Dtilde_inv, MAT_FINAL_ASSEMBLY);

//    MatPtAP(Dtilde_inv, permut, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Dinv); //P^t * \tilde(D)^{-1} * P
    MatRARt(Dtilde_inv, permutation, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Dinv); //P * \tilde(D)^{-1} * P^t, fff


    if (0) {
        Mat D_Dinv, Dtilde_Dtilde_inv; //下面的计算要使得这两个矩阵为单位矩阵
        MatDuplicate(mat, MAT_DO_NOT_COPY_VALUES, &D_Dinv);
        MatDuplicate(mat, MAT_DO_NOT_COPY_VALUES, &Dtilde_Dtilde_inv);

        MatMatMult(Dinv, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D_Dinv);
        MatView(D_Dinv, PETSC_VIEWER_STDOUT_WORLD);

        MatMatMult(Dtilde, Dtilde_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Dtilde_Dtilde_inv);
        MatView(Dtilde_Dtilde_inv, PETSC_VIEWER_STDOUT_WORLD);

        Write_Mat_Matlab_txt("Dtilde_inv.m", Dtilde_inv);
        Write_Mat_Matlab_txt("Dtilde.m", Dtilde);
        Write_Mat_Matlab_txt("Dinv.m", Dinv);
        Write_Mat_Matlab_txt("D.m", D);
        Write_Mat_Matlab_txt("A.m", mat);
    }
}
void Test_ABFDecouple()
{

    Mat A;
    Read_CSRMat_txt("../csr_spmat.txt", A);

    Vec dof2vet;
    Read_Vec_txt("../csr_spmat_dof2ver.txt", dof2vet);

    Mat permut;
    FormPermutationMatrix(dof2vet, permut);

    Vec tmp1, tmp2;
    VecDuplicate(dof2vet, &tmp1);
    VecDuplicate(dof2vet, &tmp2);

    MatMultTranspose(permut, dof2vet, tmp1);
    printf("tmp1: \n"); VecView(tmp1, PETSC_VIEWER_STDOUT_WORLD);

    MatMult(permut, tmp1, tmp2);
    printf("tmp2: \n"); VecView(tmp2, PETSC_VIEWER_STDOUT_WORLD);
    printf("dof2vet: \n"); VecView(dof2vet, PETSC_VIEWER_STDOUT_WORLD);

    Mat tmp3;
//    MatRARt(A, permut, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp3);
    MatMatMult(permut, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp3);
    printf("tmp3: \n"); MatView(tmp3, PETSC_VIEWER_STDOUT_WORLD);

    Mat tmp4;
    MatRARt(A, permut, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp4);
    printf("tmp4: \n"); MatView(tmp4, PETSC_VIEWER_STDOUT_WORLD);

    Mat Dinv;
    IS idx;
    ABFDecouple(A, permut, 3, Dinv, idx);

//    printf("A: \n"); MatView(A, PETSC_VIEWER_STDOUT_WORLD);
//    printf("D: \n"); MatView(D, PETSC_VIEWER_STDOUT_WORLD);
//    printf("Dtilde: \n"); MatView(Dtilde, PETSC_VIEWER_STDOUT_WORLD);
//    printf("tmp4: \n"); MatView(tmp4, PETSC_VIEWER_STDOUT_WORLD);
//    printf("Dtilde_inv: \n"); MatView(Dtilde_inv, PETSC_VIEWER_STDOUT_WORLD);
//    printf("Dinv: \n"); MatView(Dinv, PETSC_VIEWER_STDOUT_WORLD);

}

void UsePetscBuiltinPrec(PC& pc, const char* type)
{
    if (strcmp(type, "none") == 0)
    {
        printf("Do Not Use Preconditioner.\n");
        PCSetType(pc, PCNONE);
    }
    else if (strcmp(type, "lu") == 0)
    {
        printf("Use Built-in LU Preconditioner.\n");
        PCSetType(pc, PCLU);
    }
//    else if (strcmp(type, "boomeramg") == 0)
//    {
//        printf("Use Built-in Hypre Preconditioner.\n");
////        PCSetType(pc, PCHYPRE);
//        PCHYPRESetType(pc, "boomeramg"); //好像PETSc更新后不需要这个?
//    }
    else if (strcmp(type, "ilu") == 0)
    {
        printf("Use Built-in iLU Preconditioner.\n");
        PCSetType(pc, PCILU);
    }
//    else if (strcmp(type, "mg") == 0)
//    {
//        printf("Use Built-in AMG Preconditioner.\n");
//        PCSetType(pc, PCMG);
//    }
    else
        throw "Not supported Preconditioner.";
}


#endif //LEARN_MFEM_PETSC_UTILS_HPP