//
// Created by fan on 2019/11/1.
//

#ifndef LEARN_MFEM_MFEM2PETSC_HPP
#define LEARN_MFEM_MFEM2PETSC_HPP

#include "petscksp.h"
#include "petscpc.h"

#include <cassert>

#include "mfem.hpp"
using namespace mfem;
using namespace std;

// 将串行的Vector转换成串行Petsc的Vec
PetscErrorCode Vector2Vec(const Vector& x, Vec& y)
{
    // ref:PetscParVector::PetscParVector(MPI_Comm comm, const Vector &_x, bool copy)
    PetscErrorCode ierr;

    ierr = VecCreate(MPI_COMM_WORLD, &y); CHKERRQ(ierr);
    ierr = VecSetSizes(y, PETSC_DECIDE, x.Size()); CHKERRQ(ierr);
    ierr = VecSetType(y, VECSTANDARD); CHKERRQ(ierr);

    PetscScalar* array;
    ierr = VecGetArray(y, &array); CHKERRQ(ierr);
    for (int i=0; i<x.Size(); i++) { array[i] = x[i]; }
    ierr = VecRestoreArray(y, &array); CHKERRQ(ierr);
    return ierr;
}
PetscErrorCode Vec2Vector(const Vec& x, Vector& y)
{
    PetscErrorCode ierr;

    PetscScalar* array;
    ierr = VecGetArray(x, &array); CHKERRQ(ierr);
    int size;
    VecGetSize(x, &size);
    y.SetDataAndSize(array, size);
    return ierr;
}
void Test_Vector2Vec()
{
    Vector x(1000);
    for (int i=0; i<x.Size(); i++) x[i] = rand()%10;

    Vec y;
    Vector2Vec(x, y);
    Vector z;
    Vec2Vector(y, z);

//    x.Print(cout << "x: \n");
//    VecView(y, PETSC_VIEWER_STDOUT_WORLD);

    PetscScalar* arr;
    VecGetArray(y, &arr);
    for (int i=0; i<x.Size(); i++)
    {
//        cout << i << "-th: " << x[i] << ", " << arr[i] << ", " << z[i] << endl;
        assert(abs(arr[i] - x[i]) < 1E-10);
        assert(abs(x[i] - z[i]) < 1E-10);
    }
}


// 将串行的SparseMatrix转换成串行Petsc的Mat.
PetscErrorCode SparseMatrix2Mat(const SparseMatrix& x, Mat& y) // 有bug
{
    // ref: PetscParMatrix(const SparseMatrix *sa, Operator::Type tid)
    const_cast<SparseMatrix&>(x).SortColumnIndices();

    PetscErrorCode ierr;
    int size = x.Height();
    assert(x.Width() == size);

    PetscScalar* pVals;
    PetscInt *pI, *pJ;

    const int* pi = x.GetI();
    const int* pj = x.GetJ();
    const double* pvals = x.GetData();

    ierr = PetscMalloc1(size+1, &pI); CHKERRQ(ierr);
    ierr = PetscMalloc1(pi[size], &pJ); CHKERRQ(ierr);
    ierr = PetscMalloc1(pi[size], &pVals); CHKERRQ(ierr);
    pI[0] = pi[0];
    for (int i=0; i<size; i++)
    {
        pI[i+1] = pi[i+1];
        for (int j=pi[i]; j<pi[i+1]; j++)
        {
            pJ[j] = pj[j];
            pVals[j] = pvals[j];
        }
    }

    ierr = MatCreate(PETSC_COMM_WORLD, &y); CHKERRQ(ierr);
    ierr = MatSetType(y, MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, size, size); CHKERRQ(ierr);

    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, size, size, pI, pJ, pVals, &y); CHKERRQ(ierr);
    return ierr;
}
PetscErrorCode Mat2SparseMatrix(const Mat& x, SparseMatrix& y)
{
    PetscErrorCode ierr;

    int size, size_;
    MatGetSize(x, &size, &size_);
    assert(size == size_);

    // goon fff


}
void Test_SparseMatrix2Mat()
{
    int* I = new int[5] {0, 2, 3, 5, 6};
    int* J = new int[6] {0, 2, 1, 0, 2, 0};
    double* Vals = new double[6] {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};

    SparseMatrix sp(I, J, Vals, 4, 4);

    Mat mat;
    SparseMatrix2Mat(sp, mat);

//    sp.Print(cout << "SparseMatrix: \n");
//    MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
//    MatView(mat, PETSC_VIEWER_STDOUT_SELF);
}




int Test_MFEM2PETSc()
{
    PetscErrorCode ierr = PetscInitialize(NULL, NULL, (char*)0, NULL); CHKERRQ(ierr);

    Test_Vector2Vec();
    Test_SparseMatrix2Mat();

    cout << "===> Test Pass: MFEM2PETSc.hpp" << endl;
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return ierr;
}


#endif //LEARN_MFEM_MFEM2PETSC_HPP
