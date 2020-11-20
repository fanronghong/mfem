#ifndef MFEM_PNP_PROTEIN_PRECONDITIONERS_HPP
#define MFEM_PNP_PROTEIN_PRECONDITIONERS_HPP


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

        for (int i=0; i<3; ++i) // solve 3 equations
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

class LowerBlockPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    LowerBlockPreconditionerSolver(const OperatorHandle& oh): Solver()
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
    }
    virtual ~LowerBlockPreconditionerSolver()
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
        /* [ A C1 C2
         *  B1 A1
         *  B2    A2 ]
         *  use lower triangular preconditioner
         *  */
        Vec x1, x2, x3, y1, y2, y3;
        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc，不要修改X
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &x1); // 不要修改x1,x2,x3
        VecGetSubVector(*X, index_set[1], &x2);
        VecGetSubVector(*X, index_set[2], &x3);

        VecGetSubVector(*Y, index_set[0], &y1);
        VecGetSubVector(*Y, index_set[1], &y2);
        VecGetSubVector(*Y, index_set[2], &y3);

        KSPSolve(kspblock[0], x1, y1); // solve A y1 = x1

        Vec B1_y1, x2_;
        VecDuplicate(x2, &B1_y1);
        VecDuplicate(x2, &x2_);
        VecCopy(x2, x2_);
        MatMult(sub[1][0], y1, B1_y1);
        VecAXPY(x2_, -1.0, B1_y1); // x2 - B1 y1 -> x2
        KSPSolve(kspblock[1], x2_, y2); // solve A1 y2 = x2 - B1 y1
        VecDestroy(&B1_y1);
        VecDestroy(&x2_);

        Vec B2_y1, x3_;
        VecDuplicate(x3, &B2_y1);
        VecDuplicate(x3, &x3_);
        VecCopy(x3, x3_);
        MatMult(sub[2][0], y1, B2_y1);
        VecAXPY(x3_, -1.0, B2_y1); // x3 - B2 y1 -> x3
        KSPSolve(kspblock[2], x3_, y3); // solve A2 y3 = x3 - B2 y1
        VecDestroy(&B2_y1);
        VecDestroy(&x3_);

        VecRestoreSubVector(*X, index_set[0], &x1);
        VecRestoreSubVector(*X, index_set[1], &x2);
        VecRestoreSubVector(*X, index_set[2], &x3);

        VecRestoreSubVector(*Y, index_set[0], &y1);
        VecRestoreSubVector(*Y, index_set[1], &y2);
        VecRestoreSubVector(*Y, index_set[2], &y3);

        X->ResetArray();
        Y->ResetArray();
    }
};

class UpperBlockPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    UpperBlockPreconditionerSolver(const OperatorHandle& oh): Solver()
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
    }
    virtual ~UpperBlockPreconditionerSolver()
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
        Vec x1, x2, x3, y1, y2, y3;
        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &x1);
        VecGetSubVector(*X, index_set[1], &x2);
        VecGetSubVector(*X, index_set[2], &x3);

        VecGetSubVector(*Y, index_set[0], &y1);
        VecGetSubVector(*Y, index_set[1], &y2);
        VecGetSubVector(*Y, index_set[2], &y3);

        KSPSolve(kspblock[1], x2, y2); // solve A1 y2 = x2
        KSPSolve(kspblock[2], x3, y3); // solve A2 y3 = x3

        Vec C1_y2, C2_y3, x1_;
        VecDuplicate(x1, &C1_y2);
        VecDuplicate(x1, &C2_y3);
        VecDuplicate(x1, &x1_); // 避免下面改变x1的值
        VecCopy(x1, x1_);

        MatMult(sub[0][1], y2, C1_y2);
        MatMult(sub[0][2], y3, C2_y3);
        VecAXPY(x1_, -1.0, C1_y2);
        VecAXPY(x1_, -1.0, C2_y3);
        KSPSolve(kspblock[0], x1, y1);

        VecDestroy(&C1_y2);
        VecDestroy(&C2_y3);
        VecDestroy(&x1_);

        VecRestoreSubVector(*X, index_set[0], &x1);
        VecRestoreSubVector(*X, index_set[1], &x2);
        VecRestoreSubVector(*X, index_set[2], &x3);

        VecRestoreSubVector(*Y, index_set[0], &y1);
        VecRestoreSubVector(*Y, index_set[1], &y2);
        VecRestoreSubVector(*Y, index_set[2], &y3);

        X->ResetArray();
        Y->ResetArray();
    }
};

class BlockSchurPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub, schur;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y
    Vec diag1, diag2;

public:
    BlockSchurPreconditionerSolver(const OperatorHandle& oh): Solver()
    {
        /* [ A C1 C2
         *  B1 A1
         *  B2    A2 ]
         *  use
         *  [ schur
         *          A1
         *              A2 ] for preconditioner
         *  schur = A - C1 A1^-1 B1 - C2 A2^-1 B2
         *  */
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

        MatCreateVecs(sub[1][1], &diag1, NULL);
        MatCreateVecs(sub[2][2], &diag2, NULL);
        MatGetDiagonal(sub[1][1], diag1); // diagonal of A1
        MatGetDiagonal(sub[2][2], diag2); // diagonal of A2
        VecReciprocal(diag1); // diag(A1)^-1 => diag1
        VecReciprocal(diag2); // diag(A2)^-1 => diag2

        Mat temp1, temp2;
        MatDuplicate(sub[0][1], MAT_COPY_VALUES, &temp1); // C1 => temp1
        MatDuplicate(sub[0][2], MAT_COPY_VALUES, &temp2); // C2 => temp2
        MatDiagonalScale(temp1, NULL, diag1); // C1 diag(A1)^-1 => temp1
        MatDiagonalScale(temp2, NULL, diag2); // C2 diag(A2)^-1 => temp2

        MatDuplicate(sub[0][0], MAT_COPY_VALUES, &schur); // A => schur
        Mat temp3, temp4;
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp3);
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp4);
        MatMatMult(temp1, sub[1][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); // C1 diag(A1)^-1 B1 => temp3
        MatMatMult(temp2, sub[2][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp4); // C2 diag(A2)^-1 B2 => temp4
        double schur_alpha1=1, schur_alpha2=1.0; // fff 为了能够通过编译
        MatAXPY(schur, -1.0*schur_alpha1, temp3, DIFFERENT_NONZERO_PATTERN); // A - alpha C1 diag(A1)^-1 B1 => schur
        MatAXPY(schur, -1.0*schur_alpha2, temp4, DIFFERENT_NONZERO_PATTERN); // A - alpha C1 diag(A1)^-1 B1 - C2 diag(A2)^-1 B2 => schur

        MatDestroy(&temp1); MatDestroy(&temp2); MatDestroy(&temp3); MatDestroy(&temp4);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[0]); PCHKERRQ(kspblock[0], ierr);
        ierr = KSPSetOperators(kspblock[0], schur, schur); PCHKERRQ(schur, ierr);
        KSPAppendOptionsPrefix(kspblock[0], "sub_block1_");
        KSPSetFromOptions(kspblock[0]);
        KSPSetUp(kspblock[0]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[1]); PCHKERRQ(kspblock[1], ierr);
        ierr = KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]); PCHKERRQ(sub[1][1], ierr);
        KSPAppendOptionsPrefix(kspblock[1], "sub_block2_");
        KSPSetFromOptions(kspblock[1]);
        KSPSetUp(kspblock[1]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[2]); PCHKERRQ(kspblock[2], ierr);
        ierr = KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]); PCHKERRQ(sub[2][2], ierr);
        KSPAppendOptionsPrefix(kspblock[2], "sub_block3_");
        KSPSetFromOptions(kspblock[2]);
        KSPSetUp(kspblock[2]);
    }
    virtual ~BlockSchurPreconditionerSolver()
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

class LowerBlockSchurPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub, schur;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y
    Vec diag1, diag2;

public:
    LowerBlockSchurPreconditionerSolver(const OperatorHandle& oh): Solver()
    {
        /* [ A C1 C2
         *  B1 A1
         *  B2    A2 ]
         *  use
         *  [ schur
         *          A1
         *              A2 ] for preconditioner
         *  schur = A - C1 A1^-1 B1 - C2 A2^-1 B2
         *  */
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

        MatCreateVecs(sub[1][1], &diag1, NULL);
        MatCreateVecs(sub[2][2], &diag2, NULL);
        MatGetDiagonal(sub[1][1], diag1); // diagonal of A1
        MatGetDiagonal(sub[2][2], diag2); // diagonal of A2
        VecReciprocal(diag1); // diag(A1)^-1 => diag1
        VecReciprocal(diag2); // diag(A2)^-1 => diag2

        Mat temp1, temp2;
        MatDuplicate(sub[0][1], MAT_COPY_VALUES, &temp1); // C1 => temp1
        MatDuplicate(sub[0][2], MAT_COPY_VALUES, &temp2); // C2 => temp2
        MatDiagonalScale(temp1, NULL, diag1); // C1 diag(A1)^-1 => temp1
        MatDiagonalScale(temp2, NULL, diag2); // C2 diag(A2)^-1 => temp2

        MatDuplicate(sub[0][0], MAT_COPY_VALUES, &schur); // A => schur
        Mat temp3, temp4;
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp3);
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp4);
        MatMatMult(temp1, sub[1][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); // C1 diag(A1)^-1 B1 => temp3
        MatMatMult(temp2, sub[2][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp4); // C2 diag(A2)^-1 B2 => temp4
        MatAXPY(schur, -1.0, temp3, DIFFERENT_NONZERO_PATTERN); // A - C1 diag(A1)^-1 B1 => schur
        MatAXPY(schur, -1.0, temp4, DIFFERENT_NONZERO_PATTERN); // A - C1 diag(A1)^-1 B1 - C2 diag(A2)^-1 B2 => schur

        MatDestroy(&temp1); MatDestroy(&temp2); MatDestroy(&temp3); MatDestroy(&temp4);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[0]); PCHKERRQ(kspblock[0], ierr);
        ierr = KSPSetOperators(kspblock[0], schur, schur); PCHKERRQ(schur, ierr);
        KSPAppendOptionsPrefix(kspblock[0], "sub_block1_");
        KSPSetFromOptions(kspblock[0]);
        KSPSetUp(kspblock[0]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[1]); PCHKERRQ(kspblock[1], ierr);
        ierr = KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]); PCHKERRQ(sub[1][1], ierr);
        KSPAppendOptionsPrefix(kspblock[1], "sub_block2_");
        KSPSetFromOptions(kspblock[1]);
        KSPSetUp(kspblock[1]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[2]); PCHKERRQ(kspblock[2], ierr);
        ierr = KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]); PCHKERRQ(sub[2][2], ierr);
        KSPAppendOptionsPrefix(kspblock[2], "sub_block3_");
        KSPSetFromOptions(kspblock[2]);
        KSPSetUp(kspblock[2]);
    }
    virtual ~LowerBlockSchurPreconditionerSolver()
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
        Vec x1, x2, x3, y1, y2, y3;
        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc，不要修改X
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &x1); // 不要修改x1,x2,x3
        VecGetSubVector(*X, index_set[1], &x2);
        VecGetSubVector(*X, index_set[2], &x3);

        VecGetSubVector(*Y, index_set[0], &y1);
        VecGetSubVector(*Y, index_set[1], &y2);
        VecGetSubVector(*Y, index_set[2], &y3);

        KSPSolve(kspblock[0], x1, y1); // solve Schur y1 = x1

        Vec B1_y1, x2_;
        VecDuplicate(x2, &B1_y1);
        VecDuplicate(x2, &x2_);
        VecCopy(x2, x2_);
        MatMult(sub[1][0], y1, B1_y1);
        VecAXPY(x2_, -1.0, B1_y1); // x2 - B1 y1 -> x2
        KSPSolve(kspblock[1], x2_, y2); // solve A1 y2 = x2 - B1 y1
        VecDestroy(&B1_y1);
        VecDestroy(&x2_);

        Vec B2_y1, x3_;
        VecDuplicate(x3, &B2_y1);
        VecDuplicate(x3, &x3_);
        VecCopy(x3, x3_);
        MatMult(sub[2][0], y1, B2_y1);
        VecAXPY(x3_, -1.0, B2_y1); // x3 - B2 y1 -> x3
        KSPSolve(kspblock[2], x3_, y3); // solve A2 y3 = x3 - B2 y1
        VecDestroy(&B2_y1);
        VecDestroy(&x3_);

        VecRestoreSubVector(*X, index_set[0], &x1);
        VecRestoreSubVector(*X, index_set[1], &x2);
        VecRestoreSubVector(*X, index_set[2], &x3);

        VecRestoreSubVector(*Y, index_set[0], &y1);
        VecRestoreSubVector(*Y, index_set[1], &y2);
        VecRestoreSubVector(*Y, index_set[2], &y3);

        X->ResetArray();
        Y->ResetArray();
    }
};

class UpperBlockSchurPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub, schur;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y
    Vec diag1, diag2;

public:
    UpperBlockSchurPreconditionerSolver(const OperatorHandle& oh): Solver()
    {
        /* [ A C1 C2
         *  B1 A1
         *  B2    A2 ]
         *  use
         *  [ schur
         *          A1
         *              A2 ] for preconditioner
         *  schur = A - C1 A1^-1 B1 - C2 A2^-1 B2
         *  */
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

        MatCreateVecs(sub[1][1], &diag1, NULL);
        MatCreateVecs(sub[2][2], &diag2, NULL);
        MatGetDiagonal(sub[1][1], diag1); // diagonal of A1
        MatGetDiagonal(sub[2][2], diag2); // diagonal of A2
        VecReciprocal(diag1); // diag(A1)^-1 => diag1
        VecReciprocal(diag2); // diag(A2)^-1 => diag2

        Mat temp1, temp2;
        MatDuplicate(sub[0][1], MAT_COPY_VALUES, &temp1); // C1 => temp1
        MatDuplicate(sub[0][2], MAT_COPY_VALUES, &temp2); // C2 => temp2
        MatDiagonalScale(temp1, NULL, diag1); // C1 diag(A1)^-1 => temp1
        MatDiagonalScale(temp2, NULL, diag2); // C2 diag(A2)^-1 => temp2

        MatDuplicate(sub[0][0], MAT_COPY_VALUES, &schur); // A => schur
        Mat temp3, temp4;
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp3);
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp4);
        MatMatMult(temp1, sub[1][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); // C1 diag(A1)^-1 B1 => temp3
        MatMatMult(temp2, sub[2][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp4); // C2 diag(A2)^-1 B2 => temp4
        MatAXPY(schur, -1.0, temp3, DIFFERENT_NONZERO_PATTERN); // A - C1 diag(A1)^-1 B1 => schur
        MatAXPY(schur, -1.0, temp4, DIFFERENT_NONZERO_PATTERN); // A - C1 diag(A1)^-1 B1 - C2 diag(A2)^-1 B2 => schur

        MatDestroy(&temp1); MatDestroy(&temp2); MatDestroy(&temp3); MatDestroy(&temp4);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[0]); PCHKERRQ(kspblock[0], ierr);
        ierr = KSPSetOperators(kspblock[0], schur, schur); PCHKERRQ(schur, ierr);
        KSPAppendOptionsPrefix(kspblock[0], "sub_block1_");
        KSPSetFromOptions(kspblock[0]);
        KSPSetUp(kspblock[0]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[1]); PCHKERRQ(kspblock[1], ierr);
        ierr = KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]); PCHKERRQ(sub[1][1], ierr);
        KSPAppendOptionsPrefix(kspblock[1], "sub_block2_");
        KSPSetFromOptions(kspblock[1]);
        KSPSetUp(kspblock[1]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[2]); PCHKERRQ(kspblock[2], ierr);
        ierr = KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]); PCHKERRQ(sub[2][2], ierr);
        KSPAppendOptionsPrefix(kspblock[2], "sub_block3_");
        KSPSetFromOptions(kspblock[2]);
        KSPSetUp(kspblock[2]);
    }
    virtual ~UpperBlockSchurPreconditionerSolver()
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
        Vec x1, x2, x3, y1, y2, y3;
        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc，不要修改X
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &x1); // 不要修改x1,x2,x3
        VecGetSubVector(*X, index_set[1], &x2);
        VecGetSubVector(*X, index_set[2], &x3);

        VecGetSubVector(*Y, index_set[0], &y1);
        VecGetSubVector(*Y, index_set[1], &y2);
        VecGetSubVector(*Y, index_set[2], &y3);

        KSPSolve(kspblock[1], x2, y2); // solve A1 y2 = x2
        KSPSolve(kspblock[2], x3, y3); // solve A2 y3 = x3

        Vec C1_y2, C2_y3, x4;
        VecDuplicate(x1, &C1_y2);
        VecDuplicate(x1, &x4);
        VecDuplicate(x1, &C2_y3);

        MatMult(sub[0][1], y2, C1_y2);
        MatMult(sub[0][2], y3, C2_y3);
        VecAXPY(x4, -1.0, C1_y2); // x1 - C1 y2 -> x4
        VecAXPY(x4, -1.0, C2_y3); // x1 - C1 y2 - C2 y3 -> x4
        KSPSolve(kspblock[0], x4, y1); // solve Schur y1 = x1 - C1 y2 - C2 y3
        VecDestroy(&C1_y2);
        VecDestroy(&C2_y3);
        VecDestroy(&x4);

        VecRestoreSubVector(*X, index_set[0], &x1);
        VecRestoreSubVector(*X, index_set[1], &x2);
        VecRestoreSubVector(*X, index_set[2], &x3);

        VecRestoreSubVector(*Y, index_set[0], &y1);
        VecRestoreSubVector(*Y, index_set[1], &y2);
        VecRestoreSubVector(*Y, index_set[2], &y3);

        X->ResetArray();
        Y->ResetArray();
    }
};

class UzawaPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    UzawaPreconditionerSolver(const OperatorHandle& oh): Solver()
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
    }
    virtual ~UzawaPreconditionerSolver()
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
        Vec block_phi_k, block_c1_k, block_c2_k;
        Vec block_phi, block_c1, block_c2;

        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &block_phi_k); // known
        VecGetSubVector(*X, index_set[1], &block_c1_k);
        VecGetSubVector(*X, index_set[2], &block_c2_k);
        VecGetSubVector(*Y, index_set[0], &block_phi); // unknown
        VecGetSubVector(*Y, index_set[1], &block_c1);
        VecGetSubVector(*Y, index_set[2], &block_c2);

        KSPSolve(kspblock[1], block_c1_k, block_c1); // solve A1 y_{c1} = x_{c1}
        KSPSolve(kspblock[2], block_c2_k, block_c2); // solve A2 y_{c2} = x_{c2}

        Vec temp1, temp2;
        VecDuplicate(block_phi_k, &temp1);
        VecDuplicate(block_phi_k, &temp2);

        MatMult(sub[0][1], block_c1, temp1); // B1 c1
        MatMult(sub[0][2], block_c2, temp2); // B2 c2
        VecAXPY(block_phi_k, -1.0, temp1); // f - B1 c1
        VecAXPY(block_phi_k, -1.0, temp2); // f - B1 c1 - B2 c2
        VecScale(block_phi_k, 0.1); // 0.2 * (f - B1 c1 - B2 c2)，做一个scale

//        KSPSolve(kspblock[0], block_phi_k, block_phi); // solve A y_{phi} = x_{phi}

        VecRestoreSubVector(*X, index_set[0], &block_phi_k);
        VecRestoreSubVector(*X, index_set[1], &block_c1_k);
        VecRestoreSubVector(*X, index_set[2], &block_c2_k);
        VecRestoreSubVector(*Y, index_set[0], &block_phi);
        VecRestoreSubVector(*Y, index_set[1], &block_c1);
        VecRestoreSubVector(*Y, index_set[2], &block_c2);

        X->ResetArray();
        Y->ResetArray();
    }
};

class SIMPLEPreconditionerSolver: public Solver
{
private:
    IS index_set[3];
    Mat **sub;
    Mat schur;
    KSP kspblock[3];
    mutable PetscParVector *X, *Y; // Create PetscParVectors as placeholders X and Y

public:
    SIMPLEPreconditionerSolver(const OperatorHandle& oh): Solver()
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

        Vec diag1, diag2;
        VecCreate(PETSC_COMM_WORLD, &diag1);
        VecSetSizes(diag1, PETSC_DECIDE, (PetscInt)(width/3));
        VecSetFromOptions(diag1);
        VecDuplicate(diag1, &diag2);
        MatGetDiagonal(sub[1][1], diag1); // diagonal of A1
        MatGetDiagonal(sub[2][2], diag2); // diagonal of A2
        VecReciprocal(diag1); // diag(A1)^-1 => diag1
        VecReciprocal(diag2); // diag(A2)^-1 => diag2

        Mat temp1, temp2;
        MatDuplicate(sub[0][1], MAT_COPY_VALUES, &temp1); // B1 => temp1
        MatDuplicate(sub[0][2], MAT_COPY_VALUES, &temp2); // B2 => temp2
        MatDiagonalScale(temp1, NULL, diag1); // B1 diag(A1)^-1 => temp1
        MatDiagonalScale(temp2, NULL, diag2); // B2 diag(A2)^-1 => temp2

        MatDuplicate(sub[0][0], MAT_COPY_VALUES, &schur); // A => schur
        Mat temp3, temp4;
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp3);
        MatDuplicate(schur, MAT_DO_NOT_COPY_VALUES, &temp4);
        MatMatMult(temp1, sub[1][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); // B1 diag(A1)^-1 C1 => temp3
        MatMatMult(temp2, sub[2][0], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp4); // B2 diag(A2)^-1 C2 => temp4
        MatAXPY(schur, -1.0, temp3, DIFFERENT_NONZERO_PATTERN); // A - B1 diag(A1)^-1 C1 => schur
        MatAXPY(schur, -1.0, temp4, DIFFERENT_NONZERO_PATTERN); // A - B1 diag(A1)^-1 C1 - B2 diag(A2)^-1 => schur

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[0]); PCHKERRQ(kspblock[0], ierr);
        ierr = KSPSetOperators(kspblock[0], schur, schur); PCHKERRQ(schur, ierr);
        KSPAppendOptionsPrefix(kspblock[0], "sub_block1_");
        KSPSetFromOptions(kspblock[0]);
        KSPSetUp(kspblock[0]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[1]); PCHKERRQ(kspblock[1], ierr);
        ierr = KSPSetOperators(kspblock[1], sub[1][1], sub[1][1]); PCHKERRQ(sub[1][1], ierr);
        KSPAppendOptionsPrefix(kspblock[1], "sub_block2_");
        KSPSetFromOptions(kspblock[1]);
        KSPSetUp(kspblock[1]);

        ierr = KSPCreate(MPI_COMM_WORLD, &kspblock[2]); PCHKERRQ(kspblock[2], ierr);
        ierr = KSPSetOperators(kspblock[2], sub[2][2], sub[2][2]); PCHKERRQ(sub[2][2], ierr);
        KSPAppendOptionsPrefix(kspblock[2], "sub_block3_");
        KSPSetFromOptions(kspblock[2]);
        KSPSetUp(kspblock[2]);

    }
    virtual ~SIMPLEPreconditionerSolver()
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
        Vec phi_k, c1_k, c2_k;
        Vec phi, c1, c2;

        X->PlaceArray(x.GetData()); // no copy, only the data pointer is passed to PETSc
        Y->PlaceArray(y.GetData());

        VecGetSubVector(*X, index_set[0], &phi_k); // known
        VecGetSubVector(*X, index_set[1], &c1_k);
        VecGetSubVector(*X, index_set[2], &c2_k);
        VecGetSubVector(*Y, index_set[0], &phi); // unknown
        VecGetSubVector(*Y, index_set[1], &c1);
        VecGetSubVector(*Y, index_set[2], &c2);

        KSPSolve(kspblock[1], c1_k, c1); // solve A1 c1^* = c1_k
        KSPSolve(kspblock[2], c2_k, c2); // solve A2 c2^*  c2_k

        Vec temp1, temp2;
        VecDuplicate(phi_k, &temp1);
        VecDuplicate(phi_k, &temp2);

        MatMult(sub[0][1], c1, temp1); // temp1: B1 c1
        MatMult(sub[0][2], c2, temp2); // temp2: B2 c2
        VecAXPY(phi_k, -1.0, temp1); // f: f - B1 c1
        VecAXPY(phi_k, -1.0, temp2); // f: f - B1 c1 - B2 c2

        KSPSolve(kspblock[0], phi_k, phi); // solve S y_{phi} = x_{phi}, S is schur complement

        Vec delta_c1, delta_c2, neg_C1_delta_phi, neg_C2_delta_phi;
        VecDuplicate(c1, &delta_c1);
        VecDuplicate(c2, &delta_c2);
        VecDuplicate(c1, &neg_C1_delta_phi);
        VecDuplicate(c2, &neg_C2_delta_phi);

        MatMult(sub[1][0], phi, neg_C1_delta_phi); // C1 delta_phi
        VecScale(neg_C1_delta_phi, -1.0); // - C1 delta_phi
        KSPSolve(kspblock[1], neg_C1_delta_phi, delta_c1); // solve delta_c1

        MatMult(sub[2][0], phi, neg_C2_delta_phi); // C2 delta_phi
        VecScale(neg_C2_delta_phi, -1.0); // - C2 delta_phi
        KSPSolve(kspblock[2], neg_C2_delta_phi, delta_c2); // solve delta_c2

        VecAXPY(c1, 1.0, delta_c1); // update c1: c1 + delta_c1
        VecAYPX(c2, 1.0, delta_c2); // update c2: c2 + delta_c2

        VecRestoreSubVector(*X, index_set[0], &phi_k);
        VecRestoreSubVector(*X, index_set[1], &c1_k);
        VecRestoreSubVector(*X, index_set[2], &c2_k);
        VecRestoreSubVector(*Y, index_set[0], &phi);
        VecRestoreSubVector(*Y, index_set[1], &c1);
        VecRestoreSubVector(*Y, index_set[2], &c2);

        X->ResetArray();
        Y->ResetArray();
//        cout << "in BlockPreconditionerSolver::Mult(), l2 norm y after: " << y.Norml2() << endl;
//        MFEM_ABORT("in BlockPreconditionerSolver::Mult()");
    }
};



class PreconditionerFactory: public PetscPreconditionerFactory
{
private:
    const Operator& op; // op就是Nonlinear Operator(可用来计算Residual, Jacobian)
    string name;

public:
    PreconditionerFactory(const Operator& op_, const string& name_)
            : PetscPreconditionerFactory(name_), op(op_), name(name_) {}
    virtual ~PreconditionerFactory() {}

    virtual Solver* NewPreconditioner(const OperatorHandle& oh) // oh就是当前Newton迭代步的Jacobian的句柄
    {
        if (name == "block")
            return new BlockPreconditionerSolver(oh); // block preconditioner
        else if (name == "uzawa")
            return new UzawaPreconditionerSolver(oh); // uzawa preconditioner
        else if (name == "simple")
            return new SIMPLEPreconditionerSolver(oh); // simple preconditioner
        else if(name == "lower")
            return new LowerBlockPreconditionerSolver(oh); // lower triangular
        else if(name == "upper")
            return new UpperBlockPreconditionerSolver(oh); // upper triangular
        else if (name == "blockschur")
            return new BlockSchurPreconditionerSolver(oh); // block schur
        else if (name == "lowerblockschur")
            return new LowerBlockSchurPreconditionerSolver(oh); // lower block schur
        else if (name == "upperblockschur")
            return new UpperBlockSchurPreconditionerSolver(oh); // upper block schur
    }
};


#endif //MFEM_PNP_PROTEIN_PRECONDITIONERS_HPP
