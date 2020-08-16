#include <iostream>
#include "mfem.hpp"
using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MFEMInitializePetsc(NULL, NULL, NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Mesh* mesh = new Mesh("./inline-tri-modify.mesh");
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    Array<int> vert;
    pmesh->GetElementVertices(1, vert);
    vert.Print(cout << "vert: ");

    DG_FECollection* dg_fec = new DG_FECollection(1, 2);
    ParFiniteElementSpace* dg_space = new ParFiniteElementSpace(pmesh, dg_fec);

    Array<int> dofs;
    dg_space->GetElementDofs(4, dofs);
    const Table& el2dof = dg_space->GetElementToDofTable();
    dofs.Print(cout << "dofs: ");
    el2dof.Print(cout << "element to dofs:\n");

    Array<int> null_array;

    ParBilinearForm* a = new ParBilinearForm(dg_space);
    a->AddDomainIntegrator(new MassIntegrator);
    a->Assemble();

    ParBilinearForm* b = new ParBilinearForm(dg_space);
    b->AddDomainIntegrator(new MassIntegrator);
    b->Assemble();

    ParBilinearForm* c = new ParBilinearForm(dg_space);
    c->AddDomainIntegrator(new MassIntegrator);
    c->Assemble();

    ParBilinearForm* d = new ParBilinearForm(dg_space);
    d->AddDomainIntegrator(new MassIntegrator);
    d->Assemble();

    {
        PetscParMatrix A, B, C, D;

        a->SetOperatorType(Operator::PETSC_MATAIJ);
        a->FormSystemMatrix(null_array, A);

        b->SetOperatorType(Operator::PETSC_MATAIJ);
        b->FormSystemMatrix(null_array, B);

        c->SetOperatorType(Operator::PETSC_MATAIJ);
        c->FormSystemMatrix(null_array, C);

        d->SetOperatorType(Operator::PETSC_MATAIJ);
        d->FormSystemMatrix(null_array, D);
    }

    ParLinearForm* f = new ParLinearForm(dg_space);
    ParLinearForm* g = new ParLinearForm(dg_space);

    delete pmesh;
    delete dg_fec, dg_space;
    delete a, b, c, d, f, g;

    MFEMFinalizePetsc();
    MPI_Finalize();
    return 0;
}