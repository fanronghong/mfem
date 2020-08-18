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

//    DG_FECollection* fec = new DG_FECollection(1, 2);
    H1_FECollection* fec = new H1_FECollection(1, 2);
    ParFiniteElementSpace* dg_space = new ParFiniteElementSpace(pmesh, fec);

    // Obtain DOFs indices of mesh that belong in elements with attribute number = 2
    Array<int> all_dofs, attr2_dofs;
    for (int iel=0; iel<pmesh->GetNE(); ++iel)
    {
        int attr = dg_space->GetAttribute(iel);
        Array<int> dofs;
        dg_space->GetElementDofs(iel, dofs);
        for (auto itm: dofs) {
            all_dofs.Append(dg_space->GetGlobalTDofNumber(itm));
        }

        if (attr != 2) continue;
        Array<int> elem_dofs;
        dg_space->GetElementDofs(iel, elem_dofs);
        for (auto itm: elem_dofs) {
            attr2_dofs.Append(dg_space->GetGlobalTDofNumber(itm));
        }
    }
    attr2_dofs.Sort();
    attr2_dofs.Unique();

    if (myid == 0) {
        all_dofs.Print(cout << "All DOFs indices:\n");
        attr2_dofs.Print(cout << "DOFs indices of mesh that belong in elements with attribute number=2:\n");
    }

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

        PetscParMatrix tilde_B(B, all_dofs, attr2_dofs);
        PetscParMatrix tilde_C(C, attr2_dofs, all_dofs);
        PetscParMatrix tilde_D(D, attr2_dofs, attr2_dofs);

        tilde_B.Print("B");
        tilde_C.Print("C");
        tilde_D.Print("D");
    }

    ParLinearForm* f = new ParLinearForm(dg_space);
    ParLinearForm* g = new ParLinearForm(dg_space);

    delete pmesh;
    delete fec, dg_space;
    delete a, b, c, d, f, g;

    MFEMFinalizePetsc();
    MPI_Finalize();
    return 0;
}