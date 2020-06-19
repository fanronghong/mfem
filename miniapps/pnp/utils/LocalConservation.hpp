//
// Created by fan on 2020/6/18.
//

#ifndef MFEM_LOCALCONSERVATION_HPP
#define MFEM_LOCALCONSERVATION_HPP

#include <iostream>
#include "mfem.hpp"
#include <cassert>
#include <random>
#include <numeric>
using namespace std;
using namespace mfem;


// \int_{\partial K} q grad(gf).n ds
void ComputeLocalConservation(Coefficient& q, const GridFunction& gf, Vector& error)
{
    const FiniteElementSpace* fes = gf.FESpace();
    Mesh* mesh = fes->GetMesh();
    const FiniteElement* fe;
    int dim = mesh->Dimension();

    Vector grad, nor;
    grad.SetSize(dim);
    nor.SetSize(dim);

    error.SetSize(fes->GetNE());
    error = 0.0;

    const Table& element2face = mesh->ElementToFaceTable();
    Array<int> e2f;

    for (int i=0; i<fes->GetNE(); ++i) // loop over for all elements
    {
        fe = fes->GetFE(i);

        element2face.GetRow(i, e2f);
        for (int j=0; j<e2f.Size(); ++j) // loop over for all edges for each element
        {
            FaceElementTransformations* Trans = mesh->GetFaceElementTransformations(e2f[j]);
            const IntegrationRule* ir = &(IntRules.Get(Trans->GetGeometryType(), 2*fe->GetOrder()+3));

            for (int k=0; k<ir->GetNPoints(); ++k)
            {
                const IntegrationPoint& ip = ir->IntPoint(k);
                Trans->SetIntPoint(&ip);

                gf.GetGradient(*Trans, grad); // error!
                double q_val = q.Eval(*Trans, ip);
                CalcOrtho(Trans->Jacobian(), nor);

                error[i] += ip.weight * q_val * (grad * nor);
            }
        }
    }
}


#endif //MFEM_LOCALCONSERVATION_HPP
