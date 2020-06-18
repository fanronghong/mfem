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
    const FiniteElement* fe;
    ElementTransformation* T;
    int dim = fes->GetMesh()->Dimension();

    Vector grad, nor;
    grad.SetSize(dim);
    nor.SetSize(dim);
    double q_val = 0.0;

    error.SetSize(fes->GetNE());
    error = 0.0;

    for (int i=0; i<fes->GetNE(); ++i)
    {
        fe = fes->GetFE(i);
        const IntegrationRule* ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder()+3));
        T = fes->GetElementTransformation(i);

        for (int j=0; j<ir->GetNPoints(); ++j)
        {
            const IntegrationPoint& ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            gf.GetGradient(*T, grad);
            q_val = q.Eval(*T, ip);
            CalcOrtho(T->Jacobian(), nor);

            error[i] += ip.weight * q_val * (grad * nor);
        }
    }
}


#endif //MFEM_LOCALCONSERVATION_HPP
