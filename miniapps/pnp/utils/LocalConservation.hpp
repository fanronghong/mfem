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
    double q_val;

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
            FaceElementTransformations* FaceTrans = mesh->GetFaceElementTransformations(e2f[j]);
            const IntegrationRule* ir = &(IntRules.Get(FaceTrans->GetGeometryType(), 2*fe->GetOrder()+3));

            for (int k=0; k<ir->GetNPoints(); ++k)
            {
                const IntegrationPoint& ip = ir->IntPoint(k);
                FaceTrans->SetIntPoint(&ip);

                if (i == FaceTrans->Elem1No)
                {
                    ElementTransformation& VolTrans = *FaceTrans->Elem1;
                    gf.GetGradient(VolTrans, grad);
                    q_val = q.Eval(VolTrans, VolTrans.GetIntPoint());
                    CalcOrtho(FaceTrans->Jacobian(), nor);
                }
                else
                {
                    assert(i == FaceTrans->Elem2No);
                    ElementTransformation& VolTrans = *FaceTrans->Elem2;
                    gf.GetGradient(VolTrans, grad);
                    q_val = q.Eval(VolTrans, VolTrans.GetIntPoint());
                    CalcOrtho(FaceTrans->Jacobian(), nor);
                    nor.Neg(); // nor is from Elem1 to Elem2, but we need normal is outward normal of element i.
                }

                error[i] += ip.weight * q_val * (grad * nor);
            }
        }
    }
}
// \int_{\partial K} q1 (grad(c) + q2 c grad(phi)).n ds. NP方程中浓度变量c的单元质量守恒
void ComputeLocalConservation(Coefficient& q1, const GridFunction& gf1, Coefficient& q2, const GridFunction& gf2, Vector& error)
{
    const FiniteElementSpace* fes = gf1.FESpace();
    Mesh* mesh = fes->GetMesh();
    const FiniteElement* fe;
    int dim = mesh->Dimension();

    Vector grad1, grad2, nor;
    grad1.SetSize(dim);
    grad2.SetSize(dim);
    nor.SetSize(dim);
    double q1_val, q2_val, gf1_val;

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
            FaceElementTransformations* FaceTrans = mesh->GetFaceElementTransformations(e2f[j]);
            const IntegrationRule* ir = &(IntRules.Get(FaceTrans->GetGeometryType(), 2*fe->GetOrder()+3));

            for (int k=0; k<ir->GetNPoints(); ++k)
            {
                const IntegrationPoint& ip = ir->IntPoint(k);
                FaceTrans->SetIntPoint(&ip);
                IntegrationPoint eip;

                if (i == FaceTrans->Elem1No)
                {
                    ElementTransformation& VolTrans = *FaceTrans->Elem1;
                    gf1.GetGradient(VolTrans, grad1);
                    gf2.GetGradient(VolTrans, grad2);

                    q1_val = q1.Eval(VolTrans, VolTrans.GetIntPoint());
                    q2_val = q2.Eval(VolTrans, VolTrans.GetIntPoint());

                    FaceTrans->Loc1.Transform(ip, eip);
                    gf1_val = gf1.GetValue(i, eip);

                    CalcOrtho(FaceTrans->Jacobian(), nor);
                }
                else
                {
                    assert(i == FaceTrans->Elem2No);
                    ElementTransformation& VolTrans = *FaceTrans->Elem2;

                    gf1.GetGradient(VolTrans, grad1);
                    gf2.GetGradient(VolTrans, grad2);

                    q1_val = q1.Eval(VolTrans, VolTrans.GetIntPoint());
                    q2_val = q2.Eval(VolTrans, VolTrans.GetIntPoint());

                    FaceTrans->Loc2.Transform(ip, eip);
                    gf1_val = gf1.GetValue(i, eip);

                    CalcOrtho(FaceTrans->Jacobian(), nor);
                    nor.Neg(); // nor is from Elem1 to Elem2, but we need normal is outward normal of element i.
                }

                grad2 *= q2_val * gf1_val;
                grad2 += grad1;
                error[i] += ip.weight * q1_val * (grad2 * nor);
            }
        }
    }
}


#endif //MFEM_LOCALCONSERVATION_HPP
