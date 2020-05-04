//
// Created by fan on 2020/4/12.
//

#ifndef LEARN_MFEM_DGSELFBDRFACEINTEGRATOR_HPP
#define LEARN_MFEM_DGSELFBDRFACEINTEGRATOR_HPP

#include <iostream>
#include "mfem.hpp"
#include <cassert>

using namespace std;
using namespace mfem;

/* 计算(区域边界积分)
 *   q<u_D, v grad(w).n>_E
 * q, u_D are Coefficient
 * w is GridFunction
 * */
class DGSelfBdrFaceIntegrator: public LinearFormIntegrator
{
protected:
    Coefficient *q, *u_D;
    GradientGridFunctionCoefficient *gradw;

    Vector nor, shape, grad_w;

public:
    DGSelfBdrFaceIntegrator(Coefficient *q_, Coefficient *u_D_, GridFunction* w)
            : q(q_), u_D(u_D_) { gradw = new GradientGridFunctionCoefficient(w); }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        mfem_error("Not Support");
    }

    virtual void AssembleRHSElementVect(const FiniteElement& el,
                                        FaceElementTransformations& Tr,
                                        Vector& elvect)
    {
        int dim  = el.GetDim();
        int ndof = el.GetDof();

        nor.SetSize(dim);
        grad_w.SetSize(dim);

        shape.SetSize(ndof);
        elvect.SetSize(ndof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            // a simple choice for the integration order; is this OK?
            int order = 2*el.GetOrder();
            ir = &IntRules.Get(Tr.FaceGeom, order);
        }

        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            IntegrationPoint eip;

            Tr.Face->SetIntPoint(&ip);
            if (dim == 1)
                nor(0) = 2*eip.x - 1.0; //1维参考单元的法向量(不是单位的)
            else
                CalcOrtho(Tr.Face->Jacobian(), nor);

            Tr.Loc1.Transform(ip, eip);
            el.CalcShape(eip, shape);

            gradw->Eval(grad_w, *Tr.Elem1, eip);

            double val = ip.weight * q->Eval(*Tr.Face, ip) * u_D->Eval(*Tr.Face, ip) * (nor * grad_w);

            shape *= val;
            elvect += shape;
        }
    }
};


double sin_cfun_DGSelfBdrFaceIntegrator(const Vector& x)
{
    return sin(x[0]) * sin(x[1]); // sin(x) * sin(y)
}
double cos_cfun_DGSelfBdrFaceIntegrator(const Vector& x)
{
    return cos(x[0]) * cos(x[1]);
}

void Test_DGSelfBdrFaceIntegrator_1()
{
    Mesh* mesh = new Mesh(50, 50, Element::TRIANGLE, true, 1.0, 1.0);

    DG_FECollection h1_fec(1, mesh->Dimension());
    FiniteElementSpace h1_space(mesh, &h1_fec);

    ConstantCoefficient one(1.0);
    ConstantCoefficient neg(-1.0);
    FunctionCoefficient sin_coeff(sin_cfun_DGSelfBdrFaceIntegrator);
    FunctionCoefficient cos_coeff(cos_cfun_DGSelfBdrFaceIntegrator);
    GridFunction sin_gf(&h1_space), one_gf(&h1_space);
    sin_gf.ProjectCoefficient(sin_coeff);
    one_gf.ProjectCoefficient(one);
    GradientGridFunctionCoefficient grad_sin(&sin_gf);

    {
        LinearForm lf1(&h1_space);
        lf1.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(cos_coeff, grad_sin, 2.0, 0.0));
        lf1.Assemble();

        LinearForm lf2(&h1_space);
        lf2.AddBdrFaceIntegrator(new DGSelfBdrFaceIntegrator(&one, &cos_coeff, &sin_gf));
        lf2.Assemble();

//        lf1.Print(cout << "lf1: " , h1_space.GetVSize());
//        lf2.Print(cout << "lf2: ", h1_space.GetVSize());
        for (int i=0; i<h1_space.GetVSize(); ++i)
        {
            assert(abs(lf1[i] - lf2[i]) < 1E-10);
        }
    }

    delete mesh;
}



void Test_DGSelfBdrFaceIntegrator()
{
    Test_DGSelfBdrFaceIntegrator_1();

    cout << "===> Test Pass: DGSelfBdrFaceIntegrator.hpp" << endl;
}

#endif //LEARN_MFEM_DGSELFBDRFACEINTEGRATOR_HPP
