#ifndef MFEM_PROTEINWATERINTERFACEINTEGRATORS_HPP
#define MFEM_PROTEINWATERINTERFACEINTEGRATORS_HPP

#include <iostream>
#include "mfem.hpp"
#include "DGDiffusion_Edge_Symmetry_Penalty.hpp"
using namespace std;
using namespace mfem;


/* 计算溶剂(水)和溶质(蛋白)之间的界面积分
 *
 *     q <w \cdot n, v>,
 *
 * v is TestFunction, fff v是从哪个单元限制到边界上的?.
 * q is Coefficient, w is VectorCoefficient, n是Facet的法向量
 * */
class ProteinWaterInterfaceIntegrator1: public LinearFormIntegrator
{
protected:
    Coefficient* q;
    VectorCoefficient* w;
    Mesh* mesh;
    int protein_marker, water_marker;

    Vector shape, normal, w_val;

public:
    ProteinWaterInterfaceIntegrator1(Coefficient* q_, VectorCoefficient* w_, Mesh* mesh_, int protein, int water)
            : q(q_), w(w_), mesh(mesh_), protein_marker(protein), water_marker(water) {}
    ~ProteinWaterInterfaceIntegrator1() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement& fe1,
                                        const FiniteElement& fe2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        const Element* e1 = mesh->GetElement(Trans.Elem1No);
        const Element* e2 = mesh->GetElement(Trans.Elem2No);
        int attr1 = e1->GetAttribute(); //(对特定的mesh)蛋白和溶液的标记分别为1,2,但只想在蛋白的那个面积分,且法向量应该是蛋白区域的外法向
        int attr2 = e2->GetAttribute();

        int dim = fe1.GetDim();
        normal.SetSize(dim);
        w_val.SetSize(dim);

        int ndof1 = fe1.GetDof();
        int ndof2 = fe2.GetDof();
        int ndofs = ndof1 + ndof2;

        shape.SetSize(ndofs);
        elvect.SetSize(ndofs);
        elvect = 0.0;

        if (attr1 == attr2) return; // only integrate on interface of protein and water

        Geometry::Type geo_type = mesh->GetFaceGeometryType(Trans.Face->ElementNo);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // 计算Facet的中点,用来计算facet的法向量
        Trans.Face->SetIntPoint(center);
        // 下面这种计算facet的normal其方向始终从单元编号较小(肯定就是Elem1No)的指向单元编号较大(肯定就是Elem2No)的.
        CalcOrtho(Trans.Face->Jacobian(), normal); // not unit normal vector
        assert(Trans.Elem1No < Trans.Elem2No);

        const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, 2 * fe1.GetOrder()); //得到facet上的积分点集

        if (attr1 == protein_marker && attr2 == water_marker)
        {
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                Trans.SetAllIntPoints(&ip); // Set the integration point in the face and the neighboring element

                const IntegrationPoint &eip = Trans.GetElement1IntPoint();

                fe1.CalcShape(eip, shape);

                w->Eval(w_val, *(Trans.Elem1), eip);

                double val = ip.weight * q->Eval(*(Trans.Elem1), eip) * (w_val * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j = 0; j < ndof1; j++)
                {
                    elvect[j] += val * shape[j];
                }
            }
        }
        else
        {
            MFEM_ASSERT(attr1 == water_marker && attr2 == protein_marker, "Mesh must be with two attributes: protein marker and water marker");
            normal.Neg(); // 要取attribute为1(蛋白单元)的element的外法向量
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint& ip = ir->IntPoint(i);
                Trans.SetAllIntPoints(&ip); // Set the integration point in the face and the neighboring element

                const IntegrationPoint &eip = Trans.GetElement2IntPoint();

                fe2.CalcShape(eip, shape);

                w->Eval(w_val, *(Trans.Elem2), eip);

                double val = ip.weight * q->Eval(*(Trans.Elem2), eip) * (w_val * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j=0; j<ndof2; j++)
                {
                    elvect[j + ndof1] += val * shape[j];
                }
            }
        }
    }
};

/* 计算溶剂(水)和溶质(蛋白)之间的界面积分
 *
 *     q <w \cdot n, {v}>,
 *
 * v is TestFunction, fff v是从哪个单元限制到边界上的?.
 * q is Coefficient, w is VectorCoefficient, n是Facet的法向量
 * */
class ProteinWaterInterfaceIntegrator2: public LinearFormIntegrator
{
protected:
    Coefficient* q;
    VectorCoefficient* w;
    Mesh* mesh;
    int protein_marker, water_marker;

    Vector shape1, shape2, normal, w_val;

public:
    ProteinWaterInterfaceIntegrator2(Coefficient* q_, VectorCoefficient* w_, Mesh* mesh_, int protein, int water)
            : q(q_), w(w_), mesh(mesh_), protein_marker(protein), water_marker(water) {}
    ~ProteinWaterInterfaceIntegrator2() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    virtual void AssembleRHSElementVect(const FiniteElement& fe1,
                                        const FiniteElement& fe2,
                                        FaceElementTransformations &Trans,
                                        Vector &elvect)
    {
        const Element* e1 = mesh->GetElement(Trans.Elem1No);
        const Element* e2 = mesh->GetElement(Trans.Elem2No);
        int attr1 = e1->GetAttribute(); //(对特定的mesh)蛋白和溶液的标记分别为1,2,但只想在蛋白的那个面积分,且法向量应该是蛋白区域的外法向
        int attr2 = e2->GetAttribute();

        int dim = fe1.GetDim();
        normal.SetSize(dim);
        w_val.SetSize(dim);
        int ndof1 = fe1.GetDof();
        int ndof2 = fe2.GetDof();
        int ndofs = ndof1 + ndof2;
        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);
        elvect.SetSize(ndofs);
        elvect = 0.0;

        if (attr1 == attr2) return; // only integrate on interface of protein and water

        Geometry::Type geo_type = mesh->GetFaceGeometryType(Trans.Face->ElementNo);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // 计算Facet的中点,用来计算facet的法向量
        Trans.Face->SetIntPoint(center);
        // 下面这种计算facet的normal其方向始终从单元编号较小(肯定就是Elem1No)的指向单元编号较大(肯定就是Elem2No)的.
        CalcOrtho(Trans.Face->Jacobian(), normal); // not unit normal vector
        assert(Trans.Elem1No < Trans.Elem2No);

        const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, 2 * fe1.GetOrder()); //得到facet上的积分点集
        IntegrationPoint eip;
        if (attr1 == protein_marker && attr2 == water_marker)
        {
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                Trans.Face->SetIntPoint(&ip);

                Trans.Loc1.Transform(ip, eip); //把facet上的积分点变换到第一个与该face相连的单元的参考单元上
                fe1.CalcShape(eip, shape1);
                shape2 = 0.0; //只在attribute为1的单元积分

                Trans.Elem1->SetIntPoint(&eip);
                w->Eval(w_val, *(Trans.Elem1), eip);

                double val = ip.weight * q->Eval(*(Trans.Elem1), eip) * (w_val * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j = 0; j < ndof1; j++)
                {
                    elvect[j] += val * shape1[j];
                }
                for (int j=0; j<ndof2; j++)
                {
                    elvect[j + ndof1] -= val * shape2[j];
                }
            }
        }
        else if (attr1 == water_marker && attr2 == protein_marker)
        {
            normal.Neg(); // 要取attribute为1(蛋白单元)的element的外法向量
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint& ip = ir->IntPoint(i);
                Trans.Face->SetIntPoint(&ip);

                Trans.Loc2.Transform(ip, eip); //把facet上的积分点变换到第一个与该face相连的单元的参考单元上
                fe2.CalcShape(eip, shape2);
                shape1 = 0.0; //只在attribute为1的单元积分

                Trans.Elem2->SetIntPoint(&eip);
                w->Eval(w_val, *(Trans.Elem2), eip);

                double val = ip.weight * q->Eval(*(Trans.Elem2), eip) * (w_val * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j = 0; j < ndof1; j++)
                {
                    elvect[j] += val * shape1[j];
                }
                for (int j=0; j<ndof2; j++)
                {
                    elvect[j + ndof1] -= val * shape2[j];
                }
            }
        }
    }
};


/* Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
 *
 * 思路: https://github.com/mfem/mfem/issues/1093
 * 由 BoundaryNormalLFIntegrator 修改而得, 理论上不会有bug.
 * 修改目标: 把 BoundaryNormalLFIntegrator 用于内部边界积分, 比如水和蛋白的界面积分;
 *           其中界面上的法向量就是单元attribute为marker的单元外法向量.
 * */
class BoundaryNormalLFIntegrator_1 : public LinearFormIntegrator
{
    Vector shape;
    VectorCoefficient &Q;
    int marker, oa, ob;
    Mesh* mesh;

public:
    /// Constructs a boundary integrator with a given Coefficient QG
    BoundaryNormalLFIntegrator_1(VectorCoefficient &QG, int a = 1, int b = 1)
            : Q(QG), mesh(NULL), oa(a), ob(b) { }
    // 第1处修改: 增加一个构造函数, 保证上面的构造函数和 BoundaryNormalLFIntegrator 一致
    BoundaryNormalLFIntegrator_1(VectorCoefficient &QG, Mesh* mesh_, int marker_, int a = 1, int b = 1)
            : Q(QG), mesh(mesh_), marker(marker_), oa(a), ob(b) { }

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)

    {
        int dim = el.GetDim()+1;
        int dof = el.GetDof();
        Vector nor(dim), Qvec;

        shape.SetSize(dof);
        elvect.SetSize(dof);
        elvect = 0.0;

        const IntegrationRule *ir = IntRule;
        if (ir == NULL)
        {
            int intorder = oa * el.GetOrder() + ob;  // <----------
            ir = &IntRules.Get(el.GetGeomType(), intorder);
        }

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);

            Tr.SetIntPoint(&ip);
            CalcOrtho(Tr.Jacobian(), nor);
            Q.Eval(Qvec, Tr, ip);

            el.CalcShape(ip, shape);

            elvect.Add(ip.weight*(Qvec*nor), shape);
        }
    }

    using LinearFormIntegrator::AssembleRHSElementVect;
};


namespace protein_water_interface_integrators
{
    void Test_ProteinWaterInterfaceIntegrator1_by_BoundaryNormalLFIntegrator_1()
    {
        Mesh* mesh = new Mesh("../../../data/self_defined.msh");

        const int interface_marker = 7;
        Array<int> marker(mesh->bdr_attributes.Max());
        marker = 0;
        marker[interface_marker - 1] = 1;

        auto* fec = new H1_FECollection(1, mesh->Dimension());
        auto* fes = new FiniteElementSpace(mesh, fec);

        ConstantCoefficient one(3.1415926);
        Vector ones(3);
        ones[0] = 0.123;
        ones[1] = 3.654;
        ones[2] = 5.6789;
        VectorConstantCoefficient vec(ones);
        ScalarVectorProductCoefficient scale_vec(3.1415926, vec);

        auto *l0 = new LinearForm(fes);
        // b0: one <vec.n, v>_{\Gamma}
        auto* integ1 = new ProteinWaterInterfaceIntegrator1(&one, &vec, mesh, 2, 1);
        l0->AddInteriorFaceIntegrator(integ1);
        l0->Assemble();

        auto *l1 = new LinearForm(fes);
        auto* integ2 = new BoundaryNormalLFIntegrator_1(scale_vec, mesh, 1);
        l1->AddBoundaryIntegrator(integ2, marker);
        l1->Assemble();

//        cout << "l2 norm of l0: " << l0->Norml2() << endl;
//        cout << "l2 norm of l1: " << l1->Norml2() << endl;
        assert(fabs(l0->Norml2() - l1->Norml2()) < 1E-10);
        for (int i=0; i<fes->GetTrueVSize(); ++i)
        {
            assert(fabs(fabs((*l0)[i]) - fabs((*l1)[i])) < 1E-10);
        }

        delete mesh;
    }

}

void Test_ProteinWaterInterfaceIntegrators()
{
    using namespace protein_water_interface_integrators;

    Test_ProteinWaterInterfaceIntegrator1_by_BoundaryNormalLFIntegrator_1();


    cout << "===> Test Pass: ProteinWaterInterfaceIntegrators.hpp" << endl;
}

#endif //MFEM_PROTEINWATERINTERFACEINTEGRATORS_HPP
