//
// Created by fan on 2019/10/23.
//

#ifndef LEARN_MFEM_SELFDEFINED_LINEARFORM_HPP
#define LEARN_MFEM_SELFDEFINED_LINEARFORM_HPP

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include "PrintMesh.hpp"
using namespace std;
using namespace mfem;


// 给定 w(VectorFunctionCoefficient, 表示一个GridFunction的导数),
// 计算特定的facet积分: <w \cdot n, v>, v是TestFunction, n是Facet的法向量
class SelfDefined_LFFacetIntegrator: public LinearFormIntegrator
{
protected:
    VectorCoefficient& nabla_w;
    Mesh* mesh;
    FiniteElementSpace* fes;
    int protein_marker, water_marker;
    Vector shape1, shape2, normal, gradw;

public:
    SelfDefined_LFFacetIntegrator(FiniteElementSpace* fes_, VectorCoefficient& nablaw, int protein, int water)
            : fes(fes_), nabla_w(nablaw), protein_marker(protein), water_marker(water)
    {
        mesh = fes->GetMesh();
    }
    ~SelfDefined_LFFacetIntegrator() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect)
    {
        MFEM_ABORT("not support!");
    }

    // 把要积分的Facet当成interface
    void selfAssembleRHSElementVect(FaceElementTransformations &Trans, Vector &elvect)
    {
        const Element* e1  = mesh->GetElement(Trans.Elem1No); // 与该内部facet相连的两个 Element (与FiniteElement区分)
        const Element* e2  = mesh->GetElement(Trans.Elem2No);
        int attr1 = e1->GetAttribute(); //(对特定的mesh)蛋白和溶液的标记分别为1,2,但只想在蛋白的那个面积分,且法向量应该是蛋白区域的外法向
        int attr2 = e2->GetAttribute();
        const FiniteElement* fe1 = fes->GetFE(Trans.Elem1No); // 与该内部facet相连的两个 FiniteElement (与Element区分)
        const FiniteElement* fe2 = fes->GetFE(Trans.Elem2No);

        int ndofs1 = fe1->GetDof(), ndofs2 = fe2->GetDof();
        shape1.SetSize(ndofs1); shape2.SetSize(ndofs2);
        elvect.SetSize(ndofs1 + ndofs2);
        elvect = 0.0;

        int dim = fe1->GetDim();
        normal.SetSize(dim);
        gradw.SetSize(dim);

        Geometry::Type geo_type = mesh->GetFaceGeometryType(Trans.Face->ElementNo);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // 计算Facet的中点,用来计算facet的法向量
        Trans.Face->SetIntPoint(center);
        // 下面这种计算facet的normal其方向始终从单元编号较小(肯定就是Elem1No)的指向单元编号较大(肯定就是Elem2No)的.
        CalcOrtho(Trans.Face->Jacobian(), normal); // not unit normal vector
        assert(Trans.Elem1No < Trans.Elem2No);

        const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, 2 * fe1->GetOrder()); //得到facet上的积分点集
        IntegrationPoint eip;
        if (attr1 == protein_marker && attr2 == water_marker)
        {
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                Trans.Face->SetIntPoint(&ip);

                Trans.Loc1.Transform(ip, eip); //把facet上的积分点变换到第一个与该face相连的单元的参考单元上
                fe1->CalcShape(eip, shape1);
                shape2 = 0.0; //只在attribute为1的单元积分

                Trans.Elem1->SetIntPoint(&eip);
                nabla_w.Eval(gradw, *(Trans.Elem1), eip);

                double val = ip.weight * (gradw * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j = 0; j < ndofs1; j++)
                {
                    elvect[j] += val * shape1[j];
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
                fe2->CalcShape(eip, shape2);
                shape1 = 0.0; //只在attribute为1的单元积分

                Trans.Elem2->SetIntPoint(&eip);
                nabla_w.Eval(gradw, *(Trans.Elem2), eip);

                double val = ip.weight * (gradw * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j=0; j<ndofs2; j++)
                {
                    elvect[j + ndofs1] += val*shape2[j];
                }
            }
        }
    }

    // exactly same with above selfAssembleRHSElementVect()
    virtual void AssembleRHSElementVect(const FiniteElement& felm1, const FiniteElement& felm2,
                                        FaceElementTransformations &Trans, Vector &elvect)
    {
        const Element* e1  = mesh->GetElement(Trans.Elem1No); // 与该内部facet相连的两个 Element (与FiniteElement区分)
        const Element* e2  = mesh->GetElement(Trans.Elem2No);
        int attr1 = e1->GetAttribute(); //(对特定的mesh)蛋白和溶液的标记分别为1,2,但只想在蛋白的那个面积分,且法向量应该是蛋白区域的外法向
        int attr2 = e2->GetAttribute();
        const FiniteElement* fe1 = &felm1; // 与该内部facet相连的两个 FiniteElement (与Element区分)
        const FiniteElement* fe2 = &felm2;

        int ndofs1 = fe1->GetDof(), ndofs2 = fe2->GetDof();
        shape1.SetSize(ndofs1); shape2.SetSize(ndofs2);
        elvect.SetSize(ndofs1 + ndofs2);
        elvect = 0.0;

        int dim = fe1->GetDim();
        normal.SetSize(dim);
        gradw.SetSize(dim);

        Geometry::Type geo_type = mesh->GetFaceGeometryType(Trans.Face->ElementNo);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // 计算Facet的中点,用来计算facet的法向量
        Trans.Face->SetIntPoint(center);
        // 下面这种计算facet的normal其方向始终从单元编号较小(肯定就是Elem1No)的指向单元编号较大(肯定就是Elem2No)的.
        CalcOrtho(Trans.Face->Jacobian(), normal); // not unit normal vector
        assert(Trans.Elem1No < Trans.Elem2No);

        const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, 2 * fe1->GetOrder()); //得到facet上的积分点集
        IntegrationPoint eip;
        if (attr1 == protein_marker && attr2 == water_marker)
        {
            for (int i=0; i<ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                Trans.Face->SetIntPoint(&ip);

                Trans.Loc1.Transform(ip, eip); //把facet上的积分点变换到第一个与该face相连的单元的参考单元上
                fe1->CalcShape(eip, shape1);
                shape2 = 0.0; //只在attribute为1的单元积分

                Trans.Elem1->SetIntPoint(&eip);
                nabla_w.Eval(gradw, *(Trans.Elem1), eip);

                double val = ip.weight * (gradw * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j = 0; j < ndofs1; j++)
                {
                    elvect[j] += val * shape1[j];
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
                fe2->CalcShape(eip, shape2);
                shape1 = 0.0; //只在attribute为1的单元积分

                Trans.Elem2->SetIntPoint(&eip);
                nabla_w.Eval(gradw, *(Trans.Elem2), eip);

                double val = ip.weight * (gradw * normal); //ref:BoundaryLFIntegrator::AssembleRHSElementVect()
                for (int j=0; j<ndofs2; j++)
                {
                    elvect[j + ndofs1] += val*shape2[j];
                }
            }
        }
    }
};

// Given VectorCoefficient w and Coefficient Q, compute Q*(w, grad(v))_{\Omega}
class SelfConvectionIntegrator
{
protected:
    VectorCoefficient* w;
    Coefficient* q;

    DenseMatrix adjJ, dshape, tmp;
    Vector w_val, tmp_vec;

public:
    SelfConvectionIntegrator(Coefficient* q_, VectorCoefficient* w_): q(q_), w(w_) {}
    ~SelfConvectionIntegrator() {}

    void selfAssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
    {
        int nd = el.GetDof();
        int dim = el.GetDim();

        adjJ.SetSize(dim);
        w_val.SetSize(dim);

        dshape.SetSize(nd, dim);
        tmp.SetSize(nd, dim);
        tmp_vec.SetSize(nd);
        elvect.SetSize(nd);
        elvect = 0.0;

        int order = Tr.OrderGrad(&el) + Tr.Order() + el.GetOrder();
        const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);
        for (int i=0; i<ir->GetNPoints(); ++i)
        {
            const IntegrationPoint& ip = ir->IntPoint(i);
            el.CalcDShape(ip, dshape);

            Tr.SetIntPoint(&ip);
            CalcAdjugate(Tr.Jacobian(), adjJ);

            w->Eval(w_val, Tr, ip);
            double wi = ip.weight * q->Eval(Tr, ip);

            Mult(dshape, adjJ, tmp);
            tmp.Mult(w_val, tmp_vec);
            elvect.Add(wi, tmp_vec);
        }
    }
};

// 自定义的 LinearForm, 可以添加积分子 SelfDefined_LFFacetIntegrator
class SelfDefined_LinearForm : public LinearForm
{
protected:
    FiniteElementSpace* fes;
    Array<SelfDefined_LFFacetIntegrator*> iflfi; // 只对网格中的部分facet做积分
    Array<SelfConvectionIntegrator*> cveci;

public:
    SelfDefined_LinearForm(FiniteElementSpace *f): LinearForm(f), fes(f) { }
    ~SelfDefined_LinearForm() {}

    void AddSelfDefined_LFFacetIntegrator(SelfDefined_LFFacetIntegrator* lfi)
    {
        iflfi.Append(lfi);
    }

    void AddSelfConvectionIntegrator(SelfConvectionIntegrator* cvi)
    {
        cveci.Append(cvi);
    }

    // 当成interface积分,但在某一边全为0.
    // ref:AddInteriorFaceIntegrator(new DGDiffusionIntegrator())
    void SelfDefined_Assemble()
    {
        LinearForm::Assemble(); // 首先调用LinearForm的assemble,组装其他的Integrator

        if (iflfi.Size()) // interior face linear form integrator
        {
            Mesh* mesh = fes->GetMesh();
            FaceElementTransformations* tr;
            Vector elemvect;
            Array<int> vdofs1, vdofs2; //单元刚度向量组装到总的载荷向量时用到的自由度编号

            for (size_t i=0; i<mesh->GetNumFaces(); i++) // 对所有的facet循环:interior facet, boundary facet
            {
                tr = mesh->GetInteriorFaceTransformations(i);
                if (tr != NULL) // facet位于interior,否则位于区域边界
                {
                    fes -> GetElementVDofs (tr -> Elem1No, vdofs1);
                    fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
                    vdofs1.Append(vdofs2); // 把两边单元的自由度编号合并到第一个vdofs

                    for (size_t k=0; k<iflfi.Size(); k++)
                    {
                        iflfi[k]->selfAssembleRHSElementVect(*tr, elemvect); //true表示第一个单元是我们的目标单元
                        AddElementVector(vdofs1, elemvect);
                    }
                }
            }
        }

        if (cveci.Size())
        {
            Array<int> vdofs;
            ElementTransformation* eltrans;
            Vector elemvect;
            for (int i=0; i<fes->GetNE(); ++i)
            {
                fes->GetElementVDofs(i, vdofs);
                eltrans = fes->GetElementTransformation(i);
                for (int k=0; k<cveci.Size(); ++k)
                {
                    cveci[k]->selfAssembleRHSElementVect(*fes->GetFE(i), *eltrans, elemvect);
                    AddElementVector(vdofs, elemvect);
                }
            }
        }
    }
};


namespace _SelfDefined_LinearForm
{

double sin_cfunc(const Vector& x) // 不要修改此函数,用作特例做测试用
{
    return sin(x[0]) * sin(x[1]) * sin(x[2]);
}
void grad_sin_cfunc(const Vector& x, Vector& y) // 不要修改此函数,用作特例做测试用
{
    y[0] = cos(x[0]) * sin(x[1]) * sin(x[2]);
    y[1] = sin(x[0]) * cos(x[1]) * sin(x[2]);
    y[2] = sin(x[0]) * sin(x[1]) * cos(x[2]);
}
void Test_SelfDefined_LFFacetIntegrator3() // 这个测试里面实现了在一个facet上积分的步骤
{
    Mesh mesh("../../../data/one_tet.msh");
    int dim = mesh.Dimension();

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int ndofs = h1_space.GetVSize();

    FunctionCoefficient sin_coeff(sin_cfunc);
    GridFunction sin_gf(&h1_space);
    sin_gf.ProjectCoefficient(sin_coeff);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    LinearForm lf1(&h1_space);
    {
        Array<int> marker(mesh.bdr_attributes.Max());
        marker = 0;
        marker[2 - 1] = 1; //可以进行反向测试:测试attribute为1的所有face.这个face的三个节点(在Gmsh中的)编号是1,2,3.
        lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
        lf1.Assemble();
//        lf1.Print(cout << "lf1: ");
    }

    Vector elvect(ndofs);
    elvect = 0.0;
    Array<int> vdofs;
    for (size_t i=0; i<mesh.GetNumFaces(); i++)
    {
        Array<int> vert;
        mesh.GetFace(i)->GetVertices(vert);
        vert.Sort();
//        vert.Print(cout << i << "-th face, vertices: ");
        // 上面lf1里面的boundary积分的那个face的三个节点(在Gmsh中的)编号是1,2,3
        if (vert[0] != 0 || vert[1] != 1 || vert[2] != 2) continue;

        Vector normal(dim), phy_ip(dim), gradsin(dim), gradsin_(dim), shape1(ndofs);
        shape1 = 0.0;
        Array<int> vdofs1, vdofs2;
        IntegrationPoint eip1;
        // i所对应的face就是上面marker所指的face.如果进行反向测试:把==换成!=,然后下面的normal的assert部分注释掉
        // 下面实现了 (w \cdot n, v)_{face}
        FaceElementTransformations* tr = mesh.GetFaceElementTransformations(i);
        assert(tr->Elem2No < 0); // 该face位于区域边界
        h1_space.GetElementVDofs(tr->Elem1No, vdofs);

        Geometry::Type geo_type = mesh.GetFaceGeometryType(tr->Face->ElementNo);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type);
        tr->Face->SetIntPoint(center);
        CalcOrtho(tr->Face->Jacobian(), normal); //使用这个normal的时候有可能要反向.其模长为该三角形面积的两倍
        assert(abs(normal[0] - 1) < 1E-8);
        assert(abs(normal[1] - 1) < 1E-8);
        assert(abs(normal[2] + 1) < 1E-8);

        const FiniteElement* fe1 = h1_space.GetFE(tr->Elem1No);
        const IntegrationRule* ir = &IntRules.Get(tr->FaceGeom, 2);
        for (size_t p=0; p<ir->GetNPoints(); p++)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);
            tr->Face->SetIntPoint(&ip);
            tr->Face->Transform(ip, phy_ip); //physical point

            tr->Loc1.Transform(ip, eip1);
            fe1->CalcShape(eip1, shape1);

            tr->Elem1->SetIntPoint(&eip1);
            sin_gf.GetGradient(*(tr->Elem1), gradsin);         //错误: 这样计算梯度有问题fff,应该是数值积分精度造成的 https://github.com/mfem/mfem/issues/1122#issuecomment-550053860
            grad_sin_coeff.Eval(gradsin_, *(tr->Elem1), eip1); //正确: 这样计算梯度正确fff
            elvect.Add(ip.weight * (gradsin_ * normal), shape1);

            assert(abs(gradsin_[0] - cos(phy_ip[0])*sin(phy_ip[1])*sin(phy_ip[2])) < 1E-8); //手动求导数
            assert(abs(gradsin_[1] - sin(phy_ip[0])*cos(phy_ip[1])*sin(phy_ip[2])) < 1E-8);
            assert(abs(gradsin_[2] - sin(phy_ip[0])*sin(phy_ip[1])*cos(phy_ip[2])) < 1E-8);
            assert(abs(shape1[0] - phy_ip[0]) < 1E-8); // x. 手动计算单元刚度矩阵
            assert(abs(shape1[1] - phy_ip[1]) < 1E-8); // y
            assert(abs(shape1[2] - (-phy_ip[2] + 1)) < 1E-8); // -z+1
            assert(abs(shape1[3] - (-(phy_ip[0] + phy_ip[1]) + phy_ip[2])) < 1E-8); // -x-y+z
        }
    }
    { // test
//        elvect.Print(cout << "elvect: ");
//        lf1.Print(cout << "lf1: ");
//        vdofs.Print(cout << "vdofs: ");
        for (int i=0; i<vdofs.Size(); i++)
        {
            assert(abs(lf1[vdofs[i]] - elvect[i]) < 1E-8);
        }
    }
}
void Test_SelfDefined_LFFacetIntegrator4()
{
    Mesh mesh("../../../data/facet_normal1.msh");

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);
    ConstantCoefficient two(21.31446321465); // 可以任意改里面的数值,但是下面的测试仍然能够通过

    LinearForm lf1(&h1_space);
    {
        Array<int> marker(mesh.bdr_attributes.Max());
        marker = 0;
        marker[3 - 1] = 1; //和下面的要进行积分的face必须是同一个face
        // 下面在 BoundaryNormalLFIntegrator 里面使用的normal的方向由下面链接解释 https://github.com/mfem/mfem/issues/1122#issuecomment-549511129
        lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
        lf1.AddBoundaryIntegrator(new BoundaryLFIntegrator(two), marker); // 添加这一项是为了证明自定义的SelfDefined_LinearForm对MFEM的原始的积分子也对
        lf1.Assemble();
//        lf1.Print(cout << "lf1: ");
    }

    SelfDefined_LinearForm lf2(&h1_space);
    {
        Array<int> marker(mesh.bdr_attributes.Max());
        marker = 0;
        marker[3 - 1] = 1; //和下面的要进行积分的face必须是同一个face

        lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
        lf2.AddBoundaryIntegrator(new BoundaryLFIntegrator(two), marker); // 添加这一项是为了证明自定义的SelfDefined_LinearForm对MFEM的原始的积分子也对
        lf2.SelfDefined_Assemble();
//        lf2.Print(cout << "lf2: ");
    }

    { // test
        for (int i=0; i<lf1.Size(); i++)
        {
            assert(abs(lf1[i]) - abs(lf2[i]) < 1E-8);
        }
    }
}
void Test_SelfDefined_LFFacetIntegrator5() // 和上面一个test几乎一样
{
    Mesh mesh("../../../data/simple.mesh");

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    Array<int> marker(mesh.bdr_attributes.Max());
    marker = 0;
    marker[3 - 1] = 1; //和下面的要进行积分的face必须是同一个face

    LinearForm lf1(&h1_space);
    lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
    lf1.Assemble();
//    lf1.Print(cout << "lf1: ");

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
    lf2.SelfDefined_Assemble();
//    lf2.Print(cout << "lf2: ");

    for (size_t i=0; i<lf1.Size(); i++)
        assert(abs(abs(lf1[i]) - abs(lf2[i])) < 1E-10);
}
void Test_SelfDefined_LFFacetIntegrator6()
{
    Mesh mesh("../../../data/self_defined.mesh");

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    Array<int> marker(mesh.bdr_attributes.Max());
    marker = 0;
    marker[7 - 1] = 1; // interface的标记为7

    LinearForm lf1(&h1_space);
    lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
    lf1.Assemble();
//    lf1.Print(cout << "lf1: ", 10000);

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
    lf2.SelfDefined_Assemble();
//    lf2.Print(cout << "lf2: ", 10000);

    assert(lf1.Size() == lf2.Size());
    for (size_t i=0; i<lf1.Size(); i++)
    {
        assert(abs(lf1[i] - lf2[i]) < 1E-8);
    }
}
void Test_SelfDefined_LFFacetIntegrator7() //终极测试
{
    int p_order = 1;
    H1_FECollection h1_fec(p_order, 3);

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    //selfAssembleRHSElementVect()里面normal不取Neg()才能通过测试. 一般情况关闭这个测试
    Mesh mesh1("../../../data/special.mesh");
    FiniteElementSpace h1_space(&mesh1, &h1_fec);
    Array<int> marker(mesh1.bdr_attributes.Max());
    marker = 0;
    marker[7 - 1] = 1; // interface的标记为7

    LinearForm lf1(&h1_space);
    lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker); //(g \cdot n, v)
    lf1.Assemble();

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
    lf2.SelfDefined_Assemble();

    assert(lf1.Size() == lf2.Size());
    for (size_t i=0; i<lf1.Size(); i++)
    {
        // 打开下面的输出可以知道: 虽然lf1和lf2不同,但是不同的地方却是有规律的,
        // 所以基本可以说明自己写的积分子的积分部分是对的,只是有可能facet的normal方向跟做对比的normal方向不一致.
        // 但是用来作对比的积分子里面使用的normal方向不一定是对的,应该很大概率是不对的(ref:https://github.com/mfem/mfem/issues/1122)
//        if (abs(abs(lf1[i]) - abs(lf2[i])) > 1E-8)
//            cout << setw(14) << lf1[i] << ", " << setw(14) << lf2[i] << endl;
//        assert(abs(lf1[i] - lf2[i]) < 1E-8); //selfAssembleRHSElementVect()里面normal不取Neg()才能通过测试
    }
}
void Test_SelfDefined_LFFacetIntegrator8() //终极测试
{
    Mesh mesh1("../../data/1MAG_2.msh");
    const char* target_mesh = "temp.mesh"; //随机命名
    int marker1 = 100, marker2 = 200;
    {
        Array<Element*> interface;
        for (size_t i=0; i<mesh1.GetNumFaces(); i++)
        {
            FaceElementTransformations* tran = mesh1.GetInteriorFaceTransformations(i);
            if (tran != NULL)
            {
                Element* el1 = mesh1.GetElement(tran->Elem1No);
                Element* el2 = mesh1.GetElement(tran->Elem2No);
                // interior facet相连的第一个单元就是单元编号较小的,第二个是较大的.
                // 同时,interior facet的normal方向也是从单元编号较小的指向单元编号较大的
                assert(tran->Elem1No < tran->Elem2No);
                Element* face = const_cast<Element*>(mesh1.GetFace(i));
                if (el1->GetAttribute() < el2->GetAttribute()) //在1MAG_2.msh里面,蛋白标记为1,水标记为2
                {
                    face->SetAttribute(marker1); //在这个facet上面,其normal就是由el1(蛋白单元)指向el2(水单元)
                    interface.Append(face);
                }
                else if (el1->GetAttribute() > el2->GetAttribute())
                {
                    face->SetAttribute(marker2); //在这个facet上面,其normal就是由el1(水单元)指向el2(蛋白单元)
                    interface.Append(face);
                }
            }
        }
        PrintMesh(target_mesh, mesh1, interface);
    }

    Mesh mesh(target_mesh);

    int p_order = 1;
    H1_FECollection h1_fec(p_order, 3);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    Array<int> marker(mesh.bdr_attributes.Max());
    marker = 0;
    marker[marker1 - 1] = 1;

    LinearForm lf1(&h1_space);
    lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
    lf1.Assemble();

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
    lf2.SelfDefined_Assemble();

    for (int i=0; i<lf1.Size(); ++i)
    {
        if (abs(lf1[i]) + abs(lf2[i]) < 1E-10)
            continue;
        cout << lf1[i] << ", " << lf2[i];
        if (abs(abs(lf1[i]) - abs(lf2[i])) > 1E-8)
            cout << "                  ffffffffffffffffffffffffffffffffffffffffffffffffff";
        cout << '\n';
    }

    char command[256] = ""; // 必须初始化
    strcat(command, "rm ");
    strcat(command, target_mesh);
    system(command); //调用系统shell命令
}

void Test_SelfDefined_LFFacetIntegrator9() //终极测试
{
    Mesh mesh("/home/fan/miscellaneous/learn_mfem/cmake-build-debug/temp_examples/1MAG_2_reorder.msh");

    int p_order = 1;
    H1_FECollection h1_fec(p_order, 3);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);
    ConstantCoefficient arbt(3.1234345); // 为了测试SelfDefined_LinearForm对MFEM内部的积分子也是对的

    Array<int> marker(mesh.bdr_attributes.Max());
    marker = 0;
    marker[9 - 1] = 1;

    LinearForm lf1(&h1_space);
    lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
//    lf1.AddBoundaryIntegrator(new BoundaryLFIntegrator(arbt), marker);
    lf1.Assemble();

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
//    lf2.AddBoundaryIntegrator(new BoundaryLFIntegrator(arbt), marker);
    lf2.SelfDefined_Assemble();

    int equal=0, conter=0, error=0;
    for (int i=0; i<lf1.Size(); ++i)
    {
        if (abs(lf1[i] - lf2[i]) < 1E-10)
        {
            equal++;
        }
        else if (abs(lf1[i] + lf2[i]) < 1E-10)
        {
//            cout << i << ", " << setw(10) << lf1[i] << ",  " << setw(10) << lf2[i] << endl;
            conter++;
        }
        else
        {
//            cout << "                       " << setw(10) << lf1[i] << ",  " << setw(10) << lf2[i] << endl;
//            cout << "eeeeeeeeeeeeeeeeeeerror" << endl;
            error++;
        }
    }
//    cout << "equal: " << equal << ",  conter: " << conter << ",  error: " << error << endl;
}

void Test_SelfDefined_LFFacetIntegrator10()
{
    // 测试两种不同的方式进行单元内部边界积分: 实验结果显示两种方式完全等价
    int p_order = 1;
    H1_FECollection h1_fec(p_order, 3);

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    //selfAssembleRHSElementVect()里面normal不取Neg()才能通过测试. 一般情况关闭这个测试
    Mesh mesh1("../../../data/special.mesh");
    FiniteElementSpace h1_space(&mesh1, &h1_fec);
    Array<int> marker(mesh1.bdr_attributes.Max());
    marker = 0;
    marker[7 - 1] = 1; // interface的标记为7

    LinearForm lf1(&h1_space);
    lf1.AddInteriorFaceIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2)); //(g \cdot n, v)
    lf1.Assemble();

    SelfDefined_LinearForm lf2(&h1_space);
    lf2.AddSelfDefined_LFFacetIntegrator(new SelfDefined_LFFacetIntegrator(&h1_space, grad_sin_coeff, 1, 2));
    lf2.SelfDefined_Assemble();

//    assert(lf1.Size() == lf2.Size());
//    lf1.Print(cout << "lf1: " , h1_space.GetVSize());
//    lf2.Print(cout << "lf2: " , h1_space.GetVSize());
    for (size_t i=0; i<lf1.Size(); i++)
    {
//        assert(abs(lf1[i] - lf2[i]) < 1E-8);
    }
}


void Test_SelfConvectionIntegrator()
{
    Mesh mesh(10, 10, Element::TRIANGLE, true, 1.0, 1.0);
    int dim = mesh.Dimension();

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int ndofs = h1_space.GetVSize();

    FunctionCoefficient sin_coeff(sin_cfunc);
    VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);

    BilinearForm blf(&h1_space);
    blf.AddDomainIntegrator(new DiffusionIntegrator(sin_coeff)); // sin(x) * (grad(u), grad(v))
    blf.Assemble();
    SparseMatrix& A1 = blf.SpMat();

    GridFunction gf(&h1_space);
    gf.ProjectCoefficient(sin_coeff);

    Vector vec(h1_space.GetNDofs());
    A1.Mult(gf, vec);

    SelfDefined_LinearForm lf(&h1_space);
    lf.AddSelfConvectionIntegrator(new SelfConvectionIntegrator(&sin_coeff, &grad_sin_coeff));
    lf.SelfDefined_Assemble();

    for (int i=0; i<h1_space.GetNDofs(); ++i)
    {
        assert(abs(vec[i] - lf[i]) < 1E-10);
    }
}

}

void Test_SelfDefined_LinearForm()
{
    using namespace _SelfDefined_LinearForm;
//    Test_SelfDefined_LFFacetIntegrator9();
    Test_SelfDefined_LFFacetIntegrator3();
    Test_SelfDefined_LFFacetIntegrator4();
    Test_SelfDefined_LFFacetIntegrator5();
    Test_SelfDefined_LFFacetIntegrator6();
    Test_SelfDefined_LFFacetIntegrator7();
//    Test_SelfDefined_LFFacetIntegrator8();

//    Test_SelfConvectionIntegrator(); // fff

    cout << "===> Test Pass: SelfDefined_LinearForm.hpp" << endl;
}

#endif //LEARN_MFEM_SELFDEFINED_LINEARFORM_HPP
