#ifndef __SELECTEDELEMENT_DIFFUSIONINTEGRATOR_HPP__
#define __SELECTEDELEMENT_DIFFUSIONINTEGRATOR_HPP__
#include <iostream>
#include <cassert>

#include "mfem.hpp"

using namespace mfem;
using namespace std;


// 计算被指定attribute的单元做积分,对没有指定的单元的单元刚度矩阵总是一个0矩阵
class SelectedElement_DiffusionIntegrator: public BilinearFormIntegrator
{
private:
    Vector shape;
    DenseMatrix dshape, dshapedxt;
    Coefficient *Q;

    Array<int>& miss_attrs;

public:
    SelectedElement_DiffusionIntegrator (Coefficient &q, Array<int>& miss_attrs_) : Q(&q), miss_attrs(miss_attrs_) {}
    ~SelectedElement_DiffusionIntegrator() {}

    virtual void AssembleElementMatrix(const FiniteElement& el,
                                       ElementTransformation& Trans, DenseMatrix& elmat)
    {
        int nd = el.GetDof(); //单元自由度个数
        int dim = el.GetDim();//单元的reference space dimension
        int spaceDim = Trans.GetSpaceDim(); //dimension of the target (physical) space
        bool square = (dim == spaceDim);
        double w;

        dshape.SetSize(nd,dim);
        dshapedxt.SetSize(nd,spaceDim);

        elmat.SetSize(nd); //给elmat分配内存空间
        elmat = 0.0; //单元刚度矩阵
        assert(Trans.Attribute == 1 || Trans.Attribute == 2);
        if (!miss_attrs[Trans.Attribute - 1]) {
            return; //对部分单元不进行组装刚度矩阵,直接返回0矩阵
        }

        const IntegrationRule *ir = IntRule; //NonlinearFormIntegrator -> BilinearFormIntegrator -> DiffusionIntegrator
        if (ir == NULL)
        {
            int order;
            if (el.Space() == FunctionSpace::Pk)
            {
                order = 2*el.GetOrder() - 2; //el.GetOrder(): order/degree of the shape functions
            }
            else
                // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            {
                order = 2*el.GetOrder() + dim - 1;
            }

            if (el.Space() == FunctionSpace::rQk)
            {
                ir = &RefinedIntRules.Get(el.GetGeomType(), order);
            }
            else
            {
                ir = &IntRules.Get(el.GetGeomType(), order);
            }
        }

        for (int i = 0; i < ir->GetNPoints(); i++) //对所有积分点循环
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            //Evaluate the gradients of all shape functions of a scalar finite
            // element in reference space at the given point ip
            el.CalcDShape(ip, dshape); //每个shape function的各个偏导数放在同一行fffff

            Trans.SetIntPoint(&ip);
            w = Trans.Weight();
            w = ip.weight / (square ? w : w*w*w);
            // AdjugateJacobian = / adj(J),         if J is square
            //                    \ adj(J^t.J).J^t, otherwise
            Mult(dshape, Trans.AdjugateJacobian(), dshapedxt); //dshape * Trans.AdjugateJacobian() -> dshapedxt
            if (Q)
            {
                w *= Q->Eval(Trans, ip);
            }
            AddMult_a_AAt(w, dshapedxt, elmat); // w * dshapedxt * dshapedxt^T -> elmat
        }
    }
};


// 计算被指定attribute的单元做积分,对没有指定的单元的单元向量总是一个0向量
class SelectedElement_DomainLFIntegrator: public DomainLFIntegrator {
private:
    Array<int> &miss_attrs;
    Vector shape;
    int oa, ob;
    Coefficient &Q;

public:
    SelectedElement_DomainLFIntegrator(Coefficient &QF, Array<int> &miss_attrs_, int a = 2, int b = 0)
            : DomainLFIntegrator(QF, a, b), miss_attrs(miss_attrs_), oa(a), ob(b), Q(QF) {}

    ~SelectedElement_DomainLFIntegrator() {}

    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect) //一个单元一个单元的组装fffffffff
    {
        int dof = el.GetDof(); //el单元的自由度个数
//        cout << Tr.ElementNo << ", ";
//        cout << Tr.Attribute << endl; //ffffffffffffffffffff靠 el 和 Tr 就可以还原出每个积分单元的所有信息?
        shape.SetSize(dof); // vector of size dof
        elvect.SetSize(dof);
        elvect = 0.0;
//        cout << "Trans.Attribute: " << Tr.Attribute << endl;
//        cout << "miss: \n" << miss_attrs[Tr.Attribute - 1] << endl;
        assert(Tr.Attribute == 1 || Tr.Attribute == 2);
        if (!miss_attrs[Tr.Attribute - 1]) {
            return; //对部分单元不进行组装刚度向量,直接返回0向量
        }

        const IntegrationRule *ir = IntRule; //IntegrationRule是由许多积分点组成的Array
        if (ir == NULL) {
            ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        }

        for (int i = 0; i < ir->GetNPoints(); i++) {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetIntPoint(&ip);
            //fffffff重点. Q.Eval(Tr, ip)就是函数值f(ip), 而Tr.Weight()就是改单元的测度(面积, 体积, 也即从reference element 到 physical element变换矩阵的行列式)
            double val = Tr.Weight() * Q.Eval(Tr, ip);
            el.CalcShape(ip, shape); //计算所有shape function在ip处的取值, 结果为一个Vector(即shape), shape = (v1(ip), v2(ip), v3(ip))
            add(elvect, ip.weight * val, shape, elvect); //ffff重点: elvect + ip.weight*val*shape => elvect. f(ip) * (v1(ip), v2(ip), v3(ip)) * w_i
        }
    }
};


void Test1(Array<int>& interfacedofs, Array<int>& attr1dofs,
           SparseMatrix& blf1_mat, SparseMatrix& blf2_mat,
           LinearForm& lf1, LinearForm& lf2)
{ //for test: 除了特别的单元以外,其余部分的单刚和载荷向量应该相同
    Array<int> rest_dofs;
    rest_dofs.Append(interfacedofs);
    rest_dofs.Append(attr1dofs);

    for (size_t i=0; i<rest_dofs.Size(); i++)
    {
        assert(abs(lf1[rest_dofs[i]] - lf2[rest_dofs[i]]) < 1e-10); //单元刚度向量相同

        const double* val1 = blf1_mat.GetRowEntries(rest_dofs[i]);
        const double* val2 = blf2_mat.GetRowEntries(rest_dofs[i]);
        for (size_t j=0; j<blf1_mat.RowSize(rest_dofs[i]); j++)
        {
            assert(abs(val1[j] - val2[j]) < 1e-10); //单元刚度矩阵相同
        }
    }
}


void Test2(LinearForm& lf1, LinearForm& lf2, Array<int>& attr2dofs)
{ // for test:线性型在attribute为2的element没有积分,故相应元素为0
    for (size_t i=0; i<attr2dofs.Size(); i++) //对应的右端项为0
    {
        assert(lf1[attr2dofs[i]] < 1e-8);
        assert(lf2[attr2dofs[i]] < 1e-8);
    }

    //应该整个lf1和lf2都相同
    for (size_t i=0; i<lf1.Size(); i++)
    {
        assert(abs(lf1[i] - lf2[i]) < 1E-8);
    }
}

void Test3(SparseMatrix& blf1_mat, SparseMatrix& blf2_mat, Array<int>& attr2dofs)
{ // for test: 双线性型blf2在attribute为2的element的单元刚度矩阵为0,blf1也是
    for (size_t i=0; i<attr2dofs.Size(); i++)
    {
        //val1应该是0元素,因为这些单元的单元刚度矩阵是0位阵
        const double* val1 = blf1_mat.GetRowEntries(attr2dofs[i]);
        for (int i=0; i<blf1_mat.RowSize(attr2dofs[i]); i++)
        {
            assert(val1[i] == 0);
        }

        //val2应该没有元素,因为包含此vertex的单元刚度矩阵为0,没有非零元素,val2表示对应行的非零元的起始地址
        const double* val2 = blf2_mat.GetRowEntries(attr2dofs[i]);
        for (int i=0; i<blf2_mat.RowSize(attr2dofs[i]); i++)
        {
            assert(val2[i] == 0);
        }
    }
}

void Test4(int ndofs, SparseMatrix& blf1_mat, SparseMatrix& blf2_mat, LinearForm& lf1, LinearForm& lf2, Array<int>& attr2dofs)
{ //for test:
    // 进行线性方程组求解之前必须修改blf1的刚度矩阵:对应attr2dofs的行变成只有对角线为1,其余为0
    Vector diag1(ndofs), diag2(ndofs);

    blf1_mat.GetDiag(diag1);
    blf2_mat.GetDiag(diag2);
    for (int i=0; i<attr2dofs.Size(); i++)
    {
        assert(diag1[attr2dofs[i]] == 0);
        assert(diag2[attr2dofs[i]] == 0);
    }

    for (size_t i=0; i<attr2dofs.Size(); i++) // 消去对应的行列,但是保证主对角线元素为1.0
    {
        blf1_mat.EliminateRowCol(attr2dofs[i], Matrix::DIAG_ONE);
        blf2_mat.EliminateRowCol(attr2dofs[i], Matrix::DIAG_ONE);
    }

    blf1_mat.GetDiag(diag1);
    blf2_mat.GetDiag(diag2);
    for (int i=0; i<attr2dofs.Size(); i++)
    {
        assert(diag1[attr2dofs[i]] == 1);
        assert(diag2[attr2dofs[i]] == 1);
    }
}

class MissSomeElements : public Coefficient
{
public:
    MissSomeElements() {}
    virtual ~MissSomeElements() { }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        assert(T.Attribute == 1 || T.Attribute == 2);
        if (T.Attribute == 2) //attr为2的单元不积分
            return 0.0;
        return 1.0;
    }
};

void Generate_Dofs(Array<int>& interfacedofs, Array<int>& attr1dofs,
                   Array<int>& attr2dofs, Array<int>& numElms, Mesh& mesh, FiniteElementSpace& h1_space)
{
    {
        const Table& vertex2element = *(mesh.GetVertexToElementTable());
        Array<int> v2e, vdofs;
        for (size_t i=0; i<mesh.GetNV(); i++)
        {
            h1_space.GetVertexVDofs(i, vdofs);
            vertex2element.GetRow(i, v2e);

            Array<int> flags; //存储与vertex相连的element的attribute
            for (int j=0; j<v2e.Size(); j++)
            {
                flags.Append(mesh.GetElement(v2e[j])->GetAttribute());
            }
            flags.Sort();
            flags.Unique();

            if (flags.Size() == 2) //interface上的vertex
            {
                interfacedofs.Append(vdofs);
            }
            else if (flags[0] == 1) //attribute为1的单元的vertex
            {
                attr1dofs.Append(vdofs);
            }
            else  //attribute为2的单元的vertex
            {
                assert(flags[0] == 2);
                attr2dofs.Append(vdofs);
                numElms.Append(v2e.Size());
            }
        }
        interfacedofs.Sort(); //排序后去掉重复的元素
        interfacedofs.Unique();
        attr1dofs.Sort();
        attr1dofs.Unique();
        attr2dofs.Sort();
        attr2dofs.Unique();
        assert(interfacedofs.Size() + attr1dofs.Size() + attr2dofs.Size() == h1_space.GetNVDofs());
    }
}


void Test_SelectedElement1()
{
    Mesh mesh("../../../data/1MAG_2.msh");
    int dim = mesh.Dimension();

    Array<int> marker_attr(mesh.attributes.Size());
    marker_attr = 0;
    marker_attr[1 - 1] = 1; //只对attr为1的element积分,对上述网格,单元标记只有1,2

    int p_order = 1;
    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int ndofs = h1_space.GetNDofs();

    ConstantCoefficient one(1.0);
    MissSomeElements mycoeff;

    LinearForm lf1(&h1_space);
    lf1.AddDomainIntegrator(new SelectedElement_DomainLFIntegrator(one, marker_attr));
    lf1.Assemble();

    LinearForm lf2(&h1_space);
    lf2.AddDomainIntegrator(new DomainLFIntegrator(mycoeff));
    lf2.Assemble();

    BilinearForm blf1(&h1_space);
    blf1.AddDomainIntegrator(new SelectedElement_DiffusionIntegrator(one, marker_attr));
    blf1.Assemble(0);
    blf1.Finalize(0);
    SparseMatrix blf1_mat(blf1.SpMat());

    BilinearForm blf2(&h1_space);
    blf2.AddDomainIntegrator(new DiffusionIntegrator(mycoeff));
    blf2.Assemble(0);
    blf2.Finalize(0);
    SparseMatrix blf2_mat(blf2.SpMat());

    Array<int> interfacedofs, attr1dofs, attr2dofs, numElms;
    Generate_Dofs(interfacedofs, attr1dofs, attr2dofs, numElms, mesh, h1_space);


    Test1(interfacedofs, attr1dofs, blf1_mat, blf2_mat, lf1, lf2);
    Test2(lf1, lf2, attr2dofs);
    Test3(blf1_mat, blf2_mat, attr2dofs);
    Test4(ndofs, blf1_mat, blf2_mat, lf1, lf2, attr2dofs);
}


void Test_SelectedElement_DiffusionIntegrator()
{
    Test_SelectedElement1();

    cout << "===> Test Pass: SelectedElement_DiffusionIntegrator.hpp" << endl;
}
#endif