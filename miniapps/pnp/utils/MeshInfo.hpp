/* 如果一读入网格就形成的数据结构就没有 Table, 一般是 Array<int>;
 * 而Table是在读入数据之后由MFEM内部算法形成的
 * MFEM 在读取Gmsh网格的时候，对3D网格，只保存vertex，element，boundary element，不保存其他东西
 * 区分: mesh->GetBdrFaceTransformations(i), mesh->GetBdrElementTransformation(i), i是boundary element的编号
 * FaceElementTransformations* trans = mesh->GetBdrFaceTransformations(i); //如果i对应的face在区域内部,则trans为NULL
 * ElementTransformation* eltran = mesh->GetBdrElementTransformation(i);
 *
 * 特别注意: 如果在Gmsh的3D网格中标记一部分三角形为interface或者boundary, 那么读入MFEM后,
 * 不管在Gmsh中这些facets是interface还说boundary, 在MFEM里面全部都是BdrElement,
 * 并且这些facets的normal方向由在Gmsh中组成这个facet的节点顺序的右手定则决定, MFEM不对这个顺序作调整,
 * 除非这个facet确实是整个区域的外边界, 那么MFEM会强制这个normal方向朝外.
 *
 *
 * */

#ifndef __MESHINFO_HPP__
#define __MESHINFO_HPP__

#include "mfem.hpp"
#include <cassert>
#include <cmath>

using namespace std;
using namespace mfem;

namespace _MeshInfo
{

const double TOL_MeshInfo = 1E-10;

void grad_sin_cfunc(const Vector& x, Vector& y) // 不要修改此函数,用作特例做测试用
{
    y[0] = cos(x[0]) * sin(x[1]) * sin(x[2]);
    y[1] = sin(x[0]) * cos(x[1]) * sin(x[2]);
    y[2] = sin(x[0]) * sin(x[1]) * cos(x[2]);
}

}

void VertexInfo(Mesh& mesh)
{
    using namespace _MeshInfo;
    Vector coord;
    mesh.GetVertices(coord); //得到网格的所有vertex的坐标

    // 得到mesh中每个点的坐标值
    double* ver;
    for (int i=0; i<mesh.GetNV(); i++)
    {
        ver = mesh.GetVertex(i);
//        cout << i+1 << "-th vertex: " << ver[0] << ", " << ver[1] << ", " << ver[2] << endl; // 输出vertex的坐标

        for (int j=0; j<mesh.SpaceDimension(); j++)
        {
            assert(abs(ver[j] - coord[i + j*mesh.GetNV()]) < TOL_MeshInfo);
        }
    }


    const Table& vertex2element = *(mesh.GetVertexToElementTable());
    for (int i=0; i<mesh.GetNV(); i++)
    {
        Array<int> v2e;
        vertex2element.GetRow(i, v2e);
//        v2e.Print(cout << i+1 << "-th vertex, element indices: ", v2e.Size()); //每个vertex相连的element的编号

        Element* el;
        for (int j=0; j<v2e.Size(); j++)
        {
            el = mesh.GetElement(v2e[j]);
            assert(el->GetAttribute() == 1 || el->GetAttribute() == 2);
//            cout << v2e[j] << "-th element, attribute: " << el->GetAttribute() << endl;
        }
    }
}

void VertexInfo2(Mesh& mesh)
{
    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> interfacedofs, attr1dofs, attr2dofs;
    {
        const Table& vertex2element = *(mesh.GetVertexToElementTable());
        Array<int> v2e, vdofs;
        for (size_t i=0; i<mesh.GetNV(); i++)
        {
            h1_space.GetVertexVDofs(i, vdofs);
            vertex2element.GetRow(i, v2e);
            Array<int> flags;
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
            else if (flags[0] == 1) { //attribute为1的单元的vertex
                attr1dofs.Append(vdofs);
            }
            else { //attribute为2的单元的vertex
                assert(flags[0] == 2);
                attr2dofs.Append(vdofs);
            }
        }
        interfacedofs.Sort();
        interfacedofs.Unique();
        attr1dofs.Sort();
        attr1dofs.Unique();
        attr2dofs.Sort();
        attr2dofs.Unique();
        assert(interfacedofs.Size() + attr1dofs.Size() + attr2dofs.Size() == h1_space.GetNVDofs());
    }
}

void VertexInfo3(Mesh& mesh)
{
    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> interface_dofs, attr1_dofs, attr2_dofs; //分别保存interface,attr为1,attr为2上的自由度编号
    {
        const Table& face2element = *(mesh.GetFaceToElementTable());
        for (size_t i=0; i<mesh.GetNFaces(); i++)
        {
            Array<int> f2e;
            Array<int> vdofs;
            face2element.GetRow(i, f2e);
            if (f2e.Size() == 2) //位于内部facet
            {
                int attr1 = mesh.GetElement(f2e[0])->GetAttribute();
                int attr2 = mesh.GetElement(f2e[1])->GetAttribute();
                if (attr1 != attr2) //蛋白与溶液的界面
                {
                    h1_space.GetFaceVDofs(i, vdofs);
                    interface_dofs.Append(vdofs);
                }
                else
                {
                    if (attr1 == 1) //位于蛋白区域
                    {
                        h1_space.GetFaceVDofs(i, vdofs);
                        attr1_dofs.Append(vdofs);
                    }
                    else //位于溶液区域
                    {
                        assert(attr1 == 2);
                        h1_space.GetFaceVDofs(i, vdofs);
                        attr2_dofs.Append(vdofs);
                    }
                }
            }
            else //位于边界的facet
            {
                assert(f2e.Size() == 1);
                int attr = mesh.GetElement(f2e[0])->GetAttribute();
                if (attr == 1) //位于蛋白区域
                {
                    h1_space.GetFaceVDofs(i, vdofs);
                    attr1_dofs.Append(vdofs);
                }
                else //位于溶液区域
                {
                    assert(attr == 2);
                    h1_space.GetFaceVDofs(i, vdofs);
                    attr2_dofs.Append(vdofs);
                }
            }
        }
        interface_dofs.Sort();
        interface_dofs.Unique();
        attr1_dofs.Sort();
        attr1_dofs.Unique();
        attr2_dofs.Sort();
        attr2_dofs.Unique();
    }
}


void ElementInfo(Mesh& mesh)
{
    Element* elm;
    for (int i=0; i<mesh.GetNE(); i++)
    {
        int* ver;
        elm = mesh.GetElement(i);
        assert(elm->GetAttribute() == 1 || elm->GetAttribute() == 2);
        ver = elm->GetVertices(); //得到element的attribute和组成该element的vertex编号
//        cout << i+1 << "-th element: attribute " << elm->GetAttribute() << ", " << ver[0] << ", " << ver[1] << ", " << ver[2] << ", " << ver[3] << endl;
    }

    // 组成每个element(不包括boundary element)的的vertex编号
    Array<int> e2v;
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        mesh.GetElementVertices(i, e2v);
//        e2v.Print(cout << i+1 << "-th element, vertex indices: ", e2v.Size());
    }

    // 组成每个element(不包括boundary element)的facet编号
    const Table& element2face = mesh.ElementToFaceTable();
    Array<int> e2f;
    Element* el;
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        element2face.GetRow(i, e2f);
        el = mesh.GetElement(i); //打印出element的attribute就可以知道这些element确实不包括boundary element
//        e2f.Print(cout << i+1 << "-th element, attribute: " << el->GetAttribute()
//                       << ", face indices: ", e2f.Size());
    }
}


void BdrElementInfo(Mesh& mesh)
{
    Element* bdr;
    for (int i=0; i<mesh.GetNBE(); i++)
    {
        Array<int> ver;
        bdr = mesh.GetBdrElement(i);
        bdr->GetVertices(ver);
        assert(ver.Size() == 3); //boundary element的vertex只有3个(3D mesh)，说明boundary element是facet，不是真正的四面体单元
        assert(bdr->GetAttribute() == 4 || bdr->GetAttribute() == 5 ||
                bdr->GetAttribute() == 6 || bdr->GetAttribute() == 7 ||
                bdr->GetAttribute() == 8 || bdr->GetAttribute() == 9);
//        cout << i+1 << "-th boundary element: attribute " << bdr->GetAttribute() << ", " << ver[0] << ", " << ver[1] << ", " << ver[2] << endl;
    }

    // 组成每个boundary element的vertex编号
    Array<int> be2vertex;
    Element* el;
    for (size_t i=0; i<mesh.GetNBE(); i++)
    {
        el = mesh.GetBdrElement(i);
        el->GetVertices(be2vertex);
//        be2vertex.Print(cout << i+1 << "-th boundary element, attribute: "
//                             << el->GetAttribute()
//                             << ", vertex indices: ", be2vertex.Size());
    }


    H1_FECollection h1_fec(1, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    for (size_t i=0; i<h1_space.GetNBE(); i++)
    {
//        const FiniteElement& be = *(h1_space.GetBE(i)); //和下面一种方式相同
        const Element& be = *(mesh.GetBdrElement(i)); //获取第i个边界单元

        Array<int> be2vertex;
        be.GetVertices(be2vertex);
        assert(be2vertex.Size() == 3); //间接证明boundary element不是element
//            be2vertex.Print(cout << i+1 << "-th boundary element, attribute: " << be.GetAttribute() << ", vertex indices: ", be2vertex.Size());

        Array<int> vdofs, vdofs1, vdofs2;
        if (be.GetAttribute() == 9)
        {
            h1_space.GetBdrElementVDofs(i, vdofs);

            int el, info;
            mesh.GetBdrElementAdjacentElement(i, el, info);

            int f, o;
            mesh.GetBdrElementFace(i, &f, &o);

            ElementTransformation* eltran;
            eltran = h1_space.GetBdrElementTransformation(i);

            FaceElementTransformations* trans;
            trans = mesh.GetBdrFaceTransformations(i); //如果i对应的face在边界上,则trans为NULL
            if (trans != NULL) //第i个boundary element的实际上位于区域内部的facet,不位于边界
            {
                h1_space.GetElementVDofs(trans->Elem1No, vdofs1);
                h1_space.GetElementVDofs(trans->Elem2No, vdofs2);
                vdofs1.Append(vdofs2);
            }
        }
    }


    int water_face=0, protein_face=0, inter_face=0, bdr_face=0; //记录各种Facet的数目
    FaceElementTransformations *tr;
    const Element *e1, *e2;
    for (size_t i=0; i<mesh.GetNumFaces(); i++) //所有的Facet里面包含了boundary element
    {
        tr = mesh.GetInteriorFaceTransformations(i);
        if (tr != NULL) //整个区域的 interior face
        {
            e1 = mesh.GetElement(tr->Elem1No);
            e2 = mesh.GetElement(tr->Elem2No);
            int attr1 = e1->GetAttribute();
            int attr2 = e2->GetAttribute();
            if (attr1 == 1 && attr2 == 1)
            {
//                cout << "water face." << endl;
                water_face += 1;
            }
            else if (attr1 == 2 && attr2 == 2)
            {
//                cout << "protein face." << endl;
                protein_face += 1;
            }
            else if ((attr1 == 1 && attr2 == 2) || (attr1 == 2 && attr2 == 1))
            {
//                cout << "interior face, Elem1 attr: " << attr1 << ",  Elem2 attr: " << attr2 << endl;
                inter_face += 1;
            }
            else throw "Wrong attribute!";
        }
        else //整个区域的 boundary face
        {
            tr = mesh.GetFaceElementTransformations(i);
            e1 = mesh.GetElement(tr->Elem1No); // 这个时候只存在一个单元,没有Elem2No
            int attr1 = e1->GetAttribute();
            MFEM_ASSERT((attr1 == 1) || (attr1 == 2), "Not right for boundary faces!!")
//            cout << "boundary face, Elem1(no Elem2) attr: " << attr1 << endl;
            bdr_face += 1;
        }
    }
    assert(mesh.GetNFaces() == water_face+protein_face+inter_face+bdr_face);
    assert(inter_face + bdr_face == mesh.GetNBE()); //间接证明boundary element只包括从Gmsh里面读入的除vertex，element以外的东西


    {
        int idx;
        Element *bdr;
        const Element *facet;
        Array<int> ver1(3), ver2(3);
        for (int i=0; i<mesh.GetNBE(); i++)
        {
            idx = mesh.GetBdrElementEdgeIndex(i); //对3D网格,第i个boundary element对应的facet的编号idx

            bdr = mesh.GetBdrElement(i);
            bdr->GetVertices(ver1);

            facet = mesh.GetFace(idx);
            facet->GetVertices(ver2);

            ver1.Sort();
            ver2.Sort();
            assert(ver1 == ver2);
        }
    }
}


void FacetInfo(Mesh& mesh)
{
    const Table& face2element = *(mesh.GetFaceToElementTable());
    Array<int> f2e;
    for (size_t i=0; i<mesh.GetNumFaces(); i++)
    {
        face2element.GetRow(i, f2e);
        if (f2e.Size() == 1) // 边界上的face
        {
            int attr = mesh.GetElement(f2e[0])->GetAttribute(); //与该Facet相连的第一个element的attribute
            assert(attr == 1 || attr == 2);
        }
        else // interior facet
        {
            int attr1 = mesh.GetElement(f2e[0])->GetAttribute();
            int attr2 = mesh.GetElement(f2e[1])->GetAttribute();
            if (attr1 == 1 && attr2 == 1) {
//                cout << "in protein" << endl;
            }
            else if (attr1 == 2 && attr2 == 2) {
//                cout << "in solution" << endl;
            }
            else if ((attr1 == 2 && attr2 == 1) || (attr1 == 1 && attr2 == 2)) {
//                cout << "in interface" << endl;
            }
            else {
                throw "Wrong attribute!";
            }
        }
    }
}


void FacetInfo2(Mesh& mesh)
{
    const Table& element2facet = mesh.ElementToFaceTable();
    Array<int> e2f;
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        element2facet.GetRow(i, e2f);
        for (int j=0; j<e2f.Size(); j++)
        {
            const Element* face = mesh.GetFace(e2f[j]);
//            cout << face->GetAttribute() << endl;
        }
    }
}


void FacetInfo3()
{
    Mesh mesh("../../../data/1MAG_2.msh");
    int dim = mesh.Dimension();
//    mesh.PrintInfo(cout << "mesh information:\n");

    const int p_order = 1; //有限元基函数的多项式次数
    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    Array<int> protein_dofs, water_dofs, interface_dofs;
    for (int i=0; i<h1_space.GetNE(); i++) //h1_space.GetNE()和mesh.GetNE()完全一样
    {
        const FiniteElement* fe = h1_space.GetFE(i);
        Element* el = mesh.GetElement(i);
        int attr = el->GetAttribute();
        if (attr == 1)
        {
            Array<int> dofs;
            h1_space.GetElementDofs(i, dofs);
            protein_dofs.Append(dofs);
        }
        else
        {
            assert(attr == 2);
            Array<int> dofs;
            h1_space.GetElementDofs(i, dofs);
            water_dofs.Append(dofs);
        }
    }
    for (int i=0; i<mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations* tran = mesh.GetFaceElementTransformations(i);
        if (tran->Elem2No > 0) // interior facet
        {
            const Element* e1  = mesh.GetElement(tran->Elem1No);
            const Element* e2  = mesh.GetElement(tran->Elem2No);
            int attr1 = e1->GetAttribute();
            int attr2 = e2->GetAttribute();
            if (attr1 != attr2) // interface facet
            {
                Array<int> fdofs;
                h1_space.GetFaceVDofs(i, fdofs);
                interface_dofs.Append(fdofs);
            }
        }
    }
    protein_dofs.Sort();
    protein_dofs.Unique();
    water_dofs.Sort();
    water_dofs.Unique();
    interface_dofs.Sort();
    interface_dofs.Unique();
    assert(protein_dofs.Size() + water_dofs.Size() - interface_dofs.Size() == mesh.GetNV());
    for (int i=0; i<interface_dofs.Size(); i++) // 去掉water中的interface上的dofs
    {
        water_dofs.DeleteFirst(interface_dofs[i]);
    }
    assert(protein_dofs.Size() + water_dofs.Size() == mesh.GetNV());
}


void FaceTransformationInfo()
{
    Mesh mesh("../../../data/simple.mesh");

    Vector coors;
    mesh.GetVertices(coors);

    for (size_t i=0; i<mesh.GetNumFaces(); i++)
    {
        const Element* el = mesh.GetFace(i);
        Array<int> vert;
        el->GetVertices(vert);
//        cout << i << "-th face,\n";
        for (int k=0; k<vert.Size(); k++)
        {
//            cout << vert[k] << "-th vertex: ";
            for (int l=0; l<mesh.SpaceDimension(); l++)
            {
//                cout << coors(vert[k] + mesh.GetNV()*l) << ", ";
            }
//            cout << '\n';
        }

        Geometry::Type geo_type1 = el->GetGeometryType();
        Geometry::Type geo_type2 = mesh.GetFaceGeometryType(i);
        assert(geo_type1 == geo_type2);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type1);

        ElementTransformation* eltran = mesh.GetFaceTransformation(i);
        eltran->SetIntPoint(center);
        DenseMatrix jacobi = eltran->Jacobian();
//        jacobi.Print(cout << "jacobi: \n");

        Vector normal(mesh.Dimension());
        CalcOrtho(jacobi, normal);
//        normal.Print(cout << "normal: ");

//        cout << endl;
    }
//    cout << endl;
}


void EdgeInfo(Mesh& mesh)
{
    const Table& facet2edge = *(mesh.GetFaceEdgeTable()); // Returns the face-to-edge Table (3D)
    Table edge2facet;
    Transpose(facet2edge, edge2facet); // Determine edge-to-face connections

    const Table& element2edge = mesh.ElementToEdgeTable();
    Table edge2element;
    Transpose(element2edge, edge2element); // Determine edge-to-element connections


    Array<int> e2f, e2el;
    for (size_t i=0; i<mesh.GetNEdges(); i++)
    {
//        cout << i+1 << "-th edge, facet indices: ";
        edge2facet.GetRow(i, e2f);
        for (size_t j=0; j<e2f.Size(); j++) {
//            cout << e2f[j] << ", ";
            assert(e2f[j] < mesh.GetNumFaces());
        }
//        cout << endl;

//        cout << i+1 << "-th edge, element indices: ";
        edge2element.GetRow(i, e2el);
        for (size_t j=0; j<e2el.Size(); j++)
        {
//            cout << e2el[j] << ", ";
            assert(e2el[j] < mesh.GetNE());
        }
//        cout << endl;
    }
//    mesh.PrintInfo(cout << "mesh info:\n");
}


void NormalInfo1(Mesh& mesh)
{
//    Mesh mesh(5, 5, 5, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0); //通过这个网格实验可以看出:下面5个normal没有关系!!!
    int dim = mesh.Dimension();
    Vector norm0(dim), norm1(dim), norm2(dim), norm3(dim), norm4(dim);

    for (size_t i = 0; i < mesh.GetNumFaces(); i++) // Evaluate the Jacobian at the center of the face
    {
        FaceElementTransformations *Trans = mesh.GetInteriorFaceTransformations(i);
        if (Trans != NULL) //当前facet位于interior
        {
            assert(Trans->Face->ElementNo == i);
            Geometry::Type geo_type = mesh.GetFaceGeometryType(Trans->Face->ElementNo);
            const IntegrationPoint* center = &Geometries.GetCenter(geo_type);

            Trans->Face->SetIntPoint(center);
            CalcOrtho(Trans->Face->Jacobian(), norm0); // 这个normal应该和Tran->Elem1No在这个Face上所对应的normal一样?

            /* Trans.Loc1.Transf.SetIntPoint(&ip);
             * CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
             * For example, if the face is a triangle, and the first adjacent element is a tetrahedron then
             * Trans.Loc1.Transf is the transformation that maps the reference triangle into one of
             * the four faces of the reference tetrahedron. Consequently, normal is a vector orthogonal to
             * that face of the reference tetrahedron and its length is the ratio of the area of
             * that face over the area of the reference triangle.*/
            Trans->Loc1.Transf.SetIntPoint(center);
            Trans->Loc2.Transf.SetIntPoint(center);
            CalcOrtho(Trans->Loc1.Transf.Jacobian(), norm1);
            CalcOrtho(Trans->Loc2.Transf.Jacobian(), norm2);

            IntegrationPoint eip1, eip2;
            Trans->Loc1.Transform(*center, eip1);
            Trans->Loc2.Transform(*center, eip2);
            Trans->Elem1->SetIntPoint(&eip1);
            Trans->Elem2->SetIntPoint(&eip2);
//            CalcOrtho(Trans->Elem1->Jacobian(), norm3); // fff
//            CalcOrtho(Trans->Elem2->Jacobian(), norm4); // fff

//            cout << "Integration point: " << center->x << ", " << center->y << ", " << center->z << ", " << center->weight << endl;
//            norm0.Print(cout << "norm0: ");
//            norm1.Print(cout  << "norm1: ");
//            norm2.Print(cout  << "norm2: ");
//            norm3.Print(cout  << "norm3: ");
//            norm4.Print(cout  << "norm4: ");
//            cout << endl;
        }
    }
}


void NormalInfo2()
{
    // 下面进行测试的两个网格 simple_.msh 和 simple_2.msh只是单元的编号恰好反过来,其余全部相同
    // 测试目的: 位于区域内部的facet的normal是不是从单元编号较小的指向单元编号较大的. 结果:是的
    // ref: https://github.com/mfem/mfem/issues/1122#issuecomment-547714451
    {
        Mesh mesh("../../../data/simple_.msh");
        int dim = mesh.Dimension();

        Vector norm0(dim), norm1(dim), norm2(dim), norm3(dim), norm4(dim);

        for (size_t i = 0; i < mesh.GetNumFaces(); i++)
        {
            FaceElementTransformations *Trans = mesh.GetInteriorFaceTransformations(i);
            if (Trans != NULL) //当前facet位于interior
            {
                assert(Trans->Face->ElementNo == i);
                Geometry::Type geo_type = mesh.GetFaceGeometryType(Trans->Face->ElementNo);
                const IntegrationPoint* center = &Geometries.GetCenter(geo_type);

                Trans->Face->SetIntPoint(center);
                CalcOrtho(Trans->Face->Jacobian(), norm0); // 这个normal应该和Tran->Elem1No在这个Face上所对应的normal一样?fff
                assert(abs(norm0[0] - 1.0) + abs(norm0[1] - 1.0) + abs(norm0[2] - 1.0) < 1E-10);
                Vector physical_center(dim);
                Trans->Face->Transform(*center, physical_center);
                assert(abs(physical_center[0] - 1.0/3) + abs(physical_center[1] - 1.0/3) + abs(physical_center[2] - 1.0/3) < 1E-8);

                /* Trans.Loc1.Transf.SetIntPoint(&ip);
                 * CalcOrtho(Trans.Loc1.Transf.Jacobian(), normal);
                 * For example, if the face is a triangle, and the first adjacent element is a tetrahedron then
                 * Trans.Loc1.Transf is the transformation that maps the reference triangle into one of
                 * the four faces of the reference tetrahedron. Consequently, normal is a vector orthogonal to
                 * that face of the reference tetrahedron and its length is the ratio of the area of
                 * that face over the area of the reference triangle.*/
                Trans->Loc1.Transf.SetIntPoint(center);
                Trans->Loc2.Transf.SetIntPoint(center);
                CalcOrtho(Trans->Loc1.Transf.Jacobian(), norm1);
                CalcOrtho(Trans->Loc2.Transf.Jacobian(), norm2);

                IntegrationPoint eip1, eip2;
                Trans->Loc1.Transform(*center, eip1);
                Trans->Loc2.Transform(*center, eip2);
                Trans->Elem1->SetIntPoint(&eip1);
                Trans->Elem2->SetIntPoint(&eip2);
//                CalcOrtho(Trans->Elem1->Jacobian(), norm3); // fff
//                CalcOrtho(Trans->Elem2->Jacobian(), norm4); // fff

//            cout << "Integration point: " << center->x << ", " << center->y << ", " << center->z << ", " << center->weight << endl;
//            norm0.Print(cout << "norm0: ");
//            norm1.Print(cout  << "norm1: ");
//            norm2.Print(cout  << "norm2: ");
//            norm3.Print(cout  << "norm3: ");
//            norm4.Print(cout  << "norm4: ");
//            cout << endl;
            }
        }
    }
    {
        Mesh mesh("../../../data/simple_2.msh");
        int dim = mesh.Dimension();

        Vector norm0(dim);
        for (size_t i = 0; i < mesh.GetNumFaces(); i++)
        {
            FaceElementTransformations *Trans = mesh.GetInteriorFaceTransformations(i);
            if (Trans != NULL) //当前facet位于interior
            {
                assert(Trans->Face->ElementNo == i);
                Geometry::Type geo_type = mesh.GetFaceGeometryType(Trans->Face->ElementNo);
                const IntegrationPoint* center = &Geometries.GetCenter(geo_type);

                Trans->Face->SetIntPoint(center);
                // 内部face的法向量从单元编号小的那一侧指向单元编号大的那一侧
                CalcOrtho(Trans->Face->Jacobian(), norm0);
//                norm0.Print(cout << "norm0: ");
                assert(abs(norm0[0] + 1.0) + abs(norm0[1] + 1.0) + abs(norm0[2] + 1.0) < 1E-10);
            }
        }
    }
}


void NormalInfo3()
{
    Mesh mesh("../../../data/simple__.mesh");
    int dim = mesh.Dimension();

    Vector norm0(dim), norm1(dim), norm2(dim), norm3(dim), norm4(dim), physical_center(dim);

    for (size_t i = 0; i < mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations *Trans = mesh.GetInteriorFaceTransformations(i);
        if (Trans != NULL) //当前facet位于interior
        {
            const Element* e1  = mesh.GetElement(Trans->Elem1No); // 与该内部facet相连的两个 Element (与FiniteElement区分)
            const Element* e2  = mesh.GetElement(Trans->Elem2No);
            int attr1 = e1->GetAttribute();
            int attr2 = e2->GetAttribute();

            assert(Trans->Face->ElementNo == i);
            Geometry::Type geo_type = mesh.GetFaceGeometryType(Trans->Face->ElementNo);
            const IntegrationPoint* center = &Geometries.GetCenter(geo_type);
            Trans->Face->Transform(*center, physical_center);
            if (abs(physical_center[2]) < 1E-8) break;

            Trans->Face->SetIntPoint(center);
            CalcOrtho(Trans->Face->Jacobian(), norm0); // 这个normal应该和Tran->Elem1No在这个Face上所对应的normal一样?fff

            Trans->Loc1.Transf.SetIntPoint(center);
            Trans->Loc2.Transf.SetIntPoint(center);
            CalcOrtho(Trans->Loc1.Transf.Jacobian(), norm1);
            CalcOrtho(Trans->Loc2.Transf.Jacobian(), norm2);

            IntegrationPoint eip1, eip2;
            Trans->Loc1.Transform(*center, eip1);
            Trans->Loc2.Transform(*center, eip2);
            Trans->Elem1->SetIntPoint(&eip1);
            Trans->Elem2->SetIntPoint(&eip2);
//            CalcOrtho(Trans->Elem1->Jacobian(), norm3); // fff
//            CalcOrtho(Trans->Elem2->Jacobian(), norm4); // fff

//            cout << "reference Integration point: " << center->x << ", " << center->y << ", " << center->z << endl;
//            cout << "physicl Integration point: " << physical_center[0] << ", " << physical_center[0] << ", " << physical_center[0] << endl;
//            norm0.Print(cout << "norm0: ");
//            norm1.Print(cout  << "norm1: ");
//            norm2.Print(cout  << "norm2: ");
//            norm3.Print(cout  << "norm3: ");
//            norm4.Print(cout  << "norm4: ");
//            cout << endl;
        }
    }
}

void NormalInfo4()
{
    // 再次测试: 区域内部的facet的normal是不是从单元编号较小的单元指向编号较大的单元
    // ref: https://github.com/mfem/mfem/issues/1122#issuecomment-547714451
    // 结论: 是的
    Mesh mesh("../../../data/1MAG_2.msh"); //该网格单元编号只有1和2
    int dim = mesh.Dimension();

    Vector facet_normal(dim), center1(dim), center2(dim);
    for (size_t i = 0; i < mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations *Trans = mesh.GetInteriorFaceTransformations(i);
        if (Trans != NULL) //当前facet位于interior
        {
            assert(Trans->Face->ElementNo == i);
            assert(Trans->Elem1No < Trans->Elem2No); //结论成立

            Geometry::Type type = mesh.GetFaceGeometryType(i);
            const IntegrationPoint* cent = &Geometries.GetCenter(type);
            Trans->Face->SetIntPoint(cent);
            CalcOrtho(Trans->Face->Jacobian(), facet_normal);

            // 计算与facet相连的两个四面体单元的中心
            Geometry::Type geo_type = mesh.GetElementBaseGeometry(Trans->Elem1No);
            const IntegrationPoint* center = &Geometries.GetCenter(geo_type);

            Trans->Elem1->SetIntPoint(center);
            Trans->Elem1->Transform(*center, center1);

            Trans->Elem2->SetIntPoint(center);
            Trans->Elem2->Transform(*center, center2);

            center2 -= center1;
            assert(center2 * facet_normal + 1E-10 > 0); // 结论成立.facet normal和相连单元的中心连接而成的向量同向.
        }
    }
}

void NormalInfo5()
{
    using namespace _MeshInfo;

    /* MFEM读取Gmsh的3D网格之后,只保留3部分内容:vetex的坐标,Element(四面体)的节点编号,其余剩下的都是BdrElement
     * 如果把Gmsh中的某个(些)interior facet(s)当成interface,那么这些facet在MFEM里面变成BdrElement.
     * 那么在利用 BoundaryNormalLFIntegrator 在这些facet上积分时,这个facet normal的确定方式是右手定则(由组成这个
     * facet的节点顺序形成的右手定则).
     * ref: https://github.com/mfem/mfem/issues/1122
     *      https://github.com/mfem/mfem/issues/1122#issuecomment-550053860
     * 下面来验证这个解释是否正确.
     * facet_normal1.msh和facet_normal2.msh只有interface facet的节点顺序不一致,其余完全相同
     * */
    Mesh mesh1("../../../data/facet_normal1.msh");
    H1_FECollection h1_fec1(1, mesh1.Dimension());
    FiniteElementSpace h1_space1(&mesh1, &h1_fec1);
    LinearForm lf1(&h1_space1);
    {
        Array<int> marker(mesh1.bdr_attributes.Max());
        marker = 0;
        marker[3 - 1] = 1;
        VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);
        lf1.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
        lf1.Assemble();
//        lf1.Print(cout << "lf1: ");
    }

    Mesh mesh2("../../../data/facet_normal2.msh");
    H1_FECollection h1_fec2(1, mesh2.Dimension());
    FiniteElementSpace h1_space2(&mesh2, &h1_fec2);
    LinearForm lf2(&h1_space2);
    {
        Array<int> marker(mesh2.bdr_attributes.Max());
        marker = 0;
        marker[3 - 1] = 1;
        VectorFunctionCoefficient grad_sin_coeff(3, grad_sin_cfunc);
        lf2.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_sin_coeff), marker);
        lf2.Assemble();
//        lf2.Print(cout << "lf2: ");
    }

    for (int i=0; i<lf1.Size(); ++i)
    {
        assert(abs(lf1[i] + lf2[i]) < 1E-10); // 结论成立
    }
}


void Test_MeshInfo()
{
    Mesh mesh("../../../data/1MAG_2.msh"); //不要更改这里的网格: 下面函数的调用都是针对这个网格的,换一个网格有可能会有异常
//    mesh.attributes.Print(cout << "mesh.attributes: ", mesh.attributes.Size());
//    mesh.bdr_attributes.Print(cout << "mesh.bdr_attributes: ", mesh.bdr_attributes.Size());

    VertexInfo(mesh);
    VertexInfo2(mesh);
    VertexInfo3(mesh);
    ElementInfo(mesh);
    EdgeInfo(mesh);
    BdrElementInfo(mesh);
    FacetInfo(mesh);
    FacetInfo2(mesh);
    FacetInfo3();
    FaceTransformationInfo();
    NormalInfo1(mesh);
    NormalInfo2();
    NormalInfo3();
    NormalInfo4();
    NormalInfo5();

    cout << "===> Test Pass: MeshInfo.hpp" << endl;
}

#endif //LEARN_MFEM_VERTEX_ELEMENT_BDRELEMENT_HPP
