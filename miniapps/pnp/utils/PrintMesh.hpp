
#ifndef _PRINT_MESH_HPP_
#define _PRINT_MESH_HPP_

#include <iostream>
#include <fstream>

#include "mfem.hpp"

using namespace mfem;
using namespace std;

// additional 是自己想输出的其他entities,比如interface
void PrintMesh(const string filename, const Mesh& mesh, const Array<Element*>& additional)
{
    ofstream out(filename.c_str());

    out << "MFEM mesh v1.0\n";
    // optional
    out <<
        "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
        "# POINT       = 0\n"
        "# SEGMENT     = 1\n"
        "# TRIANGLE    = 2\n"
        "# SQUARE      = 3\n"
        "# TETRAHEDRON = 4\n"
        "# CUBE        = 5\n"
        "# PRISM       = 6\n"
        "#\n";
    out << "\ndimension\n" << mesh.Dimension();

    out << "\n\nelements\n" << mesh.GetNE() << '\n';
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        const Element* el = mesh.GetElement(i);
        out << el->GetAttribute() << " ";
        out << el->GetGeometryType();
        const int nv = el->GetNVertices();
        const int* v = el->GetVertices();
        for (int j=0; j<nv; j++)
        {
            out << " " << v[j];
        }
        out << "\n";
    }

    if (additional.Size() > 0)
        out << "\nboundary\n" << mesh.GetNBE() + additional.Size() << "\n";
    else
        out << "\nboundary\n" << mesh.GetNBE() << "\n";
    for (int i=0; i<mesh.GetNBE(); i++)
    {
        const Element* el = mesh.GetBdrElement(i);
        out << el->GetAttribute() << " ";
        out << el->GetGeometryType();
        const int nv = el->GetNVertices();
        const int* v = el->GetVertices();
        for (int j=0; j<nv; j++)
        {
            out << " " << v[j];
        }
        out << "\n";
    }
    for (int i=0; i<additional.Size(); i++)
    {
        const Element* el = additional[i];
        out << el->GetAttribute() << " ";
        out << el->GetGeometryType();
        const int nv = el->GetNVertices();
        const int* v = el->GetVertices();
        for (int j=0; j<nv; j++)
        {
            out << " " << v[j];
        }
        out << "\n";
    }

    out << "\nvertices\n" << mesh.GetNV() << "\n";
    out << mesh.SpaceDimension() << "\n";
    for (int i=0; i<mesh.GetNV(); i++)
    {
        const double* v = mesh.GetVertex(i);
        out << v[0];
        for (int j=1; j<mesh.SpaceDimension(); j++)
        {
            out << " " << v[j];
        }
        out << "\n";
    }
    out << endl;
}

void Modify_Mesh()
{
    Mesh mesh("./1MAG_2.msh");

    int marker1 = 100, marker2 = 200;
    Array<Element*> interface, protein, water, boundary;
    Array<int> index_sets;
    for (size_t i=0; i<mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations* tran = mesh.GetInteriorFaceTransformations(i);
        if (tran != NULL)
        {
            int id1 = tran->Elem1No;
            int id2 = tran->Elem2No;
            Element* el1 = mesh.GetElement(id1);
            Element* el2 = mesh.GetElement(id2);

            int has_id1 = index_sets.Find(id1);
            int has_id2 = index_sets.Find(id2);
            if (has_id1 == -1) index_sets.Append(id1);
            if (has_id2 == -1) index_sets.Append(id2);

            if (el1->GetAttribute() == 1) {
                if (has_id1 == -1) protein.Append(el1);
                if (has_id2 == -1) water.Append(el2);
            } else {
                assert(el1->GetAttribute() == 2);
                if (has_id2 == -1) protein.Append(el2);
                if (has_id1 == -1) water.Append(el1);
            }

            // interior facet相连的第一个单元就是单元编号较小的,第二个是较大的.
            // 同时,interior facet的normal方向也是从单元编号较小的指向单元编号较大的
            assert(tran->Elem1No < tran->Elem2No);
            Element* face = const_cast<Element*>(mesh.GetFace(i));
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

    cout << "index_sets size: " << index_sets.Size() << ", #element of mesh: " << mesh.GetNE()
         << ", protein size: " << protein.Size() << ", water size: " << water.Size() << endl;
    assert(index_sets.Size() == mesh.GetNE());

    Array<int> reorder;
    int size = protein.Size();
    int protein_idx=0, water_idx=size;
    for (int i=0; i<mesh.GetNE(); ++i)
    {
        Element* el = mesh.GetElement(i);
        if (protein.Find(el) != -1) {
            reorder.Append(protein_idx);
            protein_idx++;
        } else {
            assert(water.Find(el) != -1);
            reorder.Append(water_idx);
            water_idx++;
        }
    }
    cout << "reorder size: " << reorder.Size() << endl;
    mesh.ReorderElements(reorder);

    PrintMesh("reorder_1MAG_2.mesh", mesh, interface);
}

void PrintInfo_mesh()
{
    Mesh mesh("./reorder_1MAG_2.mesh");
    for (int i=0; i<mesh.GetNE(); ++i) {
        Element* el = mesh.GetElement(i);
        cout << i << endl;
        if (i < 45399)
            assert(el->GetAttribute() == 1);
        else
            assert(el->GetAttribute() == 2);
    }
}

void Test_PrintMesh()
{
    Mesh mesh(10, 10, 10, Element::TETRAHEDRON, true, 1.0, 1.0, 1.0);
    int dim = mesh.Dimension();

    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        Geometry::Type geo_type = mesh.GetElementBaseGeometry(i);
        const IntegrationPoint *center = &Geometries.GetCenter(geo_type);

        ElementTransformation* Tran = mesh.GetElementTransformation(i);
        Vector phy_center(mesh.Dimension());
        Tran->Transform(*center, phy_center);

        double tol = 1E-8;
        if (phy_center[2] < 0.4+tol || phy_center[2] > 0.6-tol)
        {
            mesh.GetElement(i)->SetAttribute(1);
        }
        else
        {
            mesh.GetElement(i)->SetAttribute(2);
        }
    }


    Array<Element*> interface;
    for (size_t i=0; i<mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations* tran = mesh.GetInteriorFaceTransformations(i);
        if (tran != NULL) { //界面单元
            Element *el1 = mesh.GetElement(tran->Elem1No);
            Element *el2 = mesh.GetElement(tran->Elem2No);
            Element* face = const_cast<Element*>(mesh.GetFace(i));
            if (el1->GetAttribute() != el2->GetAttribute()) {
                face->SetAttribute(10);
                interface.Append(face);
            }
        }
    }

    string temp = "gooooooo.mesh";
    PrintMesh(temp, mesh, interface);
//    system(("rm " + temp).c_str()); // 调用shell命令
    cout << "===> Test Pass: PrintMesh.hpp" << endl;
}


#endif