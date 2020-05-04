
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
        if ((phy_center[0] < 0.4+tol || phy_center[0] > 0.6-tol)
            && (phy_center[1] < 0.4+tol || phy_center[1] > 0.6-tol)
            && (phy_center[2] < 0.4+tol || phy_center[2] > 0.6-tol))
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
                face->SetAttribute(7);
                interface.Append(face);
            }
        }
    }

    string temp = "gooooooo.mesh";
    PrintMesh(temp, mesh, interface);
    system(("rm " + temp).c_str()); // 调用shell命令
    cout << "===> Test Pass: PrintMesh.hpp" << endl;
}


#endif