#include <iostream>
#include "mfem.hpp"
#include "../utils/PrintMesh.hpp"
using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    Mesh mesh(2, 2, Element::TRIANGLE, true);

    const Table& element2edge = mesh.ElementToEdgeTable();
    Array<int> el2e;
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        element2edge.GetRow(i, el2e);
        Element* el = mesh.GetElement(i); // 打印出element的attribute就可以知道这些element确实不包括boundary element
        el2e.Print(cout << i+1 << "-th element, attribute: " << el->GetAttribute() << ", edge indices: ", el2e.Size());
    }

    const Table& element2element = mesh.ElementToElementTable();
    Array<int> el2el;
    for (size_t i=0; i<mesh.GetNE(); i++)
    {
        element2element.GetRow(i, el2el);
        el2el.Print(cout << i << "-th element's neighbor element indices: ");
    }

    ofstream mesh_ofs("haha.mesh");
    mesh_ofs.precision(9);
    mesh.Print(mesh_ofs);
}