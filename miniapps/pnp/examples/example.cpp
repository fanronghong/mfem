
#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
using namespace std;
using namespace mfem;

int main()
{
//    using namespace _DGSelfTraceIntegrator;
//    Test_DGSelfTraceIntegrator();

    Mesh mesh(16, 16, 16, Element::TETRAHEDRON, true, 2.0, 2.0, 2.0);
    {
        ofstream mesh_file("./16_16_16.mesh");
        mesh_file.precision(14);
        mesh.Print(mesh_file);
    }

}