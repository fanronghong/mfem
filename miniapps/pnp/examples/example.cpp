
#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "../utils/PrintMesh.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
#include "../utils/mfem_utils.hpp"
#include "../utils/python_utils.hpp"
using namespace std;
using namespace mfem;

double sin_func(const Vector& x) {
    return sin(x[0]) * sin(x[1]) * sin(x[2]);
}
double cos_func(const Vector& x) {
    return cos(x[0]) * cos(x[1]) * cos(x[2]);
}

int main()
{
    const int interface_marker = 9;
    Mesh mesh("../pnp_protein/1MAG_2.msh");

    int nv = mesh.GetNV();
    Vector coors;
    mesh.GetVertices(coors);

    Array<double> z_coor;
    for (int i=0; i<nv; ++i)
    {
        cout << i << "-th vertex: " << coors[i] << ", " << coors[i + nv] << ", " << coors[i + 2*nv] << endl;
        z_coor.Append(coors[i + 2*nv]);
    }

    z_coor.Sort();
//    z_coor.Unique();
    z_coor.Print(cout << "z_coor: ", z_coor.Size());
    cout << "Max z: " << z_coor.Max() << "; Mix z: " << z_coor.Min() << endl;

}