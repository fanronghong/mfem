
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


int main()
{
    Test_PrintMatrix();
}