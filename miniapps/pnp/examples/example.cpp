#include <iostream>
#include "mfem.hpp"
#include "../utils/DGDiffusion_Edge_Symmetry_Penalty.hpp"
#include "../utils/DGEdgeIntegrator.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
    Test_DGDiffusion_Edge_Symmetry_Penalty();
}
