#include <iostream>
#include "mfem.hpp"
#include "../utils/DGDiffusion_Edge_Symmetry_Penalty.hpp"
using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    using namespace _DGDiffusion_Edge_Symmetry_Penalty;

    Test_DGDiffusion_Penalty2();

    Test_DGDiffusion_Symmetry1();
}