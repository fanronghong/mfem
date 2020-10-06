#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include "mfem.hpp"
#include "../utils/DGEdgeIntegrator.hpp"
#include "../utils/ProteinWaterInterfaceIntegrators.hpp"
using namespace mfem;


int main(int argc, char *argv[])
{
    Test_DGEdgeIntegrator();

}