
#include <iostream>
#include "mfem.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
#include "../utils/DGSelfTraceIntegrator.hpp"
using namespace std;
using namespace mfem;

int main()
{
    using namespace _DGSelfTraceIntegrator;
    Test_DGSelfTraceIntegrator_5();
}