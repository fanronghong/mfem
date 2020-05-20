
#include <iostream>
#include "mfem.hpp"
#include "../utils/SelfDefined_LinearForm.hpp"
using namespace std;
using namespace mfem;

int main()
{
    using namespace _SelfDefined_LinearForm;
    Test_SelfDefined_LFFacetIntegrator10();
}