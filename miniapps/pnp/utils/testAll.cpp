#include "AnalyticSolutionsWithMultiplePointCharges.hpp"
#include "DGSelfTraceIntegrator.hpp"
#include "EAFE_ModifyStiffnessMatrix.hpp"
#include "GradConvection_Integrator.hpp"
#include "LocalConservation.hpp"
#include "MeshInfo.hpp"
#include "MFEM2PETSc.hpp"
#include "mfem_utils.hpp"
#include "NonlinearConvection_Integrator.hpp"
#include "NonlinearPoisson_Integrator.hpp"
#include "NonlinearReaction_Integrator.hpp"
#include "PQR_GreenFunc_PhysicalParameters.hpp"
#include "PrintMesh.hpp"
#include "SelectedElement_DiffusionIntegrator.hpp"
#include "SelfDefined_LinearForm.hpp"
#include "StdFunctionCoefficient.hpp"
#include "SUPG_Integrator.hpp"

#include "DGDiffusion_Edge_Symmetry_Penalty.hpp"

#include <iostream>
using namespace std;

int main()
{
    try 
    {
        Test_AnalyticSolutionsWithMultiplePointCharges();
        Test_DGSelfTraceIntegrator();
        Test_EAFE_ModifyStiffnessMatrix();
        Test_GradConvection_Integrator();
        Test_MeshInfo();
        Test_MFEM2PETSc();
        Test_mfem_utils();
        Test_NonlinearConvection_Integrator();
        Test_NonlinearPoisson_Integrator();
        Test_NonlinearReaction_Integrator();
        Test_PQR_GreenFunc_PhysicalParameters();
        Test_PrintMesh();
        Test_SelectedElement_DiffusionIntegrator();
        Test_SelfDefined_LinearForm();
        Test_StdFunctionCoefficient();
        Test_SUPG_Integrator();

        cout << "\n========== All Tests Passed! =========" << endl;
    }
    catch (const char* msg) //如果try语句块中有throw抛出字符串,则这里catch这个字符串
    {
        cerr << msg << endl;
    }
}
