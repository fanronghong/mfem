/* ref:
 * 1. D. Xie, H. W. Volkmer, J. Ying, Analytical solutions of nonlocal Poisson dielectric models with multiple
 *    point charges inside a dielectric sphere, Physical Review E (2016)
 *
 * 2. D. Xie, J. Ying, A new box iterative method for a class of nonlinear interface problems with application
 *    in solving Poisson-Boltzmann equation, Journal of Computational and Applied Mathematics (2016)
 *
 *
 *
 * */

#ifndef LEARN_MFEM_ANALYTICSOLUTIONSWITHMULTIPLEPOINTCHARGES_HXX
#define LEARN_MFEM_ANALYTICSOLUTIONSWITHMULTIPLEPOINTCHARGES_HXX

#include <iostream>
#include <vector>

using namespace std;


class AnalyticSolutionsWithMultiplePointCharges
{
private:
    int num_s;
    int num_charge;

public:
    AnalyticSolutionsWithMultiplePointCharges() {}

    void LocalPoisson_Coeff(vector<vector<double>>& AN, vector<vector<double>>& BN,
                            vector<double> R_charge, vector<vector<double>>& charge_coor)
    {

    }

};


void Test_AnalyticSolutionsWithMultiplePointCharges()
{

    cout << "===> Test Pass: AnalyticSolutionsWithMultiplePointCharges.hpp" << endl;
}

#endif //LEARN_MFEM_ANALYTICSOLUTIONSWITHMULTIPLEPOINTCHARGES_HXX
