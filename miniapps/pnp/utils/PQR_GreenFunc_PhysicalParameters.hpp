/* 一般来讲, 此文件中的内容不要修改!!!
 * 保存一些常用的物理量
 * */
#ifndef PQR_GreenFunc_PhysicalParameters_HPP
#define PQR_GreenFunc_PhysicalParameters_HPP

#include <iostream>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "StdFunctionCoefficient.hpp"
using namespace std;


/* 不同的计算尺度对应的物理量的无量纲化有差别
 * Angstrom_SCALE: 埃米
 * Nano_SCALE: 纳米
 * Micro_SCALE: 微米
 * */
#define Angstrom_SCALE

// ---------------------- 计算区域的尺度为 Angstrom (或者 µm), 时间尺度为 ps. 下面进行无量纲化和单位统一 ------------------------
const double T                        = 298.15;                  // Absolute temperature, unit [K]
const double e_c                      = 1.60217657e-19;          // Elementary charge quantity, unit [C]
const double k_B                      = 1.3806488e-23;           // Boltzmann constant, unit [J/K]
const double N_A                      = 6.02214129e+23;          // Avogadro's constant, unit [mol^{-1}]
const double protein_rel_permittivity = 2.0;                     // Dielectric constants for solute(like protein)
const double water_rel_permittivity   = 80.0;                    // Dielectric constants for solution(like water)
#if defined(Angstrom_SCALE)
const double vacuum_permittivity      = 8.854187817e-12 * 1e-10; // Vacuum permittivity, unit [F/m], 乘以1e-10是把单位m变成Å(埃米,Angstrom), 与Poisson方程右端项自由移动离子的浓度单位一致(Poisson方程的右端项自由移动的离子浓度为mol/L,在计算时被转换成了Angstrom)
#elif defined(Nano_SCALE)
const double vacuum_permittivity      = 8.854187817e-12 * 1e-9; // Vacuum permittivity, unit [F/m], 乘以1e-6是把单位m变成nm(纳米)
#elif defined(Micro_SCALE)
const double vacuum_permittivity      = 8.854187817e-12 * 1e-6; // Vacuum permittivity, unit [F/m], 乘以1e-6是把单位m变成µm(微米)
#endif
ConstantCoefficient epsilon_protein(protein_rel_permittivity); //蛋白质的相对介电常数
ConstantCoefficient epsilon_water(water_rel_permittivity);     //水的相对介电常数

const double alpha1 = e_c / (k_B * T);                             //电势(V)前面的无量纲化系数
const double alpha2 = e_c * e_c / (k_B * T * vacuum_permittivity); //Poisson方程右端项浓度前面的无量纲化系数
#if defined(Angstrom_SCALE)
const double alpha3 = 1.0E-27 * N_A;                               //浓度mol/L转化为1/(\mathring{A}^3)前面的系数
#elif defined(Nano_SCALE)
const double alpha3 = 1.0E-24 * N_A;                               //浓度mol/L转化为1/(\mathring{nm}^3)前面的系数
#elif defined(Micro_SCALE)
const double alpha3 = 1.0E-15 * N_A;                               //浓度mol/L转化为1/(\mathring{µm}^3)前面的系数
#endif
ConstantCoefficient alpha1_coeff(alpha1);
ConstantCoefficient alpha2_coeff(alpha2);
ConstantCoefficient alpha3_coeff(alpha3);

#if defined(Angstrom_SCALE)
const double D_K  = 0.196;       // K+  扩散系数, unit [Å^2/ps]
const double D_Cl = 0.203;       // Cl- 扩散系数, unit [Å^2/ps]
const double D_Na = 0.133;       // Na+ 扩散系数, unit [Å^2/ps]
#elif defined(Nano_SCALE)
const double D_K  = 0.196*1E-1; // K+ 扩散系数,  unit [nm^2/ps]
const double D_Cl = 0.203*1E-1; // Cl- 扩散系数, unit [nm^2/ps]
const double D_Na = 0.133*1E-1; // Na+ 扩散系数, unit [nm^2/ps]
#elif defined(Micro_SCALE)
const double D_K  = 0.196*1E-2; // K+ 扩散系数,  unit [µm^2/ps]
const double D_Cl = 0.203*1E-2; // Cl- 扩散系数, unit [µm^2/ps]
const double D_Na = 0.133*1E-2; // Na+ 扩散系数, unit [µm^2/ps]
#endif
ConstantCoefficient D_K_(D_K);
ConstantCoefficient D_Cl_(D_Cl);
ConstantCoefficient D_Na_(D_Na);

const int v_K  = +1; // K+ 化合价(valence)
const int v_Cl = -1; // Cl- 化合价
const int v_Na = +1; // Na+ 化合价
ConstantCoefficient v_K_coeff(v_K);
ConstantCoefficient v_Cl_coeff(v_Cl);
ConstantCoefficient v_Na_coeff(v_Na);

DenseMatrix D_K_mat(3, 3), D_Cl_mat(3, 3), D_Na_mat(3, 3);
void GenerateDiffusionMatrices() {
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            if (i == j) {
                D_K_mat(i, i) = D_K;
                D_Cl_mat(i, i) = D_Cl;
                D_Na_mat(i, i) = D_Na;
            } else {
                D_K_mat(i, j) = 0.0;
                D_Cl_mat(i, j) = 0.0;
                D_Na_mat(i, j) = 0.0;
            }
        }
    }
}

ConstantCoefficient D_K_prod_v_K(D_K * v_K);
ConstantCoefficient D_Cl_prod_v_Cl(D_Cl * v_Cl);
ConstantCoefficient D_Na_prod_v_Na(D_Na * v_Na);

ConstantCoefficient alpha2_prod_alpha3_prod_v_K(alpha2 * alpha3 * v_K);
ConstantCoefficient alpha2_prod_alpha3_prod_v_Cl(alpha2 * alpha3 * v_Cl);
ConstantCoefficient alpha2_prod_alpha3_prod_v_Na(alpha2 * alpha3 * v_Na);

void Test_PhysicalParameters()
{
#if defined(Angstrom_SCALE)
    double tol = 1E-8;
    // below tests must be in Angstrom units, not others (like micro, nano, meter)
    if (abs(vacuum_permittivity - 8.854187817e-12 * 1e-10) > tol) return;
    // 后面的数字由应金勇的fenics程序算出
    assert(abs(alpha2 - 7042.940010604046) < tol);
    assert(abs(alpha2*alpha3 - 4.241357984085165) < tol);
    assert(abs(alpha1*0.02569257642558086 - 1.0) < tol);
    assert(abs(alpha2/(4*M_PI*protein_rel_permittivity) - 280.22967914682994) < tol);

    cout << "======> Test Pass: Test_PhysicalParameters()" << endl;
#else
    cout << "======> Not  Test: Do not test if not Angstrom scale!" << endl;
#endif
}


// ------------------------------------------ 读取蛋白的PQR信息 -----------------------------------------
void ReadPQR(const std::string& filename, std::vector<std::vector<double>>& pqr)
{
    std::ifstream input;
    input.open(filename, std::ios::in);
    if (!input.is_open()) MFEM_ABORT("PQR file not exist!");

    std::string x, y, z, valence, buff;
    std::vector<double> entry(4); //分别存放每个ATOM的x,y,z坐标值和化合价

    while (input)
    {
        input >> buff;
        if (buff == "ATOM")
        {
            input >> buff >> buff >> buff >> buff >> x >> y >> z >> valence;
            entry[0] = stod(x);
            entry[1] = stod(y);
            entry[2] = stod(z);
            entry[3] = stod(valence);
            pqr.push_back(entry);
        }
        // 取出当前流指针所在行剩下的部分,更重要的是让流指针input移动到下一行开始位置
        getline(input, buff);
    }
    input.close();
}
void Test_ReadPQR() // 只针对一个pqr文件做了测试
{
    std::vector<std::vector<double>> pqr;
    ReadPQR("../../../data/1MAG.pqr", pqr);
    for (int i=0; i<pqr.size(); i++)
    {
        if (i == 0)
        {
            assert(pqr[i][0] == -3.690);
            assert(pqr[i][1] == -1.575);
            assert(pqr[i][2] == -2.801);
            assert(pqr[i][3] == 0.550);
        }
        if (i == 99)
        {
            assert(pqr[i][0] == -0.919);
            assert(pqr[i][1] == -4.652);
            assert(pqr[i][2] == 2.785);
            assert(pqr[i][3] == 0.000);
        }
        if (i == 299)
        {
            assert(pqr[i][0] == -0.506);
            assert(pqr[i][1] == 2.978);
            assert(pqr[i][2] == -5.983);
            assert(pqr[i][3] == 0.250);
        }
        if (i == 551)
        {
            assert(pqr[i][0] == 3.492);
            assert(pqr[i][1] == -0.565);
            assert(pqr[i][2] == -16.560);
            assert(pqr[i][3] == 0.000);
        }
    }

    cout << "======> Test Pass: Test_ReadPQR()" << endl;
}


// ---------------------- 由蛋白质里面的奇异电荷产生的电势phi1(三项分解还有phi2,phi3), 及其导数 -----------------------
class Green_func // 蛋白里面的奇异电荷部分产生的电势
{
private:
    std::vector<std::vector<double>> pqr_info;
public:
    Green_func(const char* filename)
    {
        ReadPQR(filename, pqr_info);
    }
    double operator()(const mfem::Vector& x)
    {
        double alpha2_4pi_epsilon = alpha2 / (4 * M_PI * protein_rel_permittivity);
        double sum = 0.0;
        for (size_t i=0; i<pqr_info.size(); i++)
        {
            double dist =   pow(x[0] - pqr_info[i][0], 2)
                          + pow(x[1] - pqr_info[i][1], 2)
                          + pow(x[2] - pqr_info[i][2], 2);
            sum += pqr_info[i][3] / sqrt(dist); // 带电量除以距离
        }
        return alpha2_4pi_epsilon * sum;
    }
};
class gradGreen_func
{
private:
    std::vector<std::vector<double>> pqr_info;
public:
    gradGreen_func(const char* filename)
    {
        ReadPQR(filename, pqr_info);
    }
    void operator()(const mfem::Vector& x, mfem::Vector& y)
    {
        double alpha2_4pi_epsilon = alpha2 / (4 * M_PI * protein_rel_permittivity);

        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (size_t i=0; i<pqr_info.size(); i++)
        {
            double dist = pow(x[0] - pqr_info[i][0], 2) + pow(x[1] - pqr_info[i][1], 2) + pow(x[2] - pqr_info[i][2], 2);

            sum_x += pqr_info[i][3] * (pqr_info[i][0] - x[0]) * pow(dist, -3.0/2);
            sum_y += pqr_info[i][3] * (pqr_info[i][1] - x[1]) * pow(dist, -3.0/2);
            sum_z += pqr_info[i][3] * (pqr_info[i][2] - x[2]) * pow(dist, -3.0/2);
        }
        y[0] = alpha2_4pi_epsilon * sum_x;
        y[1] = alpha2_4pi_epsilon * sum_y;
        y[2] = alpha2_4pi_epsilon * sum_z;
    }
};
void Test_G_gradG_cfun() // 只针对特定的网格和特定的参数做测试
{
    const char* pqr_file = "../../../data/1MAG.pqr";
    mfem::Mesh mesh("../../../data/1MAG_2.msh", 1, 1);
    mfem::H1_FECollection h1_fec(1, 3);
    mfem::FiniteElementSpace h1_space(&mesh, &h1_fec);
    mfem::FiniteElementSpace h1_vec_space(&mesh, &h1_fec, 3);
    mfem::GridFunction phi1(&h1_space), grad_phi1(&h1_vec_space);

    mfem::ConstantCoefficient zero(0.0);
    mfem::Vector zero_(3);
    zero_ = 0.0;
    mfem::VectorConstantCoefficient zero_vec(zero_);

    Green_func G_cfun(pqr_file);
    StdFunctionCoefficient G_coeff(G_cfun);
    phi1.ProjectCoefficient(G_coeff);

    gradGreen_func gradG_cfun(pqr_file);
    VectorStdFunctionCoefficient grad_phi1_coeff(3, gradG_cfun);
    grad_phi1.ProjectCoefficient(grad_phi1_coeff);

    LinearForm lf(&h1_space);
    lf.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_phi1_coeff)); // Neumann bdc on Gamma_m, take negative below
    lf.Assemble();

    double L2norm = phi1.ComputeL2Error(zero);
    double L2norm_ = grad_phi1.ComputeL2Error(zero_vec);
    assert(abs(L2norm - 2.1067E+03) < 10); //数据由张倩如提供
    assert(abs(L2norm_ - 9.2879E+03) < 10); //数据由张倩如提供

    cout << "======> Test Pass: Test_G_gradG_cfun()" << endl;
}


void Test_PQR_GreenFunc_PhysicalParameters()
{
    Test_PhysicalParameters();
    Test_ReadPQR();
    Test_G_gradG_cfun();

    cout << "===> Test Pass: PQR_GreenFunc_PhysicalParameters.hpp" << endl;
}

#endif //LEARN_MFEM_PQR_GREENFUNC_HXX
