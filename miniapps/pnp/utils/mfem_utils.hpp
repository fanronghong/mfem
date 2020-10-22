//
// Created by fan on 2019/9/21.
//

#ifndef __MFEM_UTILITIES_HPP__
#define __MFEM_UTILITIES_HPP__

#include <iostream>
#include <cassert>
#include <string>
#include <unistd.h>
//#include <Python.h>
#include <fstream>

//#include "../utils/matplotlibcpp.hpp" // not work in computer cluster
//#include "../utils/gnuplot_cpp.hpp" //Gnuplot class handles POSIX-Pipe-communikation with Gnuplot
#include "mfem.hpp"

using namespace std;
using namespace mfem;


double TOL_utilities = 1E-10;

double pi = 3.141592653589793e0;



// -------------------------- 一些辅助功能的函数 -------------------------------------
Array<double> compute_convergence(Array<double> errornorms, Array<double> sizes)
{
    Array<double> rates;
    for (int i=0; i<errornorms.Size() - 1; i++){
        rates.Append(log(errornorms[i]/errornorms[i+1]) / log(sizes[i]/sizes[i+1]));
    }
    return rates;
}

void PecletInfo(std::vector<double> vec)
{
    int size = vec.size();
    double sum=0, average=0.0, smallest=0.0, biggest=0.0;

    for (const auto& itm: vec) {
        sum += itm;
        if (itm > smallest) smallest = itm;
        if (itm < biggest) biggest = itm;
    }

    average = sum / size;
    double _1_4 = average / 2.0;
    double _3_4 = _1_4 * 3.0;

    int num_1_4=0, num_2_4=0, num_3_4=0, num_4_4=0; // (0,1/4), (1/4, 2/4), (2/4, 3/4), (3/4, 1)
    for (const auto& itm: vec) {
        if (itm <= _1_4) num_1_4++;
        if (itm > _1_4 && itm <= average) num_2_4++;
        if (itm > average && itm <= _3_4) num_3_4++;
        if (itm > _3_4) num_4_4++;
    }

    cout << "Number of all Peclets: " << size << '\n'
         << "averge: " << average << '\n'
         << "number of (0, 1/4): " << num_1_4 << '\n'
         << "number of (1/4, 1/2): " << num_2_4 << '\n'
         << "number of (2/4, 3/4): " << num_3_4 << '\n'
         << "number of (3/4, 1): " << num_4_4 << endl;
}





void WriteCSR(const char* file, const SparseMatrix& A)
{
    cout << "===> Begin Writing CSR matrix, to " << file << endl;
    // 按照FASP solver中的 faspsolver/base/src/BlaIO.c中的函数
    // void fasp_dcsr_read (const char *filename, dCSRmat *A)
    int nrows = A.Size();
    const int *row_offsets = A.GetI();

    int nnz = row_offsets[nrows] - row_offsets[0];

    const int* colnindices = A.GetJ();
    const double* values = A.GetData();

    ofstream out(file);
    out.precision(16);
    out << nrows << '\n';
    for (int i=0; i<=nrows; i++)
    {
        out << row_offsets[i] << '\n';
    }
    for (int i=0; i<nnz; i++)
    {
        out << colnindices[i] << '\n';
    }
    for (int i=0; i<nnz; i++)
    {
        out << values[i] << '\n';
    }

    cout << "number of rows: " << nrows << endl;
    cout << "number of non-zero: " << nnz << endl;
    cout << "===> End Writing CSR matrix" << endl;
}

void ReadCSR(const char* file, SparseMatrix& spmat)
{
    cout << "\n===> Begin reading CSR SparseMatrix, from " << file << endl;
    ifstream in(file);
    int size = 0;
    in >> size;
    cout << "dimension of CSR matrix: " << size << endl;
    Array<int> row_offsets(size);
    for (int i=0; i<size; i++)
        in >> row_offsets[i];

    int nnz = row_offsets[size-1] - row_offsets[0];
    Array<int> coln_indices(nnz);
    for (int i=0; i<nnz; i++)
        in >> coln_indices[i];

    Array<double> values(nnz);
    for (int i=0; i<nnz; i++)
        in >> values[i];

    cout << "===> End reading CSR SparseMatrix\n" << endl;
}

void BinaryWriteCSR(const char* file, const SparseMatrix& A)
{
    cout << "===> Begin Writing CSR matrix, to " << file << endl;
    // 按照FASP solver中的 faspsolver/base/src/BlaIO.c中的函数
    // void fasp_dcsr_read (const char *filename, dCSRmat *A)
    int nrows = A.Size();
    const int *row_offsets = A.GetI();

    int nnz = row_offsets[nrows] - row_offsets[0];

    const int* colnindices = A.GetJ();

    const double* values = A.GetData();

    ofstream tofile(file, ios::binary);
    tofile.precision(16);
    tofile.write((char*)&nrows, sizeof(nrows));
    for (int i=0; i<=nrows; i++)
    {
        tofile.write((char*)&(row_offsets[i]), sizeof(row_offsets[i]));
    }
    for (int i=0; i<nnz; i++)
    {
        tofile.write((char*)&(colnindices[i]), sizeof(colnindices[i]));
    }
    for (int i=0; i<nnz; i++)
    {
        tofile.write((char*)&(values[i]), sizeof(values[i]));
    }

//    delete []row_offsets;
//    delete []colnindices;
//    delete []values;
    cout << "number of rows: " << nrows << endl;
    cout << "number of non-zero: " << nnz << endl;
    cout << "===> End Writing CSR matrix" << endl;
}

void BinaryReadCSR(const char* file, SparseMatrix& spmat)
{
    cout << "\n===> Begin reading CSR SparseMatrix, from " << file << endl;
    int* I = spmat.GetI();
    int* J = spmat.GetJ();
    double* data = spmat.GetData();

    ifstream tomemory(file, ios::binary);

    int size = 0;
    tomemory.read((char*)&size, sizeof(size));
    cout << "dimension of CSR matrix: " << size << endl;

    int* row_offsets = new int[size + 1];
    int idx;
    for (int i=0; i<size+1; i++)
    {
        tomemory.read((char*)&idx, sizeof(idx));
        row_offsets[i] = idx;
    }

    int nnz = row_offsets[size] - row_offsets[0];
    int* coln_indices = new int[nnz];
    int col_idx;
    for (int i=0; i<nnz; i++)
    {
        tomemory.read((char*)&col_idx, sizeof(col_idx));
        coln_indices[i] = col_idx;
    }

    double* values = new double[nnz];
    double val;
    for (int i=0; i<nnz; i++)
    {
        tomemory.read((char*)&val, sizeof(val));
        values[i] = val;
    }

    SparseMatrix spmat_(row_offsets, coln_indices, values, size, size);
    spmat = spmat_; //fff not good.

    tomemory.close();

    cout << "===> End reading CSR SparseMatrix\n" << endl;
//    delete []row_offsets;
//    delete []coln_indices;
//    delete []values;
//    spmat_.Print(cout << "spmat_: \n");
}

void WriteVector(const char *file, const Vector &vec)
{
    cout << "===> Begin Writing Vector, to " << file << endl;
    const int nelm = vec.Size();
    const double *values = vec.GetData();

    ofstream tofile(file);
    tofile.precision(16);
    tofile << nelm << '\n';
    for (int i=0; i<nelm; i++)
    {
        tofile << values[i] << '\n';
    }

    cout << "number of elements: " << nelm << endl;
    cout << "===> End Writing Vector" << endl;
}

void BinaryWriteVector(const char *file, const Vector &vec)
{
    cout << "===> Begin Writing Vector, to " << file << endl;
    const int nelm = vec.Size();
    const double *values = vec.GetData();

    ofstream tofile(file, ios::binary);
    tofile.precision(16);
    tofile.write((char*)&nelm, sizeof(int));

    for (int i=0; i<nelm; i++)
    {
        tofile.write((char*)&(values[i]), sizeof(double));
    }
    tofile.close();

    cout << "number of elements: " << nelm << endl;
    cout << "===> End Writing Vector" << endl;

}

void ReadVector(const char* file, Vector &vec)
{
    cout << "\n===> Begin reading Vector, from " << file << endl;
    ifstream in(file);
    int size = 0;
    in >> size;
    cout << "dimension of vector: " << size << endl;
    vec.SetSize(size);
    for (int i=0; i<size; i++)
    {
        in >> vec[i];
    }
    cout << "===> End reading Vector\n" << endl;
}

void BinaryReadVector(const char* file, Vector &vec)
{
    cout << "\n===> Begin reading Vector, from " << file << endl;
    ifstream tomemory(file, ios::binary);
    int size = 0;
    tomemory.read((char*)&size, sizeof(int));

    cout << "dimension of vector: " << size << endl;
    vec.SetSize(size);
    for (int i=0; i<size; i++)
    {
        tomemory.read((char*)&(vec[i]), sizeof(double));
    }
    tomemory.close();

    cout << "===> End reading Vector\n" << endl;
}

void AboutVector(ostream &out, const Vector &vec)
{
    vec.Print(out << '\n', 10000);
    out << "Maximum of Vector: " << vec.Max() << endl;
    out << "Minimum of Vector: " << vec.Min() << endl;
    out << "l1 norm of Vector: " << vec.Norml1() << endl;
    out << "l2 norm of Vector: " << vec.Norml2() << endl;
    out << "lInf norm of Vector: " << vec.Normlinf() << endl;

    out << endl;
}




// 串行的可视化
int Wx = 0, Wy = 0;            // window position
int Ww = 350, Wh = 350;        // window size
int offx = Ww+5, offy = Wh+25; // window offsets
void Visualize(VisItDataCollection& dc, string field, string title="No title", string caption="No caption", int x=Wx, int y=Wy)
{
    int w = Ww, h = Wh;

    char vishost[] = "localhost";
    int  visport   = 19916;

    socketstream sol_sockL2(vishost, visport);
    sol_sockL2.precision(10);
    sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField(field)
               << "window_geometry " << x << " " << y << " " << w << " " << h
               << "window_title '" << title << "'"
               << "plot_caption '" << caption << "'" << flush; // 主要单引号和单引号前面(关键字window_title, plot_caption后面)的空格
}
void ShowMesh(Mesh& mesh, string title="", string caption="", int x=Wx, int y=Wy)
{
    int w = Ww, h = Wh;

    char vishost[] = "localhost";
    int  visport   = 19916;

    socketstream sol_sockL2(vishost, visport);
    sol_sockL2.precision(10);
    sol_sockL2 << "mesh\n" << mesh
               << "window_geometry " << x << " " << y << " " << w << " " << h
               << "window_title '" << title << "'"
               << "plot_caption '" << caption << "'" << flush; // 主要单引号和单引号前面(关键字window_title, plot_caption后面)的空格
}
void SaveMesh(Mesh& mesh, const string file_name)
{
    ofstream out_file(file_name);
    out_file.precision(14);
    mesh.Print(out_file);
    std::cout << "Save mesh to " << file_name << std::endl;
}



void Get_ess_tdof_list(FiniteElementSpace& fsp, const Mesh& mesh_, const int marker, Array<int>& ess_tdof_list_)
{
    if (mesh_.bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh_.bdr_attributes.Max());
        ess_bdr = 0;
        ess_bdr[marker - 1] = 1;
        fsp.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);
    }
}
void Test_Get_ess_tdof_list()
{
    Mesh mesh("../../../data/1MAG_2.msh", 1, 1);
    H1_FECollection h1_fec(1, 3);
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int protein_marker = 1;
    int water_marker = 2;
    int interface_marker = 9;

    Array<int> protein_dofs, water_dofs, interface_dofs; // 最终protein_dofs和water_dofs里面不包含interface_dofs
    for (int i=0; i<h1_space.GetNE(); i++)
    {
        Element* el = mesh.GetElement(i);
        int attr = el->GetAttribute();
        if (attr == protein_marker)
        {
            Array<int> dofs;
            h1_space.GetElementDofs(i, dofs);
            protein_dofs.Append(dofs);
        }
        else
        {
            assert(attr == water_marker);
            Array<int> dofs;
            h1_space.GetElementDofs(i, dofs);
            water_dofs.Append(dofs);
        }
    }
    for (int i=0; i<mesh.GetNumFaces(); i++)
    {
        FaceElementTransformations* tran = mesh.GetFaceElementTransformations(i);
        if (tran->Elem2No > 0) // interior facet
        {
            const Element* e1  = mesh.GetElement(tran->Elem1No);
            const Element* e2  = mesh.GetElement(tran->Elem2No);
            int attr1 = e1->GetAttribute();
            int attr2 = e2->GetAttribute();
            if (attr1 != attr2) // interface facet
            {
                Array<int> fdofs;
                h1_space.GetFaceVDofs(i, fdofs);
                interface_dofs.Append(fdofs);
            }
        }
    }

    protein_dofs.Sort();
    protein_dofs.Unique();
    water_dofs.Sort();
    water_dofs.Unique();
    interface_dofs.Sort();
    interface_dofs.Unique();
    assert(protein_dofs.Size() + water_dofs.Size() - interface_dofs.Size() == mesh.GetNV()); //目前protein_dofs和water_dofs都含有interface_dofs

    for (int i=0; i<interface_dofs.Size(); i++) // 去掉protein和water中的interface上的dofs
    {
        protein_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后protein_dofs里面不可能有相同的元素
        water_dofs.DeleteFirst(interface_dofs[i]); //经过上面的Unique()函数后water_dofs里面不可能有相同的元素
    }
    assert(protein_dofs.Size() + water_dofs.Size() + interface_dofs.Size() == mesh.GetNV());

    Array<int> top_ess_tdof_list, bottom_ess_tdof_list, interface_ess_tdof_list, ess_tdof_list;
    Get_ess_tdof_list(h1_space, mesh, interface_marker, interface_ess_tdof_list);
    assert(interface_dofs == interface_ess_tdof_list); //从侧面验证了函数Get_ess_tdof_list()是正确的,上面自己形成的interface_dofs的过程也正确

}



// 在蛋白区域单元取值为1.0, 在水单元取值为0.0
class MarkProteinCoefficient : public Coefficient
{
private:
    int protein_marker, water_marker;
public:
    MarkProteinCoefficient(int protein_marker_, int water_marker_)
                    : protein_marker(protein_marker_), water_marker(water_marker_) {}
    virtual ~MarkProteinCoefficient() { }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        if (T.Attribute == protein_marker) // 蛋白区域
            return 1.0;
        else if (T.Attribute == water_marker) // 溶液区域
            return 0.0;
        else
            MFEM_ABORT("Something wrong with mesh markers!");
    }
};
class MarkWaterCoefficient : public Coefficient
{
private:
    int protein_marker, water_marker;
public:
    MarkWaterCoefficient(int protein_marker_, int water_marker_)
                : protein_marker(protein_marker_), water_marker(water_marker_) {}

    virtual ~MarkWaterCoefficient() { }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        if (T.Attribute == protein_marker) // 蛋白区域
            return 0.0;
        else if (T.Attribute == water_marker) // 溶液区域
            return 1.0;
        else
            throw "Something wrong with mesh markers!";
    }
};
class EpsilonCoefficient : public Coefficient
{
private:
    int protein_marker, water_marker;
    double epsilon_protein, epsilon_water;

public:
    EpsilonCoefficient(int protein_marker_, int water_marker_, double epsilon_protein_, double epsilon_water_)
            : protein_marker(protein_marker_), water_marker(water_marker_),
              epsilon_protein(epsilon_protein_), epsilon_water(epsilon_water_) {}

    virtual ~EpsilonCoefficient() { }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        if (T.Attribute == protein_marker) // 蛋白区域
            return epsilon_protein;
        else if (T.Attribute == water_marker) // 溶液区域
            return epsilon_water;
        else
            throw "Something wrong with mesh markers!";
    }
};
void Test_Mark_Protein_Water_Coefficient()
{
    MarkProteinCoefficient protein(1, 2);
    MarkWaterCoefficient   water(1, 2);

    Mesh mesh("../../../data/1MAG_2.msh");
    int dim = mesh.Dimension();

    int p_order = 1;
    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    GridFunction gf1(&h1_space), gf2(&h1_space);
    gf1.ProjectCoefficient(protein);
    gf2.ProjectCoefficient(water);

    for (int i=0; i<h1_space.GetNDofs(); ++i)
    {
        if (gf1[i] == 1) assert(gf2[i] == 0);
        if (gf1[i] == 0) assert(gf2[i] == 1);
    }
}


// 计算给定physical point的函数(包括Coefficient, GridFunction)值
double ComputePhysicalPointValue(Coefficient& coeff, const Vector& phy_point, Mesh& mesh)
{
    int elem_idx=-1; // 该physical point所对应的单元编号
    IntegrationPoint ip; // 该physical point所对应的在参考单元内部的reference point
    ElementTransformation* tran; // 从reference point到physical point的变换
    for (int i=0; i<mesh.GetNE(); ++i)
    {
        tran = mesh.GetElementTransformation(i);
        InverseElementTransformation invtran(tran); // 从physical point到reference point的变换
        int ret = invtran.Transform(phy_point, ip);
        if (ret == 0)
        {
            elem_idx = i;
            break;
        }
    }

    if (elem_idx == -1) {
        throw "The physical point is not located in the given mesh";
    }

    return coeff.Eval(*tran, ip);
}
double ComputePhysicalPointValue(GridFunction& gf, const Vector& phy_point, Mesh& mesh)
{
    int elem_idx=-1; // 该physical point所对应的单元编号
    IntegrationPoint ip; // 该physical point所对应的在参考单元内部的reference point
    ElementTransformation* tran; // 从reference point到physical point的变换
    for (int i=0; i<mesh.GetNE(); ++i)
    {
        tran = mesh.GetElementTransformation(i);
        InverseElementTransformation invtran(tran); // 从physical point到reference point的变换
        int ret = invtran.Transform(phy_point, ip);
        if (ret == 0)
        {
            elem_idx = i;
            break;
        }
    }

    if (elem_idx == -1) {
        throw "The physical point is not located in the given mesh";
    }

    return gf.GetValue(elem_idx, ip);
}
void ComputePhysicalPointsValues(Coefficient& coeff, DenseMatrix& phy_points, Vector& vals, Mesh& mesh)
{
    // phy_points是一个DenseMatrix, 每一列表示一个physical point的坐标值
    int width = phy_points.Width();
    vals.SetSize(width);

    Array<int> elem_ids(width); // 所有的physical points所属的单元编号
    Array<IntegrationPoint> ips(width); // 所有的physical points在参考单元所对应的积分点

    mesh.FindPoints(phy_points, elem_ids, ips);

    if (elem_ids.Find(-1) != -1) throw "Some point is not located in the given mesh";

    for (int i=0; i<width; ++i)
    {
        ElementTransformation* trans = mesh.GetElementTransformation(elem_ids[i]);
        vals(i) = coeff.Eval(*trans, ips[i]);
    }
}
void ComputePhysicalPointsValues(GridFunction& gf, DenseMatrix& phy_points, Vector& vals, Mesh& mesh)
{
    // phy_points是一个DenseMatrix, 每一列表示一个physical point的坐标值
    int width = phy_points.Width();
    vals.SetSize(width);

    Array<int> elem_ids(width); // 所有的physical points所属的单元编号
    Array<IntegrationPoint> ips(width); // 所有的physical points在参考单元所对应的积分点

    mesh.FindPoints(phy_points, elem_ids, ips);

    if (elem_ids.Find(-1) != -1) throw "Some point is not located in the given mesh";

    for (int i=0; i<width; ++i)
    {
        vals(i) = gf.GetValue(elem_ids[i], ips[i]);
    }
}
// just for tests
double func_Compute_PhysicalPoints_Values(Vector& x)
{
    return 1.5 * x[0] + 3.3 * x[1];
}
void Test_Compute_PhysicalPoints_Values()
{
    Mesh mesh(100, 100, Element::TRIANGLE, true, 1.0, 1.0);

    int dim = mesh.Dimension();

    int p_order = 1;
    H1_FECollection h1_fec(p_order, mesh.Dimension());
    FiniteElementSpace h1_space(&mesh, &h1_fec);

    GridFunction gf(&h1_space);
    FunctionCoefficient coeff(func_Compute_PhysicalPoints_Values);
    gf.ProjectCoefficient(coeff);

    { // test ComputePhysicalPointValue(): only one physical point
        Vector phy_point(dim);
        phy_point(0) = 0.85;
        phy_point(1) = 0.25;
        assert(abs(ComputePhysicalPointValue(coeff, phy_point, mesh) - (1.5*0.85 + 3.3*0.25)) < 1E-10);
        assert(abs(ComputePhysicalPointValue(gf, phy_point, mesh) - (1.5*0.85 + 3.3*0.25)) < 1E-10);
    }

    { // test ComputePhysicalPointsValues(): multiple physical points
        int width = 20;
        DenseMatrix phy_points(dim, width);
        for (int i=0; i<dim; ++i) {
            for (int j=0; j<phy_points.Width(); ++j) {
                phy_points(i, j) = rand() / double(RAND_MAX); // 取得0～1之间的浮点数
            }
        }

        Vector vals1, vals2;
        ComputePhysicalPointsValues(coeff, phy_points, vals1, mesh);
        ComputePhysicalPointsValues(gf, phy_points, vals2, mesh);

        for (int i=0; i<phy_points.Width(); ++i) {
            assert(abs(vals1(i) - vals2(i)) < 1E-10);
            assert(abs(vals2(i) - (1.5 * phy_points(0, i) + 3.3 * phy_points(1, i))) < 1E-10);
        }
    }
}


// comment for not use in computer cluster
///* style: points, dots, filledcurves
// * */
//void PlotSparsePattern()
//{
//    namespace plt = matplotlibcpp;
//
//    int II[5] = {0, 2, 4, 7, 9};
//    int JJ[9] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
//    double VVals[9] = {1, 7, 2, 8, 5, 3, 9, 6, 4};
//    SparseMatrix mat(II, JJ, VVals, 4, 4);
//
//    const int* I = mat.GetI();
//    const int* J = mat.GetJ();
//    const double* Vals = mat.GetData();
//    int size = mat.Size();
//
//    std::vector<int> x, y;
//    std::vector<double> vals;
//    for (int i=0; i<size; ++i) {
//        for (int j=I[i]; j<I[i+1]; ++j) {
//            if (abs(Vals[j]) < 1E-10) continue;
//            x.push_back(i);
//            y.push_back(J[j]);
//            vals.push_back(Vals[j]);
//        }
//    }
//
//    // Set the size of output image = 1200x780 pixels
//    plt::figure();
//
//    // Plot line from given x and y data. Color is selected automatically.
////    plt::scatter(x, y);
//    plt::plot(x, y);
//
//    plt::show();
//}
//void PrintSparsePattern(const SparseMatrix& mat,
//        const string title="",
//        const string style="points")
//{
//    const int* I = mat.GetI();
//    const int* J = mat.GetJ();
//    const double* Vals = mat.GetData();
//    int size = mat.Size();
//
//    std::vector<int> x, y;
//    std::vector<double> vals;
//    for (int i=0; i<size; ++i) {
//        for (int j=I[i]; j<I[i+1]; ++j) {
//            if (abs(Vals[j]) < 1E-10) continue;
//            x.push_back(i);
//            y.push_back(J[j]);
////            cout << Vals[j] << endl;
//            vals.push_back(Vals[j]);
//        }
//    }
//
//    double min_val = 0.0, max_val = 0.0;
//    for (const auto& itm: vals) {
////        cout << itm << endl;
//        if (min_val > itm) min_val = itm;
//        if (max_val < itm) max_val = itm;
//    }
//
//    // points, dots, filledcurves, ...
//    Gnuplot gp(style);
//    gp.reset_plot();
//    gp.set_xrange(0.0-size/10.0, size + size/10.0);
//    gp.set_yrange(0.0-size/10.0, size + size/10.0);
//    gp.set_zrange(min_val, max_val);
//
//    gp.set_pointsize(3);
//    gp.set_title(title);
//
//    if (style == "dots" || style == "points") {
//        gp.plot_xy(x, y);
//    } else {
//        assert(style == "filledcurves");
//        gp.plot_xyz(x, y, vals);
//    }
//
//    cout << endl << "Press ENTER to continue..." << endl;
//    std::cin.clear();
//    std::cin.ignore(std::cin.rdbuf()->in_avail());
//    std::cin.get();
//}
//void Test_PrintSparsePattern()
//{
//    int I[5] = {0, 2, 4, 7, 9};
//    int J[9] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
//    double Vals[9] = {1, 7, 2, 8, 5, 3, 9, 6, 4};
//
//    SparseMatrix sp(I, J, Vals, 4, 4);
////    sp.Print(cout);
//
//    PrintSparsePattern(sp);
//}


void PrintMatrix(const SparseMatrix& sp, ostream& output= std::cout)
{
    DenseMatrix* mat = sp.ToDenseMatrix();
    int size = mat->Size();
    for (int i=0; i<size; ++i)
    {
        for (int j=0; j<size; ++j)
        {
            output << std::setiosflags(ios::showpos)
                    << std::setw(10) << std::setfill(' ') << std::scientific
                   << std::setprecision(2) << std::left;
            if (abs((*mat)(i, j)) > 1E-10)
                output << (*mat)(i, j);
            else
                output << ' ';
        }
        output << '\n';
    }
}
void PrintMatrix(const DenseMatrix& mat, ostream& output= std::cout)
{
    int size = mat.Size();
    for (int i=0; i<size; ++i)
    {
        for (int j=0; j<size; ++j)
        {
            output << std::setw(10) << std::setfill(' ') << std::scientific
                    << std::setprecision(2) << std::left;
            if (abs(mat(i, j)) > 1E-10)
                output << mat(i, j);
            else
                output << ' ';
        }
        output << '\n';
    }
}
void Test_PrintMatrix()
{
    int I[5] = {0, 2, 4, 7, 9};
    int J[9] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
    double Vals[9] = {-1.0, 70001, 2, 8, 5, 3, 9, -0.00126, 4.0213};

    SparseMatrix sp(I, J, Vals, 4, 4);
    DenseMatrix den;
    sp.ToDenseMatrix(den);
    den.Print(cout << "Dense matrix:\n");
    PrintMatrix(sp, cout << "Sparse matrix print:\n");
    PrintMatrix(den, cout << "Dense matrix print:\n");
    cout << "hhhhhhhhhhh" << endl;
}



void Test_mfem_utils()
{
    Test_Get_ess_tdof_list();
    Test_Compute_PhysicalPoints_Values();
    Test_Mark_Protein_Water_Coefficient();

    cout << "===> Test Pass: mfem_utils.hpp" << endl;
}
#endif //LEARN_MFEM_UTILITIES_H
