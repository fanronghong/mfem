#include "mfem.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    const char* mesh_file   = "./1MAG_2.msh"; // 带有蛋白的网格,与PQR文件必须匹配
    int refine_times        = 0;
    const char* Linearize   = "gummel"; // newton, gummel
    const char* Discretize  = "cg"; // cg, dg

    string mesh_temp(mesh_file);
    mesh_temp.erase(mesh_temp.find(".msh"), 4);
    mesh_temp.erase(mesh_temp.find("./"), 2);

    string name = "_ref" + to_string(refine_times) + "_" + mesh_temp + "_" + string(Discretize) + "_" + string(Linearize);
    string title1  = "Peclet/c1_Peclet" + name;

    int i=0;
    string temp1 = title1 + to_string(i);
    cout << "temp1: " << temp1 << endl;

    ofstream file(temp1, std::ios::out);
    if (file.is_open())
    {
        cout << "file is open" << endl;
    } else cout << "Not open" << endl;
}