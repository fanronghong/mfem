//
// Created by plan on 2019/12/26.
//

#ifndef LEARN_MFEM_PYTHON_UTILS_HPP
#define LEARN_MFEM_PYTHON_UTILS_HPP

#include <Python.h>
//#include <numpy/arrayobject.h> // wrong in computer cluster
#include <vector>
#include <string>
#include <iostream>

#include "matplotlibcpp.hpp" // not work in computer cluster


void Python_Basics()
{
    using namespace std;
    setenv("PYTHONPATH", ".", 0);
    Py_Initialize();// 初始化

    // 将Python工作路径切换到待调用模块所在目录，一定要保证路径名的正确性
    string original_path = getcwd(NULL, 0);
    cout << "original_path:\n" << original_path << endl;

    PyRun_SimpleString("import sys, os, pprint");
    cout << "==> os.getcwd():\n";   PyRun_SimpleString("print(os.getcwd())");
    cout << "==> os.chdir(../);\n"; PyRun_SimpleString("os.chdir('../')");
    cout << "==> os.getcwd():\n";   PyRun_SimpleString("print(os.getcwd())");
    PyRun_SimpleString("sys.path.append(os.getcwd())"); //把下面要导入的模块的路径加入到Python的搜索路径
    PyRun_SimpleString("pprint.pprint(sys.path)");

    string ModuleName = "utils";
    cout << "\n==> 下面引入自己写的python模块: " << ModuleName << ".py" << endl;
    PyObject* pyModuleName = PyUnicode_FromString(ModuleName.c_str()); //模块名，不是文件名
    PyObject* pyModule = PyImport_Import(pyModuleName);
    if (!pyModule) {
        cout << "==> 引入失败!\n" << endl;
        chdir(original_path.c_str()); //一定要要C++运行环境回到原来的路径下
        return;
    }
    cout << "==> 引入成功!\n" << endl;


    const char* Func = "ReadMatlabMatTxt";
//    const char* Func = "draw";
    PyObject* pyFunc = PyObject_GetAttrString(pyModule, Func);
    if (!pyFunc || !PyCallable_Check(pyFunc)) { // 验证函数是否 加载 成功
        cout << "==> 加载函数 " << Func << ": 失败!" << endl;
        chdir(original_path.c_str()); //一定要要C++运行环境回到原来的路径下
        return;
    }
    cout << "==> 加载函数 " << Func << ": 成功!" << endl;
    PyObject* args = NULL; //函数参数
    PyObject* pyFuncRet = NULL; //函数返回值

    {
        if (0) {
            args = Py_BuildValue("()"); //多个参数: PyObject* args = Py_BuildValue("(ifs)", 100, 3.14, "hello");
            pyFuncRet = PyObject_CallObject(pyFunc, args);
        }
        else if (1) {
            const char* filename = "/home/fan/PNP_data/Dinv.m";
            PyObject* pyfilename = Py_BuildValue("s", filename);
            args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, pyfilename);
            pyFuncRet = PyObject_CallObject(pyFunc, args);
        }
    }


    if (pyFuncRet) { // 验证函数是否 运行 成功
        cout << "==> 运行函数 " << Func << ": 成功!" << endl;
    }
    else {
        cout << "==> 运行函数 " << Func << ": 失败!" << endl;
    }


    chdir(original_path.c_str()); //一定要要C++运行环境回到原来的路径下
    Py_Finalize(); // 释放资源
    cout << "\n--------------------- C++ 嵌入 Python 代码成功! ------------------\n" << endl;
}


int Test_Demo()
{
    setenv("PYTHONPATH", ".", 0);

    Py_Initialize();
    import_array();

    // Build the 2D array in C++
    const int SIZE = 3;
    npy_intp shape[2]{SIZE, SIZE};
    const int dim = 2;
    long double(*c_arr)[SIZE]{ new long double[SIZE][SIZE] };

    for (int i = 0; i < SIZE; i++){
        for (int j = 0; j < SIZE; j++){
            c_arr[i][j] = i + j;
        }
    }

    // Convert it to a NumPy array.
    PyObject *pArray = PyArray_SimpleNewFromData(dim, shape, NPY_LONGDOUBLE, reinterpret_cast<void*>(c_arr));

    // import mymodule
    std::string module_name = "mymodule";
    PyObject *pName = PyUnicode_FromString(module_name.c_str());
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    // import function
    std::string func_name = "array_tutorial";
    PyObject *pFunc = PyObject_GetAttrString(pModule, func_name.c_str());
    PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);
    PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);

    // Convert back to C++ array and print.
    int len = PyArray_SHAPE(np_ret)[0];
    long double* c_out = reinterpret_cast<long double*>(PyArray_DATA(np_ret));
    std::cout << "Printing output array - C++" << std::endl;
    for (int i = 0; i < len; i++) {
        std::cout << c_out[i] << ' ';
    }
    std::cout << std::endl << std::endl;


    // import function without arguments
    std::string func_name2 = "myfunction";
    PyObject *pFunc2 = PyObject_GetAttrString(pModule, func_name2.c_str());
    PyObject *pReturn2 = PyObject_CallFunctionObjArgs(pFunc2, NULL);
    PyArrayObject *np_ret2 = reinterpret_cast<PyArrayObject*>(pReturn2);

    // convert back to C++ array and print
    int len2 = PyArray_SHAPE(np_ret2)[0];
    long double* c_out2 = reinterpret_cast<long double*>(PyArray_DATA(np_ret2));
    std::cout << "Printing output array 2 - C++" << std::endl;
    for (int i = 0; i < len2; i++) {
        std::cout << c_out2[i] << ' ';
    }
    std::cout << std::endl << std::endl;

    Py_Finalize();
}


template <typename T>
std::vector<T> array2vector(const T* array, int size)
{
    std::vector<T> vec(array, array + size);
    return vec;
}
template <typename T>
PyObject* vector2numpy(std::vector<T>& vec)
{
    return matplotlibcpp::get_array(vec);
}
void Test_array2vector()
{
    int II[5] = {0, 2, 4, 7, 9};
    int JJ[9] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
    double VVals[9] = {1, 7, 2, 8, 5, 3, 9, 6, 4};
    SparseMatrix mat(II, JJ, VVals, 4, 4);

    const int* I = mat.GetI();
    const int* J = mat.GetJ();
    const double* Vals = mat.GetData();
    int size = mat.Size();
    int nnz  = mat.NumNonZeroElems();

    const std::vector<int> vec_I = array2vector(I, size+1);
    const std::vector<int> vec_J = array2vector(J, nnz);
    const std::vector<double> vec_Vals = array2vector(Vals, nnz);
    for (auto& item: vec_I) std::cout << item << std::endl;
    for (auto& item: vec_J) std::cout << item << std::endl;
    for (auto& item: vec_Vals) std::cout << item << std::endl;
}


template <typename T>
void Plot_vector(std::vector<T>& vec)
{
    setenv("PYTHONPATH", ".", 0);
    Py_Initialize();
    std::string original_path = getcwd(NULL, 0);

    PyRun_SimpleString("import sys, os, pprint");

    PyObject *pArray = vector2numpy(vec);

    // import mymodule
    std::string module_name = "python_utils";
    std::string module_path_name = __FILE__;
    std::string module_path;

    // 获得python模块的路径
//    std::cout << "==> os.getcwd():\n";   PyRun_SimpleString("print(os.getcwd())");
    const size_t last_back_slash_idx = module_path_name.rfind('/');
    // remove file name, only need file path
    if (std::string::npos != last_back_slash_idx) {
        module_path = module_path_name.substr(0, last_back_slash_idx);
    }

    // 跳转到python模块的所在路径
    std::string change_path_command = "os.chdir('" + module_path + "')";
    PyRun_SimpleString(change_path_command.c_str());
    PyRun_SimpleString("sys.path.append(os.getcwd())"); //把下面要导入的模块的路径加入到Python的搜索路径
//    PyRun_SimpleString("pprint.pprint(sys.path)");

    // 导入需要的python模块
    PyObject *pModuleName = PyUnicode_FromString(module_name.c_str());
    PyObject *pModule = PyImport_Import(pModuleName);
    assert(pModule != NULL);

    // import function
    std::string func_name = "PlotArray";
    PyObject *pFunc = PyObject_GetAttrString(pModule, func_name.c_str());
    PyObject_CallFunctionObjArgs(pFunc, pArray, NULL);

    Py_DECREF(pArray);
    Py_DECREF(pModuleName);
    Py_DECREF(pModule);
    Py_DECREF(pFunc);

    chdir(original_path.c_str());
    Py_Finalize();
}


// 更好的方式是阅读 matplotlibcpp.hpp, 然后把画sparse pattern的功能加进去
void PlotSparseMatrix(const SparseMatrix& sp, const std::string& title)
{
//    setenv("PYTHONPATH", ".", 0);
    Py_Initialize();
    std::string original_path = getcwd(NULL, 0);

    PyRun_SimpleString("import sys, os, pprint");

    int size = sp.Size();
    int nnz  = sp.NumNonZeroElems();
    const int* I = sp.GetI();
    const int* J = sp.GetJ();
    const double* Vals = sp.GetData();
    std::vector<int> vec_I = array2vector(I, size+1);
    std::vector<int> vec_J = array2vector(J, nnz);
    std::vector<double> vec_Vals = array2vector(Vals, nnz);

    PyObject* pArray_I = matplotlibcpp::get_array(vec_I);
    PyObject* pArray_J = matplotlibcpp::get_array(vec_J);
    PyObject* pArray_Vals = matplotlibcpp::get_array(vec_Vals);
    PyObject* pic_title = PyUnicode_FromString(title.c_str());


    // import mymodule
    std::string module_name = "python_utils";
    std::string module_path_name = __FILE__;
    std::string module_path;

    // 获得python模块的路径
//    std::cout << "==> os.getcwd():\n";   PyRun_SimpleString("print(os.getcwd())");
    const size_t last_back_slash_idx = module_path_name.rfind('/');
    // remove file name, only need file path
    if (std::string::npos != last_back_slash_idx) {
        module_path = module_path_name.substr(0, last_back_slash_idx);
    }

    // 跳转到python模块的所在路径
    std::string change_path_command = "os.chdir('" + module_path + "')";
    PyRun_SimpleString(change_path_command.c_str());
//    PyRun_SimpleString("pprint.pprint(sys.path)");
//    PyRun_SimpleString("pprint.pprint(os.listdir('.'))");
    PyRun_SimpleString("sys.path.append(os.getcwd())"); //把下面要导入的模块的路径加入到Python的搜索路径

    // 导入需要的python模块
    PyObject* pModuleName = PyUnicode_FromString(module_name.c_str());
    PyObject* pModule = PyImport_Import(pModuleName);
    assert(pModule != NULL);

    // import function
    std::string func_name = "PlotCSR";
    PyObject* pFunc = PyObject_GetAttrString(pModule, func_name.c_str());

    // construct arguments: 3 ways
//    PyObject *args = Py_BuildValue("OOO", pArray_I, pArray_J, pArray_Vals);
//    PyObject *args = Py_BuildValue("SSSs", pArray_I, pArray_J, pArray_Vals, pic_title);
//    PyObject *args = PyTuple_New(3);
//    PyTuple_SetItem(args, 0, pArray_I);
//    PyTuple_SetItem(args, 1, pArray_J);
//    PyTuple_SetItem(args, 2, pArray_Vals);

    // run python function: 2 ways
    PyObject* pRet = PyObject_CallFunctionObjArgs(pFunc, pArray_I, pArray_J, pArray_Vals, pic_title, NULL);

//    PyObject* pRet = PyObject_CallObject(pFunc, args);
//    PyObject* pRet = PyObject_CallObject(pFunc, NULL); // if no need arguments


    // 获取python函数返回值
    if (pRet == NULL) MFEM_ABORT("Run " + func_name + " Failed!") // 验证是否调用成功

    Py_DECREF(pArray_I);
    Py_DECREF(pArray_J);
    Py_DECREF(pArray_Vals);
    Py_DECREF(pic_title);
    Py_DECREF(pModuleName);
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pRet);

    chdir(original_path.c_str());
//    Py_Finalize(); // 同时画多幅图的时候, 打开Py_Finalize()会出现segmentation
}
void Test_PlotSparseMatrix()
{
    int II[5] = {0, 2, 4, 7, 9};
    int JJ[9] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
    double VVals[9] = {1, 7, 2, 8, 5, 3, 9, 6, 4};
    SparseMatrix mat(II, JJ, VVals, 4, 4);

    PlotSparseMatrix(mat, "test");
}


void Demo_Plotline()
{
    Py_Initialize();
    std::string original_path = getcwd(NULL, 0);

    PyRun_SimpleString("import sys, os, pprint");
    PyRun_SimpleString("import matplotlib.pyplot as plt");
    PyRun_SimpleString("import numpy as np");

    // 第一种方式: 适合简单的python命令
    PyRun_SimpleString("fig = plt.figure();"
                        "ax = plt.axes();"
                        "x = np.linspace(0, 10, 100);"
                        "ax.plot(x, np.sin(x));"
                        "plt.show();");


    // 第二种方式: 能够够好的用于现成的python模块, 可扩展性强
    std::string module_name = "python_utils";
    std::string module_path_name = __FILE__;
    std::string module_path;

    // 获得python模块的路径
//    std::cout << "==> os.getcwd():\n";   PyRun_SimpleString("print(os.getcwd())");
    const size_t last_back_slash_idx = module_path_name.rfind('/');
    // remove file name, only need file path
    if (std::string::npos != last_back_slash_idx) {
        module_path = module_path_name.substr(0, last_back_slash_idx);
    }

    // 跳转到python模块的所在路径
    std::string change_path_command = "os.chdir('" + module_path + "')";
    PyRun_SimpleString(change_path_command.c_str());
    PyRun_SimpleString("sys.path.append(os.getcwd())"); //把下面要导入的模块的路径加入到Python的搜索路径
//    PyRun_SimpleString("pprint.pprint(sys.path)");

    // 导入需要的python模块
    PyObject* pyModuleName = PyUnicode_FromString(module_name.c_str());
    PyObject* pyModule = PyImport_Import(pyModuleName);
    assert(pyModule != NULL);

    // 从相应模块中导入函数
    std::string func_name = "Demo_PlotLine";
    PyObject* func_pt = PyObject_GetAttrString(pyModule, func_name.c_str());
    PyObject* funcReturn = PyObject_CallFunctionObjArgs(func_pt, NULL); // 执行

    chdir(original_path.c_str());
    Py_Finalize();
}






#endif //LEARN_MFEM_PYTHON_UTILS_HPP
