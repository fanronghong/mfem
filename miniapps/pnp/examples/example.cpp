#include "mfem.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    vector<Array<int>> block; // cannot use Array< Array<int> >

    {
        Array<int>* arr1 = new Array<int>;
        for (int i=0; i<3; ++i)
        {
            arr1->Append(i);
        }
        block.push_back(*arr1);
        delete arr1;

        Array<int>* arr2 = new Array<int>;
        for (int i=0; i<3; ++i)
        {
            arr2->Append(i*10);
        }
        block.push_back(*arr2);
        delete arr2;
    }


    block[0].Print(cout << "block[0]:\n");
    block[1].Print(cout << "block[1]:\n");
}