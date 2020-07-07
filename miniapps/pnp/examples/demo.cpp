#include <iostream>
#include <string>
using namespace std;

int main()
{
    const char* mesh="./1MAG_2.msh";
    int ref=1;
    const char* lin = "gummel";
    const char* dis = "dg";

    string mesh_(mesh);
    string target(".msh");
    mesh_.erase(mesh_.find(".msh"), 4);
    mesh_.erase(mesh_.find("./"), 2);
    cout << mesh_ << endl;

    string s = "_ref" + to_string(ref) + "_" + string(lin) + "_" + string(dis) + "_" + mesh_;
    cout << s << endl;
}