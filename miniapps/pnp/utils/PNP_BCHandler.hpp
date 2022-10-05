#ifndef MFEM_PNP_BCHANDLER_HPP
#define MFEM_PNP_BCHANDLER_HPP

#include "mfem.hpp"
using namespace mfem;

class PNP_Newton_BCHandler: public PetscBCHandler
{
private:
    // ref: miniapps/mhd/imResistiveMHDOperatorp.hpp 的 class myBCHandler.
    // component表示未知量个数, 比如phi, c1, c2就为3
    // true_vsize表示每个未知量的维数, 即有限元空间的自由度个数
    int component, true_vsize;
    Vector bc_vals;

public:
    PNP_Newton_BCHandler(PetscBCHandler::Type type_,Array<int>& ess_tdof_list_, int component_, int true_vsize_)
        : PetscBCHandler(type_), component(component_), true_vsize(true_vsize_)
    {
        SetTDofs(ess_tdof_list_);
    }
    ~PNP_Newton_BCHandler() {}

    void SetTDofs(Array<int>& list) //overwrite SetTDofs
    {
        int size = list.Size();
        ess_tdof_list.SetSize(size * component);
        
        for (PetscInt j=0; j<component; ++j)
        {
            for (PetscInt i=0; i<size; ++i)
            {
                ess_tdof_list[i + j*size] = j*true_vsize + list[i];
            }
        }
        setup = false;
    }

    void SetBoundarValues(const Vector& bc_vals_)
    {
        MFEM_ASSERT(bc_vals_.Size() == component*true_vsize, "Wrong boundary values!");
        if (setup) return;
        bc_vals = bc_vals_;
    }

    // 这个是需要自己实现的
    virtual void Eval(double t, Vector &g)
    {
        g = 0.0;
        for (PetscInt i=0; i<ess_tdof_list.Size(); ++i)
        {
            g[ess_tdof_list[i]] = bc_vals[ess_tdof_list[i]];
        }
    }
};


#endif //MFEM_PNP_BCHANDLER_HPP
