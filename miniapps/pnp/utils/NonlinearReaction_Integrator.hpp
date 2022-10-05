//
// Created by fan on 2019/9/24.
//

#ifndef LEARN_MFEM_NONLINEARREACTION_INTEGRATOR_HPP
#define LEARN_MFEM_NONLINEARREACTION_INTEGRATOR_HPP

#include "mfem.hpp"
using namespace mfem;
using namespace std;


// 对u^2 计算 单元刚度向量(u^2, v) 和 Jacobian 2*u*(du, v)
class ReactionNLFIntegrator: public NonlinearFormIntegrator
{
private:
    Vector elvect, shape;
    DenseMatrix elmat;
public:
    ReactionNLFIntegrator() {}
    void AssembleElementVector(const FiniteElement& el,
                               ElementTransformation& Tr,
                               const Vector& elfun,
                               Vector& elvect)
    {
        int dof = el.GetDof();
        shape.SetSize(dof);
        elvect.SetSize(dof);
        elvect = 0.0;

        const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + Tr.OrderW());

        for (int i=0; i<ir->GetNPoints(); i++) {
            const IntegrationPoint& ip = ir->IntPoint(i);
            el.CalcShape(ip, shape);
            Tr.SetIntPoint(&ip);

            double val_qp = (shape * elfun) * (shape * elfun); // 计算 u^2
            double w = Tr.Weight() * val_qp;
            add(elvect, ip.weight*w, shape, elvect);
        }
    }
    void AssembleElementGrad(const FiniteElement& el,
                             ElementTransformation& Tr,
                             const Vector& elfun,
                             DenseMatrix& elmat)
    {
        int dof = el.GetDof();
        shape.SetSize(dof);
        elmat.SetSize(dof);
        elmat = 0.0;

        const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + Tr.OrderW());

        for (int i=0; i<ir->GetNPoints(); i++) {
            const IntegrationPoint& ip = ir->IntPoint(i);
            el.CalcShape(ip, shape);
            Tr.SetIntPoint(&ip);

            double val_qp = shape*elfun * 2; // 计算 2*u 在physical单元的值. 注意: u(x) = u(x(x_hat)) = u_hat(x_hat).
            double w = ip.weight * Tr.Weight() * val_qp;
            AddMult_a_VVt(w, shape, elmat);
        }
    }
};


double cfunc_NonlinearReaction_Integrator(const Vector& x)
{
    return x[0]*x[0] + 2*x[1];
}
double cfunc2_NonlinearReaction_Integrator(const Vector& x)
{
    return cfunc_NonlinearReaction_Integrator(x) * cfunc_NonlinearReaction_Integrator(x);
}
void Test_NonlinearReaction_Integrator()
{
    Mesh mesh(80, 80, Element::TRIANGLE, true, 1.0, 1.0);
    int dim = mesh.Dimension();
    const int p_order = 1;

    H1_FECollection h1_fec(p_order, dim);
    FiniteElementSpace h1_space(&mesh, &h1_fec);
    int size = h1_space.GetVSize();

    FunctionCoefficient coeff(cfunc_NonlinearReaction_Integrator);
    FunctionCoefficient coeff2(cfunc2_NonlinearReaction_Integrator);
    ConstantCoefficient two(2.0);

    GridFunction u(&h1_space), u2(&h1_space);
    u.ProjectCoefficient(coeff);
    u2.ProjectCoefficient(coeff2);

    {
        // 测试 ReactionNLFIntegrator::AssembleElementVector() 对不对
        LinearForm lf(&h1_space);
        lf.AddDomainIntegrator(new DomainLFIntegrator(coeff2)); // (f, v): f就是 cfunc^2
        lf.Assemble();

        NonlinearForm nlf(&h1_space);
        nlf.AddDomainIntegrator(new ReactionNLFIntegrator);
        Vector res1(size);
        nlf.Mult(u, res1);
        for (int i=0; i<size; i++) {
            if (abs(lf[i] - res1[i]) > 1E-5) // 出现这种情况有可能是网格不够密
                throw "Error Source"; //这段字符串可以被catch关键字捕获
        }
    }

    {
        // 测试　ReactionNLFIntegrator::AssembleElementGrad() 对不对
        BilinearForm blf(&h1_space);
        ProductCoefficient two_prod_coeff(two, coeff); // 2 * cfunc
        blf.AddDomainIntegrator(new MassIntegrator(two_prod_coeff));
        blf.Assemble();

        NonlinearForm nlf(&h1_space);
        nlf.AddDomainIntegrator(new ReactionNLFIntegrator);
        SparseMatrix Jacobian;
        Jacobian = dynamic_cast<SparseMatrix&> (nlf.GetGradient(u)); // 得到在当前 u 处的Jacobian

        Jacobian *= -1.0;
        Jacobian += blf.SpMat();
        DenseMatrix* Jacobian_(Jacobian.ToDenseMatrix());
        for (int i=0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (abs((*Jacobian_)(i, j)) > 1E-8)
                    throw "Error Jacobian"; //这段字符串可以被catch关键字捕获
            }
        }
    }

    cout << "===> Test Pass: NonlinearReaction_Integrator.hpp" << endl;
}

#endif //LEARN_MFEM_NONLINEARREACTION_INTEGRATOR_HPP
