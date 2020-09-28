//
// Created by fan on 2020/9/28.
//

#ifndef MFEM_DGEDGEINTEGRATOR1_HPP
#define MFEM_DGEDGEINTEGRATOR1_HPP

/* 单元边界和计算区域边界 的Facet积分:
 *
 *     q <{u grad(w).n}, [v]>_E,
 *
 * u is Trial function, v is Test function
 * q is given Coefficient, q在边E的两边连续
 * w is GridFunction, 但是w是不连续的(至少grad_w是不连续的)
 * */
class DGEdgeIntegrator1: public BilinearFormIntegrator
{
protected:
    Coefficient *Q;
    GradientGridFunctionCoefficient* gradw;

    Vector nor, shape1, shape2, grad_w;
    double val1, val2;

public:
    DGEdgeIntegrator1(GridFunction &w) : Q(NULL)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    DGEdgeIntegrator1(Coefficient &q, GridFunction &w) : Q(&q)
    { gradw = new GradientGridFunctionCoefficient(&w); }
    ~DGEdgeIntegrator1() { delete gradw; }

    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat)
    {
        int dim, ndof1(0), ndof2(0);

        dim = el1.GetDim();

        grad_w.SetSize(dim);
        nor.SetSize(dim);

        ndof1 = el1.GetDof();
        if (Trans.Elem2No != Trans.Elem1No) // 内部边界
        {
            ndof2 = el2.GetDof();
        }

        shape1.SetSize(ndof1);
        shape2.SetSize(ndof2);

        elmat.SetSize(ndof1 + ndof2);
        elmat = 0.0;

        const IntegrationRule *ir = IntRule; // ref: DGTraceIntegrator::AssembleFaceMatrix
        if (ir == NULL) {
            int order;
            // Assuming order(u)==order(mesh)
            if (ndof2)
                order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                         2 * max(el1.GetOrder(), el2.GetOrder()));
            else {
                order = Trans.Elem1->OrderW() + 2 * el1.GetOrder();
            }
            if (el1.Space() == FunctionSpace::Pk) {
                order++;
            }
            ir = &IntRules.Get(Trans.FaceGeom, order); //得到face上的积分规则(里面包含积分点)
        }

        for (int p=0; p<ir->GetNPoints(); ++p)
        {
            const IntegrationPoint& ip = ir->IntPoint(p);

            Trans.SetAllIntPoints(&ip);

            const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
            const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

            el1.CalcShape(eip1, shape1);
            el2.CalcShape(eip2, shape2);

            if (dim == 1) {
                nor(0) = 2 * eip1.x - 1.0;
            }
            else {
                CalcOrtho(Trans.Jacobian(), nor); // 计算Face的法向量
            }

            gradw->Eval(grad_w, *Trans.Elem1, eip1);
            val1 = ip.weight * Q->Eval(*Trans.Elem1, eip1) * (grad_w * nor);

            if (Trans.Elem2No != Trans.Elem1No) // 内部边界
            {
                val1 *= 0.5;
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);

                gradw->Eval(grad_w, *Trans.Elem2, eip2);
                val2 = ip.weight * Q->Eval(*Trans.Elem2, eip2) * (grad_w * nor) * 0.5;

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(ndof1 + i, j) -= shape2(i) * val1 * shape1(j);

                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(i, ndof1+j) += shape1(i) * val2 * shape2(j);

                for (int i=0; i<ndof2; ++i)
                    for (int j=0; j<ndof2; ++j)
                        elmat(ndof1 + i, ndof1 + j) -= val2 * shape2(i) * shape2(j);
            }
            else
            {
                for (int i=0; i<ndof1; ++i)
                    for (int j=0; j<ndof1; ++j)
                        elmat(i, j) += val1 * shape1(i) * shape1(j);
            }

        }
    }
};

#endif //MFEM_DGEDGEINTEGRATOR1_HPP
