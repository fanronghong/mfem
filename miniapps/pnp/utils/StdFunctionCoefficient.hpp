/* Wrap FunctionCoefficient.
 * 原本MFEM里面的FunctionCoefficient之类的构造函数的参数只有 (const Vector& x, Vector& y),
 * 但实际使用的时候,这类函数还依赖其他的数据,比如在PNP中的奇异项Green函数需要PQR文件的信息.
 * */
#ifndef LEARN_MFEM_STDFUNCTIONCOEFFICIENT_HXX
#define LEARN_MFEM_STDFUNCTIONCOEFFICIENT_HXX

#include <functional>
#include <cassert>
#include "mfem.hpp"

using namespace mfem;

class StdFunctionCoefficient : public Coefficient
{
// ref: https://github.com/ianabel/mfem/blob/fe9de1fc3eeb93fe8333fafa3358507814cd541f/fem/coefficient.hpp#L163
protected:
    using RealFunc = std::function< double( const mfem::Vector & )>;
    using RealTimeFunc = std::function< double( const mfem::Vector &, double )>;

    RealFunc function;
    RealTimeFunc timeFunction;

public:
    StdFunctionCoefficient( RealFunc F ): function( F ) {}
    StdFunctionCoefficient( RealTimeFunc F ): timeFunction( F ) {}

    /// Evaluate coefficient
    virtual double Eval(ElementTransformation &T,
                        const IntegrationPoint &ip) override
    {
        double x[3];
        Vector transip(x, 3);

        T.Transform(ip, transip);

        if ( ( bool )( timeFunction ) )
        {
            return timeFunction( transip, GetTime() );
        }
        else if ( ( bool )( function ) )
        {
            return function( transip );
        }
        else
        {
            MFEM_ABORT( "No valid function object inside this FunctionCoefficient!" );
            return std::nan( "" );
        }
    }

};

class VectorStdFunctionCoefficient : public VectorCoefficient
{
private:
    using VectorFunc = std::function< void( const mfem::Vector &, mfem::Vector & )>;
    using VectorTimeFunc = std::function< void( const mfem::Vector &, double, mfem::Vector & )>;

    VectorFunc vectorFunction;
    VectorTimeFunc vectorTimeFunction;
    Coefficient *Q;
    int Dim;

public:
    /// Construct a time-independent vector coefficient from a C-function
    VectorStdFunctionCoefficient(int dim, VectorFunc VF, Coefficient *q = nullptr)
            : Dim(dim), VectorCoefficient(dim), vectorFunction( VF ), Q(q) { }

    /// Construct a time-dependent vector coefficient from a C-function
    VectorStdFunctionCoefficient(int dim, VectorTimeFunc VF, Coefficient *q = nullptr)
            : Dim(dim), VectorCoefficient(dim), vectorTimeFunction( VF ), Q(q) { }

    virtual ~VectorStdFunctionCoefficient() { }

    using VectorCoefficient::Eval;
    virtual void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        double x[3];
        Vector transip(x, 3);
        V.SetSize(Dim);

        T.Transform(ip, transip);

        if ( ( bool )( vectorTimeFunction ) )
        {
            std::cerr << " nope " << std::endl;
            vectorTimeFunction( transip, GetTime(), V );
        }
        else if ( ( bool )( vectorFunction ) )
        {
            vectorFunction( transip, V );
        }
        else
        {
            MFEM_ABORT( "No valid function object inside this VectorStdFunctionCoefficient!" );
        }

        if ( Q != nullptr )
        {
            V *= Q->Eval( T, ip, GetTime() );
        }
    }

};


class PFunction
{
// ref: https://github.com/mfem/mfem/issues/1151
private:
    int _p;
public:
    PFunction( int Parameter ) : _p( Parameter ) {}
    double operator()( const mfem::Vector & point )
    {
        return _p;
    }
};
void Test1_StdFunctionCoefficient()
{
    mfem::Mesh mesh(100, 100, mfem::Element::TRIANGLE, true, 1.0, 1.0);
    mfem::H1_FECollection h1_fec(1, 2);
    mfem::FiniteElementSpace h1_space(&mesh, &h1_fec);
    mfem::GridFunction gf1(&h1_space), gf2(&h1_space);

    PFunction F( 10 ); // Initialize the function with a given parameter.
    StdFunctionCoefficient MyFunctionCoeff( F ); // Now you can use this function coefficient, and change the parameter between uses by calling F.SetParam()

    mfem::ConstantCoefficient ten(10.0);

    gf1.ProjectCoefficient(MyFunctionCoeff);
    gf2.ProjectCoefficient(ten);
    for (int i=0; i<gf1.Size(); ++i)
    {
        assert(abs(gf1[i] - gf2[i]) < 1E-10);
    }
}


class vecfunc // 自己构造一个三维的向量函数
{
private:
    int a, b, c;
public:
    vecfunc(int a_, int b_, int c_): a(a_), b(b_), c(c_) {}
    void operator()(const mfem::Vector& x, mfem::Vector& y)
    {
        y[0] = a;
        y[1] = b;
        y[2] = c;
    }
};
void Test2_StdFunctionCoefficient()
{
    mfem::Mesh mesh(100, 100, mfem::Element::TRIANGLE, true, 1.0, 1.0);
    mfem::H1_FECollection h1_fec(1, 2);
    mfem::FiniteElementSpace h1_space(&mesh, &h1_fec, 3);
    mfem::GridFunction gf1(&h1_space), gf2(&h1_space);

    vecfunc func1(1.0, 2.0, 3.0);
    VectorStdFunctionCoefficient MyFunctionCoeff( 3, func1 );

    mfem::Vector vec(3);
    vec[0] = 1.0;
    vec[1] = 2.0;
    vec[2] = 3.0;
    mfem::VectorConstantCoefficient coeff(vec);

    gf1.ProjectCoefficient(MyFunctionCoeff);
    gf2.ProjectCoefficient(coeff);
    for (int i=0; i<gf1.Size(); ++i) assert(abs(gf1[i] - gf2[i]) < 1E-10);
}



void Test_StdFunctionCoefficient()
{
    Test1_StdFunctionCoefficient();
    Test2_StdFunctionCoefficient();
    std::cout << "===> Test Pass: StdFunctionCoefficient.hpp" << std::endl;
}
#endif //LEARN_MFEM_STDFUNCTIONCOEFFICIENT_HXX
