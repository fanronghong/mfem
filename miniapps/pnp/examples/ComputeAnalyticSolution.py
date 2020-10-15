#coding: utf-8
import sympy

pi = 3.14159265359

def ComputeSteadyStateSolutions_():
    '''
    在验证带蛋白通道的PNP方程的代码的正确性的时候, 利用简化模型, 即不带蛋白的立方体盒子, 来验证.
    这个简化模型矩阵解析解, 下面就是来计算相关的解析表达式.
    计算区域为: [-L/2, L/2]^3, 所有未知量的边界条件都是Dirichlet
    ref Fan Ronghong PhD thesis
    Poisson Equation:
            div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = 0
    NP Equation:
            div( -D_i (grad(c_i) + z_i c_i grad(phi) ) ) = f_i
    '''
    x, y, t = sympy.symbols("x[0] x[1] t")

    u = (1 + sympy.cos(pi * x) * sympy.cos(pi * y)) * sympy.exp(t)

    u_t = u.diff(t, 1)
    u_x = u.diff(x, 1)
    u_y = u.diff(y, 1)

    alpha = 1.0e-2
    kappa = 0.5

    flux = [(kappa + alpha*u) * u_x,
            (kappa + alpha*u) * u_y]

    f = u_t - (flux[0].diff(x, 1) + flux[1].diff(y, 1))

    # 把下面的输出结果复制粘贴到C++源码里面
    print("\n\n//======> Time Dependent Analytic Solutions:")
    print("double u_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(u)))
    print("\ndouble f_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f)))


if __name__ == '__main__':
    ComputeSteadyStateSolutions_()