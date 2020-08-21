#coding: utf-8
import sympy


# 模型的计算尺度: Angstrom, Nano, Micro
SCALE = "Angstrom"

# 下面的参数给定参考C++的hpp文件,不要轻易修改
pi                       = 3.14159265359
T                        = 298.15
e_c                      = 1.60217657e-19
k_B                      = 1.3806488e-23
N_A                      = 6.02214129e+23
water_rel_permittivity   = 80.0
z1, z2                   = +1, -1 # K+或Na+离子, Cl-离子
if SCALE == "Angstrom":
    vacuum_permittivity  = 8.854187817e-12 * 1e-10 # F/m -> F/Å
    alpha1               = e_c / (k_B * T)
    alpha2               = e_c * e_c / (k_B * T * vacuum_permittivity)
    alpha3               = 1.0E-27 * N_A           # mol/L -> 1/(Å^3)
    D1, D2               = 0.196, 0.203            # Å^2/ps
elif SCALE == "Nano":
    vacuum_permittivity  = 8.854187817e-12 * 1e-9  # F/m -> F/nm
    alpha1               = e_c / (k_B * T)
    alpha2               = e_c * e_c / (k_B * T * vacuum_permittivity)
    alpha3               = 1.0E-24 * N_A           # mol/L -> 1/(nm^3)
    D1, D2               = 0.196*1E-2, 0.203*1E-2  # nm^2/ps
elif SCALE == "Micro":
    vacuum_permittivity  = 8.854187817e-12 * 1e-6  # F/m -> F/µm
    alpha1               = e_c / (k_B * T)
    alpha2               = e_c * e_c / (k_B * T * vacuum_permittivity)
    alpha3               = 1.0E-15 * N_A           # mol/L -> 1/(µm^3)
    D1, D2               = 0.196*1E-8, 0.203*1E-8  # µm^2/ps
else: raise "Not support the computation scale!!!"


def ComputeSteadyStateSolutions():
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
    x, y, z = sympy.symbols("x[0] x[1] x[2]")

    # 下面的等式右端的第一项是对应的Dirichlet边界条件
    phi3 = phi_bulk * (z + L/2.0) / L \
           + (alpha2 * alpha3 * c_bulk * L * L / (3 * water_rel_permittivity * pi * pi)) \
           * (sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L))
    c1   = c_bulk + 0.5*c_bulk * (sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L))
    c2   = c_bulk - 0.5*c_bulk * (sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L))

    phi3_x = phi3.diff(x, 1)
    phi3_y = phi3.diff(y, 1)
    phi3_z = phi3.diff(z, 1)

    c1_x   = c1.diff(x, 1)
    c1_y   = c1.diff(y, 1)
    c1_z   = c1.diff(z, 1)

    c2_x   = c2.diff(x, 1)
    c2_y   = c2.diff(y, 1)
    c2_z   = c2.diff(z, 1)

    phi_flux = [-water_rel_permittivity * phi3_x,
                -water_rel_permittivity * phi3_y,
                -water_rel_permittivity * phi3_z] # - epsilon_s grad(phi)
    c1_flux = [-D1 * (c1_x + z1 * c1 * phi3_x),
               -D1 * (c1_y + z1 * c1 * phi3_y),
               -D1 * (c1_z + z1 * c1 * phi3_z)] # - D1 (grad(c1) + z1 c1 grad(phi))
    c2_flux = [-D2 * (c2_x + z2 * c2 * phi3_x),
               -D2 * (c2_y + z2 * c2 * phi3_y),
               -D2 * (c2_z + z2 * c2 * phi3_z)] # - D2 (grad(c2) + z2 c2 grad(phi))

    # 真实的对流速度
    adv1 = [D1 * z1 * phi3_x,
            D1 * z1 * phi3_y,
            D1 * z1 * phi3_z]
    adv2 = [D2 * z2 * phi3_x,
            D2 * z2 * phi3_y,
            D2 * z2 * phi3_z]

    div_adv1 = adv1[0].diff(x, 1) + adv1[1].diff(y, 1) + adv1[2].diff(z, 1)
    div_adv2 = adv2[0].diff(x, 1) + adv2[1].diff(y, 1) + adv2[2].diff(z, 1)

    f1 = c1_flux[0].diff(x, 1) + c1_flux[1].diff(y, 1) + c1_flux[2].diff(z, 1)
    f2 = c2_flux[0].diff(x, 1) + c2_flux[1].diff(y, 1) + c2_flux[2].diff(z, 1)

    # 把下面的输出结果复制粘贴到C++源码里面
    print("\n//======> Steady State Analytic Solutions:")
    print("double phi_exact_(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(phi3)))
    print("\ndouble c1_exact_(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c1)))
    print("\ndouble c2_exact_(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c2)))
    print("\ndouble f1_analytic_(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f1)))
    print("\ndouble f2_analytic_(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f2)))

    print("\nvoid J_(const Vector& x, Vector& y)\n{{    \ny[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(phi_flux[0]), sympy.printing.ccode(phi_flux[1]), sympy.printing.ccode(phi_flux[2])))
    print("\nvoid J1_(const Vector& x, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(c1_flux[0]), sympy.printing.ccode(c1_flux[1]), sympy.printing.ccode(c1_flux[2])))
    print("\nvoid J2_(const Vector& x, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(c2_flux[0]), sympy.printing.ccode(c2_flux[1]), sympy.printing.ccode(c2_flux[2])))

    print("\nvoid adv1(const Vector& x, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(adv1[0]), sympy.printing.ccode(adv1[1]), sympy.printing.ccode(adv1[2])))
    print("\nvoid adv2(const Vector& x, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(adv2[0]), sympy.printing.ccode(adv2[1]), sympy.printing.ccode(adv2[2])))

    print("\ndouble div_adv1(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(div_adv1)))
    print("\ndouble div_adv2(const Vector& x)\n{{\n    return {};\n}}".format(sympy.printing.ccode(div_adv2)))


if __name__ == '__main__':
    phi_bulk = alpha1 * 1.0*10**(-7) # 1.0*10**(-7) V 变成无量纲的，取值这么小是为了是phi的真解由cos部分主导，而不是一个常数
    c_bulk   = alpha3 * 1.0 # 1.0 mol/L 变成无量纲的
    L        = 2            # 区域尺寸，本身无量纲

    print("epsilon_s: {}".format(water_rel_permittivity))
    print("alpha1: {}\nalpha2: {}\nalpha3: {}".format(alpha1, alpha2, alpha3))
    print("phi bulk (dimensionless): {}".format(phi_bulk))
    print("c   bulk (dimensionless): {}".format(c_bulk))

    print("alpha2 * alpha3 * c_bulk * L^2 / (3 * epsilon_s * pi^2): {}".format(alpha2 * alpha3 * c_bulk * L*L / (3 * water_rel_permittivity * pi*pi)))

    ComputeSteadyStateSolutions()
