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


def ComputeTimeDependentSolutions():
    '''
    在验证带蛋白通道的PNP方程的代码的正确性的时候, 利用简化模型, 即不带蛋白的立方体盒子, 来验证.
    这个简化模型矩阵解析解, 下面就是来计算相关的解析表达式.
    计算区域为: [-L/2, L/2]^3, 所有未知量的边界条件都是Dirichlet
    Ref Fan Ronghong PhD thesis
    Poisson Equation:
            div( -epsilon_s grad(phi) ) - alpha2 alpha3 \sum_i z_i c_i = f
    NP Equation:
            dc_i / dt = div( D_i (grad(c_i) + z_i c_i grad(phi) ) ) + f_i
    '''
    x, y, z, t = sympy.symbols("x[0] x[1] x[2] t")

    phi3 = sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L) * sympy.cos(pi * t / Tn)
    c1   = sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L) * sympy.cos(pi * t / Tn)
    c2   = sympy.cos(pi * x / L) * sympy.cos(pi * y / L) * sympy.cos(pi * z / L) * sympy.cos(pi * t / Tn)

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

    f  = phi_flux[0].diff(x, 1) + phi_flux[1].diff(y, 1) + phi_flux[2].diff(z, 1) \
         - alpha2 * alpha3 * (z1 * c1 + z2 * c2)
    f1 = c1.diff(t, 1) + ( c1_flux[0].diff(x, 1) + c1_flux[1].diff(y, 1) + c1_flux[2].diff(z, 1) )
    f2 = c2.diff(t, 1) + ( c2_flux[0].diff(x, 1) + c2_flux[1].diff(y, 1) + c2_flux[2].diff(z, 1) )

    # 真实的对流速度
    adv1 = [D1 * z1 * phi3_x,
            D1 * z1 * phi3_y,
            D1 * z1 * phi3_z]
    adv2 = [D2 * z2 * phi3_x,
            D2 * z2 * phi3_y,
            D2 * z2 * phi3_z]

    div_adv1 = adv1[0].diff(x, 1) + adv1[1].diff(y, 1) + adv1[2].diff(z, 1)
    div_adv2 = adv2[0].diff(x, 1) + adv2[1].diff(y, 1) + adv2[2].diff(z, 1)

    # 把下面的输出结果复制粘贴到C++源码里面
    print("\n\n//======> Time Dependent Analytic Solutions:")
    print("double phi_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(phi3)))
    print("\ndouble c1_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c1)))
    print("\ndouble c2_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c2)))

    print("\ndouble dphidt_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(phi3.diff(t, 1))))
    print("\ndouble dc1dt_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c1.diff(t, 1))))
    print("\ndouble dc2dt_exact_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(c2.diff(t, 1))))

    print("\ndouble f0_analytic_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f)))
    print("\ndouble f1_analytic_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f1)))
    print("\ndouble f2_analytic_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(f2)))

    print("\nvoid J_time(const Vector& x, double t, Vector& y)\n{{    \ny[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(phi_flux[0]),
                                                                                                                               sympy.printing.ccode(phi_flux[1]),
                                                                                                                               sympy.printing.ccode(phi_flux[2])))
    print("\nvoid J1_time(const Vector& x, double t, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(c1_flux[0]),
                                                                                                                                sympy.printing.ccode(c1_flux[1]),
                                                                                                                                sympy.printing.ccode(c1_flux[2])))
    print("\nvoid J2_time(const Vector& x, double t, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(c2_flux[0]),
                                                                                                                                sympy.printing.ccode(c2_flux[1]),
                                                                                                                                sympy.printing.ccode(c2_flux[2])))

    print("\nvoid adv1_time(const Vector& x, double t, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(adv1[0]), sympy.printing.ccode(adv1[1]), sympy.printing.ccode(adv1[2])))
    print("\nvoid adv2_time(const Vector& x, double t, Vector& y)\n{{\n    y[0] = {};\n    y[1] = {};\n    y[2] = {};\n}}".format(sympy.printing.ccode(adv2[0]), sympy.printing.ccode(adv2[1]), sympy.printing.ccode(adv2[2])))

    print("\ndouble div_adv1_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(div_adv1)))
    print("\ndouble div_adv2_time(const Vector& x, double t)\n{{\n    return {};\n}}".format(sympy.printing.ccode(div_adv2)))


if __name__ == '__main__':
    phi_bulk = alpha1 * 1.0*10**(-7) # 1.0*10**(-7) V 变成无量纲的，取值这么小是为了是phi的真解由cos部分主导，而不由边界条件主导
    c_bulk   = alpha3 * 100.0 # 1.0 mol/L 变成无量纲的
    L        = 2            # 区域尺寸，本身无量纲
    Tn       = 1            # 时间 [0, Tn]

    print("epsilon_s: {}".format(water_rel_permittivity))
    print("alpha1: {}\nalpha2: {}\nalpha3: {}".format(alpha1, alpha2, alpha3))
    print("phi bulk (dimensionless): {}".format(phi_bulk))
    print("c   bulk (dimensionless): {}".format(c_bulk))

    print("alpha2 * alpha3 * c_bulk * L^2 / (3 * epsilon_s * pi^2): {}".format(alpha2 * alpha3 * c_bulk * L*L / (3 * water_rel_permittivity * pi*pi)))

    ComputeTimeDependentSolutions()