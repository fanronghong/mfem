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

def SymbolCompute():
    '''
    在验证带蛋白通道的PNP方程的代码的正确性的时候, 利用简化模型, 即不带蛋白的立方体盒子, 来验证.
    这个简化模型矩阵解析解, 下面就是来计算相关的解析表达式.
    计算区域为: [-L/2, L/2]^3, 所有未知量的边界条件都是Dirichlet
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

    c1_flux = [c1_x + z1 * c1 * phi3_x,
               c1_y + z1 * c1 * phi3_y,
               c1_z + z1 * c1 * phi3_z]
    c2_flux = [c2_x + z2 * c2 * phi3_x,
               c2_y + z2 * c2 * phi3_y,
               c2_z + z2 * c2 * phi3_z]

    f1 = -D1 * (c1_flux[0].diff(x, 1) + c1_flux[1].diff(y, 1) + c1_flux[2].diff(z, 1))
    f2 = -D2 * (c2_flux[0].diff(x, 1) + c2_flux[1].diff(y, 1) + c2_flux[2].diff(z, 1))

    # 把下面的输出结果复制粘贴到C++源码里面
    print("\nphi3:\n{}".format(sympy.printing.ccode(phi3)))
    print("\nc1:\n{}".format(sympy.printing.ccode(c1)))
    print("\nc2:\n{}".format(sympy.printing.ccode(c2)))
    print("\nf1:\n{}".format(sympy.printing.ccode(f1)))
    print("\nf2:\n{}".format(sympy.printing.ccode(f2)))
    print("\nphi_x:\n{}".format(sympy.printing.ccode(phi3_x)))
    print("\nphi_y:\n{}".format(sympy.printing.ccode(phi3_y)))
    print("\nphi_z:\n{}".format(sympy.printing.ccode(phi3_z)))


if __name__ == '__main__':
    phi_bulk = alpha1 * 1.0 # 1.0 V
    c_bulk   = alpha3 * 1.0 # 1.0 mol/L
    L        = 2            # 区域尺寸

    SymbolCompute()
