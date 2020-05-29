#coding: utf-8

import sympy

pi = 3.14159265359
diff = 100.0
adv = [1.0, 1.0]

def SymbolCompute():
    x, y, z = sympy.symbols("x[0] x[1] x[2]")

    u = sympy.cos(pi * x / 2) * sympy.cos(pi * y / 2) * sympy.sin(pi * x / 2) * sympy.sin(pi * y / 2)

    u_x = u.diff(x, 1)
    u_y = u.diff(y, 1)

    u_flux = [diff * u_x + adv[0] * u,
              diff * u_y + adv[1] * u]

    f = -( u_flux[0].diff(x, 1) + u_flux[1].diff(y, 1) )

    # 把下面的输出结果复制粘贴到C++源码里面
    print("\nu:\n{}".format(sympy.printing.ccode(u)))
    print("\nf:\n{}".format(sympy.printing.ccode(f)))


if __name__ == '__main__':
    SymbolCompute()
