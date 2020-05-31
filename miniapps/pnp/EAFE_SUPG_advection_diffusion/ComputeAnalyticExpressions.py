# coding: utf-8

import sympy, os
import re, argparse

pi = 3.141592653589793e0

# 将C++语言中的函数及变量替换为Python中的函数及变量
ReplaceRules = {r"sin": r"sympy.sin",
                r"cos": r"sympy.cos",
                r"exp": r"sympy.exp",
                r"x[0]": r"x",
                r"x[1]": r"y",
                r"x[2]": r"z"}
math_compute_symbols = re.compile(r"[\+\-\*\/]{1}") #匹配4种+-*/数学运算符号
# math_compute_symbols = re.compile("[" + re.escape("+-*/") + "]{1}") #同上,只是不需要自己写转义字符
def ReplaceExpressions(string=None):
    # print("Old expression: {!r}".format(string))
    for key, val in ReplaceRules.items():
        string = string.replace(key, val)
    # print("New expression: {!r}".format(string))
    return string

# 计算真解对应的rhs, 已经将它们转换成C++能够识别的字符串
def SymbolCompute(**kwargs):
    '''
    -\nabla\cdot(\alpha \nabla u + \beta u) = f
    '''
    x, y, z = sympy.symbols("x[0] x[1] x[2]")

    u = sympy.cos(pi * x) * sympy.cos(pi * y) * sympy.cos(pi * z)
    # u = sympy.exp(2 * pi * x) * sympy.exp(2 * pi * y) * sympy.exp(2 * pi * z)

    u_x = u.diff(x, 1)
    u_y = u.diff(y, 1)
    u_z = u.diff(z, 1)

    alpha = 1E-1
    beta = 1E-0
    kwargs = {"diff": [[alpha, 0, 0], # 扩散系数矩阵
                      [0, alpha, 0],
                      [0, 0, alpha]],
              "adv": [beta + 0*x,     # 对流速度向量
                      beta + 0*y,
                      beta + 0*z]}


    flux = [kwargs["diff"][0][0] * u_x + kwargs["diff"][0][1] * u_y + kwargs["diff"][0][2] * u_z + kwargs["adv"][0] * u,
            kwargs["diff"][1][0] * u_x + kwargs["diff"][1][1] * u_y + kwargs["diff"][1][2] * u_z + kwargs["adv"][1] * u,
            kwargs["diff"][2][0] * u_x + kwargs["diff"][2][1] * u_y + kwargs["diff"][2][2] * u_z + kwargs["adv"][2] * u]

    f = -(flux[0].diff(x, 1) + flux[1].diff(y, 1) + flux[2].diff(z, 1))
    div_adv = -(kwargs["adv"][0].diff(x, 1) + kwargs["adv"][1].diff(y, 1) + kwargs["adv"][2].diff(z, 1)) #取负是因为在MFEM中定义MassIntegrator就不用管负号

    print("u_exact: {}\n".format(sympy.printing.ccode(u)))
    print("source term: {}\n".format(sympy.printing.ccode(f)))
    print("divergence of advection: {}\n".format(sympy.printing.ccode(div_adv)))
    return sympy.printing.ccode(u), sympy.printing.ccode(f), sympy.printing.ccode(div_adv)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='输入parameters.hpp,根据里面的参数,算出PDEs的真解和右端项,并将它们写入到parameters.hpp中.')
    # parser.add_argument("-parameters_hpp", nargs="?", default="./adv_diff.hpp", help="Path of parameters.hpp")
    # args = parser.parse_args()
    #
    # # 提取C++语言中的参数, 转换为Python能识别的对象
    # alpha = None # 扩散系数的量级
    # DiffusionTensor = [[], []]
    # AdvectionVector = []
    # with open(args.parameters_hpp, "r") as f:
    #     for line in f:
    #         if line.startswith(r"void DiffusionTensor("):
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             DiffusionTensor[0].append(re.split(r"[=;]", newline)[-2])
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             DiffusionTensor[0].append(re.split(r"[=;]", newline)[-2])
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             DiffusionTensor[1].append(re.split(r"[=;]", newline)[-2])
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             DiffusionTensor[1].append(re.split(r"[=;]", newline)[-2])
    #         elif line.startswith(r"void AdvectionVector("):
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             AdvectionVector.append(re.split(r"[=;]", newline)[-2])
    #             line = f.readline()
    #             newline = ReplaceExpressions(line)
    #             AdvectionVector.append(re.split(r"[=;]", newline)[-2])
    #         elif line.startswith(r"double alpha"):
    #             alpha = (re.split(r"[=;\s]+", line)[2])
    #
    # print("Magnitude of diffusion: {}".format(alpha))
    # print(DiffusionTensor)
    # print(AdvectionVector)
    # if alpha:
    #     for i in range(len(DiffusionTensor)):
    #         for j in range(len(DiffusionTensor[i])):
    #             DiffusionTensor[i][j] = re.sub(r"alpha", str(alpha), DiffusionTensor[i][j])
    # print(DiffusionTensor)
    #
    # x, y = sympy.symbols("x[0] x[1]")
    # params = {"diff": [[eval(DiffusionTensor[0][0]), eval(DiffusionTensor[0][1])],
    #                    [eval(DiffusionTensor[1][0]), eval(DiffusionTensor[1][1])]],
    #           "adv": [eval(AdvectionVector[0]),
    #                   eval(AdvectionVector[1])],
    #           }
    # result = SymbolCompute(**params)
    # print("Succeed Computing Analytic Expressions in {}".format(args.parameters_hpp))
    #
    # all_lines = None
    # with open(args.parameters_hpp, "r") as f:
    #     all_lines = f.readlines()
    #
    # # 将Python中的对象转换为C++能识别的变量和函数, 并写入到C++源码中
    # for idx, line in enumerate(all_lines):
    #     if line.startswith(r"double analytic_solution"):
    #         all_lines[idx] = re.sub(r"{.*}", r"{{return {};}}".format(result[0]), line)
    #     elif line.startswith(r"double analytic_rhs"):
    #         all_lines[idx] = re.sub(r"{.*}", r"{{return {};}}".format(result[1]), line)
    #     elif line.startswith(r"double div_advection"):
    #         all_lines[idx] = re.sub(r"{.*}", r"{{return {};}}".format(result[2]), line)
    #
    # with open(args.parameters_hpp, "w") as f:
    #     f.writelines(all_lines)
    # print("Succeed Modifying Analytic Expressions in {}".format(args.parameters_hpp))



    SymbolCompute()