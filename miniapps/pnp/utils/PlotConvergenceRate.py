#-*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
import os

def PlotConvergRate(p_order, mesh_sizes=None, errornorms1=None, errornorms2=None, errornorms3=None, log_tranform=True, xaxis="x", yaxis="y"):
    sizes = [float(size) for size in mesh_sizes.split()]
    norms1 = [float(norm) for norm in errornorms1.split()]
    norms2 = [float(norm) for norm in errornorms2.split()]
    norms3 = [float(norm) for norm in errornorms3.split()]

    x_coor = []
    y_coor = []
    y1_coor = []
    y2_coor = []
    y3_coor = []
    if (log_tranform): # log transformation
        for i in range(len(sizes)):
            x_coor.append(math.log(sizes[i]))
            y_coor.append((p_order +1)*math.log(sizes[i]) - 6)
            y1_coor.append(math.log(norms1[i]))
            y2_coor.append(math.log(norms2[i]))
            y3_coor.append(math.log(norms3[i]))
    else: # no transformation
        for i in range(len(sizes)):
            x_coor.append(sizes[i])
            y_coor.append((p_order +1) * sizes[i] - 6)
            y1_coor.append(norms1[i])
            y2_coor.append(norms2[i])
            y3_coor.append(norms3[i])

    xticks = [49152, 2*49152, 49152*3, 49152*4]

    fig,ax = plt.subplots()
    # ax.plot(x_coor, y_coor, '-d', label='exact convergence rate')
    ax.plot(xticks, y1_coor, ':s', label='$\phi_h$ convergence rate')
    ax.plot(xticks, y2_coor, ':o', label='$c_{1,h}$ convergence rate')
    ax.plot(xticks, y3_coor, ':*', label='$c_{2,h}$ convergence rate')
    # ax.set_xticks([49152, 2*49152, 49152*3, 49152*4])
    # ax.xaxis.set_major_locator(MultipleLocator(8))
    # ax.invert_xaxis()
    # plt.xticks(x_coor, [49152, 2*49152, 49152*3, 49152*4])
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    # plt.autoscale(enable=True, axis='y')
    plt.legend()
    plt.show()

def demo():
    x = [0.00001,0.001,0.01,0.1,0.5,1,5]
    # create an index for each tick position
    xi = list(range(len(x)))
    y = [0.945,0.885,0.893,0.9,0.996,1.25,1.19]
    plt.ylim(0.8,1.4)
    # plot the index for the x-values
    plt.plot(xi, y, marker='o', linestyle='--', color='r', label='Square')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xticks(xi, x)
    plt.title('compare')
    plt.legend()
    plt.show()
    os._exit(0)


if __name__ == '__main__':
    # demo()

    if 1:
        # 数据来源dddd5
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611"
        refine = "0 1 2 3"
        DOFs = "375 2187 14739 107811"
        Elements = "384 3072 24576 196608"
        errornorm1 = "1.709663142867e-06 5.331941427823e-07 1.417857507295e-07 3.6017364967346e-08"
        errornorm2 = "7.3887406793338e-05 2.0872575658785e-05 5.3954068610653e-06 1.360507562863e-06"
        errornorm3 = "7.3887625976381e-05 2.0872659731248e-05 5.3954305107203e-06 1.3605136545165e-06"
        # PlotConvergRate(p_order, mesh_size, errornorm1, errornorm2, errornorm3)
        # PlotConvergRate(p_order, refine, errornorm1, errornorm2, errornorm3, False)
        PlotConvergRate(p_order, Elements, errornorm1, errornorm2, errornorm3, False)
    elif 0:
        # 数据来源dddd6
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.1908213316503e-06 3.8176261280444e-07 1.0404646102519e-07 2.6787429280004e-08 6.7692348870455e-09"
        errornorm2 = "4.9368227106152e-05 1.4896002572748e-05 3.985692356082e-06 1.0213733304598e-06 2.5784222485067e-07"
        errornorm3 = "4.9368410620442e-05 1.4896070498027e-05 3.9857115708548e-06 1.0213783077032e-06 2.5784348358464e-07"
        PlotConvergRate(p_order, mesh_size, errornorm1, errornorm2, errornorm3)
    elif 0:
        # 数据来源ggggg5
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.7092319731061e-06 5.3315837672936e-07 1.4178332158692e-07 3.6017208808574e-08 9.0407196268916e-09"
        errornorm2 = "7.3887406791091e-05 2.0872575655984e-05 5.3954068742075e-06 1.3605075677237e-06 3.4086567163204e-07"
        errornorm3 = "7.3887625978123e-05 2.0872659733973e-05 5.3954304971389e-06 1.360513648315e-06 3.408672081231e-07"
        PlotConvergRate(p_order, mesh_size, errornorm1, errornorm2, errornorm3)
    elif 0:
        # 数据来源ggggg6
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.4080441191044e-06 4.3726876679914e-07 1.17468502077e-07 3.0054706107393e-08 7.5735988504231e-09"
        errornorm2 = "4.9368238642349e-05 1.4896013648534e-05 3.9856962348929e-06 1.0213744233093e-06 2.5784251776647e-07"
        errornorm3 = "4.9368422295529e-05 1.4896080001237e-05 3.9857148334939e-06 1.0213791973692e-06 2.5784370382021e-07"
        PlotConvergRate(p_order, mesh_size, errornorm1, errornorm2, errornorm3)

