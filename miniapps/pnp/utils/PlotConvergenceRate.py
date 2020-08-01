#-*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import math

def PlotConvergRate(p_order, mesh_sizes=None, errornorms1=None, errornorms2=None, errornorms3=None):
    sizes = [float(size) for size in mesh_sizes.split()]
    norms1 = [float(norm) for norm in errornorms1.split()]
    norms2 = [float(norm) for norm in errornorms2.split()]
    norms3 = [float(norm) for norm in errornorms3.split()]

    x_coor = []
    y_coor = []
    y1_coor = []
    y2_coor = []
    y3_coor = []
    for i in range(len(sizes)):
        x_coor.append(math.log(sizes[i]))
        y_coor.append((p_order +1)*math.log(sizes[i]) - 6)
        y1_coor.append(math.log(norms1[i]))
        y2_coor.append(math.log(norms2[i]))
        y3_coor.append(math.log(norms3[i]))

    fig,ax = plt.subplots()
    ax.plot(x_coor, y_coor, '-d', label='exact convergence rate')
    ax.plot(x_coor, y1_coor, ':s', label='$\phi_h$ convergence rate')
    ax.plot(x_coor, y2_coor, ':o', label='$c_{1,h}$ convergence rate')
    ax.plot(x_coor, y3_coor, ':*', label='$c_{2,h}$ convergence rate')
    ax.invert_xaxis()
    plt.xlabel("mesh sizes (log)")
    plt.ylabel("errornorms (log)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if 0:
        # 数据来源dddd5
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.709663142867e-06 5.331941427823e-07 1.417857507295e-07 3.6017364967346e-08 9.0407294951329e-09"
        errornorm2 = "7.3887406793338e-05 2.0872575658785e-05 5.3954068610653e-06 1.360507562863e-06 3.4086567120304e-07"
        errornorm3 = "7.3887625976381e-05 2.0872659731248e-05 5.3954305107203e-06 1.3605136545165e-06 3.4086720560745e-07"
    elif 0:
        # 数据来源dddd6
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.1908213316503e-06 3.8176261280444e-07 1.0404646102519e-07 2.6787429280004e-08 6.7692348870455e-09"
        errornorm2 = "4.9368227106152e-05 1.4896002572748e-05 3.985692356082e-06 1.0213733304598e-06 2.5784222485067e-07"
        errornorm3 = "4.9368410620442e-05 1.4896070498027e-05 3.9857115708548e-06 1.0213783077032e-06 2.5784348358464e-07"
    elif 0:
        # 数据来源ggggg5
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.7092319731061e-06 5.3315837672936e-07 1.4178332158692e-07 3.6017208808574e-08 9.0407196268916e-09"
        errornorm2 = "7.3887406791091e-05 2.0872575655984e-05 5.3954068742075e-06 1.3605075677237e-06 3.4086567163204e-07"
        errornorm3 = "7.3887625978123e-05 2.0872659733973e-05 5.3954304971389e-06 1.360513648315e-06 3.408672081231e-07"
    elif 1:
        # 数据来源ggggg6
        p_order = 1
        mesh_size  = "0.38268343236509 0.19134171618254 0.095670858091283 0.047835429045611 0.023917714522614"
        errornorm1 = "1.4080441191044e-06 4.3726876679914e-07 1.17468502077e-07 3.0054706107393e-08 7.5735988504231e-09"
        errornorm2 = "4.9368238642349e-05 1.4896013648534e-05 3.9856962348929e-06 1.0213744233093e-06 2.5784251776647e-07"
        errornorm3 = "4.9368422295529e-05 1.4896080001237e-05 3.9857148334939e-06 1.0213791973692e-06 2.5784370382021e-07"

    PlotConvergRate(p_order, mesh_size, errornorm1, errornorm2, errornorm3)

