#-*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import MultipleLocator
import os
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

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
    # 数据来源dddd5,dddd7,dddd9
    dofs1 = [375, 2187, 14739, 107811] # CG
    phi_L2err_p1_Gummel_CG = [1.709663142867e-06 ,5.331941427823e-07  ,1.417857507295e-07  ,3.6017364967346e-08]
    phi_L2err_p2_Gummel_CG = [9.0677006036163e-08, 9.6411904332864e-09, 1.109203806505e-09 ,1.3505173221464e-10]
    phi_L2err_p3_Gummel_CG = [7.1194678823885e-09, 4.0253823886387e-10, 2.3856537329199e-11, 1.4556205114757e-12]
    c1_L2err_p1_Gummel_CG  = [7.3887406793338e-05, 2.0872575658785e-05, 5.3954068610653e-06, 1.360507562863e-06]
    c1_L2err_p2_Gummel_CG  = [4.8228081439797e-06, 5.9971712999153e-07, 7.4754960106047e-08, 9.340389628465e-09]
    c1_L2err_p3_Gummel_CG  = [4.8436574995652e-07, 2.7996570372119e-08, 1.6643325888192e-09, 1.0160209458625e-10]
    c2_L2err_p1_Gummel_CG  = [7.3887625976381e-05, 2.0872659731248e-05, 5.3954305107203e-06, 1.3605136545165e-06]
    c2_L2err_p2_Gummel_CG  = [4.8228132396781e-06, 5.9971734542429e-07, 7.4754967450763e-08, 9.3403897286654e-09]
    c2_L2err_p3_Gummel_CG  = [4.8436581086501e-07, 2.7996570874086e-08, 1.6643325840855e-09, 1.016021018976e-10]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Gummel-CG Algorithm')

    plt.subplot(131)
    plt.plot(dofs1, phi_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(dofs1, phi_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(dofs1, phi_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(dofs1, c1_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(dofs1, c1_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(dofs1, c1_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(dofs1, c2_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(dofs1, c2_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(dofs1, c2_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源dddd6,dddd8,dddd10
    dofs2 = [4607, 36864, 294912, 2359296] # DG
    phi_L2err_p1_Gummel_DG = [1.1908213316503e-06, 3.8176261280444e-07, 1.0404646102519e-07, 2.6787429280004e-08]
    phi_L2err_p2_Gummel_DG = [5.580632641125e-08 , 6.0640956661286e-09, 7.0743246617863e-10, 8.6736711096234e-11]
    phi_L2err_p3_Gummel_DG = [4.6405897152029e-09, 2.9565306186731e-10, 1.8573609966271e-11, 1.1640495522987e-12]
    c1_L2err_p1_Gummel_DG  = [4.9368227106152e-05, 1.4896002572748e-05, 3.985692356082e-06 , 1.0213733304598e-06]
    c1_L2err_p2_Gummel_DG  = [3.1422496583221e-06, 3.865702123161e-07 , 4.8032839183601e-08, 6.0102797904016e-09]
    c1_L2err_p3_Gummel_DG  = [3.1921634593105e-07, 2.0585565098525e-08, 1.2959191897289e-09, 8.1249149295828e-11]
    c2_L2err_p1_Gummel_DG  = [4.9368410620442e-05, 1.4896070498027e-05, 3.9857115708548e-06, 1.0213783077032e-06]
    c2_L2err_p2_Gummel_DG  = [3.1422521951085e-06, 3.8657027271244e-07, 4.8032834638581e-08, 6.010278539527e-09]
    c2_L2err_p3_Gummel_DG  = [3.1921646668613e-07, 2.0585571858461e-08, 1.2959195459689e-09, 8.1249162907996e-11]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Gummel-DG Algorithm')

    plt.subplot(131)
    plt.plot(dofs1, phi_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(dofs1, phi_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(dofs1, phi_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(dofs1, c1_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(dofs1, c1_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(dofs1, c1_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(dofs1, c2_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(dofs1, c2_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(dofs1, c2_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源dddd5,dddd7,dddd9
    dofs1 = [375, 2187, 14739, 107811] # CG
    total_time_cg_p1 = [0.210650833, 0.741110076 , 5.315967528  , 46.369986687 ]
    total_time_cg_p2 = [1.236275491, 11.600974869, 121.869654467, 1199.248494011]
    total_time_cg_p3 = [6.909309446, 88.314832273, 962.684786833, 9293.58390373]
    # 数据来源dddd6,dddd8,dddd10
    dofs2 = [4607, 36864, 294912, 2359296] # DG
    total_time_dg_p1 = [0.511482366 , 5.675165956  , 60.855592931  , 590.498538606]
    total_time_dg_p2 = [3.260492066 , 37.243939857 , 359.543541736 , 3234.251345357]
    total_time_dg_p3 = [15.379614678, 161.333081452, 1511.730181773, 13465.790963216]

    fig = plt.figure()
    plt.subplot(121)
    plt.plot(dofs1, total_time_cg_p1, ':s', label='p=1')
    plt.plot(dofs1, total_time_cg_p2, ':o', label='p=2')
    plt.plot(dofs1, total_time_cg_p3, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("time (s)")
    plt.title("Gummel-CG Algorithm")
    plt.legend()

    plt.subplot(122)
    plt.plot(dofs2, total_time_dg_p1, ':s', label='p=1')
    plt.plot(dofs2, total_time_dg_p2, ':o', label='p=2')
    plt.plot(dofs2, total_time_dg_p3, ':*', label='p=3')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("DOFs")
    plt.ylabel("time (s)")
    plt.title("Gummel-DG Algorithm")
    plt.legend()

    plt.show()

    os._exit(0)


if __name__ == '__main__':
    demo()

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

