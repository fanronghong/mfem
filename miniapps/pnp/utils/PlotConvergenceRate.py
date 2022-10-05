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

def demo(): # Gummel线性化对应的CG和DG离散格式
    # 数据来源dddd5,dddd7,dddd9
    mesh_size = [0.38268343236509, 0.19134171618254, 0.095670858091283, 0.047835429045611] # CG, refine 3

    phi_L2err_p1_Gummel_CG = [1.709663142867e-06 ,5.331941427823e-07  ,1.417857507295e-07  ,3.6017364967346e-08]
    c1_L2err_p1_Gummel_CG  = [7.3887406793338e-05, 2.0872575658785e-05, 5.3954068610653e-06, 1.360507562863e-06]
    c2_L2err_p1_Gummel_CG  = [7.3887625976381e-05, 2.0872659731248e-05, 5.3954305107203e-06, 1.3605136545165e-06]

    phi_L2err_p2_Gummel_CG = [9.0677006036163e-08, 9.6411904332864e-09, 1.109203806505e-09 ,1.3505173221464e-10]
    c1_L2err_p2_Gummel_CG  = [4.8228081439797e-06, 5.9971712999153e-07, 7.4754960106047e-08, 9.340389628465e-09]
    c2_L2err_p2_Gummel_CG  = [4.8228132396781e-06, 5.9971734542429e-07, 7.4754967450763e-08, 9.3403897286654e-09]

    phi_L2err_p3_Gummel_CG = [7.1194678823885e-09, 4.0253823886387e-10, 2.3856537329199e-11, 1.4556205114757e-12]
    c1_L2err_p3_Gummel_CG  = [4.8436574995652e-07, 2.7996570372119e-08, 1.6643325888192e-09, 1.0160209458625e-10]
    c2_L2err_p3_Gummel_CG  = [4.8436581086501e-07, 2.7996570874086e-08, 1.6643325840855e-09, 1.016021018976e-10]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Gummel-CG Algorithm')

    plt.subplot(131)
    plt.plot(mesh_size, phi_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(mesh_size, phi_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(mesh_size, phi_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(mesh_size, c1_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(mesh_size, c1_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(mesh_size, c1_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(mesh_size, c2_L2err_p1_Gummel_CG, ':s', label='p=1')
    plt.plot(mesh_size, c2_L2err_p2_Gummel_CG, ':o', label='p=2')
    plt.plot(mesh_size, c2_L2err_p3_Gummel_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源dddd6,dddd8,dddd10
    mesh_size = [0.38268343236509, 0.19134171618254, 0.095670858091283, 0.047835429045611] # DG, refine 3

    phi_L2err_p1_Gummel_DG = [1.1908213316503e-06, 3.8176261280444e-07, 1.0404646102519e-07, 2.6787429280004e-08]
    c1_L2err_p1_Gummel_DG  = [4.9368227106152e-05, 1.4896002572748e-05, 3.985692356082e-06 , 1.0213733304598e-06]
    c2_L2err_p1_Gummel_DG  = [4.9368410620442e-05, 1.4896070498027e-05, 3.9857115708548e-06, 1.0213783077032e-06]

    phi_L2err_p2_Gummel_DG = [5.580632641125e-08 , 6.0640956661286e-09, 7.0743246617863e-10, 8.6736711096234e-11]
    c1_L2err_p3_Gummel_DG  = [3.1921634593105e-07, 2.0585565098525e-08, 1.2959191897289e-09, 8.1249149295828e-11]
    c2_L2err_p2_Gummel_DG  = [3.1422521951085e-06, 3.8657027271244e-07, 4.8032834638581e-08, 6.010278539527e-09]

    phi_L2err_p3_Gummel_DG = [4.6405897152029e-09, 2.9565306186731e-10, 1.8573609966271e-11, 1.1640495522987e-12]
    c1_L2err_p2_Gummel_DG  = [3.1422496583221e-06, 3.865702123161e-07 , 4.8032839183601e-08, 6.0102797904016e-09]
    c2_L2err_p3_Gummel_DG  = [3.1921646668613e-07, 2.0585571858461e-08, 1.2959195459689e-09, 8.1249162907996e-11]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Gummel-DG Algorithm')

    plt.subplot(131)
    plt.plot(mesh_size, phi_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(mesh_size, phi_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(mesh_size, phi_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(mesh_size, c1_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(mesh_size, c1_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(mesh_size, c1_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(mesh_size, c2_L2err_p1_Gummel_DG, ':s', label='p=1')
    plt.plot(mesh_size, c2_L2err_p2_Gummel_DG, ':o', label='p=2')
    plt.plot(mesh_size, c2_L2err_p3_Gummel_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源dddd5,dddd7,dddd9
    mesh_size = [0.38268343236509, 0.19134171618254, 0.095670858091283, 0.047835429045611]
    total_time_cg_p1 = [0.210650833, 0.741110076 , 5.315967528  , 46.369986687 ]
    total_time_cg_p2 = [1.236275491, 11.600974869, 121.869654467, 1199.248494011]
    total_time_cg_p3 = [6.909309446, 88.314832273, 962.684786833, 9293.58390373]
    # 数据来源dddd6,dddd8,dddd10
    total_time_dg_p1 = [0.511482366 , 5.675165956  , 60.855592931  , 590.498538606]
    total_time_dg_p2 = [3.260492066 , 37.243939857 , 359.543541736 , 3234.251345357]
    total_time_dg_p3 = [15.379614678, 161.333081452, 1511.730181773, 13465.790963216]

    fig = plt.figure()
    plt.subplot(121)
    plt.plot(mesh_size, total_time_cg_p1, ':s', label='p=1')
    plt.plot(mesh_size, total_time_cg_p2, ':o', label='p=2')
    plt.plot(mesh_size, total_time_cg_p3, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("time (s)")
    plt.title("Gummel-CG Algorithm")
    plt.legend()

    plt.subplot(122)
    plt.plot(mesh_size, total_time_dg_p1, ':s', label='p=1')
    plt.plot(mesh_size, total_time_dg_p2, ':o', label='p=2')
    plt.plot(mesh_size, total_time_dg_p3, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("time (s)")
    plt.title("Gummel-DG Algorithm")
    plt.legend()

    plt.show()

    os._exit(0)

def demo1(): # Newton线性化对应的CG和DG离散格式
    # 数据来源ggggg5,ggggg7,ggggg9
    mesh_size = [0.38268343236509, 0.19134171618254, 0.095670858091283, 0.047835429045611] # refine 3 times

    phi_L2err_p1_Newton_CG = [4.9076706008227e-06, 4.3338167765831e-07, 1.3435952134685e-07, 3.5531787552875e-08]
    c1_L2err_p1_Newton_CG  = [0.00029850795683089, 2.0872575499734e-05, 5.3954068397671e-06, 1.360507579623e-06]
    c2_L2err_p1_Newton_CG  = [9.187202621044e-05 , 2.0872659889465e-05, 5.3954305313618e-06, 1.3605136383262e-06]

    phi_L2err_p2_Newton_CG = [2.9629110978087e-07, 1.8670159593064e-08, 1.4400357417177e-09, 1.4563874484459e-10]
    c1_L2err_p2_Newton_CG  = [4.822807658265e-06 , 5.9971712471177e-07, 7.4754960755148e-08, 9.3403893481035e-09]
    c2_L2err_p2_Newton_CG  = [4.8228137250288e-06, 5.9971735082251e-07, 7.4754966924163e-08, 9.3403899759726e-09]

    phi_L2err_p3_Newton_CG = [4.8677615998827e-08, 2.8337052716854e-09, 1.6031905025776e-10, 9.3186025279306e-12]
    c1_L2err_p3_Newton_CG  = [4.8436573442466e-07, 2.7996570195992e-08, 1.6643325814789e-09, 1.016020955722e-10]
    c2_L2err_p3_Newton_CG  = [4.8436582638666e-07, 2.7996571019712e-08, 1.6643325869872e-09, 1.016020964525e-10]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Newton-CG Algorithm')

    plt.subplot(131)
    plt.plot(mesh_size, phi_L2err_p1_Newton_CG, ':s', label='p=1')
    plt.plot(mesh_size, phi_L2err_p2_Newton_CG, ':o', label='p=2')
    plt.plot(mesh_size, phi_L2err_p3_Newton_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(mesh_size, c1_L2err_p1_Newton_CG, ':s', label='p=1')
    plt.plot(mesh_size, c1_L2err_p2_Newton_CG, ':o', label='p=2')
    plt.plot(mesh_size, c1_L2err_p3_Newton_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(mesh_size, c2_L2err_p1_Newton_CG, ':s', label='p=1')
    plt.plot(mesh_size, c2_L2err_p2_Newton_CG, ':o', label='p=2')
    plt.plot(mesh_size, c2_L2err_p3_Newton_CG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源ggggg6,ggggg8,ggggg10
    phi_L2err_p1_Newton_DG = [1.4080441169605e-06, 4.3726876673531e-07, 1.1746850212489e-07, 3.0054706270964e-08]
    c1_L2err_p1_Newton_DG  = [4.9368238499083e-05, 1.4896013036878e-05, 3.9856960363718e-06, 1.0213743536607e-06]
    c2_L2err_p1_Newton_DG  = [4.9368422163591e-05, 1.4896079499274e-05, 3.9857146848341e-06, 1.0213791825122e-06]

    phi_L2err_p2_Newton_DG = [6.910269731495e-08 , 7.5963296171886e-09, 8.943292837865e-10 ,1.0992319969451e-10]
    c1_L2err_p2_Newton_DG  = [3.1422444583083e-06, 3.8656915698253e-07, 4.8032688690171e-08, 6.0102600548903e-09]
    c2_L2err_p2_Newton_DG  = [3.1422481054841e-06, 3.8656949356228e-07, 4.8032724096439e-08, 6.0102650690159e-09]

    phi_L2err_p3_Newton_DG = [6.6971220347722e-09, 3.8179577110911e-10, 2.2585154425281e-11, 1.3742935237902e-12]
    c1_L2err_p3_Newton_DG  = [3.1921669647897e-07, 2.058561385377e-08 , 1.295925698231e-09 , 8.1249979320893e-11]
    c2_L2err_p3_Newton_DG  = [3.1921684466935e-07, 2.0585621723581e-08, 1.2959262263142e-09, 8.1250023141216e-11]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Newton-DG Algorithm')

    plt.subplot(131)
    plt.plot(mesh_size, phi_L2err_p1_Newton_DG, ':s', label='p=1')
    plt.plot(mesh_size, phi_L2err_p2_Newton_DG, ':o', label='p=2')
    plt.plot(mesh_size, phi_L2err_p3_Newton_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||\phi_e - \phi_h||_{L^2}$")
    plt.legend()

    plt.subplot(132)
    plt.plot(mesh_size, c1_L2err_p1_Newton_DG, ':s', label='p=1')
    plt.plot(mesh_size, c1_L2err_p2_Newton_DG, ':o', label='p=2')
    plt.plot(mesh_size, c1_L2err_p3_Newton_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{1,e} - c_{1,h}||_{L^2}$")
    plt.legend()

    plt.subplot(133)
    plt.plot(mesh_size, c2_L2err_p1_Newton_DG, ':s', label='p=1')
    plt.plot(mesh_size, c2_L2err_p2_Newton_DG, ':o', label='p=2')
    plt.plot(mesh_size, c2_L2err_p3_Newton_DG, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("$||c_{2,e} - c_{2,h}||_{L^2}$")
    plt.legend()

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # 数据来源ggggg5,ggggg7,ggggg9
    total_time_cg_p1 = [0.214845806, 0.597663727 , 4.69957973   , 26.865184912]
    total_time_cg_p2 = [0.612946536, 5.182390011 , 49.280601455 , 302.793871018]
    total_time_cg_p3 = [3.068604385, 32.420339671, 317.660134215, 1981.021138444]
    # 数据来源dddd6,dddd8,dddd10
    total_time_dg_p1 = [0.821611292, 9.782432553 , 108.581384507, 1057.423638407]
    total_time_dg_p2 = [5.195099324, 58.013185158, 560.675797105, 5058.251439714]
    total_time_dg_p3 = [23.018890112, 239.32950567, 2288.632139155, 25945.60045763]

    fig = plt.figure()
    plt.subplot(121)
    plt.plot(mesh_size, total_time_cg_p1, ':s', label='p=1')
    plt.plot(mesh_size, total_time_cg_p2, ':o', label='p=2')
    plt.plot(mesh_size, total_time_cg_p3, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("time (s)")
    plt.title("Newton-CG Algorithm")
    plt.legend()

    plt.subplot(122)
    plt.plot(mesh_size, total_time_dg_p1, ':s', label='p=1')
    plt.plot(mesh_size, total_time_dg_p2, ':o', label='p=2')
    plt.plot(mesh_size, total_time_dg_p3, ':*', label='p=3')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.gca().invert_xaxis()
    plt.xlabel("mesh size (h)")
    plt.ylabel("time (s)")
    plt.title("Newton-DG Algorithm")
    plt.legend()

    plt.show()

    os._exit(0)

def demo2(): # 使用EAFE和SUPG求解NP方程的时间差异
    # 数据来源ee1,ee2,ee3,ss1,ss2,ss3
    mesh_size = [0.38268343236509, 0.19134171618254, 0.095670858091283, 0.047835429045611, 0.047835429045611/2, 0.047835429045611/4] # refine 5

    np1_eafe = [0.003897635 , 0.01361938   , 0.1076190626, 0.9105708488 , 24.3784115578 , 210.2011713252]
    np2_eafe = [0.0043495255, 0.0141814942 , 0.1082109336, 0.9121379526 , 24.3529989016 , 215.2970149852]
    np1_supg = [0.0040086875, 0.01509560775, 0.1098770995, 0.98384646975, 27.23847211525, 244.3053652115]
    np2_supg = [0.004266588 , 0.01528169725, 0.1102986135, 0.98290693375, 27.19731748675, 243.967417715]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    # fig.suptitle('Gummel-CG Algorithm')
    plt.plot(mesh_size, np1_eafe, ':s', label='EAFE for NP1')
    plt.plot(mesh_size, np2_eafe, ':o', label='EAFE for NP2')
    plt.plot(mesh_size, np1_supg, ':*', label='SUPG for NP1')
    plt.plot(mesh_size, np2_supg, ':x', label='SUPG for NP2')
    # plt.xscale('symlog') # symlog, logit, log
    # plt.yscale('logit')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.xlabel("mesh size (h)")
    plt.ylabel("average time (s)")
    # plt.title("Gummel-CG Algorithm")
    plt.legend()
    plt.show()

    os._exit(0)

def demo3(): # 测试强可扩展性
    import matplotlib
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    # 数据来源 block1, block2, block3, block4, block5, block6, block7
    # 和 block1_2, block2_2, block3_2, block4_2, block5_2, block6_2, block7_2
    # num_process = [1, 2, 4, 8, 16, 32, 64]
    # T1 = [260.21, 132.89, 79.78, 102.98, 106.40, 166.69, 276.98]
    # T2 = [200.25, 104.66, 58.89, 47.74, 40.64, 55.73, 112.74]
    # 下面把进程数增加求解时间也增加的部分去掉
    num_process = [1, 2, 4, 8, 16]
    T1 = [260.21, 132.89, 79.78, 102.98, 106.40]
    T2 = [200.25, 104.66, 58.89, 47.74, 40.64]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    plt.plot(num_process, T1, ':s', label="$P_1$")
    plt.plot(num_process, T2, '-o', label="$P_5$")
    plt.xscale('log') # symlog, logit, log
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("Number of processors", fontsize=20)
    plt.ylabel("Time (s)", fontsize=20)
    plt.title("Strong Scaling", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    os._exit(0)

def demo4(): # 测试强可扩展性
    import matplotlib
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    # 数据来源 block1_1, block2_1, block3_1, block4_1, block5_1, block6, block7_1 和 block1_3, block2_3, block3_3, block4_3, block5_3, block6_3, block7_3
    # num_process = [2, 4, 8, 16, 32, 64, 128]
    # T1 = [436.17, 291.57, 128.29, 104.46, 85.91, 103.63, 174.16]
    # T2 = [491.35, 331.55, 141.00, 109.75, 80.12, 94.79, 167.89]
    # 下面把进程数增加求解时间也增加的部分去掉
    num_process = [2, 4, 8, 16, 32]
    T1 = [436.17, 291.57, 128.29, 104.46, 85.91]
    T2 = [491.35, 331.55, 141.00, 109.75, 80.12]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    plt.plot(num_process, T1, ':s', label="$P_1$")
    plt.plot(num_process, T2, '-o', label="$P_5$")
    plt.xscale('log') # symlog, logit, log
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("Number of processors", fontsize=20)
    plt.ylabel("Time (s)", fontsize=20)
    plt.title("Strong Scaling", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    os._exit(0)

def demo5(): # 测试强可扩展性
    import matplotlib
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    # 数据来源 scalability0,scalability1,scalability2,scalability3,scalability4,scalability5,scalability6
    num_process1 = [1, 8, 64]
    T1 = [2.09, 10.64, 46.32]
    num_process2 = [4, 32, 256]
    T2 = [255.62, 338.64, 5108.96]

    fig = plt.figure()
    fig.canvas.set_window_title('Window Title')
    plt.plot(num_process1, T1, ':s', label="ref_h: 1, 2, 3")
    plt.plot(num_process2, T2, '-o', label="ref_h: 4, 5, 6")
    plt.xscale('log') # symlog, logit, log
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel("Number of processors", fontsize=20)
    plt.ylabel("Time (s)", fontsize=20)
    plt.title("Weak Scaling", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    os._exit(0)


if __name__ == '__main__':
    # demo1()

    if 0:
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
    elif 1:
        demo3()
    elif 0:
        demo4()
    elif 0:
        demo5()