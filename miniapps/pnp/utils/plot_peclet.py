# coding: utf-8
import matplotlib.pyplot as plt
import os


def PlotPeclet(filename=None):
    peclet = []
    for i in range(filename[1]):
        name = filename[0] + "{}".format(i+1)
        each = []
        try:
            with open(name, "r") as f:
                for line in f:
                    each.append(float(line))
        except:
            print("Not found file: {}".format(name))
            os._exit(1)

        peclet.append(each)

    num_subfig = len(peclet)
    plt.figure(num_subfig)
    for i, data in enumerate(peclet[:]):
        title = filename[0] + "{}".format(i+1)
        zeros = range(len(data))
        ax = plt.subplot(num_subfig, 1, i+1)
        plt.title(title)
        ax.scatter(zeros, data, s=0.1)  # s is size of dot
        plt.hlines(2, 0, len(data), colors="red")

    plt.xlabel("Number of mesh elements")  # 设置横轴标签
    plt.ylabel("Element Peclet number")  # 设置纵轴标签
    plt.show()


if __name__ == '__main__':
    file = [
        "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/Peclet/c1_Peclet_ref0_cg_1MAG_2_gummel",
        2
    ]
    PlotPeclet(file)