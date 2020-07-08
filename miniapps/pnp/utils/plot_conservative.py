#coding: utf-8
import matplotlib.pyplot as plt
import os

def PlotConservative(filename=None, title=None):
    local_conserv = []
    for name in filename:
        local = []
        try:
            with open(name, "r") as f:
                for line in f:
                    local.append(float(line))
        except:
            print("Not found file: {}".format(name))
            os._exit(1)

        local_conserv.append(local)


    num_subfig = len(local_conserv)
    subfig_tag = int("{}11".format(num_subfig))
    plt.figure(num_subfig)
    for i, data in enumerate(local_conserv):
        zeros = range(len(data))
        ax = plt.subplot(subfig_tag + i)
        plt.title(title[i])
        ax.scatter(zeros, data, s=1) # s is size of dot
        # plt.plot(data, linestyle=":")
        plt.hlines(0, 0, len(data), colors="red")

    plt.xlabel("Number of mesh elements") # 设置横轴标签
    plt.ylabel("Element mass") # 设置纵轴标签
    plt.show()


if __name__ == '__main__':
    file = [
        "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c1_conserv_ref0_gummel_cg_1MAG_2",
        "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c1_conserv_ref0_gummel_dg_1MAG_2",
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c2_conserv_ref0_gummel_cg_1MAG_2",
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c2_conserv_ref0_gummel_dg_1MAG_2",
    ]
    
    title = [
        "Gummel, CG, 1MAG_2, ref=0",
        "Gummel, DG, 1MAG_2, ref=0",
    ]
    PlotConservative(file, title)