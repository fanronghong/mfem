#coding: utf-8
import matplotlib.pyplot as plt
import os

def PlotConservative(filename=None):
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
        plt.title(filename[i])
        ax.scatter(zeros, data)
        # plt.plot(data, linestyle=":")
        plt.hlines(0, 0, len(data), colors="red")

    plt.xlabel("element number") # 设置横轴标签
    plt.ylabel("local mass") # 设置纵轴标签
    plt.show()


if __name__ == '__main__':
    file = [
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_box/c1_local_conservation_CG_Gummel_box.txt",
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_box/c1_local_conservation_DG_Gummel_box.txt",
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_box/phi_local_conservation_DG_Gummel_box.txt",
        # "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_box/phi_local_conservation_CG_Gummel_box.txt"
        "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c1_local_conservation.txt",
        "/home/fan/mfem/build_pnp/miniapps/pnp/pnp_protein/c2_local_conservation.txt"
    ]
    PlotConservative(file)