#coding: utf-8
from __future__ import division
import matplotlib.pyplot as plt
import os, numpy as np

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
        Statistics(data)
        # Statistics_(data)
        base = range(len(data))
        ax = plt.subplot(subfig_tag + i)
        plt.title(title[i])
        ax.scatter(base, data, s=1) # s is size of dot
        plt.hlines(0, 0, len(data), colors="red")

    plt.xlabel("Number of mesh elements") # 设置横轴标签
    plt.ylabel("Element mass") # 设置纵轴标签
    plt.show()

def Statistics(data):
    abs_data = map(abs, data)
    length = len(abs_data)
    min_val = min(abs_data)
    max_val = max(abs_data)
    avg_val = np.mean(abs_data)
    scale = (avg_val - min_val) / 10.0
    a1 = a2 = a3 = a4 = 0
    for val in abs_data:
        if   0 <= val and val < 1e-7:
            a1 += 1
        elif 1e-7 <= val and val < 1e-6:
            a2 += 1
        elif 1e-6 <= val and val < 1e-5:
            a3 += 1
        elif 1e-5 <= val:
            a4 += 1
        else:
            raise ValueError

    # 计算比例
    a1 = a1/length
    a2 = a2/length
    a3 = a3/length
    a4 = a4/length

    print("[{}, {}): {:.2%}".format(0, 1e-7, a1))
    print("[{}, {}): {:.2%}".format(1e-7, 1e-6, a2))
    print("[{}, {}): {:.2%}".format(1e-6, 1e-5, a3))
    print("[{}, {}): {:.2%}".format(1e-5, max_val, a4))
    print("\n")

def Statistics_(data):
    abs_data = map(abs, data)
    length = len(abs_data)
    max_val = max(abs_data)
    min_val = min(abs_data)
    avg_val = np.mean(abs_data)
    med_val = np.median(abs_data)
    scale = (avg_val - min_val) / 10.0
    a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = a10 = 0
    for val in abs_data:
        if   0*scale <= val and val < 1*scale:
            a1 += 1
        elif 1*scale <= val and val < 2*scale:
            a2 += 1
        elif 2*scale <= val and val < 3*scale:
            a3 += 1
        elif 3*scale <= val and val < 4*scale:
            a4 += 1
        elif 4*scale <= val and val < 5*scale:
            a5 += 1
        elif 5*scale <= val and val < 6*scale:
            a6 += 1
        elif 6*scale <= val and val < 7*scale:
            a7 += 1
        elif 7*scale <= val and val < 8*scale:
            a8 += 1
        elif 8*scale <= val and val < 9*scale:
            a9 += 1
        elif 9*scale <= val:
            a10 += 1
        else:
            raise ValueError

    # 计算比例
    a1 = a1/length
    a2 = a2/length
    a3 = a3/length
    a4 = a4/length
    a5 = a5/length
    a6 = a6/length
    a7 = a7/length
    a8 = a8/length
    a9 = a9/length
    a10 = a10/length

    print("[{}, {}): {}".format(0*scale, 1*scale, a1))
    print("[{}, {}): {}".format(1*scale, 2*scale, a2))
    print("[{}, {}): {}".format(2*scale, 3*scale, a3))
    print("[{}, {}): {}".format(3*scale, 4*scale, a4))
    print("[{}, {}): {}".format(4*scale, 5*scale, a5))
    print("[{}, {}): {}".format(5*scale, 6*scale, a6))
    print("[{}, {}): {}".format(6*scale, 7*scale, a7))
    print("[{}, {}): {}".format(7*scale, 8*scale, a8))
    print("[{}, {}): {}".format(8*scale, 9*scale, a9))
    print("[{}, {}): {}".format(9*scale, max_val, a10))
    print("\n")


if __name__ == '__main__':
    file = [
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c1_conserv_ref0_gummel_cg_1MAG_2",
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c1_conserv_ref0_gummel_dg_1MAG_2",
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c2_conserv_ref0_gummel_cg_1MAG_2",
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c2_conserv_ref0_gummel_dg_1MAG_2",
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c1_conserv_ref0_gummel_cg_1bl8_tu",
        # "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c1_conserv_ref0_gummel_dg_1bl8_tu",
        "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c2_conserv_ref0_gummel_cg_1bl8_tu",
        "/home/fan/shared_by_two_accounts/phdthesis/numerical_experiments_data/local_conservation/c2_conserv_ref0_gummel_dg_1bl8_tu",
    ]
    
    title = [
        "Gummel, CG, 1MAG_2, ref=0",
        "Gummel, DG, 1MAG_2, ref=0",
        # "Gummel, CG, 1BL8, ref=0",
        # "Gummel, DG, 1BL8, ref=0",
    ]
    PlotConservative(file, title)