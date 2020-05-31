import os, shutil

alphas = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
Ns = [128, 256, 512, 1024, 2048, 4096]
# alphas = [1]
# Ns = [1024]


option_files_top_path = "/home/fan/eafe_supg_0817/"
# option_files_top_path = "/tmp/fff/"
if os.path.exists(option_files_top_path): # 如果顶级目录已经存在,先删除,小心这个操作
    shutil.rmtree(option_files_top_path)
    # raise ValueError("目录已经存在: {}".format(option_files_top_path))


# 下面分别创建多层目录,以及在最底层目录创建参数文件
for n in Ns:
    path1 = "N_" + repr(n) + "/"
    os.makedirs(option_files_top_path + path1)

    for alpha in alphas:
        path2 = "alpha" + repr(alpha).replace(".", "_") + "/"
        os.makedirs(option_files_top_path + path1 + path2)

        with open(option_files_top_path + path1 + path2 + "options.txt", "w") as f:
            f.write("-alpha {}\n".format(alpha))
            f.write("-N {}\n".format(n))


if True: # 打印查看上面的过程有不有错误
    for path, dirlist, filelist in os.walk(option_files_top_path, topdown=True):
        if "options.txt" in filelist:
            opt_file = os.path.join(path, "options.txt")
            print("\n\ncurrent path of options.txt: {}".format(path))
            with open(opt_file) as f:
                for line in f:
                    print(line, end="")




