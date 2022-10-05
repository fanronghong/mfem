#coding: utf-8
'''
这个脚本是用来作自动化测试的: 根据文件目录(目录名暗含了用来作测试的参数), 修改参数文件parameters.hpp, 然后用这个参数文件进行计算
'''
import os, shutil, re

meshsize = {128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12}
option_files_top_path = "/home/fan/eafe_supg_0817/"
source1 = "/home/fan/materials/learn_mfem/parameters.hpp"
source2 = "/home/fan/materials/learn_mfem/adv_diff.cpp"
source3 = "/home/fan/materials/learn_mfem/utils.hpp"
source4 = "/home/fan/materials/learn_mfem/inline-tri.mesh"
source5 = "/home/fan/materials/learn_mfem/ComputeAnalyticExpressions.py"
source6 = "/home/fan/materials/learn_mfem/CMakeLists.txt"

for path, dirlist, filelist in os.walk(option_files_top_path, topdown=True):
    if "options.txt" in filelist:
        opt_file = os.path.join(path, "options.txt")
        print("\n\ncurrent path of options.txt: {}".format(path))
        alpha = None
        N = None
        with open(opt_file) as f:
            for line in f:
                if line.startswith("-alpha"):
                    alpha = float(line.strip().split()[-1])
                if line.startswith("-N"):
                    N = int(line.strip().split()[-1])

        target1 = os.path.join(path, "parameters.hpp")
        target2 = os.path.join(path, "adv_diff.cpp")
        target3 = os.path.join(path, "utils.hpp")
        target4 = os.path.join(path, "inline-tri.mesh")
        target5 = os.path.join(path, "ComputeAnalyticExpressions.py")
        target6 = os.path.join(path, "CMakeLists.txt")
        try:
            shutil.copy(source1, target1) #将parameters.hpp复制到options.txt所在目录
            shutil.copy(source2, target2)
            shutil.copy(source3, target3)
            shutil.copy(source4, target4)
            shutil.copy(source5, target5)
            shutil.copy(source6, target6)
        except IOError as e:
            print("Unable to copy file. {}".format(e))
            os._exit(1)
        except:
            print("Fail to copy file.")
            os._exit(1)
        print("Copy file succeed: from {} to {}".format(source1, target1))
        print("Copy file succeed: from {} to {}".format(source2, target2))
        print("Copy file succeed: from {} to {}".format(source3, target3))
        print("Copy file succeed: from {} to {}".format(source4, target4))
        print("Copy file succeed: from {} to {}".format(source5, target5))
        print("Copy file succeed: from {} to {}".format(source6, target6))

        all_lines = []
        with open(target1, "r") as f:
            for line in f:
                all_lines.append(line)

        # 修改parameters.hpp中的两个参数, 使得与options.txt匹配
        for idx, line in enumerate(all_lines):
            if line.startswith("int refine_times"):
                all_lines[idx] = "int refine_times = {}; \n".format(meshsize[N])
            if line.startswith("double alpha"):
                all_lines[idx] = "double alpha = {}; \n".format(alpha)

        with open(target1, "w") as f:
            for line in all_lines:
                f.write(line)
        print("Succeed: modify ./parameters.hpp to match ./options.txt")
















