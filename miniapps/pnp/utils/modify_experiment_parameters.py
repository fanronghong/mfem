#coding: utf-8
import os, fileinput, fnmatch, argparse
from pprint import pprint
from operator import itemgetter
from collections import OrderedDict

top = "/home/fan/_data_supg/differ_Ncoarse_ref2_differ_beta_differ_epsilon"
# show = ("-nc", "-epsilon", "-beta", "-refinetimes", "-SUPG") # 想要打印的参数
show = ("-refinetimes",)
target = 2

if 0: # 修改文件中的参数
    for path, dirlist, filelist in os.walk(top, topdown=True):
        if "options.database" in filelist:
            opt_file = os.path.join(path, "options.database")
            lines_list = None
            with open(opt_file, "r") as f:
                lines_list = f.readlines()

            with open(opt_file, "w") as f:
                for line in lines_list:
                    # if line.startswith("-nc "):
                    #     f.write("-nc {}\n".format(target_nc))
                    if line.startswith("-refinetimes "):
                        f.write("-refinetimes {}\n".format(target))
                    else:
                        f.write(line)


elif 1: # 显示文件中的参数
    for path, dirlist, filelist in os.walk(top, topdown=True):
        if "options.database" in filelist:
            opt_file = os.path.join(path, "options.database")
            with open(opt_file, "r") as f:
                for line in f:
                    if line.startswith(show):
                        print(line, end="")
            print()



