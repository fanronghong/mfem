#coding: utf-8
import os, fileinput, fnmatch, argparse
from pprint import pprint
from operator import itemgetter
from collections import OrderedDict, deque

parser = argparse.ArgumentParser()
parser.add_argument("-top", help="top path of options.database, twogridpc.out, hypre.out")
parser.add_argument("-sort1", help="1st sorted by some character")
parser.add_argument("-sort2", help="2nd sorted by some character")
parser.add_argument("-sort3", help="2nd sorted by some character")
parser.add_argument("-group", help="how many elements in a group")
args = parser.parse_args()

# top = "/home/fan/_data_/粗256_细refine2times__epsilon1_不同的beta"
# top = "./"
datas = []
for path, dirlist, filelist in os.walk(args.top, topdown=True):
    if "options.database" in filelist:
        if "twogridpc.out" in filelist:
            tg_file  = os.path.join(path, "twogridpc.out")
        elif "tg.out" in filelist:
            tg_file = os.path.join(path, "tg.out")
        else:
            raise ValueError("No output of twogridpc.cpp with twogrid-pc!")
        if "hypre.out" in filelist:
            amg_file = os.path.join(path, "hypre.out")
        elif "amg.out" in filelist:
            amg_file = os.path.join(path, "amg.out")
        else:
            pass
            # raise ValueError("No output of twogridpc.cpp with hypre-boomeramg!")
        
        opt_file = os.path.join(path, "options.database")
                    
        converged = "CONVERGED_HAPPY_BREAKDOWN" # 要搜索的字符串
        L2_norm = "L2 norm of |u_exact - u_h|:" # 要搜索的字符串
        # data = OrderedDict()
        data = {}
        # with open(opt_file, "r") as ops, open(tg_file, "r") as tg:
        with open(opt_file, "r") as ops, open(tg_file, "r") as tg, open(amg_file, "r") as amg:
            for line in ops:
                if line.startswith("-epsilon"):
                    data["epsilon"] = float(line.strip().split(" ")[1])
                if line.startswith("-beta"):
                    data["beta"] = float(line.strip().split(" ")[1])
                if line.startswith("-nc"):
                    data["nc"] = int(line.strip().split(" ")[1])
                if line.startswith("-refinetimes"):
                    data["refinetimes"] = int(line.strip().split(" ")[1])
                if line.startswith("-SUPG"):
                    data["SUPG"] = int(line.strip().split(" ")[1])

            data["tg"] = data["tg_L2norm"] = None
            for line in tg:
                ls = line.strip().split(" ")
                if converged in ls:
                    data["tg"] = int(ls[-1])
                if L2_norm in line:
                    data["tg_L2norm"] = float(ls[-1])

            data["amg"] = None
            for line in amg:
                ls = line.strip().split(" ")
                if converged in ls:
                    data["amg"] = int(ls[-1])

            datas.append(data)


if args.sort1 and (not args.sort2) and (not args.sort3):
    sort_datas = sorted(datas, key=itemgetter(args.sort1))
elif (not args.sort1) and args.sort2 and (not args.sort3):
    sort_datas = sorted(datas, key=itemgetter(args.sort2))
elif (not args.sort1) and (not args.sort2) and args.sort3:
    sort_datas = sorted(datas, key=itemgetter(args.sort2))
elif args.sort1 and args.sort2 and (not args.sort3):
    sort_datas = sorted(datas, key=itemgetter(args.sort1, args.sort2))
elif (not args.sort1) and args.sort2 and args.sort3:
    sort_datas = sorted(datas, key=itemgetter(args.sort2, args.sort3))
elif args.sort1 and (not args.sort2) and args.sort3:
    sort_datas = sorted(datas, key=itemgetter(args.sort1, args.sort3))
elif args.sort1 and args.sort2 and args.sort3:
    sort_datas = sorted(datas, key=itemgetter(args.sort1, args.sort2, args.sort3))
else:
    pprint(datas)
    print("Do not sort!")
    os._exit(0)


if args.group:
    previous = deque(maxlen=int(args.group))
write_latex = False
for idx, item in enumerate(sort_datas):
    print(item, end="\n")
    if args.group:
        previous.append(item)
    if args.group and (idx+1) % int(args.group) == 0:
        print()
        if write_latex:
            with open("./out.txt", "a") as f:
                k = 0
                line = "{} & ".format(item["beta"])
                for i in range(int(args.group)):
                    iters = previous.popleft()["tg"]
                    line += "{}, ".format(iters) if iters else "--, "
                    k += 1
                    if k%3 == 0:
                        line = line[:-2]
                        line += " & "
                line = line[:-2]
                line += "\\\\ \n\\hline\n"
                f.write(line)
                



if __name__ == "__main__":
    top = "/home/fan/_data_supg/differ_Ncoarse_ref1_differ_beta_differ_epsilon"
