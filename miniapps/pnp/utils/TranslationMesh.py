
from itertools import islice
import argparse


def TranslateMesh(filename=None, mv_x=0.0, mv_y=0.0, mv_z=0.0):
    '''
    把MFEM默认支持的3D网格整体平移: x, y, z 坐标分别移动 mv_x, mv_y, mv_z 这么多距离, 其余不变
    :param filename:
    :return:
    '''
    string = ""
    mv_vertices = ""
    print("Input .mesh file: {}".format(filename))
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(r"vertices"):
                num_lines = int(f.readline().strip().split()[0])
                assert int(f.readline().strip().split()[0]) == 3 # 只针对3D网格
                string += "vertices\n{}\n{}\n".format(num_lines, 3)

                for ver in islice(f, num_lines):
                    x, y, z = ver.strip().split()
                    mv_vertices += "{} {} {}\n".format(float(x) + mv_x,
                                                       float(y) + mv_y,
                                                       float(z) + mv_z)

            else:
                string += line


    # print(string)
    # print("\n\n\n\n\n" + mv_vertices)
    string += mv_vertices

    mesh = filename.replace(".mesh", "_translate.mesh")
    print("Output .mesh file: {}".format(mesh))
    with open(mesh, "w") as f:
        f.write(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input .mesh file format, generate .msh file format')
    parser.add_argument("mesh", type=str, help="Path to .mesh file")
    parser.add_argument("x", type=float, help="translate x")
    parser.add_argument("y", type=float, help="translate y")
    parser.add_argument("z", type=float, help="translate z")
    args = parser.parse_args()

    TranslateMesh(args.mesh, args.x, args.y, args.z)







