
from itertools import islice
import argparse


def Medit2Gmsh(filename=None):
    '''
    注意: 这里的.mesh格式不是MFEM的默认的网格格式!!!
    把 .mesh 格式文件转换成 .msh
    .mesh 格式参考: https://www.xyzgate.com/article?id=5b43109b4f022e535f1d18e9
                    https://hal.inria.fr/inria-00069921/document
    .msh 格式参考: Gmsh文档, http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    暂时只支持Gmsh 2.3版本, 4.0版本参考: http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    查看 mfem::Mesh::ReadGmshMesh() 就知道为啥这么写了: $PhysicalNames 不用, 每个element的elementary entity tag不用(设为0, 重要的是第一个physical entity tag)
    :param filename:
    :return:
    '''
    vertices = []
    tetrahedrons = []
    triangles = []
    print("Input .mesh file: {}".format(filename))
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(r"Vertices"):
                try: # 注意: 有的是把单元(vertex,tetrahedron,triagle)个数放到下一行
                    num_lines = int(line.split()[-1])
                except ValueError:
                    num_lines = int(f.readline().strip().split()[0])
                for ver in islice(f, num_lines): # 一次性读取所有的点, 这些verteices最终在MFEM里面就是Mesh对象的vertices成员变量
                    x, y, z, marker = ver.strip().split()
                    vertices.append([float(x), float(y), float(z), int(marker)])

            elif line.startswith(r"Tetrahedra"): # 一次性读取所有的四面体, 这些tetrahedrons最终在MFEM里面就是Mesh对象的elements成员变量
                try:
                    num_lines = int(line.split()[-1])
                except ValueError:
                    num_lines = int(f.readline().strip().split()[0])
                for tet in islice(f, num_lines):
                    id1, id2, id3, id4, marker = [int(item) for item in tet.strip().split()]
                    tetrahedrons.append([id1, id2, id3, id4, marker])

            elif line.startswith(r"Triangles"): # 一次性读取所有的被标记的三角形, 这些triangles最终在MFEM里面就是Mesh对象的boundary成员变量
                try:
                    num_lines = int(line.split()[-1])
                except ValueError:
                    num_lines = int(f.readline().strip().split()[0])
                for tri in islice(f, num_lines):
                    id1, id2, id3, marker = [int(item) for item in tri.strip().split()]
                    triangles.append([id1, id2, id3, marker])

    # Note: 只需要 $MeshFormat, $Nodes 和 $Elements, 其他都不需要, 参考 mfem::Mesh::ReadGmshMesh()
    string = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"

    string += "$Nodes\n{}\n".format(len(vertices))
    for index, ver in enumerate(vertices):
        string += "{} {} {} {}\n".format(index+1, ver[0], ver[1], ver[2])
    string += "$EndNodes\n"

    string += "$Elements\n{}\n".format(len(tetrahedrons) + len(triangles))
    for index, tet in enumerate(tetrahedrons):
        string += "{elm_num} {elm_type} {num_tag} {tags} {node1} {node2} {node3} {node4}\n".format(elm_num=index+1, elm_type=4, num_tag=2, tags="{} {}".format(tet[-1], tet[-1]), node1=tet[0], node2=tet[1], node3=tet[2], node4=tet[3])
    for index, tri in enumerate(triangles): # 为了能够在Gmsh里面可视化，下面把elementary entity tag(第二个tag)也取成和physical entity tag(第一个tag)一样
        string += "{elm_num} {elm_type} {num_tag} {tags} {node1} {node2} {node3}\n".format(elm_num=index+1 + len(tetrahedrons), elm_type=2, num_tag=2, tags="{} {}".format(tri[-1], tri[-1]), node1=tri[0], node2=tri[1], node3=tri[2])
    string += "$EndElements\n"


    msh = filename.replace(".mesh", ".msh")
    print("Output .msh file: {}".format(msh))
    with open(msh, "w") as f:
        f.write(string)


if __name__ == '__main__':
    # filename = "/home/fan/miscellaneous/learn_mfem/data/1MAG_2.mesh"

    parser = argparse.ArgumentParser(description='Input .mesh file format, generate .msh file format')
    parser.add_argument("mesh", type=str, help="Path to .mesh file")
    args = parser.parse_args()

    Medit2Gmsh(args.mesh)