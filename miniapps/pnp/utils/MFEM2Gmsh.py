
from itertools import islice
import argparse


def MFEM2Gmsh(filename=None):
    '''
    只针对3D网格, 把MFEM的.mesh格式转换成Gmsh的.msh网格格式
    把 .mesh 格式文件转换成 .msh
    .mesh 格式参考: https://mfem.org/mesh-formats/#mfem-mesh-v10
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
            if line.startswith(r"vertices"):
                num_lines = int(f.readline().strip().split()[0])
                assert int(f.readline().strip().split()[0]) == 3 # 只针对3D网格
                for ver in islice(f, num_lines):
                    x, y, z = ver.strip().split()
                    vertices.append([float(x), float(y), float(z)])

            elif line.startswith(r"elements"):
                num_lines = int(f.readline().strip().split()[0])
                for tet in islice(f, num_lines):
                    marker, type, id1, id2, id3, id4 = [int(item) for item in tet.strip().split()]
                    assert type == 4 # 只能是四面体
                    tetrahedrons.append([id1+1, id2+1, id3+1, id4+1, marker]) # .mesh中vertex的编号从0开始,而.msh中以1开始

            elif line.startswith(r"boundary"):
                num_lines = int(f.readline().strip().split()[0])
                for tri in islice(f, num_lines):
                    marker, type, id1, id2, id3 = [int(item) for item in tri.strip().split()]
                    assert type == 2 # 只能是三角形
                    triangles.append([id1+1, id2+1, id3+1, marker]) # .mesh中vertex的编号从0开始,而.msh中以1开始

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

    MFEM2Gmsh(args.mesh)