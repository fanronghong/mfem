'''
然后对tetrahedron重新排序:首先排attribute较小的,然后较大的.
想达到的目的: 让所有蛋白单元的index比水单元的index小,这样在MFEM里面,所有的facet normal的方向
            就自然的从蛋白单元指向水单元.
'''

from itertools import islice
from pprint import pprint

def ReorderElement(filename=None):
    # 首先读取 .msh 里面的所有内容
    nodes = []
    tetrahedrons = []
    triangles = []
    print("Input  .msh file: {}".format(filename))
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("$Nodes"):
                num_nodes = int(f.readline().strip().split()[0])
                for node in islice(f, num_nodes):
                    idx, x, y, z = node.strip().split()
                    nodes.append([int(idx), float(x), float(y), float(z)])

            elif line.startswith("$Elements"):
                num_elms = int(f.readline().strip().split()[0])
                for elm in islice(f, num_elms):
                    idx, geom, *other = elm.strip().split()
                    if (int(geom) == 4): # 四面体
                        tetrahedrons.append([int(idx), 4] + [int(itm) for itm in other])
                    else:
                        assert int(geom) == 2 # 暂时只支持2中Element:四面体和三角形
                        triangles.append([int(idx), 2] + [int(itm) for itm in other])

    # 然后对tetrahedron重新排序:首先排attribute较小的,然后较大的.
    # 想达到的目的: 让所有蛋白单元的index比水单元的index小,这样在MFEM里面,所有的facet normal的方向就自然的从蛋白单元指向水单元
    idx = 1
    new_tetrahedrons = []
    for tet in tetrahedrons:
        if tet[3] == 1:
            assert tet[3] == tet[4]
            tet[0] = idx
            idx += 1
            new_tetrahedrons.append(tet)
    for tet in tetrahedrons:
        if tet[3] == 2:
            assert tet[3] == tet[4]
            tet[0] = idx
            idx += 1
            new_tetrahedrons.append(tet)

    # 重新写入一个新的 .msh 文件
    # Note: 只需要 $MeshFormat, $Nodes 和 $Elements, 其他都不需要, 参考 mfem::Mesh::ReadGmshMesh()
    string = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"

    string += "$Nodes\n{}\n".format(len(nodes))
    for nod in nodes:
        string += " ".join(map(lambda x: str(x), nod)) + "\n"
    string += "$EndNodes\n"

    string += "$Elements\n{}\n".format(len(new_tetrahedrons) + len(triangles))
    for tet in new_tetrahedrons:
        string += " ".join(map(lambda x: str(x), tet)) + "\n"
    for tri in triangles:
        string += " ".join(map(lambda x: str(x), tri)) + "\n"
    string += "$EndElements\n"


    msh = filename.replace(".msh", "_reorder.msh")
    print("Output .msh file: {}".format(msh))
    with open(msh, "w") as f:
        f.write(string)




if __name__ == '__main__':
    filename = "/home/fan/miscellaneous/learn_mfem/cmake-build-debug/temp_examples/1MAG_2.msh"
    ReorderElement(filename=filename)
