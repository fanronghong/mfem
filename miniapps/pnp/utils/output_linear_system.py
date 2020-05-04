# coding: utf-8
'''

'''

from fenics import *
import petsc4py, inspect, sympy, argparse, os, sys


parameters["linear_algebra_backend"] = "PETSc"

def refine_times(mesh, t):
    for i in range(t):
        mesh = refine(mesh)
    return mesh

def write_CSR(filename=None, matrix=None):
    print("Write csr matrix to " + filename)
    with open(filename, "w") as f:
        row_shifts, cols, vals = as_backend_type(matrix).mat().getValuesCSR()
        f.write("number of row_offsets: {}".format(len(row_shifts)))
        for i in row_shifts:
            f.write("{}\n".format(i))
        f.write("number of columns: {}".format(len(cols)))
        for j in cols:
            f.write("{}\n".format(j))
        f.write("number of values: {}".format(len(vals)))
        for k in vals:
            f.write("{}\n".format(k))

def write_Vector(filename=None, vector=None):
    print("Write vector to " + filename)
    with open(filename, "w") as f:
        vec = as_backend_type(vector).vec().array
        f.write("number of dimension: {}".format(len(vec)))
        for i in vec:
            f.write("{}\n".format(i))


def solve_original_problem(V=None, bc=None, direct_solver=False, iterate_solver=False, **kwargs):
    epsilon =   kwargs["epsilon"]
    sigma =     kwargs["sigma"]
    f =         kwargs["f"]
    u_exact =   kwargs["u_exact"]
    SUPG =      kwargs["SUPG"]

    uh, vh = TrialFunction(V), TestFunction(V)
    F1 = epsilon*inner(grad(uh), grad(vh)) * dx + inner(sigma, grad(uh)) * vh * dx - f * vh * dx

    r = -epsilon*div(grad(uh)) + dot(sigma, grad(uh)) - f
    sigma_norm = sqrt(dot(sigma, sigma))
    h = CellDiameter(V.mesh())
    tao_K_1 = h / (2.0 * sigma_norm)
    tao_K_2 = h ** 2 / (12 * epsilon)
    Pe_K = sigma_norm * h / (6 * epsilon)  # 网格Peclet数
    tao_K = conditional(gt(Pe_K, 1.0), tao_K_1, tao_K_2) # Pe_K >=1, take tao_K_1; Pe_K < 1, take tao_K_2
    supg_term = tao_K * dot(sigma, grad(vh)) * r * dx

    if SUPG:
        print("Use SUPG in FE discretization.")
        a, L = lhs(F1 + supg_term), rhs(F1 + supg_term)
        b = epsilon*inner(grad(uh), grad(vh)) * dx + tao_K * dot(sigma, grad(uh)) * dot(sigma, grad(vh)) * dx
    else:
        print("Do Not use SUPG in FE discretization.")
        a, L = lhs(F1), rhs(F1)
        b = epsilon*inner(grad(uh), grad(vh)) * dx
    Ah = assemble(a)
    bh = assemble(L)
    As = assemble(b) # SPD 预调件子
    bc.apply(Ah)
    bc.apply(bh)
    bc.apply(As)

    # # solve the problem
    # import matplotlib.pyplot as plt
    # u = Function(V)
    # solver = LUSolver(Ah)
    # solver.solve(u.vector(), bh)
    # plt.figure()
    # plot(project(u_exact, V), title="exact soluton", mode="warp")
    # plt.figure()
    # plot(u, title="solution of FEM with SUPG", mode="warp")
    # plt.show()

    return Ah, bh, uh, As


def dummp_data(AH=None, Ah=None, bh=None, As=None, VH=None, Vh=None, **kwargs): # 导出二进制格式
    Prolongation = PETScDMCollection.create_transfer_matrix(VH, Vh) # 反过来从 Vh --> VH 形成的转移矩阵与 projection(v, V)函数 等价！！！

    print("\n======> Write AH.dat")
    viewer = petsc4py.PETSc.Viewer().createBinary("AH.dat", "w")
    viewer(as_backend_type(AH).mat())
    
    print("======> Write Ah.dat")
    viewer = petsc4py.PETSc.Viewer().createBinary("Ah.dat", "w")
    viewer(as_backend_type(Ah).mat())

    print("======> Write As.dat")
    viewer = petsc4py.PETSc.Viewer().createBinary("As.dat", "w")
    viewer(as_backend_type(As).mat())

    print("======> Write Prolongation.dat")
    viewer = petsc4py.PETSc.Viewer().createBinary("Prolongation.dat", "w")
    viewer(as_backend_type(Prolongation).mat())

    print("======> Write bh.dat")
    viewer = petsc4py.PETSc.Viewer().createBinary('bh.dat', 'w')
    viewer(as_backend_type(bh).vec())
    print("Dumpping data finish!\n")

def dummp_data_txt(AH=None, Ah=None, bh=None, As=None, VH=None, Vh=None, **kwargs):
    Prolongation = PETScDMCollection.create_transfer_matrix(VH, Vh) # 反过来从 Vh --> VH 形成的转移矩阵与 projection(v, V)函数 等价！！！

    print("\n======> Write AH.txt")
    viewer = petsc4py.PETSc.Viewer().createASCII("AH.txt", format=petsc4py.PETSc.Viewer.Format.ASCII_COMMON,
                                                 comm= petsc4py.PETSc.COMM_WORLD)
    viewer(as_backend_type(AH).mat())

    print("======> Write Ah.txt")
    viewer = petsc4py.PETSc.Viewer().createASCII("Ah.txt", format=petsc4py.PETSc.Viewer.Format.ASCII_COMMON,
                                                 comm= petsc4py.PETSc.COMM_WORLD)
    viewer(as_backend_type(Ah).mat())

    print("======> Write As.txt")
    viewer = petsc4py.PETSc.Viewer().createASCII("As.txt", format=petsc4py.PETSc.Viewer.Format.ASCII_COMMON,
                                                 comm= petsc4py.PETSc.COMM_WORLD)
    viewer(as_backend_type(As).mat())

    print("======> Write Prolongation.txt")
    viewer = petsc4py.PETSc.Viewer().createASCII("Prolongation.txt", format=petsc4py.PETSc.Viewer.Format.ASCII_COMMON,
                                                 comm= petsc4py.PETSc.COMM_WORLD)
    viewer(as_backend_type(Prolongation).mat())

    print("======> Write bh.txt")
    viewer = petsc4py.PETSc.Viewer().createASCII("bh.txt", format=petsc4py.PETSc.Viewer.Format.ASCII_COMMON,
                                                 comm= petsc4py.PETSc.COMM_WORLD)
    viewer(as_backend_type(bh).vec())
    print("Dumpping data finish!\n")

def dummp_data_csr(AH=None, Ah=None, bh=None, As=None, VH=None, Vh=None, **kwargs):
    Prolongation = PETScDMCollection.create_transfer_matrix(VH, Vh) # 反过来从 Vh --> VH 形成的转移矩阵与 projection(v, V)函数 等价！！！

    write_CSR("Prolongation.txt", Prolongation)
    write_CSR("AH.txt", AH)
    write_CSR("Ah.txt", Ah)
    write_CSR("As.txt", As)
    write_Vector("bh.txt", bh)
    print("Dumpping data finish!\n")



def setup(epsilon=None, nc=None, refinetimes=None, SUPG=None, compute_norm_only=None):
    def AllInputs(epsilon=None, nc=None, SUPG=None):
        sigma = Constant((1, 1))
        epsilon = epsilon
        SUPG = SUPG
        gamma = 1
        nc = nc
        cell = "triangle"
        FE = "CG"
        order = 1
        def AllExpressions():  # PDE的真解，右端项
            '''
            PDE: - epsilon * Laplace u + sigma * grad(u) = f
            '''
            x, y = sympy.symbols("x[0] x[1]")
            u = sympy.exp(x * y) * sympy.sin(pi * x) * sympy.sin(pi * y)  # 真解，满足边界条件为0

            u_x = u.diff(x, 1)
            u_y = u.diff(y, 1)
            u_xx = u_x.diff(x, 1)
            u_yy = u_y.diff(y, 1)
            f = - epsilon * (u_xx + u_yy) \
                + (sigma.values()[0] * u_x + sigma.values()[1] * u_y)

            u = Expression(sympy.printing.ccode(u), degree=6, cell=cell)
            f = Expression(sympy.printing.ccode(f), degree=4, cell=cell)
            return u, f
        u_exact, f = AllExpressions()
        return locals()
    data = AllInputs(epsilon=epsilon, nc=nc, SUPG=SUPG)

    coarse  = UnitSquareMesh(data["nc"], data["nc"])
    # fine    = UnitSquareMesh(data["nf"], data["nf"])
    fine = refine_times(coarse, refinetimes)
    element = FiniteElement(data["FE"], data["cell"], data["order"])

    VH = FunctionSpace(coarse, element)
    bcH = DirichletBC(VH, data["u_exact"], DomainBoundary())

    Vh = FunctionSpace(fine, element)
    bch = DirichletBC(Vh, data["u_exact"], DomainBoundary())

    if not compute_norm_only:
        AH, _, _, *ign = solve_original_problem(VH, bcH, iterate_solver=True, **data)

    Ah, bh, uh, As, *ign = solve_original_problem(Vh, bch, iterate_solver=True, **data)

    if compute_norm_only:
        f = open("tg.out", "a+")
        sys.stdout = f
        print("\n\n======> Read uh.dat")
        uh = Function(Vh)
        viewer = petsc4py.PETSc.Viewer().createBinary('./uh.dat', 'r')
        uh.vector().set_local(petsc4py.PETSc.Vec().load(viewer).getArray())
        print("L2 norm of |u_exact - u_h|: {:.8f}".format(errornorm(data["u_exact"], uh, "L2")))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plot(uh, title="FEM solution(read solution vector from file)", mode="warp")
        # plt.show()
        f.close()
        os._exit(0)

    dummp_data(AH=AH, Ah=Ah, bh=bh, As=As, VH=VH, Vh=Vh, **data)
    # dummp_data_txt(AH=AH, Ah=Ah, bh=bh, As=As, VH=VH, Vh=Vh, **data)
    # dummp_data_csr(AH=AH, Ah=Ah, bh=bh, As=As, VH=VH, Vh=Vh, **data)
    return VH.dim(), Vh.dim()


if __name__ == '__main__':
    if 1:
        with open("options.database", "r") as f:
            for line in f:
                if line.startswith("-epsilon"):
                    epsilon = float(line.strip().split(" ")[1])
                if line.startswith("-nc"):
                    nc = int(line.strip().split(" ")[1])
                if line.startswith("-refinetimes"):
                    refinetimes = int(line.strip().split(" ")[1])

        Ncoarse, Nfine = setup(epsilon=epsilon, nc=nc, refinetimes=refinetimes)
        os._exit(0)


    parser = argparse.ArgumentParser(description="Input options.database, generate linear system!")
    parser.add_argument('-options_database')
    parser.add_argument("-compute_norm_only")
    args = parser.parse_args()

    epsilon = nc = refinetimes = SUPG = 0
    with open(args.options_database, "r") as f:
        for line in f:
            if line.startswith("-epsilon"):
                epsilon = float(line.strip().split(" ")[1])
            if line.startswith("-nc"):
                nc = int(line.strip().split(" ")[1])
            if line.startswith("-refinetimes"):
                refinetimes = int(line.strip().split(" ")[1])
            if line.startswith("-SUPG"):
                SUPG = int(line.strip().split(" ")[1])

    Ncoarse, Nfine = setup(epsilon=epsilon, nc=nc, refinetimes=refinetimes, SUPG=SUPG, compute_norm_only=int(args.compute_norm_only))
    with open(args.options_database, "a") as f:
        f.write("-Ncoarse {}\n".format(Ncoarse))
        f.write("-Nfine {}\n".format(Nfine))

