import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def Demo_WithArgs(a):
    print("Demo_WithArgs() in python_utils.py")
    print("Input args: {}".format(a))

    plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.cos(x))
    plt.show()
    return 0

def Demo_WithoutArgs():
    beta = np.array([[1,2,3],[1,2,3],[1,2,3]])
    print("myfunction - python")
    print(beta)
    print("")
    firstRow = beta[0,:]
    return firstRow


def Demo_PlotLine():
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 5, 100)
    ax.plot(x, np.cos(x))
    plt.show()


def PlotArray(arr):
    print(arr)
    x = np.arange(arr.shape[0])
    plt.figure()
    ax = plt.axes()
    ax.plot(x, arr)
    plt.show()

def PlotCSR(I, J, Vals, title):
    # print("In python module: {}".format(__file__))
    # print("I: ", I)
    # print("J: ", J)
    # print("Vals: ", Vals)
    size = I.shape[0] - 1
    csr = csr_matrix((Vals, J, I), shape=(size, size))
    plt.figure()
    plt.spy(csr, precision=1E-10)
    plt.title(title)
    plt.savefig('./' + title + '.png')
    plt.show()
def Test_PlotCSR():
    I = np.array([0, 2, 3, 6])
    J = np.array([0, 2, 2, 0, 1, 2])
    Vals = np.array([1, 2, 3, 4, 5, 6])
    csr = PlotCSR(I, J, Vals, "test_title")



def ShowMesh(mesh=None):
    test = "mkdir test && ls"
    string = "glvis -m {}".format(mesh)
    os.system(string)


if __name__ == '__main__':
    Test_PlotCSR()