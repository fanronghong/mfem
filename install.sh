#!/bin/bash
#set -x
# 在脚本所在当前路径下运行此脚本即可编译、安装mfem-4.4。如果已经安装，自动退出。
# 本脚本会在当前路径下建立build目录和安装目录，build路径为./cmake-build-debug, install路径为./install(默认不安装)


# 所有依赖包的顶层目录
TOPDIR="/home/fan/ENV4PKGs"
# 下面依次列出所有的依赖包安装路径
MPIDIR="${TOPDIR}/mpich/install"
cmake="${TOPDIR}/cmake/install/bin/cmake"
HYPRE_DIR="${TOPDIR}/hypre/install"
METIS_DIR="${TOPDIR}/metis/install"
PARMETIS_DIR="${TOPDIR}/parmetis/install"
SUPERLU_DIST_DIR="${TOPDIR}/superlu_dist/install"
MUMPS_DIR="${TOPDIR}/mumps/install"
ScaLAPACK_DIR="${TOPDIR}/scalapack/scalapack-2.0.2"
BlasLapack_Lib="-L${TOPDIR}/lapack/install/lib -lblas -llapack"
PETSC_DIR="${TOPDIR}/petsc/petsc-3.15.0"
PETSC_ARCH=arch-linux-c-debug


# 系统及硬件信息
systype=$(uname -s)
cputype=$(uname -m | sed "s/\\ /_/g")
cpucount=$(nproc --all)
SHELL_PATH_NAME="$(realpath "$0")"  # 本脚本所在的路径及名称
SHELL_NAME="${SHELL_PATH_NAME##*/}" # 本脚本名称
SHELL_PATH="${SHELL_PATH_NAME%/*}"  # 本脚本路径
CURDIR=$SHELL_PATH
echo "${systype}, ${cputype}, ${cpucount}"


BUILDDIR="${CURDIR}/cmake-build-debug"
INSTALLDIR="${CURDIR}/install"
if [ "$(ls -A ${INSTALLDIR} | wc -l)" -ne 0 ]; then #判断安装目录是否为空(包括include/, lib/, share/等)
  echo "mfem already installed at: ${INSTALLDIR}"
  exit 0
fi
# 删除build目录重建
rm -rf ${BUILDDIR} ${INSTALLDIR} 
mkdir ${BUILDDIR} 


cd ${BUILDDIR} || exit 1
${cmake} ${CURDIR} \
      -DMPICXX=${MPIDIR}/bin/mpicxx \
      -DCMAKE_BUILD_TYPE=DEBUG \
      -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
      -DMFEM_USE_MPI=ON -DMPIEXEC_EXECUTABLE=${MPIDIR}/bin/mpiexec \
      -DMFEM_ENABLE_TESTING=ON -DMFEM_ENABLE_EXAMPLES=ON -DMFEM_ENABLE_MINIAPPS=ON \
      -DHYPRE_DIR=${HYPRE_DIR} \
      -DMETIS_DIR=${METIS_DIR} \
      -DParMETIS_DIR=${PARMETIS_DIR} \
      -DMFEM_USE_SUPERLU=ON -DSuperLUDist_DIR=${SUPERLU_DIST_DIR} \
      -DMFEM_USE_PETSC=ON -DPETSC_DIR=${PETSC_DIR} -DPETSC_ARCH=arch-linux-c-debug || exit 1
#      -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=${MUMPS_DIR} \
#      -DScaLAPACK_DIR=${ScaLAPACK_DIR} \

# 上面的 -D<variable> 参数都可以在 ./config/defaults.cmake 中的 option(...) 或者 set(...) 找到默认值.
# 怎么使用自己安装的MPI而不是系统的MPI? MPIEXEC_EXECUTABLE 指定了自己安装的mpi的路径下面的mpiexec, 由这个可执行文件的路径可以
# 推断出MPI的安装路径: 通过 find_package(MPI REQUIRED). 具体可以参考cmake的安装路径下面的share/cmake-3.17/Modules中的FindMPI.cmake
# 为啥使用hypre只需 -DHYPRE_DIR=${HYPRE_DIR}, 而使用mumps需要 -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=${MUMPS_DIR} ? 请查看./CMakeLists.txt.
# mumps需要scalapack, 而scalapack需要在defaults.cmake中指定scalapack-config.cmake的路径. 但是安装完scalapack之后没有这个文件!!!


echo "============================================"
echo "Configure done. Continue compile: yes or no?"
echo "============================================"
read compile 
if [ $compile = no ] ; then
    exit 0
fi
# 编译mfem
make -j ${cpucount}
# 编译examples
cd ${BUILDDIR}/examples || exit 1
make -j ${cpucount}
# 编译unittest
cd ${BUILDDIR}/tests || exit 1
make -j ${cpucount}
# 编译miniapps
cd ${BUILDDIR}/miniapps || exit 1
make -j ${cpucount}
# 编译完上述3个目录下的测试算例之后，下面进行测试
cd ${BUILDDIR} || exit 1
ctest -E "superlu" # 跳过测试名称中含有"superlu"字样的测例(太慢了)


echo "============================================"
echo "Compile done. Continue install: yes or no?"
echo "============================================"
read install
if [ $install = no ] ; then
    exit 0 # 不需要安装
fi
cd $CURDIR
mkdir ${INSTALLDIR}
make install
