
message("===> Reading utils/CMakeLists.txt")

add_mfem_miniapp(testAll
        MAIN testAll.cpp
        LIBRARIES mfem)
#add_executable(gnuplot_cpp_ex gnuplot_cpp_ex.cpp)

#add_executable(matplotlibcpp_ex matplotlibcpp_ex.cpp)

