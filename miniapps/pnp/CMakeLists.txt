
FIND_PACKAGE(PythonLibs 3.5 REQUIRED)
if (PythonLibs_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    link_libraries(${PYTHON_LIBRARIES})
    message("===> python include path: ${PYTHON_INCLUDE_DIRS}")
    message("===> python library: ${PYTHON_LIBRARIES}")
endif()


add_subdirectory(pnp_box)

add_subdirectory(pnp_protein)

add_subdirectory(utils)

add_subdirectory(examples)
