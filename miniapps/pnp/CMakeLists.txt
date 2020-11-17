
FIND_PACKAGE(PythonLibs 3.5 OPTIONAL_COMPONENTS)
if (PythonLibs_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    link_libraries(${PYTHON_LIBRARIES})
    message("===> python include path: ${PYTHON_INCLUDE_DIRS}")
    message("===> python library: ${PYTHON_LIBRARIES}")
endif()

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
message("===> CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

FIND_PACKAGE(SLEPc)
if (SLEPC_FOUND)
    include_directories(${SLEPC_INC})
    link_libraries(${SLEPC_LIB})
    message("===> slepc include path: ${SLEPC_INC}")
    message("===> slepc library: ${SLEPC_LIB}")
endif()

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/bsub.script
               ${PROJECT_BINARY_DIR}/miniapps/pnp/bsub.script
               COPYONLY)


add_subdirectory(pnp_box_steadystate)
add_subdirectory(pnp_box_timedependent)

add_subdirectory(pnp_protein_steadystate)
add_subdirectory(pnp_protein_timedependent)

add_subdirectory(pnp_data)

add_subdirectory(utils)

add_subdirectory(examples)

add_subdirectory(EAFE_SUPG_advection_diffusion)

