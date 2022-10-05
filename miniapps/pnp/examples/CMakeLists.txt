add_mfem_miniapp(project_coefficient
        MAIN project_coefficient.cpp
        LIBRARIES mfem)

add_mfem_miniapp(example
        MAIN example.cpp
        LIBRARIES mfem)

add_mfem_miniapp(Mult_TrueMult
        MAIN Mult_TrueMult.cpp
        LIBRARIES mfem)

add_mfem_miniapp(demo
        MAIN demo.cpp
        LIBRARIES mfem)

add_mfem_miniapp(inhomogeneousBdc
        MAIN inhomogeneousBdc.cpp
        LIBRARIES mfem)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/examples/adv_diff_petsc.opts
        ${PROJECT_BINARY_DIR}/miniapps/pnp/examples/adv_diff_petsc.opts
        COPYONLY)
add_mfem_miniapp(adv_diff
        MAIN adv_diff.cpp
        LIBRARIES mfem)

add_mfem_miniapp(h1_project_dg
        MAIN h1_project_dg.cpp
        LIBRARIES mfem)

if (SLEPC_FOUND)
    add_mfem_miniapp(slepc_demo
            MAIN slepc_demo.cpp
            LIBRARIES mfem slepc)
endif()






