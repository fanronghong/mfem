add_mfem_miniapp(project_coefficient
        MAIN project_coefficient.cpp
        LIBRARIES mfem)

add_mfem_miniapp(example
        MAIN example.cpp
        LIBRARIES mfem)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/examples/adv_diff_petsc.opts
        ${PROJECT_BINARY_DIR}/miniapps/pnp/examples/adv_diff_petsc.opts
        COPYONLY)
add_mfem_miniapp(adv_diff
        MAIN adv_diff.cpp
        LIBRARIES mfem)








