
configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/pnp_steadystate_box_petsc.opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/pnp_steadystate_box_petsc.opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/4_4_4_translate.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/4_4_4_translate.msh
               COPYONLY)

add_mfem_miniapp(pnp_steadystate_box
        MAIN pnp_steadystate_box.cpp
        LIBRARIES mfem)


#configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/run.sh
#        ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/run.sh
#        COPYONLY)