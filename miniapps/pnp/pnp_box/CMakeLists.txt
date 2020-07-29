configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/pnp_box_petsc_opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/pnp_box_petsc_opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/4_4_4_translate.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/4_4_4_translate.msh
               COPYONLY)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/gummel_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/gummel_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/gummel_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/gummel_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/gummel_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/gummel_lu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/newton_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/newton_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/newton_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/newton_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/newton_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/newton_lu
               COPYONLY)


add_mfem_miniapp(pnp_steadystate_box
        MAIN pnp_steadystate_box.cpp
        LIBRARIES mfem)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box/bsub.box
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box/bsub.box
               COPYONLY)