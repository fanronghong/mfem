configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/pnp_box_petsc_opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/pnp_box_petsc_opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/4_4_4_translate.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/4_4_4_translate.msh
               COPYONLY)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/gummel_lu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/newton_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/newton_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/newton_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/newton_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/newton_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/newton_lu
               COPYONLY)


add_mfem_miniapp(pnp_box_steadystate
                 MAIN pnp_box_steadystate.cpp
                 LIBRARIES mfem)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_steadystate/bsub.box
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_steadystate/bsub.box
               COPYONLY)