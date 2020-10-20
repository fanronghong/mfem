configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/gummel_lu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/newton_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/newton_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/newton_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/newton_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/newton_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/newton_lu
               COPYONLY)


add_mfem_miniapp(pnp_box_timedependent
                 MAIN pnp_box_timedependent.cpp
                 LIBRARIES mfem)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/bsub.box
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/bsub.box
               COPYONLY)