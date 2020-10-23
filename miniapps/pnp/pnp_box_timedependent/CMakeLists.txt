configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_box_timedependent/bsub.box
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_box_timedependent/bsub.box
               COPYONLY)


add_mfem_miniapp(pnp_box_timedependent
                 MAIN pnp_box_timedependent.cpp
                 LIBRARIES mfem)


