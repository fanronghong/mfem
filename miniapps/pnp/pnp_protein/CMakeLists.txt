
configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/pnp_steadystate_protein_petsc.opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/pnp_steadystate_protein_petsc.opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1MAG.pqr
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1MAG.pqr
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1MAG_2.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1MAG_2.msh
               COPYONLY)


add_mfem_miniapp(pnp_steadystate_protein
        MAIN pnp_steadystate_protein.cpp
        LIBRARIES mfem)


