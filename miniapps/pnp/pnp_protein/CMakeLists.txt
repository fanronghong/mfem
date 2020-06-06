configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1MAG.pqr
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1MAG.pqr
               COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1MAG_2.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1MAG_2.msh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1bl8_tu.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1bl8_tu.msh
               COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/1bl8.pqr
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/1bl8.pqr
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/2Jk4.msh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/2Jk4.msh
               COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/BPTI.pqr
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/BPTI.pqr
               COPYONLY)


add_mfem_miniapp(pnp_steadystate_protein
        MAIN pnp_steadystate_protein.cpp
        LIBRARIES mfem)


configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/pnp_protein_petsc_opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/pnp_protein_petsc_opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/run.sh
               COPYONLY)

