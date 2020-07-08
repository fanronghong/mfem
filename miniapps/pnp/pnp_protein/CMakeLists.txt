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

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/pnp_protein_petsc_opts
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/pnp_protein_petsc_opts
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/gummel_amg
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/gummel_amg
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/gummel_ilu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/gummel_ilu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/gummel_lu
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/gummel_lu
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/newton_shell
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/newton_shell
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/run.sh
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/run.sh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/pnp/pnp_protein/bsub.protein
               ${PROJECT_BINARY_DIR}/miniapps/pnp/pnp_protein/bsub.protein
               COPYONLY)


#add_definitions(-DTESTT=${MESH})
#if(${MESH} STREQUAL "_1MAG_2_test_case")
#    add_definitions(-D_1MAG_2_test_case)
#elseif(${MESH} STREQUAL "_1MAG_2")
#    add_definitions(-D_1MAG_2)
#elseif(${MESH} STREQUAL "_1bl8_tu")
#    add_definitions(-D_1bl8_tu)
#else()
#    add_definitions()
#endif()
#message("===> use mesh: ${MESH}")


add_mfem_miniapp(pnp_steadystate_protein
        MAIN pnp_steadystate_protein.cpp
        LIBRARIES mfem)
