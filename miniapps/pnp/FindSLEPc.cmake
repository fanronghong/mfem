#############################################################
# Try to find SLEPc                                         #
#                                                           #
# Once done this will define:                               #
#  SLEPC_FOUND     - system has SLEPc                       #
#  SLEPC_DIR       - SLEPc directory                        #
#  SLEPC_INC       - SLEPc include directory                #
#  SLEPC_LIB       - SLEPc library (static or dynamic)      #
#                                                           #
# Usage:                                                    #
#  find_package(SLEPc)                                      #
#                                                           #
# Setting these changes the behavior of the search          #
#  SLEPC_DIR       - SLEPc directory                        #
# ref: https://gitlab.onelab.info/gmsh/small_fem/blob/master/cmake/FindSLEPc.cmake
#############################################################

## Try to set SLEPC_DIR ##
##########################
message("===> environment variable SLEPc: $ENV{SLEPC_DIR}")
if(NOT DEFINED SLEPC_DIR)
    MESSAGE("HHH")
    set(SLEPC_DIR $ENV{SLEPC_DIR})
endif()
set(SLEPC_DIR $ENV{SLEPC_DIR})
message("===> cmake variable SLEPC_DIR: ${SLEPC_DIR}")

## Includes ##
##############
if(EXISTS "${SLEPC_DIR}/include" AND
        EXISTS "${SLEPC_DIR}/${PETSC_ARCH}/include")
    set(SLEPC_INC "${SLEPC_DIR}/include" "${SLEPC_DIR}/${PETSC_ARCH}/include")
else()
    message(SEND_ERROR "SLEPc includes not found")
endif()

## Library ##
#############
message("fffffffffffffffffffffffff: ${SLEPC_DIR}")
if(EXISTS "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.so")
    set(SLEPC_LIB "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.so")
elseif(EXISTS "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.a")
    set(SLEPC_LIB "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.a")
elseif(EXISTS "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.dylib")
    set(SLEPC_LIB "${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.dylib")
else()
    message(SEND_ERROR "SLEPc library not found")
endif()

## CMake check and done ##
##########################
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPc
        "SLEPc could not be found: be sure to set SLEPC_DIR in your environment variables"
        SLEPC_LIB SLEPC_INC SLEPC_DIR)
