# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../../build_pnp
#重点就是指定上面的两个路径, 然后就可以make编译了
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mhd/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
OBJ = InitialConditions.o
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = exMHD exAMRMHD 
PAR_EXAMPLES = exMHDp exAMRMHDp imMHDp imAMRMHDp
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

# Remove built-in rule
%: %.cpp
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(OBJ) $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

imMHDp: imMHDp.cpp $(OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@  $(OBJ) $(MFEM_LIBS)

test_mat: test_mat.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@  $(MFEM_LIBS)


all: $(EXAMPLES) 

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec imMHDp
.PRECIOUS: %.o


##implicitMHDc2: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) -g $< -o $@ $(MFEM_LIBS)
##
##exAMRMHD: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) -g $< -o $@ $(MFEM_LIBS)
##
##exMHD: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) -g $< -o $@ $(MFEM_LIBS)
##
##exMHDnew: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) -g $< -o $@ $(MFEM_LIBS)
##
##exMHDp: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) -g $< -o $@ $(MFEM_LIBS)
##
##exAMRMHDp: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
##	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)


.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf implicitMHD implicitMHDp

clean-data:
	rm -f *.00* *.sol
