# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

SRC = $(if $(MFEM_DIR),$(MFEM_DIR)/config/,)

# Helper print-info function
mfem-info = $(if $(filter YES,$(VERBOSE)),$(info *** [info]$(1)),)

# The list of all defines that may be enabled in $(CONFIG_HPP) and $(CONFIG_MK)
# is kept in the MFEM_DEFINES variable, exported here from ../makefile

# The list of variables to set in config.mk using their current values
# is kept in the MFEM_CONFIG_VARS variable, exported here from ../makefile

# Debug output
# $(foreach def,$(MFEM_DEFINES),$(info $(def) = $(value $(def))))
# $(foreach var,$(MFEM_CONFIG_VARS),$(info $(var) = $(value $(var))))

# Output path + file name for the config header
CONFIG_HPP = _config.hpp

# Output path + file name for the config makefile
CONFIG_MK = config.mk

.SUFFIXES:
.PHONY: all get-hypre-version header config-mk

all: header config-mk

MPI = $(MFEM_USE_MPI:NO=)
GHV_CXX ?= $(MFEM_CXX)
GHV = get_hypre_version
GHV_FLAGS = $(subst @MFEM_DIR@,$(if $(MFEM_DIR),$(MFEM_DIR),..),$(HYPRE_OPT))
SMX = $(if $(MFEM_USE_PUMI:NO=),MFEM_USE_SIMMETRIX)
SMX_PATH = $(PUMI_DIR)/include/gmi_sim.h
SMX_FILE = $(subst @MFEM_DIR@,$(if $(MFEM_DIR),$(MFEM_DIR),..),$(SMX_PATH))

$(GHV): $(SRC)$(GHV).cpp # 设定依赖目标 mfem/config/get_hypre_version.cpp
	$(call mfem-info, Determining HYPRE version ...)
	$(GHV_CXX) ${GHV_FLAGS} $(SRC)$(GHV).cpp -o $(GHV) # 编译 mfem/config/get_hypre_version.cpp 为目标文件
$(GHV).out: $(GHV)
	./$(GHV) > $(GHV).out # 运行 get_hypre_version.cpp 生成的可执行文件，输出结果从定向到 get_hypre_version.out
.INTERMEDIATE: $(GHV) $(GHV).out

get-hypre-version: $(GHV).out
	$(eval MFEM_HYPRE_VERSION:=$(shell cat $(GHV).out))
	$(if $(MFEM_HYPRE_VERSION),$(eval export MFEM_HYPRE_VERSION)\
	$(info HYPRE version: $(MFEM_HYPRE_VERSION)),\
	$(error Unable to determine HYPRE version))

check-smx:
	$(call mfem-info, Checking for Simmetrix header [$(SMX_FILE)] ...)
	$(eval MFEM_USE_SIMMETRIX:=$(if $(wildcard $(SMX_FILE)),YES,NO))
	$(call mfem-info, MFEM_USE_SIMMETRIX = $(MFEM_USE_SIMMETRIX))
	$(eval export MFEM_USE_SIMMETRIX)

header: $(if $(MPI),get-hypre-version,) $(if $(SMX),check-smx)
	$(call mfem-info, Writing $(CONFIG_HPP) ...)
	@set -- && \
	for def in $${MFEM_DEFINES} $(if $(MPI),MFEM_HYPRE_VERSION) $(SMX); do \
	   eval var=\$$$$def && \
	   if [ "NO" != "$${var}" ]; then \
	      set -- "$$@" -e "s|// \(#define $${def} \)|\1|" && \
	      set -- "$$@" -e "s|// \(#define $${def}\)$$|\1|" && \
	      set -- "$$@" -e "s#@$${def}@#$${var}#g"; \
	   fi; \
	done && \
	sed "$$@" $(SRC)config.hpp.in > $(CONFIG_HPP)

# The target below uses the values of the shell variables listed in
# the shell variables MFEM_DEFINES(来源于mfem/makefile)
# and MFEM_CONFIG_VARS(来源于mfem/makefile)
config-mk:
	$(call mfem-info, Writing $(CONFIG_MK) ...)
	# set -- : https://stackoverflow.com/questions/20088290/whats-set-progname-means-in-shell-script
	@set -- && \
	for ref in $${MFEM_DEFINES} $${MFEM_CONFIG_VARS}; do \
	   eval var=\$$$$ref && \
	   set -- "$$@" -e "s#@$${ref}@#$${var}#g"; \
	done && \
	set -- "$$@" -e 's/@\([^@]*\)@/$$(\1)/g' -e 's/  *$$//g' && \
	set -- "$$@" -e tb -e ba -e ":b" -e 's/^$$//' -e tc -e ba && \
	set -- "$$@" -e ":c" -e d -e ":a" && \
	sed "$$@" $(SRC)config.mk.in > $(CONFIG_MK) # 依照脚本的指令来处理、编辑文本文件, 主要用来自动编辑一个或多个文件、简化对文件的反复操作、编写转换程序等

clean:
	rm -f $(CONFIG_HPP) $(CONFIG_MK) sample-runs-build.log
