GCC=g++
GCCVERSION = $(shell gcc --version | grep ^gcc | sed 's/^.* //g')

ifeq "$(GCCVERSION)" "4.9.1"
	GCC = g++-4.8
endif
CUDA_PATH=/usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)  

OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS = 
# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS}
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_CCFLAGS += $(pkg-config --libs --cflags opencv)
ALL_CCFLAGS += $(pkg-config --libs --cflags opencv)
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


CFLAGS = -I. -I$(CUDA_PATH)/include `pkg-config --cflags  opencv` -Wall
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart `pkg-config --cflags --libs opencv`


GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

 ################################################################################



all: build

build: gauss_gpu

kernel.o: kernel.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cpp
	$(GCC) $(CFLAGS)  -c main.cpp  -o main.o -std=c++11

gauss_gpu: main.o kernel.o
	$(GCC) -L$(CUDA_PATH)/lib64 -lcudart -lcuda  kernel.o main.o `pkg-config --cflags --libs opencv`   -o $@

clean:
	rm -f gauss_gpu main.o kernel.o

clobber: clean

