# Location of the CUDA Toolkit
CUDA_PATH?=/usr/local/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
   ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
      TARGET_SIZE := 64
    else ifneq (,$(filter $(TARGET_ARCH),armv7l))
      TARGET_SIZE := 32
    endif
  else
    TARGET_SIZE := $(shell getconf LONG_BIT)
  endif
else
  $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
    $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
  endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
  TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux ))
  $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
    HOST_COMPILER ?= arm-linux-gnueabihf-g++
  else ifeq ($(TARGET_ARCH),aarch64)
    HOST_COMPILER ?= aarch64-linux-gnu-g++
  else ifeq ($(TARGET_ARCH),ppc64le)
    HOST_COMPILER ?= powerpc64le-linux-gnu-g++
  endif
endif
HOST_COMPILER ?= g++
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS   :=
LDFLAGS   :=

# build flags
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    ifneq ($(TARGET_FS),)
      GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
      ifeq ($(GCCVERSIONLTEQ46),1)
        CCFLAGS += --sysroot=$(TARGET_FS)
      endif
      LDFLAGS += --sysroot=$(TARGET_FS)
      LDFLAGS += -rpath-link=$(TARGET_FS)/lib
      LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
      LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
    endif
  endif
endif

# Debug build flags
ifeq ($(dbg),1)
  NVCCFLAGS += -g -G
  BUILD_TYPE := debug
else
  NVCCFLAGS += -O3
  BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := -I./include
LIBRARIES :=

################################################################################

# Makefile include to help find GL Libraries
include ./findgllib.mk

LIBRARIES += $(GLLINK)
LIBRARIES += -lGLEW -lGL -lGLU -lX11 -lglut -lopencv_core -lopencv_video -lopencv_highgui -lopencv_videoio

# Gencode arguments
SMS ?= 30 35 37 50 52 60 70

ifeq ($(GENCODE_FLAGS),)
  # Generate SASS code for each SM architecture listed in $(SMS)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

  # Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
  HIGHEST_SM := $(lastword $(sort $(SMS)))
  ifneq ($(HIGHEST_SM),)
    GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
  endif
endif


################################################################################

# Target rules
all: build

build: ParticleBot

particlebot.o:particlebot.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

particlebot_cuda.o:particlebot_cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

postprocess.o:postprocess.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

render.o:render.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

shaders.o:shaders.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ParticleBot: particlebot.o particlebot_cuda.o postprocess.o main.o render.o shaders.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	$(EXEC) cp $@ bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	$(EXEC) ./ParticleBot

clean:
	rm -f ParticleBot particlebot.o particlebot_cuda.o postprocess.o main.o render.o shaders.o
	rm -rf bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/ParticleBot

clobber: clean
