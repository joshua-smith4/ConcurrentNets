PROJECT = Scheduler

BINDIR = ./bin/
SRCDIR = ./Source/
OBJDIR = ./obj/
INCDIR = ./Header/

HOST_TYPE := $(shell ./scripts/calc_host.sh)

NC := /usr/local/cuda/bin/nvcc
CXX := g++
LDFINAL := g++

ifeq ($(HOST_TYPE), x86_64)
CXXFLAG := -m64 -O3
NCFLAG := -m64 -arch=sm_32 -O3
LDFLAG := -m64

#LIBCUTIL := -lcutil_x86_64
CULIBPATH := -L/usr/local/cuda/lib64
else
ifeq ($(HOST_TYPE), x86)
CXXFLAG := -m32 -w -O3
NCFLAG := -m32 -arch=sm_32 -O3
LDFLAG := -m32

#LIBCUTIL := -lcutil_i386
CULIBPATH := -L/usr/local/cuda/lib
endif
endif

NVCCVER := `$(NC) -V | grep release`

LIBS = -lrt
INCS = -I$(INCDIR)

#LIBCUDA := -lcuda -lcudart $(LIBCUTIL)
LIBCUDA := -lcuda -lcudart
CULIBPATH += -L/usr/local/cuda-8.0/NVIDIA_CUDA-8.0_Samples/lib/linux/x86_64/ 

LIBS += $(CULIBPATH) $(LIBCUDA)
NVINCS := $(INCS) -I/usr/local/cuda/include -I/usr/local/cuda-8.0/NVIDIA_CUDA-8.0_Samples/common/inc -I/usr/local/cuda-8.0/include -I/usr/local/cuda-8.0/targets/x86_64-linux/include

#object files to create from cu files
CUOBJ = $(OBJDIR)scheduler.cu.o
CXXOBJ = $(patsubst $(SRCDIR)%.cxx, $(OBJDIR)%.cxx.o, $(wildcard $(SRCDIR)*.cxx))

all: $(BINDIR)$(PROJECT)

$(BINDIR)$(PROJECT): $(CUOBJ) $(CXXOBJ)
	@echo "Linking the target $@"
	mkdir -p $(BINDIR)
	$(LDFINAL) $(LDFLAG) $^ -o $@ $(LIBS)

$(OBJDIR)%.cu.o: $(SRCDIR)scheduler.cu $(SRCDIR)GPUKernels.cu
	@echo "Compiling CUDA source: $<"
	mkdir -p $(OBJDIR)
	$(NC) -c $(NCFLAG) $(NVINCS) $< -o $@ 

$(OBJDIR)%.cxx.o: $(SRCDIR)%.cxx
	@echo "Compiling C++ source: $<"
	mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAG) -D_PROJECT_="\"$(PROJECT)\"" -D__NVCCVER__="\"$(NVCCVER)\"" $(DEBUG_FLAGS) $(NVINCS) $< -o $@

clean:
	@echo "Cleaning project..."
	rm -rf $(OBJDIR)
	rm -f $(BINDIR)$(PROJECT)
