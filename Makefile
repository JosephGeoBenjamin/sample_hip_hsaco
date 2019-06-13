CLANG_OCL=/opt/rocm/bin/clang-ocl
HIPCC=/opt/rocm/bin/hipcc

OCL_FLAGS= -mcpu=gfx900 -Wno-everything



.PHONY: all cl run clean

all: cl
	$(HIPCC) -g3 hsaco_brightness.cpp -o output

cl:
	$(CLANG_OCL) $(OCL_FLAGS) -c -g3 brightness_contrast.cl -o brightness_contrast.cl.o
	# $(HIPCC) $(OCL_FLAGS) -c -g3 brightness_contrast.cl -o bright_cl.o


run: all
	./output

clean:
	rm -f bright_cl.o
	rm -f output
