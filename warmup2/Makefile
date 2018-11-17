#
# Makefile
#


CUSRC=spmv_driver.cu
CPPSRC=mm_helper.cpp

EXEC=spmv_driver

OBJS=$(CUSRC:.cu=.o)
OBJS+=$(CPPSRC:.cpp=.o)

NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -O3

CC=nvcc
HCC=gcc


$(EXEC): $(OBJS)
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o : %.cu
	$(CC)  $(NVCCFLAGS) -c $< -o $@

%.o : %.cpp
	$(HCC) -O3 -c $< -o $@

clean:
	rm -f $(EXEC) $(OBJS)


# vim:ft=make
#
