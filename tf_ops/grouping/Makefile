



CUDA_HOME := /usr/local/cuda-9.0

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

LIBFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -L $(TF_LIB) -ltensorflow_framework -lcublas

arch = sm_61

NVCC = $(CUDA_HOME)/bin/nvcc

LD_LIBRARY_PATH=$(CUDA_HOME)/lib64
INCLUDE_PATH=$(CUDA_HOME)/include

all: tf_grouping_so.so 

tf_grouping_so.so: tf_grouping.cpp tf_grouping_g.cu.o
	g++ -std=c++11 -shared -fPIC -o tf_grouping_so.so tf_grouping.cpp tf_grouping_g.cu.o -I$(TF_INC) -I$(INCLUDE_PATH) -L$(LIBFLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_grouping_g.cu.o: tf_grouping_g.cu
	$(NVCC) -std=c++11 -c -o tf_grouping_g.cu.o tf_grouping_g.cu -I$(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch $(arch) --use_fast_math
		
clean:
	rm *.o *.so
