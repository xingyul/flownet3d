



CUDA_HOME := /usr/local/cuda-9.0

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

LIBFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -L $(TF_LIB) -ltensorflow_framework -lcublas

arch = sm_61

LD_LIBRARY_PATH=$(CUDA_HOME)/lib64
INCLUDE_PATH=$(CUDA_HOME)/include

all: tf_interpolate_so.so

tf_interpolate_so.so: tf_interpolate.cpp
	g++ -std=c++11 -shared -fPIC -o tf_interpolate_so.so tf_interpolate.cpp -I$(TF_INC) -I$(INCLUDE_PATH) -L$(LIBFLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm *.so
