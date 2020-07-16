# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
from scipy.misc import  imsave
from time import time

# size of matrix 
MATRIX_SIZE = 3840

# gets platform list and takes first
platforms = cl.get_platforms()
platform = platforms[0]

# gets device list and takes first in my case intel xeon
devices = platform.get_devices()
device = devices[0]

# sets the local work group x,y 
local_work_group = (16,16)
mf = cl.mem_flags

# creates context and command queue
ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device, properties=cpq.PROFILING_ENABLE)

# creates 3 random matrix size MATRIX_SIZE
matrix1 = np.random.rand(MATRIX_SIZE,MATRIX_SIZE).astype(np.float32)
matrix2 = np.random.rand(MATRIX_SIZE,MATRIX_SIZE).astype(np.float32)
matrix3 = np.random.rand(MATRIX_SIZE,MATRIX_SIZE).astype(np.float32)

# loads and builts the kernel
prg = cl.Program(ctx,open(r'kernels/matrixmp.cl').read()).build()

# setes memory buffers
mat1_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix1)
mat2_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix2)

# sets with of matrix 
wA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(matrix1.shape[1]))
wB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(matrix2.shape[1]))

# resulting buffer
result_g1 = cl.Buffer(ctx, mf.READ_WRITE, matrix3.nbytes)

#measures time and executes the kernel
time_start = time()
prg.matrixMul(queue,(matrix1.shape[0],matrix2.shape[1]), local_work_group, mat1_g,mat2_g,result_g1,wA,wB )
queue.finish()
time_end = time()
elapsed = time_end - time_start

# copies the resutl 
matrix3 = np.empty_like(matrix3)
cl.enqueue_copy(queue,matrix3,result_g1)

#prints execution time
print("Time: {0} s".format(elapsed))
