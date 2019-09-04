# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
from scipy.misc import  imsave
from time import time

platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
mf = cl.mem_flags


ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device, properties=cpq.PROFILING_ENABLE)
matrix1 = np.random.rand(3840,3840).astype(np.float32)
matrix2 = np.random.rand(3840,3840).astype(np.float32)
matrix3 = np.random.rand(3840,3840).astype(np.float32)

prg = cl.Program(ctx,open('matrixmp.cl').read()).build()

mat1_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix1)
mat2_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix2)

wA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(matrix1.shape[1]))
wB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(matrix2.shape[1]))


result_g1 = cl.Buffer(ctx, mf.READ_WRITE, matrix3.nbytes)
time_start = time()
prg.matrixMul(queue,(matrix1.shape[0],matrix2.shape[1]), (16,16), mat1_g,mat2_g,result_g1,wA,wB )
queue.finish()
time_end = time()
elapsed = time_end - time_start
matrix3 = np.empty_like(matrix3)

cl.enqueue_copy(queue,matrix3,result_g1)

print(device)
print("Time: {0} s".format(elapsed))
