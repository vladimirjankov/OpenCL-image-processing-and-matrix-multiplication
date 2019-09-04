# -*- coding: utf-8 -*-

# Order of the square matrices A, B and C

ORDER = 3840

# A elemetns are constant and equal to AVAL
AVAL = 3.0

# B elemetns are constant and equal to BVAL
BVAL = 5.0

# tolerance used in floating point comparisons
TOL = 0.001

# Max dim for NDRange
DIM = 2

# number of times to do each multiplication
COUNT = 1

#  Function to compute the matrix product (sequential algorithm, dot prod)

import pyopencl as cl
import numpy
from time import time


def seq_mat_mul_sdot(N, A, B, C):
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i*N+k] * B[k*N+j]
            C[i*N+j] = tmp

#  Function to compute errors of the product matrix
def error(N, C):
   cval = float(N) * AVAL * BVAL
   errsq = 0.0
   for i in range(N):
       for j in range(N):
            err = C[i*N+j] - cval
            errsq += err * err
   return errsq;


# Function to analyze and output results
def results(N, C, run_time):
    mflops = 2.0 * N * N * N/(1000000.0* run_time)
    print (run_time, "seconds at", mflops, "MFLOPS")
    errsq = error(N, C)
    if (errsq > TOL):
        print ("Errors in multiplication:", errsq)



# A[N][N], B[N][N], C[N][N]
N = ORDER;


# Number of elements in the matrix
size = N * N


# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)

print ("\n===== Sequential, matrix mult (dot prod), order", ORDER, "on host CPU ======\n")

for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    print ("Skipping as this takes a long time to run!")
    #seq_mat_mul_sdot(N, h_A, h_B, h_C)

    run_time = time() - start_time
    #results(N, h_C, run_time)


# Set up OpenCL



context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)



kernelsource = open("mat_mul_block.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None, None])
print ("\n==== Parallel matrix mult (blocked), order {0} on device ======\n".format(N))
# Do the multiplication COUNT times
for i in range(COUNT):
    h_C.fill(0.0)
    start_time = time()

    # Work-group computes a block of C. This size is also set
    # in a #define inside the kernel function. Note this blocksize
    # must evenly divide the matrix order
    blocksize = 32

    A_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blocksize * blocksize)
    B_block = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * blocksize * blocksize)
    mmul(queue, (N,N), (blocksize,blocksize), N,
        d_a, d_b, d_c, A_block, B_block)
    queue.finish()

    run_time = time() - start_time

    cl.enqueue_copy(queue, h_C, d_c)
    results(N, h_C, run_time)
