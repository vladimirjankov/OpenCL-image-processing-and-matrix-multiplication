import pyopencl as cl
import numpy

from utilities.constants import ORDER, AVAL, BVAL, TOL, DIM, COUNT
from utilities.supporting_functions import seq_mat_mul_sdot, error, results
from time import time

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


# Sets memory buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

# loads and builts the kernel
kernelsource = open(r"kernels/mat_mul_block.cl").read()
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
