# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np
from imageio import imread,imsave
import cv2
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# loads images
img = imread(r'images/grayscale.jpg').astype(np.float32)

# gets platform list and takes first
platforms = cl.get_platforms()
platform = platforms[0]

# gets device list and takes first in my case intel xeon
devices = platform.get_devices()
device = devices[0]

# sets the local work group x,y 
local_work_group = (4,4)
mf = cl.mem_flags

# creates context and command queue
ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)

# loads and builts the kernel
prg = cl.Program(ctx, open(r'kernels/sobel.cl').read())
prg = prg.build()
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)

# 3 buffers for image
result_g1 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g2 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g3 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

# 2 buffers for width and height of image
width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))

# executes sobel filter by x and y and adds both images 
event2 = prg.sobelXFilter(queue,img.shape,local_work_group,img_g,result_g1,width_g,height_g)
event1 = prg.sobelYFilter(queue,img.shape,local_work_group,img_g,result_g2,width_g,height_g)

# wait for queue to finish
queue.finish()
event3 = prg.sobelAdd(queue,img.shape,local_work_group,result_g1,result_g2,result_g3,width_g,height_g)

# wait for queue to finish
queue.finish()

# copyies x sobel to resultx
resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g1)

# copyies y sobel to resulty
resulty = np.empty_like(img)
cl.enqueue_copy(queue,resulty,result_g2)

# copyies image from buffer to result
result = np.empty_like(img)
cl.enqueue_copy(queue,result,result_g3)

# computation times
compute_time_e1 = (event1.profile.end-event1.profile.start)*1e-9 
compute_time_e2 = (event2.profile.end-event2.profile.start)*1e-9 
compute_time_e3 = (event3.profile.end-event3.profile.start)*1e-9 

print("CPU Time: {0} s".format(compute_time_e1+compute_time_e2+compute_time_e3))

# test of optimised kernel
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

# sobel 3x3 optimised kernel
event1 = prg.sobel3x3(queue,img.shape,local_work_group,img_g,result_g,width_g,height_g)
queue.finish()

# computation time
compute_time_e1 = (event1.profile.end-event1.profile.start)*1e-9 
print("CPU Time: {0} s".format(compute_time_e1))

# copies resulting image
resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g)
rez_fun = resultx + img

# saves both images
imsave(r'images/new_grayscale_sobel_kernel.png',resultx.astype(np.uint8 ))
imsave(r'images/sharp_grayscale_kernel.png',rez_fun.astype(np.uint8 ))
