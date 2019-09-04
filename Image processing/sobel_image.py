# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np
from imageio import imread,imsave
import cv2
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


img = imread('grayscale.jpg').astype(np.float32)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
print(device)
local_work_group = (4,4)
mf = cl.mem_flags

ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)

prg = cl.Program(ctx, open('sobel.cl').read())

prg = prg.build()
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)

result_g1 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g2 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g3 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))

#not optimised kernel
event2 = prg.sobelXFilter(queue,img.shape,local_work_group,img_g,result_g1,width_g,height_g)
event1 = prg.sobelYFilter(queue,img.shape,local_work_group,img_g,result_g2,width_g,height_g)
queue.finish()
event3 = prg.sobelAdd(queue,img.shape,local_work_group,result_g1,result_g2,result_g3,width_g,height_g)
queue.finish()
resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g1)
resulty = np.empty_like(img)
cl.enqueue_copy(queue,resulty,result_g2)


resultz = np.empty_like(img)
cl.enqueue_copy(queue,resultz,result_g3)

compute_time_e1 = (event1.profile.end-event1.profile.start)*1e-9 
compute_time_e2 = (event2.profile.end-event2.profile.start)*1e-9 
compute_time_e3 = (event3.profile.end-event3.profile.start)*1e-9 
print("GPU Time: {0} s".format(compute_time_e1+compute_time_e2+compute_time_e3))

# test of optimised kernel
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

event1 = prg.sobel3x3(queue,img.shape,local_work_group,img_g,result_g,width_g,height_g)
queue.finish()

compute_time_e1 = (event1.profile.end-event1.profile.start)*1e-9 
print("GPU Time: {0} s".format(compute_time_e1))

resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g)

rez_fun = resultx + img

imsave('new_grayscale_sobel_kernel.png',resultx.astype(np.uint8 ))

imsave('sharp_grayscale_kernel.png',rez_fun.astype(np.uint8 ))

'''
print("NVIDIA 1050 Time: {0} s".format(compute_time_e1+compute_time_e2+compute_time_e3))

platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
local_work_group = (4,4)
mf = cl.mem_flags

ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)



prg = cl.Program(ctx, open('sobel.cl').read()).build()
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g1 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g2 = cl.Buffer(ctx, mf.READ_WRITE, img.nbytes)
result_g3 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))
time_start = time()
event2 = prg.sobelXFilter(queue,img.shape,local_work_group,img_g,result_g1,width_g,height_g)
event1 = prg.sobelYFilter(queue,img.shape,local_work_group,img_g,result_g2,width_g,height_g)
queue.finish()
event3 = prg.sobelAdd(queue,img.shape,local_work_group,result_g1,result_g2,result_g3,width_g,height_g)
queue.finish()
time_end = time()
elapsed = time_end - time_start
resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g1)


resulty = np.empty_like(img)
cl.enqueue_copy(queue,resulty,result_g2)


resultz = np.empty_like(img)
cl.enqueue_copy(queue,resultz,result_g3)


compute_time_e1 = (event1.profile.end-event1.profile.start)*1e-9 
compute_time_e2 = (event2.profile.end-event2.profile.start)*1e-9 
compute_time_e3 = (event3.profile.end-event3.profile.start)*1e-9 


print("CPU Time: {0} s".format(compute_time_e1+compute_time_e2+compute_time_e3))
'''
