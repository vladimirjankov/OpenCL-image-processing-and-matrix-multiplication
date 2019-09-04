# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import cv2
from imageio import imread,imsave
from time import time


img = imread('grayscale.jpg').astype(np.float32)
window1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).astype(np.float32)
window2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).astype(np.float32)
window3 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)


platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[0]
print(device)
local_work_group = (16,16)
mf = cl.mem_flags



ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)

prg = cl.Program(ctx, open('convolution_kernel.cl').read())

prg = prg.build()



img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
window_g1 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window1)
window_g2 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window2)
window_g3 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window3)
  

result_g1 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
result_g2 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
result_g3 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)


width_i_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_i_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))

width_w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(window1.shape[1]))
height_w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(window1.shape[0]))

event1 = prg.conv(queue,img.shape,local_work_group,img_g,result_g1,window_g1,width_i_g,height_i_g,width_w_g,height_w_g)
event2 = prg.conv(queue,img.shape,local_work_group,img_g,result_g2,window_g2,width_i_g,height_i_g,width_w_g,height_w_g)
start_time = time()
event3 = prg.conv(queue,img.shape,local_work_group,img_g,result_g3,window_g3,width_i_g,height_i_g,width_w_g,height_w_g)
run_time = time() - start_time

queue.finish()

resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g1)

resulty = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g2)

result_box = np.empty_like(img)
cl.enqueue_copy(queue,result_box,result_g3)


resultx = resultx*resultx + resulty*resulty


print(run_time)

imsave('new_grayscale_sobel_conv.png',resultx.astype(np.float32))
imsave('new_grayscale_box.png',result_box)

'''
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sob = sobelx*sobelx + sobely*sobely
imsave('new_grayscale_sobel.png',sob)
'''




