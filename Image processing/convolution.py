import numpy as np
import pyopencl as cl
import cv2

from imageio import imread,imsave
from time import time

# loads images
img = imread(r'images/grayscale.jpg').astype(np.float32)

# convolution windows or kernels 
window1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).astype(np.float32)
window2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).astype(np.float32)
window3 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)

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
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)

# loads and builts the kernel
prg = cl.Program(ctx, open('convolution_kernel.cl').read())
prg = prg.build()

# init 1 one image buffer and 3 window buffers
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
window_g1 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window1)
window_g2 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window2)
window_g3 =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=window3)

# resulting convolution image buffers
result_g1 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
result_g2 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
result_g3 = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

# image width and height
width_i_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_i_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))

# kernekl width and height
width_w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(window1.shape[1]))
height_w_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(window1.shape[0]))

# set 3 exectutions and measure run time
start_time = time()
event1 = prg.conv(queue,img.shape,local_work_group,img_g,result_g1,window_g1,width_i_g,height_i_g,width_w_g,height_w_g)
event2 = prg.conv(queue,img.shape,local_work_group,img_g,result_g2,window_g2,width_i_g,height_i_g,width_w_g,height_w_g)
event3 = prg.conv(queue,img.shape,local_work_group,img_g,result_g3,window_g3,width_i_g,height_i_g,width_w_g,height_w_g)
run_time = time() - start_time

# wait for queue to finish
queue.finish()

# copies back first convolution
resultx = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g1)

# copies back second convolution
resulty = np.empty_like(img)
cl.enqueue_copy(queue,resultx,result_g2)

# copies back third convolution
result_box = np.empty_like(img)
cl.enqueue_copy(queue,result_box,result_g3)

# square image of first two kernels (sobel)
resultx = resultx*resultx + resulty*resulty

print("CPU Time: {0} s".format(run_time))

# saves images
imsave(r'images/new_grayscale_sobel_conv.png',resultx.astype(np.float32))
imsave(r'images/new_grayscale_box.png',result_box)





