import pyopencl as cl
import numpy as np


platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices()
device = devices[1]

ctx = cl.Context([device])
cpq = cl.command_queue_properties
queue = cl.CommandQueue(ctx,device,cpq.PROFILING_ENABLE)

print(device)



