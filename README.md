# OpenCL image processing and matrix multiplication

This is a basic demonstration of convolution and matrix multiplication using openCL.
The host program is written in python and all kernels are run using python.


## Image processing

Two main scripts :

	Convolution.py - runs 3 convolution kernels( Sobel X, Sobel Y and box filter kernel).
	The program also calculates execution times.
	All the calculations are done on grayscale images that can be found in images folder.
	
	sobel_image.py - Runs two versions of kernel one optimised and one note. The difference in time is
	measured. The non optimised version calculates first X then Y kernel and later adds them together while
	the optimised version does all calculations in one kernel.
	All the calculations are done on grayscale images that can be found in images folder.



# pisi kasnije
