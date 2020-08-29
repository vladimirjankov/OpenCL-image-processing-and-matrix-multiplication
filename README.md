# OpenCL image processing and matrix multiplication

This is a basic demonstration of convolution and matrix multiplication using openCL.
The host program is written in python and all kernels are run using python.


## Image processing

Two main scripts :

convolution.py - runs 3 convolution kernels( Sobel X, Sobel Y and box filter kernel).
The program also calculates execution times.
All the calculations are done on grayscale images that can be found in images folder.

sobel_image.py - Runs two versions of kernel one optimised and one note. The difference in time is
measured. The non optimised version calculates first X then Y kernel and later adds them together while
the optimised version does all calculations in one kernel.
All the calculations are done on grayscale images that can be found in images folder.


## Matrix multiplication

Two main scripts : 

mat_block_mul.py - executes matrix multiplication in block form. 
It is executed by blocks size of 32 x 32. 
Supporting functions and constants can be fount in utilities.
	
matrix.py - creates two matrix 3840 x 3840 and multies them the straight forward way. 
Works in local work grup size of 16 x 16. Slower the the mat_block_mul execution. 
	
### Results

###### Image processing:
Original image
![alt text](https://github.com/vladimirjankov/OpenCL-image-processing-and-matrix-multiplication/blob/master/Image%20processing/images/grayscale.jpg?raw=true =250x250)
