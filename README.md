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

### Sobel algorithm  

The Sobel operator, sometimes called the Sobel - Feldman operator or Sobel filter, is used in image processing and computer vision, particularly within edge detection algorithms where it creates an image emphasising edges. Technically, it is a discrete differentiation operator, computing an approximation of the gradient of the image intensity function.  At each point in the image, the result of the Sobel - Feldman operator is either the corresponding gradient vector or the norm of this vector.  The Sobel - Feldman operator is based on convolving the image with a small, separable, and integer - valued filter in the horizontal and vertical directions and is therefore, relatively inexpensive in terms of computations. On the other hand, the gradient approximation that it produces is relatively crude, in particular, for high-frequency variations in the image.    
  
The operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. If A is defined as the source image, and G_xand G_y as two images which at each point contain the vertical and horizontal derivative approximations respectively, the computations are as follows:   
<img src="https://i.ibb.co/rG7Js0V/sobel-kernel.png" alt="Results"/>  
where * represents the 2 - dimensional signal processing convolution operation. 
The resulting image is calculated by squaring Gx  and Gy and getting the square root of their addition.


## Matrix multiplication

Two main scripts : 

mat_block_mul.py - executes matrix multiplication in block form. 
It is executed by blocks size of 32 x 32. 
Supporting functions and constants can be fount in utilities.
	
matrix.py - creates two matrix 3840 x 3840 and multies them the straight forward way. 
Works in local work grup size of 16 x 16. Slower the the mat_block_mul execution. 
	
## Results

### Image processing:  
####  Original image  
<img src="https://github.com/vladimirjankov/OpenCL-image-processing-and-matrix-multiplication/blob/master/Image%20processing/images/grayscale.jpg?raw=true" alt="Grayscale image" width="350"/>  

####  Sobel image  
<img src="https://github.com/vladimirjankov/OpenCL-image-processing-and-matrix-multiplication/blob/master/Image%20processing/images/new_grayscale_sobel_kernel.png?raw=true" alt="Sobel grayscale image" width="350"/>  

####  Benchmarks on cpu on intel based cpu  
<img src="https://i.ibb.co/3FKTyRb/sobel-table.png" alt="Results"/>

### Matrix multiplication:  

#### Benchmarks
<img src="https://i.ibb.co/42tPwRc/matrix-multiplication.png" alt="Results"/>
