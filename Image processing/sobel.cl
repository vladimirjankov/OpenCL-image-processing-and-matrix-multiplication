__kernel void sobelXFilter(__global const float *img, __global  float *result, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    const int posx = get_global_id(1);
    const int posy = get_global_id(0);
    const int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        float pixel00, pixel02, pixel10, pixel12, pixel20, pixel22;
        pixel00 = -1*img[i - w-1];
        pixel02 =  img[i - w+1];
        pixel10 = -2*img[i    -1];
        pixel12 =  2*img[i    +1];
        pixel20 = -1*img[i + w-1];
        pixel22 =  img[i + w+1];
        result[i] = pixel00+pixel02+pixel10+pixel12+pixel20+pixel22;
    }
}
__kernel void sobelYFilter(__global const  float *img, __global float *result, __global const int *width, __global const int *height){
    const int w = *width;
    const int h = *height;
    const int posx = get_global_id(1);
    const int posy = get_global_id(0);
    const int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        float pixel00, pixel01, pixel02, pixel20, pixel21, pixel22;
        pixel00 =    img[i - w-1];
        pixel01 =  2*img[i - w];
        pixel02 =    img[i   -w+1];
        pixel20 =  -1*img[i  + w-1];
        pixel21 =   -2*img[i + w];
        pixel22 =  1*img[i + w+1];
        result[i] = pixel00+pixel01+pixel02+pixel20+pixel21+pixel22;
    }
}

__kernel void sobel3x3(__global const  float *img, __global float *result,
 __global const int *width, __global const int *height){
    
    __private int w;
    w = *width;
    __private int h;
    h = *height;
    __private int posx;
    posx= get_global_id(1);
    __private int posy;
    posy = get_global_id(0);
    __private int i; 
    i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        __private float pixelRez1,pixelRez2;

        // mad ( a, ,b, c ) -- a * b + c
        //sobel x
        pixelRez1 = mad(2,img[i - w],img[i   -w+1]);
        pixelRez1 = mad(-1,img[i  + w-1],pixelRez1);
        pixelRez1 = mad(-2,img[i + w],pixelRez1);
        pixelRez1 = mad(-1,img[i + w+1],pixelRez1); 
        pixelRez1 = pixelRez1 + img[i - w-1];

        //sobel y 
        pixelRez2 = mad(-1,img[i - w-1],img[i - w+1]);
        pixelRez2 = mad(-2,img[i    -1],pixelRez2);
        pixelRez2 = mad(2,img[i    +1],pixelRez2);
        pixelRez2 = mad(-1,img[i + w-1],pixelRez2);
        pixelRez2 = pixelRez2 + img[i + w+1];
        
      
        result[i] =sqrt(pown(pixelRez1,2) +pown(pixelRez2,2));
        
    }

}


__kernel void sobelAdd(__global float *imgX,__global float *imgY, __global float *result,__global int *width, __global int *height ){
    int posx = get_global_id(1);   
    int posy = get_global_id(0);  
    int w = *width;
    int h = *height;
    int i = w*posy + posx;
    result[i] = sqrt(imgX[i]*imgX[i] + imgY[i]*imgY[i]);
}

__kernel void boxFilter(__global float *img, __global float *result, __global int *width, __global int *height){
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        float pixel00, pixel01, pixel02, pixel10, pixel11, pixel12,pixel20, pixel21, pixel22;
        pixel00 = img[i - w-2];
        pixel01 = img[i -  w -1];
        pixel02 = img[i - w+1];
        pixel10 = img[i  -1];
        pixel11 = img[i ];
        pixel12 = img[i    +1];
        pixel20 = img[i + w-1];
        pixel21 = img[i + w];
        pixel22 =  img[i + w+1];
        result[i] = (pixel00+ pixel01+ pixel02+ pixel10+ pixel11+ pixel12+pixel20+ pixel21+ pixel22)/9;
        
    }
}






