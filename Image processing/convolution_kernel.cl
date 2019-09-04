

__kernel void conv(__global const float *img, __global  float *result,__global float *window,
    __global const int *width_img, __global const int *height_img,
    __global const int *width_window, __global const int *height_window){
    
    __local int w_i;
    w_i = *width_img;
    __local int h_i;
    h_i = *height_img;
    
    __local int w_window;
    w_window = *width_window;
    __local int h_window;
    h_window = *height_window;
    
    __local int posx;
    posx= get_global_id(1);
    __local int posy;
    posy = get_global_id(0);
    __local int i; 
    i = w_i*posy + posx;
    
    __local int pading_x, pading_y;
    pading_x=  w_window/2; 
    pading_y=  h_window/2; 
    __local int window_i;
    window_i = pading_y*w_window + pading_x;
    
    if(posx == pading_x || posy ==pading_y || posx==w_i-pading_x || posy == h_i-pading_y){
        result[i] = img[i] ;
        }
    else{
        __local int k,j,index;
        __local float sum;
        index = 0;
        
        sum = 0; 
        for(k = -1*pading_y*w_window-pading_x ; k <pading_y*w_window+pading_x ;++k ){
            sum+= img[i+k ] * window[window_i + k];
        }
        result[i] = sum;
    
    
        
        }        
              
              
}
