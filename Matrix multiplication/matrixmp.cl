__kernel void matmul(__global float *mat1, __global float *mat2,__global float *mat3, __global int *wA, __global int *wB){
    int wa = *wA;
    int wb = *wB;
    int posx = get_global_id(0);
    int posy = get_global_id(1);
    int i = wb*posy + posx;

    
    float value = 0;

    for(int k =0 ; k < wa; ++k){
        int idxA = posy*wa +k;
        int idxB = k*wb + posx;
        value += mat1[idxA]*mat2[idxB];
    }
    mat3[i] = value;
    
    
}

__kernel void matrixMul(__global float *mat1, __global float *mat2,__global float *mat3,__global int *wA, __global int *wB){
    
    int posx = get_global_id(0);
    int posy = get_global_id(1);
    float value = 0; 
    
    for(int i =0 ; i < *wA;++i){
        value+= mat1[posy*(*wA)+i]*mat2[i*(*wB)+posx];    
    }
    mat3[posy*(*wB)+posx] = value;
    
}