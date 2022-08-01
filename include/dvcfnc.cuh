#pragma once
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>

typedef unsigned char uc;

//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 

//関数群

__global__ void cusetcucomplex(cuComplex* com, double* Re, double* Im, int size);
__global__ void uc2cucomplex(cuComplex* com, unsigned char* Re, int num, int size);

// fft normalization
__global__ void normfft(cufftComplex* dev, int x, int y);
__global__ void pow_norm_fft1d(uc* pow, cufftComplex* dev, int num, int size);

// signal process
__global__ void BPF(cuComplex* dev, float strt, float end, int num, int size);
__global__ void HPF(cuComplex* dev, float cutrate, int num, int size);
__global__ void LPF(cuComplex* dev, float cutrate, int num, int size);


//FFT
//1d
void fft_1D_C2C(int x, cufftComplex*dev, int batch);
void ifft_1D_C2C(int x, cufftComplex*dev, int batch);
//void fft_1D_R2C(int x, cufftComplex*dev, int batch);
//2d
void fft_2D_cuda_dev(int x, int y, cufftComplex* dev);
void ifft_2D_cuda_dev(int x, int y, cufftComplex* dev);


__global__ void Hcudaf(float* Re, float* Im, int x, int y, float u, float v, float z, float lam);
__global__ void HcudacuCom(cuComplex* H, int x, int y, float z, float d, float lam);

__global__ void  shiftf(float* ore, float* oim, float* re, float* im, int x, int y);
__global__ void shiftCom(cuComplex* out, cuComplex* in, int x, int y);

//floatXcufftCom
__global__ void mulcomcufftcom(cufftComplex* out, float* re, float* im, cufftComplex* in, int s);
//doubleXcufftCom
__global__ void muldoublecomcufftcom(cufftComplex* out, double* re, double* im, cufftComplex* in, int s);
__global__ void Cmulfft(cufftComplex* out, cufftComplex* fin, cuComplex* in, int s);


__global__ void pad_cufftcom2cufftcom(cufftComplex* out, int lx, int ly, cufftComplex* in, int sx, int sy);

__global__ void elimpad(cufftComplex* out, int sx, int sy, cufftComplex* in, int lx, int ly);
__global__ void elimpad2Cmulfft(cuComplex* outmlt, cuComplex* opponent,
    int sx, int sy, cuComplex* in, int lx, int ly);

void Hcudaf_shiftf(float* devReH, float* devImH, int x, int y, float d, float z, float lamda, dim3 grid, dim3 block);
void Hcudashiftcom(cuComplex* dev, int x, int y, float z, float d, float lamda, dim3 grid, dim3 block);


__global__ void cucompower(double* power, cuComplex* dev, int s);

__global__ void elimpadcucompower(double* power, int sx, int sy, cuComplex* dev, int lx, int ly);

__global__ void cunormaliphase(cuComplex* out, double* normali, int s);


