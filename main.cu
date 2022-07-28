#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>
#include <sys/stat.h>
#include "my_all.h"
#include "Bmp_class.h"
#include "complex_array_class.h"
#include "dvcfnc.cuh"

#include <opencv2//opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

//copy
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//乱数ライブラリインクルード
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
using namespace cv;

#define sqr(x) ((x)*(x))

//CUDA
#ifndef __CUDACC__
#define __CUDACC__
#endif 

//1次元でのブロックサイズ
#define BS 1024

typedef unsigned char uc;
typedef long long ll;


// コマンド引数
// input(output) directory, input file name, output file name, data size, amount of data, L(LPF) or H(HPF), cutrate 
int main(int argc, char* argv[]){
    if(argc!=8){
        cout<<"this program need seven variables"<<endl;
        cout<<"but you input "<<argc-1<<"."<<endl;
        return 0;
    }
    //出力ファイルディレクトリ作成
    string prcsdir=argv[1]+(string)"/process";
    struct stat statbuf;
    if(stat(prcsdir.c_str(), &statbuf)!=0){
        mkdir(prcsdir.c_str(), 0777);
        cout<<"create new directory."<<endl;
    }

    //コマンドライン引数
    string inputdata=argv[1]+(string)"/"+argv[2], processdata=prcsdir+"/"+argv[3];
    ll N=(ll)atoi(argv[5]), SIZE=(ll)atoi(argv[4]);

    
    //ファイル入力・バイナリストリームオープン
    ifstream ifs(inputdata, ios::binary);
    //ファイル出力・バイナリストリームオープン
    ofstream ofs(processdata, ios::binary);

    if (ifs&&ofs){
        clock_t start, lap;
        start = clock();
        //ucdatにデータ読み込み
        uc*ucdat;
        ucdat=new uc[N*SIZE];
        ifs.read((char*)ucdat, sizeof(uc)*N*SIZE);
        cout<<"success reading file. "<<endl;

 
        //デバイス側に同じサイズのメモリ確保
        uc*devuc;
        cudaMalloc((void**)&devuc, sizeof(uc)*N*SIZE);
        //デバイスへcopy
        cudaMemcpy(devuc, ucdat, sizeof(uc)*N*SIZE, cudaMemcpyHostToDevice);
        
        //デバイスにcufftcomplexメモリ確保
        cufftComplex*devcfc;
        cudaMalloc((void**)&devcfc, sizeof(cufftComplex)*N*SIZE);

        cout<<"success allocate memory."<<endl;
        
        //Nスレッドに分割、スレッド内でSIZEだけloop
        uc2cucomplex<<<(N + BS - 1) / BS, BS >>>(devcfc, devuc, N, SIZE);

        //FFT
        fft_1D_C2C(SIZE, devcfc, N);
        cout<<"FFT"<<endl;

        //signal process
        //周波数シフトしていないことに注意
        if((string)argv[6]==(string)"H"){
            HPF<<<(N + BS - 1) / BS, BS >>>(devcfc, atof(argv[7]), N, SIZE);
            cout<<"signal processing."<<endl;
        }
        else if((string)argv[6]==(string)"L"){
            LPF<<<(N + BS - 1) / BS, BS >>>(devcfc, atof(argv[7]), N, SIZE);
            cout<<"signal processing."<<endl;

        }

        //IFFT
        ifft_1D_C2C(SIZE, devcfc, N);
        cout<<"IFFT"<<endl;

        //power
        uc*devpow;
        cudaMalloc((void**)&devpow, sizeof(uc)*N*SIZE);
        pow_norm_fft1d<<<(N + BS - 1) / BS, BS >>>(devpow, devcfc, N, SIZE);
        cudaMemcpy(ucdat, devpow, sizeof(uc)*N*SIZE, cudaMemcpyDeviceToHost);
        
        // write
        ofs.write((char*)ucdat, sizeof(uc)*N*SIZE);
        cout<<"write."<<endl;

        // memory free
        cudaFree(devuc);cudaFree(devcfc);cudaFree(devpow);
        delete[]ucdat;
        cout<<"memory free."<<endl;

        lap=clock();
        cout<<setprecision(3)<<(lap-start)/CLOCKS_PER_SEC/60<<"min"<<endl;

    }
    else{
        if(!ifs){
            cout<<"読み取りたいファイルが開けませんでした。終了します。"<<endl;
        }
        if(!ofs){
            cout<<"書き込みたいファイルが開けませんでした。終了します。"<<endl;
        }
    }

    return 0;
}
