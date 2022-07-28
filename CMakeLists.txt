set(FFTW3_SO 
libfftw3.so
libfftw3.so.3
libfftw3.so.3.6.10
libfftw3_threads.so
libfftw3_threads.so.3
libfftw3_threads.so.3.6.10
)


set(CUDA_LINKS_DIR
/usr/local/cuda-11.3/lib64
)

cmake_minimum_required(VERSION 3.13)

project(pattern_process CXX CUDA)  

find_package(OpenCV REQUIRED)

#CUDA
#enable_language(CUDA)
find_package(CUDA 11.3 REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    include_directories(/usr/include 
                        ${OpenCV_INCLUDE_DIRS} 
                        /usr/local/fftw3/include 
                        /home/kawamoto/cprjct/pattern_process/include 
                        ${CUDA_INCLUDE_DIRS}
                        )

    link_directories(/home/kawamoto/cprjct/pattern_process/sos 
                        /usr/local/lib 
                        /usr/local/fftw3/lib 
                        ${CUDA_LINKS_DIR}
                        )


    CUDA_ADD_EXECUTABLE(pattern_process main.cu dvcfnc.cu)

    target_link_libraries(pattern_process 
        libbmp_class.so 
        libcomplex_array_class.so 
        ${OpenCV_LIBRARIES} 
        ${FFTW3_SO}
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        )

else()
    message("CUDA not found !!!!!")
endif()


