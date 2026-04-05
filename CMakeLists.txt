cmake_minimum_required(VERSION 3.18)
project(my_sgemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# RTX 4090 = compute capability 8.9
set(CUDA_COMPUTE_CAPABILITY 89)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

# Main executable
add_executable(sgemm sgemm.cu)
target_link_libraries(sgemm CUDA::cublas)

# Include directories
target_include_directories(sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
