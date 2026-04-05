#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Error checking macro
#define cudaCheck(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define cublasCheck(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error at line %d\n", __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Initialize matrix with random values
void randomize_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// CPU reference implementation for verification
void cpu_sgemm(int M, int N, int K, float alpha,
               const float *A, const float *B, float beta, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// Verify results against cuBLAS using relative error
bool verify_results(const float *result, const float *reference, int N, float rel_tolerance = 1e-5f) {
    for (int i = 0; i < N; i++) {
        float diff = fabs(result[i] - reference[i]);
        float rel_err = diff / fmax(fabs(reference[i]), 1e-6f);
        if (rel_err > rel_tolerance) {
            printf("Mismatch at index %d: got %f, expected %f (rel_err=%.2e)\n",
                   i, result[i], reference[i], rel_err);
            return false;
        }
    }
    return true;
}

// Run cuBLAS for reference
void run_cublas(cublasHandle_t handle, int M, int N, int K,
                float alpha, float *A, float *B, float beta, float *C) {
    // cuBLAS uses column-major, so we compute B^T * A^T = C^T
    // which gives us C in row-major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// Benchmark a kernel
float benchmark_kernel(int kernel_num, int M, int N, int K,
                       float alpha, float *A, float *B, float beta, float *C,
                       int warmup_runs = 5, int benchmark_runs = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        run_kernel(kernel_num, M, N, K, alpha, A, B, beta, C);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_runs; i++) {
        run_kernel(kernel_num, M, N, K, alpha, A, B, beta, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / benchmark_runs;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <kernel_number> [matrix_size]\n", argv[0]);
        printf("Kernel numbers:\n");
        printf("  0: cuBLAS (reference)\n");
        printf("  1: Naive\n");
        printf("  2: Global memory coalescing\n");
        printf("  3: Shared memory caching\n");
        printf("  4: 1D blocktiling\n");
        printf("  5: 2D blocktiling\n");
        printf("  6: Vectorized memory access\n");
        printf("  9: Autotuned\n");
        printf("  10: Warptiling\n");
        return 1;
    }

    int kernel_num = atoi(argv[1]);
    int size = (argc > 2) ? atoi(argv[2]) : 4096;

    int M = size, N = size, K = size;
    float alpha = 1.0f, beta = 0.0f;

    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Running kernel %d\n\n", kernel_num);

    // Allocate host memory
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_C_ref = (float *)malloc(bytes_C);

    // Initialize matrices
    srand(42);
    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);
    memset(h_C, 0, bytes_C);
    memset(h_C_ref, 0, bytes_C);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaCheck(cudaMalloc(&d_A, bytes_A));
    cudaCheck(cudaMalloc(&d_B, bytes_B));
    cudaCheck(cudaMalloc(&d_C, bytes_C));
    cudaCheck(cudaMalloc(&d_C_ref, bytes_C));

    // Copy to device
    cudaCheck(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C_ref, h_C_ref, bytes_C, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Run cuBLAS for reference timing
    float cublas_time = 0.0f;
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        for (int i = 0; i < 5; i++) {
            run_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        }
        cudaDeviceSynchronize();

        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++) {
            run_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cublas_time, start, stop);
        cublas_time /= 20;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calculate cuBLAS GFLOPs
    double flops = 2.0 * M * N * K;
    double cublas_gflops = (flops / (cublas_time / 1000.0)) / 1e9;
    printf("cuBLAS: %.2f ms, %.1f GFLOPs\n", cublas_time, cublas_gflops);

    if (kernel_num == 0) {
        // Just running cuBLAS reference
        printf("\nDone.\n");
    } else {
        // Run the selected kernel
        printf("\nBenchmarking kernel %d...\n", kernel_num);

        float kernel_time = benchmark_kernel(kernel_num, M, N, K, alpha, d_A, d_B, beta, d_C);
        double kernel_gflops = (flops / (kernel_time / 1000.0)) / 1e9;

        printf("Kernel %d: %.2f ms, %.1f GFLOPs (%.1f%% of cuBLAS)\n",
               kernel_num, kernel_time, kernel_gflops, (kernel_gflops / cublas_gflops) * 100);

        // Verify correctness
        printf("\nVerifying results...\n");
        cudaCheck(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

        if (verify_results(h_C, h_C_ref, M * N)) {
            printf("PASSED: Results match cuBLAS!\n");
        } else {
            printf("FAILED: Results do not match cuBLAS!\n");
        }
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
