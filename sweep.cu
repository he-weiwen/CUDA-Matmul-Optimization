#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "src/9_autotuned.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define cudaCheck(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    }

void randomize_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void run_cublas(cublasHandle_t handle, int M, int N, int K,
                float alpha, float *A, float *B, float beta, float *C) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

template <int BM, int BN, int BK, int TM, int TN>
float benchmark_config(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C, float *C_ref,
                       int warmup = 10, int runs = 30) {
    constexpr int numThreads = 256;

    // Validate constraints
    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    if constexpr (BM % WM != 0 || BN % WN != 0) return -1;
    if constexpr (BK % 4 != 0) return -1;
    if constexpr (BN % 4 != 0) return -1;
    if constexpr ((numThreads * 4) % BK != 0) return -1;
    if constexpr (numThreads % (BN / 4) != 0) return -1;
    if constexpr (TM % 4 != 0 || TN % 4 != 0) return -1;

    dim3 block(numThreads);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        sgemm_autotuned<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        sgemm_autotuned<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= runs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Quick correctness check (sample a few elements)
    float *h_C = (float *)malloc(16 * sizeof(float));
    float *h_ref = (float *)malloc(16 * sizeof(float));
    cudaMemcpy(h_C, C, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, C_ref, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) {
        float rel_err = fabs(h_C[i] - h_ref[i]) / fmax(fabs(h_ref[i]), 1e-6f);
        if (rel_err > 1e-4f) {
            free(h_C); free(h_ref);
            return -2; // Incorrect
        }
    }
    free(h_C); free(h_ref);

    return ms;
}

#define TRY_CONFIG(bm, bn, bk, tm, tn) { \
    float ms = benchmark_config<bm, bn, bk, tm, tn>( \
        M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref); \
    if (ms > 0) { \
        double gflops = (2.0 * M * N * K / (ms / 1000.0)) / 1e9; \
        printf("BM=%3d BN=%3d BK=%2d TM=%d TN=%d | %6.2f ms | %8.1f GFLOPs | %5.1f%%\n", \
               bm, bn, bk, tm, tn, ms, gflops, (gflops / cublas_gflops) * 100); \
        if (gflops > best_gflops) { \
            best_gflops = gflops; best_ms = ms; \
            best_bm=bm; best_bn=bn; best_bk=bk; best_tm=tm; best_tn=tn; \
        } \
    } else if (ms == -2) { \
        printf("BM=%3d BN=%3d BK=%2d TM=%d TN=%d | INCORRECT\n", bm, bn, bk, tm, tn); \
    } \
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;

    printf("Autotuning SGEMM on %dx%dx%d\n\n", M, N, K);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    srand(42);
    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaCheck(cudaMalloc(&d_A, bytes_A));
    cudaCheck(cudaMalloc(&d_B, bytes_B));
    cudaCheck(cudaMalloc(&d_C, bytes_C));
    cudaCheck(cudaMalloc(&d_C_ref, bytes_C));
    cudaCheck(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_C, 0, bytes_C));
    cudaCheck(cudaMemset(d_C_ref, 0, bytes_C));

    // cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Thermal warm-up: run cuBLAS 50 times to stabilize GPU clocks
    printf("Warming up GPU...\n");
    for (int i = 0; i < 50; i++)
        run_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        run_cublas(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    cublas_ms /= 20;
    double cublas_gflops = (2.0 * M * N * K / (cublas_ms / 1000.0)) / 1e9;
    printf("cuBLAS: %.2f ms, %.1f GFLOPs\n\n", cublas_ms, cublas_gflops);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double best_gflops = 0, best_ms = 0;
    int best_bm=0, best_bn=0, best_bk=0, best_tm=0, best_tn=0;

    printf("%-45s | %8s | %10s | %s\n", "Config", "Time", "GFLOPs", "% cuBLAS");
    printf("----------------------------------------------+----------+------------+-------\n");

    // Sweep: BM/BN from {64,128,256}, BK from {8,16,32}, TM/TN from {4,8}
    // Constraint: BM % (TM*16) == 0 && BN % (TN*16) == 0

    // TM=8, TN=8 → WM=128, WN=128
    TRY_CONFIG(128, 128,  8, 8, 8);
    TRY_CONFIG(128, 128, 16, 8, 8);
    TRY_CONFIG(128, 128, 32, 8, 8);
    TRY_CONFIG(256, 128,  8, 8, 8);
    TRY_CONFIG(256, 128, 16, 8, 8);
    TRY_CONFIG(128, 256,  8, 8, 8);
    TRY_CONFIG(128, 256, 16, 8, 8);
    TRY_CONFIG(256, 256,  8, 8, 8);
    TRY_CONFIG(256, 256, 16, 8, 8);

    // TM=4, TN=4 → WM=64, WN=64
    TRY_CONFIG( 64,  64,  8, 4, 4);
    TRY_CONFIG( 64,  64, 16, 4, 4);
    TRY_CONFIG( 64,  64, 32, 4, 4);
    TRY_CONFIG(128, 128,  8, 4, 4);
    TRY_CONFIG(128, 128, 16, 4, 4);
    TRY_CONFIG(128, 128, 32, 4, 4);
    TRY_CONFIG(128,  64,  8, 4, 4);
    TRY_CONFIG(128,  64, 16, 4, 4);
    TRY_CONFIG( 64, 128,  8, 4, 4);
    TRY_CONFIG( 64, 128, 16, 4, 4);

    // TM=8, TN=4 → WM=128, WN=64
    TRY_CONFIG(128,  64,  8, 8, 4);
    TRY_CONFIG(128,  64, 16, 8, 4);
    TRY_CONFIG(128, 128,  8, 8, 4);
    TRY_CONFIG(128, 128, 16, 8, 4);
    TRY_CONFIG(256, 128,  8, 8, 4);
    TRY_CONFIG(256,  64,  8, 8, 4);
    TRY_CONFIG(256,  64, 16, 8, 4);

    // TM=4, TN=8 → WM=64, WN=128
    TRY_CONFIG( 64, 128,  8, 4, 8);
    TRY_CONFIG( 64, 128, 16, 4, 8);
    TRY_CONFIG(128, 128,  8, 4, 8);
    TRY_CONFIG(128, 128, 16, 4, 8);
    TRY_CONFIG( 64, 256,  8, 4, 8);
    TRY_CONFIG(128, 256,  8, 4, 8);

    printf("\n========================================\n");
    printf("BEST: BM=%d BN=%d BK=%d TM=%d TN=%d\n", best_bm, best_bn, best_bk, best_tm, best_tn);
    printf("      %.2f ms, %.1f GFLOPs (%.1f%% of cuBLAS)\n",
           best_ms, best_gflops, (best_gflops / cublas_gflops) * 100);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B);
    return 0;
}
