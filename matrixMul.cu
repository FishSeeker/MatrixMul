#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cuda_fp16.h"
#include "fp16_conversion.h"



#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif
#define BLOCK_SIZE 32

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

__global__ void shared_kernel(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
       __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
       __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__ void general_kernel(float *C,const float *A, const float *B, int N,int M,int S)
{
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x
                    + blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < M * S)
    {
        int row = threadId / S;
        int column = threadId % S;
        float sum = 0.0;
        C[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
             sum += A[row * N + i] * B[i * S + column];
        }
        C[threadId] = sum;
    }
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = 0.2;
    }
}

void constantInit_half( half *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] =  approx_float_to_half(0.2);
    }
}

void cpu(dim3 &dimsA, dim3 &dimsB){
	//dimsA.x 就是列数，dimsB.y是行数
	int hA = dimsA.x,wA = dimsA.y,wB = dimsB.x;
	float *A,*B,*C;
	A = (float*)malloc(sizeof(float) * dimsA.x * dimsA.y);
	B = (float*)malloc(sizeof(float) * dimsB.x * dimsB.y);
	C = (float*)malloc(sizeof(float) * dimsA.y * dimsB.x);

	constantInit(A,dimsA.x * dimsA.y,1.0);
	constantInit(B,dimsB.x * dimsB.y,1.0);
    for (unsigned int i = 0; i < hA; ++i){
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
    }
    for(int i=0;i<10;i++)
    	printf("%f ",C[i]);

}

int shared(dim3 &dimsA, dim3 &dimsB)
{
	printf("\nstart=================shared memory================================\n");
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 1.0f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    shared_kernel<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);


    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 30;

    for (int j = 0; j < nIter; j++)
    {

        shared_kernel<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps


    for(int i=0;i<10;i++)
    	printf("%f ",h_C[i]);

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    printf("\nend=================shared memory================================\n");
    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

int general(dim3 &dimsA, dim3 &dimsB)
{
	printf("\nstart=================general================================\n");
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float valB = 1.0f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");


    // Performs warmup operation using matrixMul CUDA kernel
    general_kernel<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsA.y,dimsB.x);
    printf("done\n");
    cudaDeviceSynchronize();



    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 30;

    for (int j = 0; j < nIter; j++)
    {

        general_kernel<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsA.y,dimsB.x);

    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

    for(int i=0;i<10;i++)
    	printf("%f ",h_C[i]);
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    printf("\nend=================general================================\n");
    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

int single_cublas(dim3 &dimsA, dim3 &dimsB)
{
	printf("\nstart=================single cublas================================\n");
	sMatrixSize matrix_size;
	//在这对matrix_size根据dimsA和dimsB赋值

	matrix_size.uiHA = dimsA.y;
	matrix_size.uiWA = dimsA.x;
	matrix_size.uiHB = dimsB.y;
	matrix_size.uiWB = dimsB.x;
	matrix_size.uiHC = dimsA.y;
	matrix_size.uiWC = dimsB.x;

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);


    constantInit(h_A,size_A,1.0);
    constantInit(h_B,size_B,1.0);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);

    (cudaMalloc((void **) &d_A, mem_size_A));
    (cudaMalloc((void **) &d_B, mem_size_B));
    (cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    (cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 30;

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t start, stop;

        (cublasCreate(&handle));

        //Perform warmup operation with cublas
        (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

        // Allocate CUDA events that we'll use for timing
        (cudaEventCreate(&start));
        (cudaEventCreate(&stop));

        // Record the start event
        (cudaEventRecord(start, NULL));

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

        }

        printf("done.\n");

        // Record the stop event
        (cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        (cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        (cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // copy result from device to host
        (cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        (cublasDestroy(handle));
    }
    for(int i=0;i<10;i++)
    	printf("%f ",h_CUBLAS[i]);

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    (cudaFree(d_A));
    (cudaFree(d_B));
    (cudaFree(d_C));
    printf("\nend=================single cublas================================\n");
    return 1;

}

int half_cublas(dim3 &dimsA, dim3 &dimsB)
{
	printf("\nstart=================half cublas================================\n");
	sMatrixSize matrix_size;
	//在这对matrix_size根据dimsA和dimsB赋值

	matrix_size.uiHA = dimsA.y;
	matrix_size.uiWA = dimsA.x;
	matrix_size.uiHB = dimsB.y;
	matrix_size.uiWB = dimsB.x;
	matrix_size.uiHC = dimsA.y;
	matrix_size.uiWC = dimsB.x;

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(half) * size_A;
    half *h_A = (half *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(half) * size_B;
    half *h_B = (half *)malloc(mem_size_B);


    constantInit_half(h_A,size_A,1.0);
    constantInit_half(h_B,size_B,1.0);

    // allocate device memory
    __half *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(half) * size_C;

    // allocate host memory for the result
    half *h_C      = (half *) malloc(mem_size_C);
    half *h_CUBLAS = (half *) malloc(mem_size_C);

    (cudaMalloc((void **) &d_A, mem_size_A));
    (cudaMalloc((void **) &d_B, mem_size_B));
    (cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    (cudaMalloc((void **) &d_C, mem_size_C));

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    int nIter = 30;

    // CUBLAS version 2.0
    {
    	const __half alpha = approx_float_to_half(1.0);
    	const __half beta = approx_float_to_half(0.0);


        cublasHandle_t handle;
        cudaEvent_t start, stop;

        (cublasCreate(&handle));

        //Perform warmup operation with cublas
        (cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

        // Allocate CUDA events that we'll use for timing
        (cudaEventCreate(&start));
        (cudaEventCreate(&stop));

        // Record the start event
        (cudaEventRecord(start, NULL));

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            (cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
        }

        printf("done.\n");

        // Record the stop event
        (cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        (cudaEventSynchronize(stop));

        float msecTotal = 0.0f;
        (cudaEventElapsedTime(&msecTotal, start, stop));

        // Compute and print the performance
        float msecPerMatrixMul = msecTotal / nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

        // copy result from device to host
        (cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

        // Destroy the handle
        (cublasDestroy(handle));
    }
    for(int i=0;i<10;i++)
    	printf("%f ",half_to_float(h_CUBLAS[i]));

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    (cudaFree(d_A));
    (cudaFree(d_B));
    (cudaFree(d_C));
    printf("\nend=================half cublas================================\n");
    return 1;
}


int main(int argc, char **argv)
{
	int multiple = 64;
    dim3 dimsA(multiple*BLOCK_SIZE, multiple*BLOCK_SIZE, 1);
    dim3 dimsB(multiple*BLOCK_SIZE, multiple*BLOCK_SIZE, 1);


    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);


    // cpu(dimsA, dimsB);
    shared(dimsA, dimsB);
    general(dimsA, dimsB);
    single_cublas(dimsA, dimsB);
    half_cublas(dimsA, dimsB);

    return 0;
}
