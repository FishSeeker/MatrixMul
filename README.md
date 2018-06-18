Used shared memory, cublas_gemm and half-precision_cublas_gemm respectively to imply matrix multiplication algorithm

NOTE:

Using

```
nvcc -arch sm_60 -lcublas matrixMul.cu -o mm
```

to compile this code.