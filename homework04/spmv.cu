#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz,
                double* x, double* b)
{
    // One thread per row. The ELL arrays are column-major (m x n, entry
    // (row, k) at k * m + row), so neighbouring threads read neighbouring
    // addresses and the loads coalesce. n is the padded row width, i.e. the
    // max number of NNZ in any row. Padding entries have value 0
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < m;
         row += gridDim.x * blockDim.x) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            size_t idx = (size_t) k * m + row;
            T v = vals[idx];
            if (v != 0) {   // skip padding (saves the extra memory load from x)
                sum += v * __ldg(&x[col_ind[idx]]);   // x via read-only cache
            }
        }
        b[row] = sum;
    }
}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread per row (the ELL kernel needs no shared memory)
    unsigned int threads = 256;
    unsigned int blocks = (m + threads - 1) / threads;
    unsigned int shared = 0;

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy ELL data to GPU and allocate memory for output
    // n here is the padded ELL width (n_new), so each array holds m * n entries.
    unsigned int ell_size = (unsigned int) ((size_t) m * n);
    CopyData<unsigned int>(col_ind, ell_size, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, ell_size, sizeof(double), dev_vals);

    // x has one entry per matrix column. The original column count isn't
    // passed in (n is the ELL width), so this assumes a square matrix (m == n),
    // which holds for cant and any symmetric matrix.
    CopyData<double>(x, m, sizeof(double), dev_x);

    checkCudaErrors(cudaMalloc((void**) dev_b, sizeof(double) * m));
}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy CSR data to GPU and allocate memory for output
    CopyData<unsigned int>(row_ptr, m + 1, sizeof(unsigned int), dev_row_ptr);
    CopyData<unsigned int>(col_ind, nnz, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, nnz, sizeof(double), dev_vals);
    CopyData<double>(x, n, sizeof(double), dev_x);

    checkCudaErrors(cudaMalloc((void**) dev_b, sizeof(double) * m));
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
            int m, int n, int nnz, double* x, double* b)
{
    // CSR vector kernel (paper Section 3.3): one warp (32 threads) per row.
    // The warp's lanes stride through the row together, so they read col_ind
    // and vals contiguously, then add their partial sums with an intra-warp
    // reduction. blockDim.x must be a multiple of 32.
    int lane      = threadIdx.x % 32;
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int num_warps = (gridDim.x * blockDim.x) / 32;

    // grid-stride over rows, so a grid smaller than m still covers every row.
    // row depends only on the warp, so all 32 lanes reach the shuffles together.
    for (int row = warp_id; row < m; row += num_warps) {
        unsigned int start = row_ptr[row];
        unsigned int end   = row_ptr[row + 1];

        double sum = 0.0;
        for (unsigned int j = start + lane; j < end; j += 32) {
            sum += vals[j] * __ldg(&x[col_ind[j]]);   // x via read-only cache
        }

        // The paper reduces in shared memory without barriers, relying on a
        // warp running in lockstep. Volta and later GPUs (A100) don't
        // guarantee that, so the same tree reduction is done in registers.
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            b[row] = sum;
        }
    }
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 warp (32 threads) per row, 8 warps per block
    // the reduction uses warp shuffles, so no shared memory is needed
    unsigned int threads = 256;
    unsigned int blocks = (m * 32 + threads - 1) / threads;
    unsigned int shared = 0;

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}
