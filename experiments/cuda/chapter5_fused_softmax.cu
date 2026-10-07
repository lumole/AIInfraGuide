#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__device__ float block_reduce_max(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_reduce_sum(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    return shared[0];
}

__global__ void scale_mask_kernel(const float* input, float* scaled,
                                  int rows, int cols, float scale) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const float* x = input + static_cast<size_t>(row) * cols;
    float* y = scaled + static_cast<size_t>(row) * cols;
    for (int col = tid; col < cols; col += blockDim.x) {
        y[col] = row >= col ? x[col] * scale : -CUDART_INF_F;
    }
}

__global__ void softmax_kernel(const float* input, float* output,
                               int rows, int cols) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const float* x = input + static_cast<size_t>(row) * cols;
    float* y = output + static_cast<size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) local_max = fmaxf(local_max, x[col]);
    const float max_value = block_reduce_max(local_max, shared);
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) local_sum += expf(x[col] - max_value);
    const float sum = block_reduce_sum(local_sum, shared);
    __syncthreads();

    const float inv_sum = 1.0f / sum;
    for (int col = tid; col < cols; col += blockDim.x) y[col] = expf(x[col] - max_value) * inv_sum;
}

__global__ void fused_scale_causal_softmax(const float* input, float* output,
                                           int rows, int cols, float scale) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const float* x = input + static_cast<size_t>(row) * cols;
    float* y = output + static_cast<size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = row >= col ? x[col] * scale : -CUDART_INF_F;
        local_max = fmaxf(local_max, value);
    }
    const float max_value = block_reduce_max(local_max, shared);
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = row >= col ? x[col] * scale : -CUDART_INF_F;
        local_sum += expf(value - max_value);
    }
    const float sum = block_reduce_sum(local_sum, shared);
    __syncthreads();

    const float inv_sum = 1.0f / sum;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float value = row >= col ? x[col] * scale : -CUDART_INF_F;
        y[col] = expf(value - max_value) * inv_sum;
    }
}

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int threads = 256;
    const float scale = 1.0f / std::sqrt(128.0f);
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t bytes = elements * sizeof(float);

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_input(elements);
    for (float& value : h_input) value = dist(rng);
    std::vector<float> h_output(elements);
    std::vector<float> h_reference(elements);
    for (int row = 0; row < rows; ++row) {
        float max_value = -INFINITY;
        for (int col = 0; col < cols; ++col) {
            float value = row >= col ? h_input[static_cast<size_t>(row) * cols + col] * scale : -INFINITY;
            max_value = std::max(max_value, value);
        }
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = row >= col ? h_input[static_cast<size_t>(row) * cols + col] * scale : -INFINITY;
            sum += std::exp(value - max_value);
        }
        for (int col = 0; col < cols; ++col) {
            float value = row >= col ? h_input[static_cast<size_t>(row) * cols + col] * scale : -INFINITY;
            h_reference[static_cast<size_t>(row) * cols + col] = std::exp(value - max_value) / sum;
        }
    }

    float *d_input = nullptr, *d_scaled = nullptr, *d_separate = nullptr, *d_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_scaled, bytes));
    CUDA_CHECK(cudaMalloc(&d_separate, bytes));
    CUDA_CHECK(cudaMalloc(&d_fused, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 grid(rows), block(threads);
    const size_t shared_bytes = threads * sizeof(float);
    auto launch_separate = [&](cudaStream_t stream) {
        scale_mask_kernel<<<grid, block, 0, stream>>>(d_input, d_scaled, rows, cols, scale);
        softmax_kernel<<<grid, block, shared_bytes, stream>>>(d_scaled, d_separate, rows, cols);
    };
    auto launch_fused = [&](cudaStream_t stream) {
        fused_scale_causal_softmax<<<grid, block, shared_bytes, stream>>>(d_input, d_fused, rows, cols, scale);
    };

    // CUDA events cannot accept capturing lambdas with a function pointer, so time manually.
    for (int i = 0; i < 10; ++i) { launch_separate(0); launch_fused(0); }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 50; ++i) launch_separate(0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float separate_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&separate_ms, start, stop));
    separate_ms /= 50.0f;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 50; ++i) launch_fused(0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float fused_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    fused_ms /= 50.0f;

    launch_fused(0);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_fused, bytes, cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    for (size_t i = 0; i < elements; ++i) max_abs = std::max(max_abs, std::fabs(h_output[i] - h_reference[i]));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU=%s rows=%d cols=%d threads=%d\n", prop.name, rows, cols, threads);
    std::printf("separate_ms=%.4f fused_ms=%.4f speedup=%.3fx max_abs_error=%.8g\n",
                separate_ms, fused_ms, separate_ms / fused_ms, max_abs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scaled));
    CUDA_CHECK(cudaFree(d_separate));
    CUDA_CHECK(cudaFree(d_fused));
    return max_abs < 2e-5f ? EXIT_SUCCESS : EXIT_FAILURE;
}
