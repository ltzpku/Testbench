// gpu_stream_b200_full.cu
// Compile: nvcc -O3 -o gpu_stream_b200_full gpu_stream_b200_full.cu stream_json_writer.cpp -I.
// Requires: json.hpp in current directory or include path

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <cuda_runtime.h>
#include "stream_json_writer.h" // Added header

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------- vectorized STREAM kernels (float4) with strided loop ----------
__global__ void copy_f4_kernel(float4 *dst, const float4 *src, size_t nvec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < nvec; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void scale_f4_kernel(float4 *dst, const float4 *src, float scalar, size_t nvec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < nvec; i += stride) {
        float4 v = src[i];
        dst[i].x = scalar * v.x;
        dst[i].y = scalar * v.y;
        dst[i].z = scalar * v.z;
        dst[i].w = scalar * v.w;
    }
}

__global__ void add_f4_kernel(float4 *dst, const float4 *a, const float4 *b, size_t nvec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < nvec; i += stride) {
        float4 A = a[i], B = b[i];
        dst[i].x = A.x + B.x;
        dst[i].y = A.y + B.y;
        dst[i].z = A.z + B.z;
        dst[i].w = A.w + B.w;
    }
}

__global__ void triad_f4_kernel(float4 *dst, const float4 *a, const float4 *b, float scalar, size_t nvec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < nvec; i += stride) {
        float4 A = a[i], B = b[i];
        dst[i].x = A.x + scalar * B.x;
        dst[i].y = A.y + scalar * B.y;
        dst[i].z = A.z + scalar * B.z;
        dst[i].w = A.w + scalar * B.w;
    }
}

// ---------- timing helper ----------
double measure_kernel_ms_copy(float4 *A, float4 *B, size_t nvec, int blocks, int threads, int iters) {
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    float best = 1e30f;
    copy_f4_kernel<<<blocks, threads>>>(B, A, nvec); // warmup
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start));
        copy_f4_kernel<<<blocks, threads>>>(B, A, nvec);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return (double)best;
}

double measure_kernel_ms_scale(float4 *dst, float4 *src, float scalar, size_t nvec, int blocks, int threads, int iters) {
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    float best = 1e30f;
    scale_f4_kernel<<<blocks, threads>>>(dst, src, scalar, nvec); // warmup
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start));
        scale_f4_kernel<<<blocks, threads>>>(dst, src, scalar, nvec);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return (double)best;
}

double measure_kernel_ms_add(float4 *dst, float4 *a, float4 *b, size_t nvec, int blocks, int threads, int iters) {
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    float best = 1e30f;
    add_f4_kernel<<<blocks, threads>>>(dst, a, b, nvec); // warmup
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start));
        add_f4_kernel<<<blocks, threads>>>(dst, a, b, nvec);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return (double)best;
}

double measure_kernel_ms_triad(float4 *dst, float4 *a, float4 *b, float scalar, size_t nvec, int blocks, int threads, int iters) {
    cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    float best = 1e30f;
    triad_f4_kernel<<<blocks, threads>>>(dst, a, b, scalar, nvec); // warmup
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start));
        triad_f4_kernel<<<blocks, threads>>>(dst, a, b, scalar, nvec);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return (double)best;
}

// ---------- Reporting and JSON Output Wrapper ----------
void process_result(const char* test_name, size_t bytes_moved, double ms, int device, const cudaDeviceProp& prop) {
    double seconds = ms * 1e-3;
    double bw_bytes_s = (double)bytes_moved / seconds;
    double bw_GB_s = bw_bytes_s / 1e9;   // decimal GB/s
    double bw_TB_s = bw_bytes_s / 1e12;  // decimal TB/s
    
    // Console Output
    printf("Bytes moved : %llu bytes\n", (unsigned long long)bytes_moved);
    printf("Elapsed     : %.3f ms\n", ms);
    printf("BW          : %.2f GB/s  (%.4f TB/s)\n", bw_GB_s, bw_TB_s);

    // JSON Output
    write_stream_result_json(std::string(test_name), 
                             (double)bytes_moved, 
                             ms, 
                             bw_GB_s, 
                             device, 
                             prop);
}

// ---------- main ----------
int main(int argc, char** argv) {
    // defaults
    unsigned long long bytes = 16ULL * 1024 * 1024 * 1024; // 16 GB
    int iters = 20;
    const char *tests = "CSAT"; // default run all

    static struct option long_options[] = {
        {"bytes", required_argument, 0, 'b'},
        {"iters", required_argument, 0, 'i'},
        {"tests", required_argument, 0, 't'},
        {0,0,0,0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "b:i:t:", long_options, NULL)) != -1) {
        if (opt == 'b') bytes = strtoull(optarg, NULL, 0);
        if (opt == 'i') iters = atoi(optarg);
        if (opt == 't') tests = optarg;
    }

    // --- Added: Get Device Properties for JSON ---
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // ---------------------------------------------

    printf("GPU STREAM (B200-tuned) ?~@~T bytes=%.2f GB  iters=%d  tests=%s\n",
           (double)bytes / (1024.0*1024.0*1024.0), iters, tests);

    // ensure bytes is multiple of sizeof(float4)
    size_t bytes_per_vec = sizeof(float4);
    unsigned long long nvec = bytes / bytes_per_vec;
    if (nvec == 0) {
        fprintf(stderr, "ERROR: bytes too small or not multiple of %zu\n", bytes_per_vec);
        return 1;
    }
    size_t bytes_actual = nvec * bytes_per_vec;

    // allocate
    float4 *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK(cudaMalloc((void**)&d_A, bytes_actual));
    CHECK(cudaMalloc((void**)&d_B, bytes_actual));
    CHECK(cudaMalloc((void**)&d_C, bytes_actual));

    // init (set to non-zero patterns)
    CHECK(cudaMemset(d_A, 0x11, bytes_actual));
    CHECK(cudaMemset(d_B, 0x22, bytes_actual));
    CHECK(cudaMemset(d_C, 0x33, bytes_actual));

    // launch config
    int threads = 256;
    int blocks = 65535; 
    printf("nvec = %llu (float4 elems), threads=%d blocks=%d\n", (unsigned long long)nvec, threads, blocks);

    // 删除旧的 stream_result.json 以防追加混淆 (可选，看你需要积累还是每次刷新)
    remove("stream_result.json"); 

    if (strchr(tests, 'C')) {
        printf("\n==== COPY ====\n");
        double ms = measure_kernel_ms_copy((float4*)d_A, (float4*)d_B, nvec, blocks, threads, iters);
        size_t bytes_moved = (size_t)2 * (size_t)bytes_actual;
        process_result("Copy", bytes_moved, ms, device, prop);
    }

    if (strchr(tests, 'S')) {
        printf("\n==== SCALE ====\n");
        double ms = measure_kernel_ms_scale((float4*)d_B, (float4*)d_A, 3.14159f, nvec, blocks, threads, iters);
        size_t bytes_moved = (size_t)2 * (size_t)bytes_actual;
        process_result("Scale", bytes_moved, ms, device, prop);
    }

    if (strchr(tests, 'A')) {
        printf("\n==== ADD ====\n");
        double ms = measure_kernel_ms_add((float4*)d_C, (float4*)d_A, (float4*)d_B, nvec, blocks, threads, iters);
        size_t bytes_moved = (size_t)3 * (size_t)bytes_actual;
        process_result("Add", bytes_moved, ms, device, prop);
    }

    if (strchr(tests, 'T')) {
        printf("\n==== TRIAD ====\n");
        double ms = measure_kernel_ms_triad((float4*)d_C, (float4*)d_A, (float4*)d_B, 2.71828f, nvec, blocks, threads, iters);
        size_t bytes_moved = (size_t)3 * (size_t)bytes_actual;
        process_result("Triad", bytes_moved, ms, device, prop);
    }

    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    return 0;
}
