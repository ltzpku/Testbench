#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "json_writer.h"

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, int(s)); exit(1);} } while(0)

// ==========================================
// 1. 初始化 Kernel (保持不变)
// ==========================================
template<typename T>
__global__ void init_data(T* p, size_t n, float scale=1.0f) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = T(float((i % 97) - 48) * 0.01f * scale);
}

template<>
__global__ void init_data<__half>(__half* p, size_t n, float scale) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2half(float((i%97)-48)*0.01f*scale);
}

template<>
__global__ void init_data<__nv_bfloat16>(__nv_bfloat16* p, size_t n, float scale) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16(float((i%97)-48)*0.01f*scale);
}

__global__ void init_data_fp8_e4m3(__nv_fp8_e4m3* p, size_t n, float scale=1.0f) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __nv_fp8_e4m3(float((i % 97) - 48) * 0.01f * scale);
}

__global__ void init_data_fp8_e5m2(__nv_fp8_e5m2* p, size_t n, float scale=1.0f) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __nv_fp8_e5m2(float((i % 97) - 48) * 0.01f * scale);
}

__global__ void init_data_int8(int8_t* p, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (int8_t)((i % 25) - 12);
}

// ==========================================
// 2. cuBLASLt 函数 (返回 json 对象)
// ==========================================
json run_cublasLt_benchmark(int M, int N, int K, std::string dtype, int iters, int device, cudaDeviceProp prop) {
    printf("  [Device %d] Running cuBLASLt Benchmark for dtype: %s\n", device, dtype.c_str());

    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    size_t sizeA, sizeB, sizeC;
    void *dA, *dB, *dC;

    cudaDataType typeA, typeB, typeC;
    cublasComputeType_t computeType;
    cudaDataType scaleType;

    float a_scale_val = 1.0f, b_scale_val = 1.0f;

    if (dtype == "fp8_e4m3") {
        typeA = CUDA_R_8F_E4M3; typeB = CUDA_R_8F_E4M3; typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F;
        scaleType = CUDA_R_32F;
        sizeA = (size_t)M * K * 1; sizeB = (size_t)K * N * 1; sizeC = (size_t)M * N * 4;
        CHECK_CUDA(cudaMalloc(&dA, sizeA)); CHECK_CUDA(cudaMalloc(&dB, sizeB)); CHECK_CUDA(cudaMalloc(&dC, sizeC));
        init_data_fp8_e4m3<<<(M*K+255)/256, 256>>>((__nv_fp8_e4m3*)dA, M*K);
        init_data_fp8_e4m3<<<(K*N+255)/256, 256>>>((__nv_fp8_e4m3*)dB, K*N);
    }
    else if (dtype == "fp8_e5m2") {
        typeA = CUDA_R_8F_E5M2; typeB = CUDA_R_8F_E4M3; typeC = CUDA_R_32F;
        computeType = CUBLAS_COMPUTE_32F;
        scaleType = CUDA_R_32F;
        sizeA = (size_t)M * K * 1; sizeB = (size_t)K * N * 1; sizeC = (size_t)M * N * 4;
        CHECK_CUDA(cudaMalloc(&dA, sizeA)); CHECK_CUDA(cudaMalloc(&dB, sizeB)); CHECK_CUDA(cudaMalloc(&dC, sizeC));
        init_data_fp8_e5m2<<<(M*K+255)/256, 256>>>((__nv_fp8_e5m2*)dA, M*K);
        init_data_fp8_e4m3<<<(K*N+255)/256, 256>>>((__nv_fp8_e4m3*)dB, K*N);
    }
    else if (dtype == "int8") {
        typeA = CUDA_R_8I; typeB = CUDA_R_8I; typeC = CUDA_R_32I;
        computeType = CUBLAS_COMPUTE_32I;
        scaleType = CUDA_R_32I;
        sizeA = (size_t)M * K * 1; sizeB = (size_t)K * N * 1; sizeC = (size_t)M * N * 4;
        CHECK_CUDA(cudaMalloc(&dA, sizeA)); CHECK_CUDA(cudaMalloc(&dB, sizeB)); CHECK_CUDA(cudaMalloc(&dC, sizeC));
        init_data_int8<<<(M*K+255)/256, 256>>>((int8_t*)dA, M*K);
        init_data_int8<<<(K*N+255)/256, 256>>>((int8_t*)dB, K*N);
    } else {
        fprintf(stderr, "Unknown cublasLt dtype: %s\n", dtype.c_str()); exit(1);
    }

    CHECK_CUDA(cudaMemset(dC, 0, sizeC));

    void *d_a_scale = nullptr, *d_b_scale = nullptr;
    if (dtype.find("fp8") != std::string::npos) {
        CHECK_CUDA(cudaMalloc(&d_a_scale, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b_scale, sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_a_scale, &a_scale_val, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_scale, &b_scale_val, sizeof(float), cudaMemcpyHostToDevice));
    }

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, typeA, M, K, M));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, typeB, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, typeC, M, N, M));

    cublasLtMatmulDesc_t matmulDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));

    if (d_a_scale) {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale)));
    }

    cublasLtMatmulPreference_t preference;
    size_t workspaceSize = 64 * 1024 * 1024;
    void* workspace;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutC, preference, 1, &heuristicResult, &returnedResults));


    if (returnedResults == 0) {
        fprintf(stderr, "Error: No valid cuBLASLt algorithm found for %s. Check GPU Arch.\n", dtype.c_str());
        exit(1);
    }
    cublasLtMatmulAlgo_t algo = heuristicResult.algo;

    float alpha_f = 1.0f, beta_f = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    void *alpha_ptr, *beta_ptr;

    if (dtype == "int8") {
        alpha_ptr = &alpha_i;
        beta_ptr = &beta_i;
    } else {
        alpha_ptr = &alpha_f;
        beta_ptr = &beta_f;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Warmup
    for(int i=0; i<10; i++) {
        CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, alpha_ptr, dA, layoutA, dB, layoutB, beta_ptr, dC, layoutC, dC, layoutC, &algo, workspace, workspaceSize, 0));
    }

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for(int i=0; i<iters; i++) {
        CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, alpha_ptr, dA, layoutA, dB, layoutB, beta_ptr, dC, layoutC, dC, layoutC, &algo, workspace, workspaceSize, 0));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / 1e12) / (ms / iters / 1e3);

    printf("  [Device %d] Avg time: %.3f ms, Perf: %.2f TFLOPS/TOPS (dtype=%s)\n", device, ms/iters, tflops, dtype.c_str());
    
    // 返回 JSON 对象而不是写文件
    json result = get_result_json(M, N, K, dtype, ms, iters, device, prop, tflops);

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(workspace));
    if (d_a_scale) { CHECK_CUDA(cudaFree(d_a_scale)); CHECK_CUDA(cudaFree(d_b_scale)); }
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc)); CHECK_CUBLAS(cublasLtDestroy(handle));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return result;
}

// ==========================================
// 3. 封装旧版 cuBLAS (FP16/FP32等) 为函数
// ==========================================
json run_legacy_benchmark(int M, int N, int K, std::string dtype, int iters, int device, cudaDeviceProp prop) {
    printf("  [Device %d] Running Standard cuBLAS Benchmark for dtype: %s\n", device, dtype.c_str());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    if (dtype=="tf32") CHECK_CUBLAS(cublasSetMathMode(handle,CUBLAS_TF32_TENSOR_OP_MATH));
    else CHECK_CUBLAS(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH));

    int lda=M, ldb=K, ldc=M;
    void *dA=nullptr, *dB=nullptr, *dC=nullptr;
    size_t elemsA=(size_t)M*K, elemsB=(size_t)K*N, elemsC=(size_t)M*N;

    cudaDataType Atype,Btype,Ctype,ComputeType;
    float alpha=1.0f,beta=0.0f;

    if (dtype=="fp16") {
        Atype=Btype=Ctype=CUDA_R_16F; ComputeType=CUDA_R_32F;
        CHECK_CUDA(cudaMalloc(&dA,elemsA*sizeof(__half))); CHECK_CUDA(cudaMalloc(&dB,elemsB*sizeof(__half))); CHECK_CUDA(cudaMalloc(&dC,elemsC*sizeof(__half)));
        int bs=256; init_data<<<(elemsA+bs-1)/bs,bs>>>((__half*)dA,elemsA); init_data<<<(elemsB+bs-1)/bs,bs>>>((__half*)dB,elemsB); CHECK_CUDA(cudaMemset(dC,0,elemsC*sizeof(__half)));
    } else if (dtype=="bf16") {
        Atype=Btype=Ctype=CUDA_R_16BF; ComputeType=CUDA_R_32F;
        CHECK_CUDA(cudaMalloc(&dA,elemsA*sizeof(__nv_bfloat16))); CHECK_CUDA(cudaMalloc(&dB,elemsB*sizeof(__nv_bfloat16))); CHECK_CUDA(cudaMalloc(&dC,elemsC*sizeof(__nv_bfloat16)));
        int bs=256; init_data<<<(elemsA+bs-1)/bs,bs>>>((__nv_bfloat16*)dA,elemsA); init_data<<<(elemsB+bs-1)/bs,bs>>>((__nv_bfloat16*)dB,elemsB); CHECK_CUDA(cudaMemset(dC,0,elemsC*sizeof(__nv_bfloat16)));
    } else if (dtype=="tf32") {
        Atype=Btype=Ctype=ComputeType=CUDA_R_32F;
        CHECK_CUDA(cudaMalloc(&dA,elemsA*sizeof(float))); CHECK_CUDA(cudaMalloc(&dB,elemsB*sizeof(float))); CHECK_CUDA(cudaMalloc(&dC,elemsC*sizeof(float)));
        int bs=256; init_data<<<(elemsA+bs-1)/bs,bs>>>((float*)dA,elemsA); init_data<<<(elemsB+bs-1)/bs,bs>>>((float*)dB,elemsB); CHECK_CUDA(cudaMemset(dC,0,elemsC*sizeof(float)));
    } else if (dtype=="fp32") {
        Atype=Btype=Ctype=ComputeType=CUDA_R_32F;
        CHECK_CUBLAS(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
        CHECK_CUDA(cudaMalloc(&dA,elemsA*sizeof(float))); CHECK_CUDA(cudaMalloc(&dB,elemsB*sizeof(float))); CHECK_CUDA(cudaMalloc(&dC,elemsC*sizeof(float)));
        int bs=256; init_data<<<(elemsA+bs-1)/bs,bs>>>((float*)dA,elemsA); init_data<<<(elemsB+bs-1)/bs,bs>>>((float*)dB,elemsB); CHECK_CUDA(cudaMemset(dC,0,elemsC*sizeof(float)));
    } else if (dtype=="fp64") {
        Atype=Btype=Ctype=ComputeType=CUDA_R_64F;
        CHECK_CUBLAS(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
        CHECK_CUDA(cudaMalloc(&dA,elemsA*sizeof(double))); CHECK_CUDA(cudaMalloc(&dB,elemsB*sizeof(double))); CHECK_CUDA(cudaMalloc(&dC,elemsC*sizeof(double)));
        int bs=256; init_data<<<(elemsA+bs-1)/bs,bs>>>((double*)dA,elemsA); init_data<<<(elemsB+bs-1)/bs,bs>>>((double*)dB,elemsB); CHECK_CUDA(cudaMemset(dC,0,elemsC*sizeof(double)));
    } else {
        fprintf(stderr,"Unknown dtype %s\n",dtype.c_str()); exit(1);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    for(int i=0;i<10;i++) CHECK_CUBLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,&alpha,dA,Atype,lda,dB,Btype,ldb,&beta,dC,Ctype,ldc,ComputeType,CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0;i<iters;i++) CHECK_CUBLAS(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,&alpha,dA,Atype,lda,dB,Btype,ldb,&beta,dC,Ctype,ldc,ComputeType,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
    float ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));

    double flops=2.0*(double)M*(double)N*(double)K;
    double tflops=(flops/1e12)/(ms/iters/1e3);
    printf("  [Device %d] Avg time: %.3f ms, Perf: %.2f TFLOPS (dtype=%s)\n", device, ms/iters, tflops, dtype.c_str());
    
    // 返回 JSON
    json result = get_result_json(M, N, K, dtype, ms, (int)iters, device, prop, tflops);

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUBLAS(cublasDestroy(handle));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return result;
}

// ==========================================
// 4. 参数解析
// ==========================================
void parse_args(int argc, char** argv, int& M, int& N, int& K, std::string& dtype, int& iters) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--M=") == 0) {
            M = std::stoi(arg.substr(4));
        } else if (arg.find("--N=") == 0) {
            N = std::stoi(arg.substr(4));
        } else if (arg.find("--K=") == 0) {
            K = std::stoi(arg.substr(4));
        } else if (arg.find("--dtype=") == 0) {
            dtype = arg.substr(8);
        } else if (arg.find("--iters=") == 0) {
            iters = std::stoi(arg.substr(8));
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --M=INT       Matrix dim M (default: 8192)\n");
            printf("  --N=INT       Matrix dim N (default: 8192)\n");
            printf("  --K=INT       Matrix dim K (default: 8192)\n");
            printf("  --dtype=STR   Data type (fp16, bf16, tf32, fp32, fp64, fp8_e4m3, fp8_e5m2, int8) (default: fp16)\n");
            printf("  --iters=INT   Number of iterations (default: 100)\n");
            exit(0);
        } else {
            printf("Warning: Unknown argument: %s\n", arg.c_str());
        }
    }
}

// ==========================================
// 5. Main 函数 (支持多卡遍历)
// ==========================================
int main(int argc, char** argv) {
  int M = 8192;
  int N = 8192;
  int K = 8192;
  std::string dtype = "fp16";
  int iters = 100;

  parse_args(argc, argv, M, N, K, dtype, iters);

  // 1. 获取设备数量
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
      fprintf(stderr, "No CUDA devices found.\n");
      return 1;
  }

  printf("Found %d CUDA devices. Starting benchmark on all devices sequentially...\n", device_count);

  // 2. 准备结果容器
  std::vector<json> all_results;

  // 3. 循环遍历所有 GPU
  for (int i = 0; i < device_count; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      cudaDeviceProp prop;
      CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

      printf("----------------------------------------------------------------\n");
      printf("Benchmarking Device %d: %s (SMs=%d, CC=%d.%d)\n", i, prop.name, prop.multiProcessorCount, prop.major, prop.minor);
      
      json res;
      // 根据 dtype 选择测试函数
      if (dtype == "fp8_e4m3" || dtype == "fp8_e5m2" || dtype == "int8") {
          res = run_cublasLt_benchmark(M, N, K, dtype, iters, i, prop);
      } else {
          res = run_legacy_benchmark(M, N, K, dtype, iters, i, prop);
      }
      
      // 收集结果
      all_results.push_back(res);
  }

  printf("----------------------------------------------------------------\n");
  
  // 4. 将所有结果一次性写入文件
  save_all_results(all_results, "gemm_result.json");

  return 0;
}
