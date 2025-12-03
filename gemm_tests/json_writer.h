#ifndef JSON_WRITER_H
#define JSON_WRITER_H

#include <string>
#include <cuda_runtime_api.h> // 仅用于 cudaDeviceProp 类型

// 声明我们的写入函数
void write_result_json(int M, int N, int K, const std::string& dtype, 
                       float ms, int iters, int device, 
                       const cudaDeviceProp& prop, double tflops);

#endif // JSON_WRITER_H
