#ifndef JSON_WRITER_H
#define JSON_WRITER_H

#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "json.hpp" // 需要包含这个以使用 nlohmann::ordered_json 作为返回类型

using json = nlohmann::ordered_json;

// 1. 生成单个 GPU 的测试结果 JSON 对象 (不写文件)
json get_result_json(int M, int N, int K, const std::string& dtype,
                     float ms, int iters, int device,
                     const cudaDeviceProp& prop, double tflops);

// 2. 将所有 GPU 的结果列表保存到文件
void save_all_results(const std::vector<json>& results, const std::string& filename);

#endif // JSON_WRITER_H
