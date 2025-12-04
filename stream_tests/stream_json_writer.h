#ifndef STREAM_JSON_WRITER_H
#define STREAM_JSON_WRITER_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "json.hpp" // 需要包含这个以使用 nlohmann::ordered_json 作为返回类型

using json = nlohmann::ordered_json;

// 1. 生成单次测试 (如 "Copy") 的结果 JSON 对象 (不写文件)
json get_stream_result_json(const std::string& test_name,
                            double bytes_moved,
                            double ms,
                            double bw_gb_s,
                            int device,
                            const cudaDeviceProp& prop);

// 2. 将所有测试结果保存到文件
void save_all_stream_results(const std::vector<json>& results, const std::string& filename);

#endif // STREAM_JSON_WRITER_H
