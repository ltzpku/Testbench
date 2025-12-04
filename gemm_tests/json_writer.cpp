#include "json_writer.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <cctype>   
#include <iostream>
#include "json.hpp"
#include <cuda_runtime_api.h>

using json = nlohmann::ordered_json;

// === 1. 智能读取驱动版本 (保持不变) ===
std::string get_driver_version_smart() {
    std::string version = "Unknown";
    std::ifstream f("/proc/driver/nvidia/version");
    if (f.is_open()) {
        std::string line;
        if (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string token;
            while (ss >> token) {
                if (token.length() > 0 && std::isdigit(token[0]) && token.find('.') != std::string::npos) {
                    version = token;
                    break;
                }
            }
        }
    }
    return version;
}

// === 2. 性能达标检查 (保持不变) ===
std::pair<std::string, double> check_performance_status(const std::string& spec_filename, const std::string& dtype, double measured_tflops) {
    std::ifstream f(spec_filename);
    if (!f.is_open()) return {"Spec File Not Found", 0.0};

    try {
        json specs = json::parse(f);
        if (!specs.contains("gemm")) return {"'gemm' Block Missing", 0.0};
        if (!specs["gemm"].contains(dtype)) return {"Dtype Missing in Spec", 0.0};

        double spec_val = specs["gemm"][dtype];
        double threshold = spec_val * 0.75; 

        if (measured_tflops >= threshold) {
            return {"Passed", spec_val};
        } else {
            return {"Failed", spec_val};
        }
    } catch (...) {
        return {"Unknown Spec Error", 0.0};
    }
}

// === 3. 生成 JSON 对象 (原 write_result_json 的核心逻辑) ===
json get_result_json(int M, int N, int K, const std::string& dtype,
                     float ms, int iters, int device,
                     const cudaDeviceProp& prop, double tflops) {

    std::string target_spec_file = "B200_specs.json"; 

    // 获取驱动版本
    std::string driver_str = get_driver_version_smart();
    if (driver_str == "Unknown") {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        driver_str = "CUDA " + std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 1000) / 10);
    }

    // 处理测试名称
    std::string raw_name = "cuBLAS " + dtype + " GEMM";
    std::string prefix = "cuBLAS ";
    size_t start_pos = raw_name.find(prefix);
    if (start_pos != std::string::npos) {
        raw_name.replace(start_pos, prefix.length(), "");
    }
    if (!raw_name.empty() && raw_name[0] == ' ') raw_name.erase(0, 1);
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // 检查 Spec
    std::pair<std::string, double> check_result = check_performance_status(target_spec_file, dtype, tflops);
    std::string status = check_result.first;
    double spec_val = check_result.second;

    // 构建 JSON
    json j;
    j["test_name"] = raw_name;
    j["configuration"]["data_type"] = dtype;
    
    j["gpu_info"]["index"] = device;
    j["gpu_info"]["name"] = prop.name;
    j["gpu_info"]["compute_capability"] = cc;
    j["gpu_info"]["driver_version"] = driver_str;

    j["status"] = status;

    j["problem_size"]["m"] = M;
    j["problem_size"]["n"] = N;
    j["problem_size"]["k"] = K;
    j["problem_size"]["l_batch"] = 1;
    
    j["performance"]["tflops"] = tflops;
    
    if (spec_val > 0.0) {
        j["performance"]["spec_tflops"] = spec_val;
        j["performance"]["threshold_75pct"] = spec_val * 0.75;
        j["performance"]["percent_of_spec"] = (tflops / spec_val) * 100.0;
    }
    
    // 这里不再写入文件，而是返回对象
    return j;
}

// === 4. 批量保存函数 ===
void save_all_results(const std::vector<json>& results, const std::string& filename) {
    std::ofstream o(filename);
    // 输出为一个 JSON 数组
    o << std::setw(4) << results << std::endl;
    o.close();
    std::cout << "All results saved to " << filename << " (Total GPUs: " << results.size() << ")" << std::endl;
}
