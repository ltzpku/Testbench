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

// === 新增辅助函数：根据设备名称动态生成 Spec 文件名 ===
std::string detect_spec_filename(std::string device_name) {
    // 1. 定义已知的高频型号列表 (可以按需补充)
    // 注意：顺序很重要，如果名字里同时包含多个关键词，前面的优先
    std::vector<std::string> known_models = {
        "B200", "B300", "GB200", "GB300", "H100", "H200", "H20", "H800", "A100", "A800", "5090",		
        "4090", "3090", "L40", "T4", "V100"
    };

    for (const auto& model : known_models) {
        if (device_name.find(model) != std::string::npos) {
            // 找到匹配型号，例如 "H100" -> "H100_specs.json"
            return model + "_specs.json";
        }
    }

    // 2. 兜底策略：如果不在列表中，使用全名，但把空格换成下划线
    // 例如 "NVIDIA GeForce RTX 5090" -> "NVIDIA_GeForce_RTX_5090_specs.json"
    std::string safe_name = device_name;
    std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
    return safe_name + "_specs.json";
}

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
    if (!f.is_open()) {
        // 返回明确的错误信息，方便 Debug
        return {"Spec File Not Found (" + spec_filename + ")", 0.0};
    }

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

// === 3. 生成 JSON 对象 (核心修改区域) ===
json get_result_json(int M, int N, int K, const std::string& dtype,
                     float ms, int iters, int device,
                     const cudaDeviceProp& prop, double tflops) {

    // [修改点]：不再硬编码 "B200_specs.json"
    // 而是通过 prop.name 动态获取
    std::string target_spec_file = detect_spec_filename(prop.name);
    
    // 如果你想在控制台确认它在找哪个文件，可以取消下面这行的注释
    // std::cout << "Debug: Looking for spec file: " << target_spec_file << std::endl;

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
    j["gpu_info"]["name"] = prop.name; // 原始名称，例如 "NVIDIA H100 80GB HBM3"
    j["gpu_info"]["target_spec_file"] = target_spec_file; // [可选] 记录实际使用的 spec 文件名
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
    
    return j;
}

// === 4. 批量保存函数 (保持不变) ===
void save_all_results(const std::vector<json>& results, const std::string& filename) {
    std::ofstream o(filename);
    o << std::setw(4) << results << std::endl;
    o.close();
    std::cout << "All results saved to " << filename << " (Total GPUs: " << results.size() << ")" << std::endl;
}
