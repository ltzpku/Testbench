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

// === 2. 核心修改：性能达标检查 ===
// 逻辑：直接读取传入的 spec_filename 文件，不再自动生成文件名
std::pair<std::string, double> check_performance_status(const std::string& spec_filename, const std::string& dtype, double measured_tflops) {
    std::ifstream f(spec_filename);
    
    // 1. 如果找不到你指定的 Spec 文件
    if (!f.is_open()) {
        // 返回 Spec Missing，方便你排查是不是文件名写错了
        return {"Spec File Not Found", 0.0};
    }

    try {
        json specs = json::parse(f);
        
        // 2. 按照你提供的 B200_specs.json 结构查找
        // 先找 "gemm" 字段
        if (!specs.contains("gemm")) {
            return {"'gemm' Block Missing", 0.0};
        }

        // 再在 "gemm" 里找对应的 dtype (比如 "fp16", "tf32")
        if (!specs["gemm"].contains(dtype)) {
            return {"Dtype Missing in Spec", 0.0};
        }

        // 3. 读取标准值并计算阈值
        double spec_val = specs["gemm"][dtype];
        double threshold = spec_val * 0.75; // 75% 达标线

        // 4. 判断 Pass/Fail
        if (measured_tflops >= threshold) {
            return {"Passed", spec_val};
        } else {
            return {"Failed", spec_val};
        }

    } catch (const json::parse_error& e) {
        return {"JSON Parse Error", 0.0};
    } catch (...) {
        return {"Unknown Spec Error", 0.0};
    }
}

// === 3. 主写入函数 ===
void write_result_json(int M, int N, int K, const std::string& dtype,
                       float ms, int iters, int device,
                       const cudaDeviceProp& prop, double tflops) {

    // A. 指定 Spec 文件名 (这里直接对应你上传的文件名)
    // 如果以后想灵活，可以把这个文件名作为参数传进来
    std::string target_spec_file = "B200_specs.json"; 

    // B. 获取驱动版本
    std::string driver_str = get_driver_version_smart();
    if (driver_str == "Unknown") {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        driver_str = "CUDA " + std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 1000) / 10);
    }

    // C. 处理测试名称
    std::string raw_name = "cuBLAS " + dtype + " GEMM";
    std::string prefix = "cuBLAS ";
    size_t start_pos = raw_name.find(prefix);
    if (start_pos != std::string::npos) {
        raw_name.replace(start_pos, prefix.length(), "");
    }
    if (!raw_name.empty() && raw_name[0] == ' ') raw_name.erase(0, 1);
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // D. 检查 Spec (传入指定的文件名)
    std::pair<std::string, double> check_result = check_performance_status(target_spec_file, dtype, tflops);
    std::string status = check_result.first;
    double spec_val = check_result.second;

    // E. 构建 JSON
    json j;
    j["test_name"] = raw_name;
    j["configuration"]["data_type"] = dtype;
    
    j["gpu_info"]["index"] = device;
    j["gpu_info"]["name"] = prop.name;
    j["gpu_info"]["compute_capability"] = cc;
    j["gpu_info"]["driver_version"] = driver_str;

    // 输出计算后的 Status
    j["status"] = status;

    j["problem_size"]["m"] = M;
    j["problem_size"]["n"] = N;
    j["problem_size"]["k"] = K;
    j["problem_size"]["l_batch"] = 1;
    
    // Performance
    j["performance"]["tflops"] = tflops;
    
    // 输出对比详情 (只有成功读到 Spec 时才显示)
    if (spec_val > 0.0) {
        j["performance"]["spec_tflops"] = spec_val;
        j["performance"]["threshold_75pct"] = spec_val * 0.75;
        j["performance"]["percent_of_spec"] = (tflops / spec_val) * 100.0;
    }

    // F. 输出到 result.json
    std::ofstream o("result.json");
    o << std::setw(4) << j << std::endl;
    o.close();
    
    std::cout << "JSON saved to result.json. (Compared against: " << target_spec_file << ")" << std::endl;
}
