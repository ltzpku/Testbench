#include "stream_json_writer.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <iostream>
#include <sys/stat.h>
#include "json.hpp" 

using json = nlohmann::ordered_json;

// --- 新增：根据设备名称动态生成 Spec 文件名 ---
std::string detect_stream_spec_filename(std::string device_name) {
    // 1. 定义已知的高频型号列表 (与 gemm 和 nvbandwidth 保持一致)
    std::vector<std::string> known_models = {
        "B200", "B300", "GB200", "GB300", 
        "H100", "H200", "H20", "H800", 
        "A100", "A800", 
        "5090", "4090", "3090", 
        "L40", "T4", "V100"
    };

    for (const auto& model : known_models) {
        if (device_name.find(model) != std::string::npos) {
            // 找到匹配型号，例如 "H100" -> "H100_specs.json"
            return model + "_specs.json";
        }
    }

    // 2. 兜底策略：如果不在列表中，使用全名，但把空格换成下划线
    // 例如 "NVIDIA GeForce RTX 6000 Ada" -> "NVIDIA_GeForce_RTX_6000_Ada_specs.json"
    std::string safe_name = device_name;
    std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
    return safe_name + "_specs.json";
}

// --- 辅助函数：获取驱动版本 (保持不变) ---
static std::string get_driver_version_smart() {
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

// --- 辅助函数：将字符串转小写 (保持不变) ---
static std::string to_lower(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// --- 核心逻辑：检查性能是否达标 ---
static std::pair<std::string, double> check_stream_performance(const std::string& spec_filename, 
                                                               const std::string& test_name, 
                                                               double measured_bw) {
    std::ifstream f(spec_filename);
    if (!f.is_open()) {
        // [优化] 返回具体找不到的文件名，方便排查
        return {"Spec File Not Found (" + spec_filename + ")", 0.0};
    }

    try {
        json specs = json::parse(f);
        std::string lower_name = to_lower(test_name); // Copy -> copy

        if (!specs.contains("stream")) return {"'stream' Block Missing", 0.0};
        if (!specs["stream"].contains(lower_name)) return {"Test Type Missing in Spec", 0.0};

        double spec_val = specs["stream"][lower_name];
        double threshold = spec_val * 0.75; // 75% 达标线

        if (measured_bw >= threshold) return {"Passed", spec_val};
        else return {"Failed", spec_val};

    } catch (...) {
        return {"Spec Parse Error", 0.0};
    }
}

// --- 1. 生成单条结果 JSON 对象 ---
json get_stream_result_json(const std::string& test_name,
                            double bytes_moved,
                            double ms,
                            double bw_gb_s,
                            int device,
                            const cudaDeviceProp& prop) {

    // [修改点] 动态获取 spec 文件名
    std::string spec_file = detect_stream_spec_filename(prop.name);

    // 1. 获取基础信息
    std::string driver_str = get_driver_version_smart();
    if (driver_str == "Unknown") {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        driver_str = "CUDA " + std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 1000) / 10);
    }
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // 2. 检查性能
    auto check_res = check_stream_performance(spec_file, test_name, bw_gb_s);
    std::string status = check_res.first;
    double spec_val = check_res.second;

    // 3. 构建单条结果 JSON 对象
    json j_entry;
    j_entry["test_name"] = test_name; 
    j_entry["status"] = status;
    
    j_entry["gpu_info"]["index"] = device;
    j_entry["gpu_info"]["name"] = prop.name;
    j_entry["gpu_info"]["target_spec_file"] = spec_file; // [新增] 记录使用的 spec 文件
    j_entry["gpu_info"]["compute_capability"] = cc;
    j_entry["gpu_info"]["driver_version"] = driver_str;

    j_entry["metrics"]["bytes_moved"] = bytes_moved;
    j_entry["metrics"]["elapsed_ms"] = ms;
    j_entry["metrics"]["bandwidth_gb_s"] = bw_gb_s;

    if (spec_val > 0.0) {
        j_entry["performance_check"]["spec_bw_gb_s"] = spec_val;
        j_entry["performance_check"]["threshold_75pct"] = spec_val * 0.75;
        j_entry["performance_check"]["percent_of_spec"] = (bw_gb_s / spec_val) * 100.0;
    }

    // 控制台输出增加 Spec 文件提示，方便调试
    // std::cout << "Debug: " << test_name << " on " << prop.name << " using " << spec_file << std::endl;

    return j_entry;
}

// --- 2. 批量保存函数 (保持不变) ---
void save_all_stream_results(const std::vector<json>& results, const std::string& filename) {
    std::ofstream o(filename);
    o << std::setw(4) << results << std::endl;
    o.close();
    std::cout << ">> All results saved to " << filename << " (Total records: " << results.size() << ")" << std::endl;
}
