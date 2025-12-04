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

// --- 核心逻辑：检查性能是否达标 (保持不变) ---
static std::pair<std::string, double> check_stream_performance(const std::string& spec_filename, 
                                                               const std::string& test_name, 
                                                               double measured_bw) {
    std::ifstream f(spec_filename);
    if (!f.is_open()) return {"Spec File Not Found", 0.0};

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

    const std::string spec_file = "B200_specs.json";

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

    return j_entry;
}

// --- 2. 批量保存函数 ---
void save_all_stream_results(const std::vector<json>& results, const std::string& filename) {
    std::ofstream o(filename);
    o << std::setw(4) << results << std::endl;
    o.close();
    std::cout << ">> All results saved to " << filename << " (Total records: " << results.size() << ")" << std::endl;
}
