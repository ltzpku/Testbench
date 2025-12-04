#include "nccl_json_writer.h"
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

// 全局静态变量，用于追踪最大带宽
static double g_max_bus_bw = 0.0;

// [已移除] static const double PASS_THRESHOLD_RATIO = 1.0; 

void nccl_init_bw_tracker() {
    g_max_bus_bw = 0.0;
}

void nccl_record_bw(double bus_bw) {
    if (bus_bw > g_max_bus_bw) {
        g_max_bus_bw = bus_bw;
    }
}

// --- 辅助函数：获取驱动版本 ---
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
    if (version == "Unknown") {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion > 0) {
            version = std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 1000) / 10);
        }
    }
    return version;
}

static std::string to_lower(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

// --- 检查性能 (增加 threshold_ratio 参数) ---
static std::pair<std::string, double> check_nccl_performance(const std::string& spec_filename, 
                                                             const std::string& test_name, 
                                                             double measured_bw,
                                                             double threshold_ratio) {
    std::ifstream f(spec_filename);
    if (!f.is_open()) return {"Spec File Not Found", 0.0};

    try {
        json specs = json::parse(f);
        std::string lower_name = to_lower(test_name); 

        if (!specs.contains("nccl")) return {"'nccl' Block Missing", 0.0};
        if (!specs["nccl"].contains(lower_name)) return {"Test Type Missing in Spec", 0.0};

        double spec_val = specs["nccl"][lower_name];
        
        // 使用传入的比例计算阈值
        double threshold = spec_val * threshold_ratio; 

        if (measured_bw >= threshold) return {"Passed", spec_val};
        else return {"Failed", spec_val};

    } catch (...) {
        return {"Spec Parse Error", 0.0};
    }
}

// --- 主写入函数 (增加 filename 和 threshold_ratio) ---
void nccl_write_json(const std::string& test_name, 
                     int root, 
                     const std::vector<int>& gpu_indices,
                     const std::string& output_filename,
                     double threshold_ratio) {
    
    int representative_device = 0;
    if (!gpu_indices.empty()) {
        representative_device = gpu_indices[0];
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, representative_device);

    // 使用传入的文件名，不再硬编码 result.json
    const std::string filename = output_filename;
    const std::string spec_file = "B200_specs.json";

    std::string driver_str = get_driver_version_smart();
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // 检查性能，传入特定的 ratio
    auto check_res = check_nccl_performance(spec_file, test_name, g_max_bus_bw, threshold_ratio);
    std::string status = check_res.first;
    double spec_val = check_res.second;

    // 构建 JSON
    json j_entry;
    j_entry["test_name"] = test_name;
    j_entry["status"] = status;
    j_entry["gpu_info"]["index"] = gpu_indices; 
    j_entry["gpu_info"]["name"] = prop.name;
    j_entry["gpu_info"]["compute_capability"] = cc;
    j_entry["gpu_info"]["driver_version"] = driver_str;

    j_entry["metrics"]["max_out_of_place_bus_bw_gb_s"] = g_max_bus_bw;

    if (spec_val > 0.0) {
        j_entry["performance_check"]["spec_bw_gb_s"] = spec_val;
        j_entry["performance_check"]["threshold_ratio"] = threshold_ratio; // 记录使用的比例
        j_entry["performance_check"]["threshold_val"] = spec_val * threshold_ratio;
        j_entry["performance_check"]["percent_of_spec"] = (g_max_bus_bw / spec_val) * 100.0;
    }

    // 追加逻辑
    json j_root;
    std::ifstream ifile(filename);
    if (ifile.good()) {
        try {
            j_root = json::parse(ifile);
        } catch (...) { j_root = json::array(); }
    } else {
        j_root = json::array();
    }
    ifile.close();

    if (!j_root.is_array()) {
        json temp = json::array();
        temp.push_back(j_root);
        j_root = temp;
    }

    j_root.push_back(j_entry);

    std::ofstream o(filename);
    o << std::setw(4) << j_root << std::endl;
    o.close();

    std::cout << ">> [JSON Writer] Max BusBW (" << g_max_bus_bw << " GB/s) saved to " << filename << " (Status: " << status << ", Threshold: " << (threshold_ratio*100) << "%)" << std::endl;
}
