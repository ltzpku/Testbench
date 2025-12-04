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

void nccl_init_bw_tracker() {
    g_max_bus_bw = 0.0;
}

void nccl_record_bw(double bus_bw) {
    if (bus_bw > g_max_bus_bw) {
        g_max_bus_bw = bus_bw;
    }
}

// === 新增：根据设备名称动态生成 Spec 文件名 ===
std::string detect_nccl_spec_filename(std::string device_name) {
    // 1. 定义已知的高频型号列表
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
    std::string safe_name = device_name;
    std::replace(safe_name.begin(), safe_name.end(), ' ', '_');
    return safe_name + "_specs.json";
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
    if (!f.is_open()) {
        // [优化] 返回具体找不到的文件名，方便排查
        return {"Spec File Not Found (" + spec_filename + ")", 0.0};
    }

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

    // [修改点] 动态获取 specs 文件名，不再硬编码 "B200_specs.json"
    std::string spec_file = detect_nccl_spec_filename(prop.name);
    const std::string filename = output_filename;

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
    j_entry["gpu_info"]["target_spec_file"] = spec_file; // [新增] 记录实际使用的 specs 文件
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

    // 控制台输出增加 Spec 文件提示，方便调试
    std::cout << ">> [JSON Writer] Max BusBW (" << g_max_bus_bw << " GB/s) saved to " << filename 
              << " (Spec: " << spec_file << ", Status: " << status << ", Threshold: " << (threshold_ratio*100) << "%)" << std::endl;
}
