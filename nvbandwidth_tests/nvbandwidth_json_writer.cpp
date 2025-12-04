#include "nvbandwidth_json_writer.h"
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <sys/stat.h>
#include "json.hpp" // Ensure json.hpp is in the include path

using json = nlohmann::ordered_json;

// Global variable to store all bandwidth samples for the current test
static std::vector<double> g_bw_samples;

// Configuration: Pass threshold is 90% of the Spec value (根据你原来的逻辑保留 0.8 或者改成 0.9)
static const double PASS_THRESHOLD_RATIO = 0.80;

void nvbandwidth_init_bw_tracker() {
    g_bw_samples.clear();
}

void nvbandwidth_record_bw(double bus_bw) {
    g_bw_samples.push_back(bus_bw);
}

// === 新增：根据设备名称动态生成 Spec 文件名 ===
std::string detect_spec_filename(std::string device_name) {
    // 1. 定义已知的高频型号列表 (按照你的需求更新列表)
    // 优先匹配更具体的型号
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

// Helper: Get Driver Version
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
    // Fallback
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

// Check performance against spec
static std::pair<std::string, double> check_performance(const std::string& spec_filename,
                                                        const std::string& test_name,
                                                        double avg_bw) {
    std::ifstream f(spec_filename);
    if (!f.is_open()) {
        // [优化] 返回具体找不到的文件名，方便排查
        return {"Spec File Not Found (" + spec_filename + ")", 0.0};
    }

    try {
        json specs = json::parse(f);
        std::string lower_name = to_lower(test_name);

        double spec_val = 0.0;
        bool found = false;

        // Logic: Support structure { "nvbandwidth": { ... } } or flat structure
        if (specs.contains("nvbandwidth") && specs["nvbandwidth"].contains(lower_name)) {
            spec_val = specs["nvbandwidth"][lower_name];
            found = true;
        } else if (specs.contains(lower_name)) {
             spec_val = specs[lower_name];
             found = true;
        }

        if (!found) return {"Test Missing in Spec", 0.0};

        double threshold = spec_val * PASS_THRESHOLD_RATIO;

        if (avg_bw >= threshold) return {"Passed", spec_val};
        else return {"Failed", spec_val};

    } catch (...) {
        return {"Spec Parse Error", 0.0};
    }
}

void nvbandwidth_write_json(const std::string& test_name, const std::vector<int>& gpu_indices) {
    // 1. Calculate Statistics
    double avg_bw = 0.0;
    if (!g_bw_samples.empty()) {
        double sum = std::accumulate(g_bw_samples.begin(), g_bw_samples.end(), 0.0);
        avg_bw = sum / g_bw_samples.size();
    }

    // 2. Get Device Info (Representative)
    int representative_device = 0;
    if (!gpu_indices.empty()) representative_device = gpu_indices[0];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, representative_device);

    // [修改点] 动态获取 specs 文件名，不再硬编码 "B200_specs.json"
    std::string spec_file = detect_spec_filename(prop.name);
    const std::string filename = "nvbandwidth_result.json";

    std::string driver_str = get_driver_version_smart();
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    // 3. Check Performance
    auto check_res = check_performance(spec_file, test_name, avg_bw);
    std::string status = check_res.first;
    double spec_val = check_res.second;

    // 4. Build JSON Entry
    json j_entry;
    j_entry["test_name"] = test_name;
    j_entry["status"] = status;

    j_entry["gpu_info"]["index"] = gpu_indices;
    j_entry["gpu_info"]["name"] = prop.name;
    j_entry["gpu_info"]["target_spec_file"] = spec_file; // [新增] 记录实际使用的 specs 文件，方便 Debug
    j_entry["gpu_info"]["compute_capability"] = cc;
    j_entry["gpu_info"]["driver_version"] = driver_str;

    // Metrics: Average first, then All Samples
    j_entry["metrics"]["average_bw_gb_s"] = avg_bw;
    j_entry["metrics"]["all_samples_gb_s"] = g_bw_samples;

    // Performance Check Details
    if (spec_val > 0.0) {
        j_entry["performance_check"]["spec_bw_gb_s"] = spec_val;
        j_entry["performance_check"]["threshold_val"] = spec_val * PASS_THRESHOLD_RATIO;
        j_entry["performance_check"]["percent_of_spec"] = (avg_bw / spec_val) * 100.0;
    }

    // 5. Append to JSON Array in file
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

    std::cout << ">> [JSON Writer] " << test_name << " Avg BW: " << avg_bw
              << " GB/s saved to " << filename << " (Spec: " << spec_file << ", Status: " << status << ")" << std::endl;
}
