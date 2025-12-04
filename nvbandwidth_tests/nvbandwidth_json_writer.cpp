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

// Configuration: Pass threshold is 90% of the Spec value
static const double PASS_THRESHOLD_RATIO = 0.80;

void nvbandwidth_init_bw_tracker() {
    g_bw_samples.clear();
}

void nvbandwidth_record_bw(double bus_bw) {
    g_bw_samples.push_back(bus_bw);
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
    if (!f.is_open()) return {"Spec File Not Found", 0.0};

    try {
        json specs = json::parse(f);
        std::string lower_name = to_lower(test_name);

        // Logic: Look for "nvbandwidth" block, fallback to "nccl" or root if needed. 
        // Assuming structure similar to: { "nvbandwidth": { "host_to_device...": 100.0 } }
        // Or if the user puts them in the same "nccl" block or root, adjust here.
        // For now, checking root or "nvbandwidth" key.
        
        double spec_val = 0.0;
        bool found = false;

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

    const std::string filename = "nvbandwidth_result.json";
    const std::string spec_file = "B200_specs.json";

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
              << " GB/s saved to " << filename << " (Status: " << status << ")" << std::endl;
}
