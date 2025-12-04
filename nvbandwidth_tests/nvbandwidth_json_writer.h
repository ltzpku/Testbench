#ifndef NVBANDWIDTH_JSON_WRITER_H
#define NVBANDWIDTH_JSON_WRITER_H

#include <string>
#include <vector>
#include <cuda_runtime.h>

// Initializes the bandwidth tracker (clears previous samples)
void nvbandwidth_init_bw_tracker();

// Records a bandwidth sample (in GB/s)
// Call this inside your test loop or where the bandwidth is calculated
void nvbandwidth_record_bw(double bus_bw);

// Writes the collected results to result.json
// Compares average bandwidth against B200_specs.json with 90% threshold
void nvbandwidth_write_json(const std::string& test_name, const std::vector<int>& gpu_indices);

#endif // NVBANDWIDTH_JSON_WRITER_H
