#ifndef NCCL_JSON_WRITER_H
#define NCCL_JSON_WRITER_H

#include <string>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief 初始化带宽追踪器
 */
void nccl_init_bw_tracker();

/**
 * @brief 记录并更新最大 Bus Bandwidth
 */
void nccl_record_bw(double bus_bw);

/**
 * @brief 将最大带宽结果写入 JSON 文件
 * @param test_name 测试名称 (如 "all_reduce", "alltoall")
 * @param root Root ID
 * @param gpu_indices GPU ID 列表
 * @param output_filename 输出的文件名 (新增)
 * @param threshold_ratio 达标阈值比例 (如 0.9 或 0.7) (新增)
 */
void nccl_write_json(const std::string& test_name, 
                     int root, 
                     const std::vector<int>& gpu_indices,
                     const std::string& output_filename,
                     double threshold_ratio);

#endif // NCCL_JSON_WRITER_H
