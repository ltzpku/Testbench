#ifndef STREAM_JSON_WRITER_H
#define STREAM_JSON_WRITER_H

#include <string>
#include <cuda_runtime.h>

/**
 * @brief 将 STREAM 测试结果写入/追加到 stream_result.json
 * * @param test_name 测试名称 (如 "Copy", "Scale", "Add", "Triad")
 * @param bytes_moved 移动的总字节数
 * @param ms 耗时 (毫秒)
 * @param bw_gb_s 带宽 (GB/s)
 * @param device GPU 设备 ID
 * @param prop GPU 属性结构体
 */
void write_stream_result_json(const std::string& test_name,
                              double bytes_moved,
                              double ms,
                              double bw_gb_s,
                              int device,
                              const cudaDeviceProp& prop);

#endif // STREAM_JSON_WRITER_H
