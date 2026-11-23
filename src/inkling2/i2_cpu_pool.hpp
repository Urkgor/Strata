// i2_cpu_pool.hpp - experts evaluated on the CPU: the decode step's cold experts (the ones that are not in the GPU's expert cache) are computed from the host
// copy while the GPU runs the others, so that only the 2304 activations cross PCIe instead of the 4.5 MB of an expert's weights.
//
// y = down(SITU(gate_up(x))) for each job, in float, from the weights as the file stores them (the same arithmetic as the GPU kernels, up to the order of the
// sums): IQ4_NL and Q8_0 rows have AVX2 + FMA kernels (chosen at run time), every other type goes through g4::dequant32_t.  The jobs are cut into rows
// shared by the threads; the calling thread works too.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace i2cpu {

struct Job {
    const uint8_t* gu = nullptr;     // [2F rows][E columns]: F gate rows, then F up rows
    int gu_type = 0;
    size_t gu_row_bytes = 0;
    const uint8_t* dn = nullptr;     // [E rows][F columns]
    int dn_type = 0;
    size_t dn_row_bytes = 0;
    float* y = nullptr;              // [E], the expert's output (without the router's weight, and without the down scalar: the GPU's sum applies it)
    float gu_scale = 1.f;            // the expert's scalar on the gate+up product (NVFP4 files)
};

/// the dot product of one weight row of `n` values (type `type`) with x
float dot_row(int type, const uint8_t* row, const float* x, int n);
/// the same through the portable path (g4::dequant32_t), for the tests
float dot_row_generic(int type, const uint8_t* row, const float* x, int n);
bool have_avx2();

class Pool {
public:
    explicit Pool(int threads);                // threads: the workers; the caller works too
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    int threads() const { return (int) workers_.size() + 1; }
    void run(const float* x, int E, int F, float beta, float lin, const Job* jobs, int n);

private:
    void parallel_for(size_t total, size_t chunk, const std::function<void(size_t, size_t)>& fn);
    void worker();

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> sleepers_{0};
    std::atomic<bool> stop_{false};
    // the work of the current generation
    const std::function<void(size_t, size_t)>* fn_ = nullptr;
    size_t total_ = 0, chunk_ = 1;
    std::atomic<size_t> next_{0};
    std::atomic<int> done_{0};
};

}  // namespace i2cpu
