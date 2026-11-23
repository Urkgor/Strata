// i2_cuda.hpp - the Inkling 2 Mini CUDA engine's interface (i2_cuda.cu).  Also built for the CPU emulation of CUDA (g4_emu.hpp) by the tests.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "i2_model.hpp"

namespace i2cuda {

struct GpuOptions {
    int device = 0;                // the CUDA device (one card: the layers, the visits' caches and the experts all live on it)
    int chunk = 512;               // tokens a prompt is read in at a time
    int slack = 1024;              // cells a sliding-window layer keeps beyond its window + one chunk, so that a conversation can step back this far
    int reserve_mib = 700;         // VRAM left free (the driver's context, the allocator)
    int host_experts = 0;          // keep the experts of (at least) this many layers in host memory, even if they would fit (a check of the streaming path)
    int stage_slots = 32;          // experts of a streaming layer copied to the card at a time (prompt reading)
    int cache_mib = -1;            // VRAM for the cache of the streamed layers' experts (decode): -1 = what is left, 0 = none
    bool profile = false;          // time the phases of a decoded token (--profile)
    int cache_slots = -1;          // the same in experts, for tests (-1: from cache_mib)
    int cpu_threads = -1;          // the CPU expert pool (decode: cold experts computed on the CPU instead of copied): -1 = on when a speed check says it beats the copy, 0 = off, N = N threads
    bool cpu_force = false;        // use the pool without the speed check (tests)
    double cache_decay = 0.97;     // an expert's use count is multiplied by this at every decoded token: the cache keeps the most used
    bool verbose = true;
};

class GpuEngine {
public:
    GpuEngine(const i2::Model& m, int ctx, GpuOptions opt);
    ~GpuEngine();
    /// Place the layers' experts (all on the card, or the last ones in host memory), allocate and upload.  "" or the reason it cannot run.
    std::string init();
    int ctx() const;
    int n_vocab() const;
    int64_t pos() const;
    void reset();
    void truncate(int64_t p);
    bool can_keep(int64_t p) const;
    /// the number of physical layers whose experts stay in host memory (valid after init())
    int streamed_layers() const;
    /// --profile's table (empty without it)
    std::string profile_report() const;
    /// the expert cache of the streamed layers: its slots, and the hits / misses of the decode steps so far
    void cache_stats(int* slots, long long* hits, long long* misses, long long* cpu_experts = nullptr, long long* shared = nullptr) const;
    /// Append n tokens; the logits of the last one are made at the end.  on_piece(done) after each chunk (false: stop reading).
    bool process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& on_piece, int64_t* done);
    void logits(std::vector<float>& out);
    int greedy();
    /// test hook: the stream (last token of a chunk) after every visit
    void set_trace(std::vector<std::vector<float>>* t, std::vector<std::vector<int>>* sel = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// strata-inkling2's main for a CUDA build: serve, run a prompt, check against the CPU reference, or benchmark.
int run(const i2::Model& m, int ctx, bool serve, const std::string& tokens, int n_gen, const std::string& dump, const std::string& gpus,
        const std::set<int32_t>& eos, const std::vector<std::string>& extra);

}  // namespace i2cuda
