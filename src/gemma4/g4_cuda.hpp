// g4_cuda.hpp - the Gemma 4 CUDA engine's interface (g4_cuda.cu).  Also built for the CPU emulation of CUDA (g4_emu.hpp) by the tests.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "g4_model.hpp"

namespace g4cuda {

struct GpuOptions {
    std::vector<int> devices;      // CUDA device ordinals in layer order; empty: every visible device, the one with the most free memory first
    int chunk = 512;               // tokens a prompt is read in at a time
    int slack = 1024;              // cells a sliding-window layer keeps beyond its window + one chunk, so that a conversation can step back this far
    int reserve_mib = 700;         // VRAM left free on each device (the driver's context, the allocator)
    int host_experts = 0;          // keep the experts of (at least) the last N layers in host memory, even if they would fit (a check of the streaming path)
    int stage_slots = 32;          // experts of a streaming layer copied to the card at a time
    bool verbose = true;
};

class GpuEngine {
public:
    GpuEngine(const g4::Model& m, int ctx, GpuOptions opt);
    ~GpuEngine();
    /// Choose the devices' layers, allocate and upload.  "" or the reason it cannot run (the VRAM that is missing, a type the kernels lack).
    std::string init();
    int ctx() const;
    int n_vocab() const;
    int64_t pos() const;
    void reset();
    void truncate(int64_t p);
    bool can_keep(int64_t p) const;
    /// the number of layers whose experts stay in host memory (valid after init())
    int streamed_layers() const;
    /// Append n tokens; the logits of the last one are made at the end.  on_piece(done) after each chunk (false: stop reading).
    bool process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& on_piece, int64_t* done);
    void logits(std::vector<float>& out);
    int greedy();
    /// test hook: the hidden state (n_embd floats of the last token) after the layers [0, il] of the last chunk (il = -1: the embedding)
    std::vector<float> debug_hidden(int il);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

/// strata-gemma4's main for a CUDA build: serve, run a prompt, check against the CPU reference, or benchmark.
int run(const g4::Model& m, int ctx, bool serve, const std::string& tokens, int n_gen, const std::string& dump, const std::string& gpus,
        const std::set<int32_t>& eos, const std::vector<std::string>& extra);

}  // namespace g4cuda
