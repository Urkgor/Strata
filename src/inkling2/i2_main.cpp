// strata-inkling2: the Inkling 2 Mini engine on its own.  --serve speaks the server's line protocol (the Gemma 4 engine's g4_serve.hpp, which is
// engine-neutral); --tokens runs a prompt of ids and prints what it generates (a check from the command line); --cpu runs the CPU reference.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "i2_model.hpp"
#include "i2_ref.hpp"
#include "../gemma4/g4_serve.hpp"
#ifdef STRATA_I2_CUDA
#include "i2_cuda.hpp"
#endif

#ifndef STRATA_I2_VERSION
#define STRATA_I2_VERSION "0.1.0-inkling2"
#endif

namespace {

/// The CPU reference behind the serve protocol's engine interface.
class CpuEngine {
public:
    CpuEngine(const i2::Model& m, int ctx, i2::RefOpts o) : m_(m), eng_(m, ctx, o) {}
    int ctx() const { return eng_.ctx(); }
    int n_vocab() const { return m_.hp.n_vocab; }
    int64_t pos() const { return eng_.pos(); }
    void reset() { eng_.reset(); }
    void truncate(int64_t p) { eng_.truncate(p); }
    bool can_keep(int64_t) const { return true; }       // the reference keeps every position of every visit
    bool process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& on_piece, int64_t* done) {
        int64_t i = 0;
        const int64_t piece = 16;
        *done = 0;
        while (i < n) {
            const int64_t k = std::min(piece, n - i);
            if (!eng_.forward(ids + i, (int) k, &last_)) return false;
            i += k;
            *done = i;
            if (!on_piece(i)) return i >= n;
        }
        return true;
    }
    void logits(std::vector<float>& out) { out = last_; }
    int greedy() { return (int) (std::max_element(last_.begin(), last_.end()) - last_.begin()); }
    i2::RefEngine& ref() { return eng_; }

private:
    const i2::Model& m_;
    i2::RefEngine eng_;
    std::vector<float> last_;
};

void usage() {
    std::fprintf(stderr,
                 "usage: strata-inkling2 --model GGUF [--serve | --tokens ID,ID,... [--n N]] [--max-context N] [--cpu] [--gpu 0,1] [--eos ID,ID,...]\n"
                 "                       [--dump-logits FILE [--all]] [--loop-passes N]\n"
                 "       GPU: [--prefill N] [--swa-slack N] [--vram-reserve-mib N] [--host-experts N] [--expert-cache-mib N] [--cpu-experts auto|N] [--cpu-int8 on|off|auto] [--cache-decay X] [--profile] [--dp4a on|off] [--check] [--bench-pp N] [--bench-tg N] [--bench-warmup N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, dump, gpus;
    int ctx = 8192, n_gen = 32, passes = 0;
    bool serve = false, cpu = false, all = false;
    std::set<int32_t> eos;
    bool eos_given = false;
    std::vector<std::string> extra;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-inkling2: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--serve") serve = true;
        else if (a == "--cpu") cpu = true;
        else if (a == "--all") all = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--dump-logits") dump = next();
        else if (a == "--loop-passes") passes = std::atoi(next().c_str());
        else if (a == "--gpu") gpus = next();
        else if (a == "--eos") {
            eos.clear();
            for (int32_t t : g4::parse_ids(next().c_str())) eos.insert(t);
            eos_given = true;
        } else if (a == "--help" || a == "-h") { usage(); return 0; }
        else extra.push_back(a);
    }
    if (model.empty()) { usage(); return 2; }
    i2::Model m;
    const std::string err = i2::load_model(model, m);
    if (!err.empty()) { std::fprintf(stderr, "strata-inkling2: %s\n", err.c_str()); return 1; }
    if (passes > 0) {
        if (!m.hp.varying && passes != m.hp.passes) { std::fprintf(stderr, "strata-inkling2: --loop-passes: this model loops %d times (only the variable-loop checkpoint takes another count)\n", m.hp.passes); return 2; }
        m.hp.passes = passes;
    }
    if (!eos_given) eos = {m.hp.eos};
    std::fprintf(stderr, "strata-inkling2: %s: %d layers, %d visits, %d experts (%d used), vocabulary %d\n", m.name.c_str(), m.hp.n_layer, m.hp.visits(), m.hp.n_expert,
                 m.hp.n_used, m.hp.n_vocab);
#ifdef STRATA_I2_CUDA
    if (!cpu) return i2cuda::run(m, ctx, serve, tokens, n_gen, dump, gpus, eos, extra);
#else
    if (!cpu) std::fprintf(stderr, "strata-inkling2: built without CUDA: running the CPU reference\n");
#endif
    i2::RefOpts ro;
    ro.threads = (int) std::max(1u, std::thread::hardware_concurrency());
    CpuEngine eng(m, ctx, ro);
    if (serve) {
        g4::Serve<CpuEngine> sv(eng, eos, STRATA_I2_VERSION);
        return sv.run();
    }
    if (tokens.empty()) { usage(); return 2; }
    const std::vector<int32_t> ids = g4::parse_ids(tokens.c_str());
    std::vector<float> lg, every;
    if (all) {
        if (!eng.ref().forward(ids.data(), (int) ids.size(), &lg, &every)) { std::fprintf(stderr, "strata-inkling2: the prompt does not fit the context\n"); return 1; }
    } else {
        int64_t done = 0;
        if (!eng.process(ids.data(), (int64_t) ids.size(), [](int64_t) { return true; }, &done)) { std::fprintf(stderr, "strata-inkling2: the prompt does not fit the context\n"); return 1; }
        eng.logits(lg);
    }
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        const std::vector<float>& src = all ? every : lg;
        std::fwrite(src.data(), 4, src.size(), f);
        std::fclose(f);
    }
    std::printf("generated:");
    for (int i = 0; i < n_gen; ++i) {
        const int nx = eng.greedy();
        std::printf(" %d", nx);
        if (eos.count(nx)) break;
        int64_t d = 0;
        const int32_t one = nx;
        if (!eng.process(&one, 1, [](int64_t) { return true; }, &d)) break;
    }
    std::printf("\n");
    return 0;
}
