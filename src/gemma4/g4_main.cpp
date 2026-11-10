// strata-gemma4: the Gemma 4 engine (the 26B-A4B mixture of experts) on its own.  --serve speaks the server's line protocol (g4_serve.hpp);
// --tokens runs a prompt of ids and prints what it generates (a check from the command line); --cpu runs the CPU reference (g4_ref.hpp).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "g4_model.hpp"
#include "g4_ref.hpp"
#include "g4_serve.hpp"
#ifdef STRATA_G4_CUDA
#include "g4_cuda.hpp"
#endif

#ifndef STRATA_G4_VERSION
#define STRATA_G4_VERSION "0.1.0-gemma4"
#endif

namespace {

/// The CPU reference behind the serve protocol's engine interface.
class CpuEngine {
public:
    CpuEngine(const g4::Model& m, int ctx, g4::RefOpts o) : m_(m), eng_(m, ctx, o) {}
    int ctx() const { return eng_.ctx(); }
    int n_vocab() const { return m_.hp.n_vocab; }
    int64_t pos() const { return eng_.pos(); }
    void reset() { eng_.reset(); }
    void truncate(int64_t p) { eng_.truncate(p); }
    bool can_keep(int64_t) const { return true; }       // every cell is kept (the reference caches the whole context for every layer)
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
private:
    const g4::Model& m_;
    g4::RefEngine eng_;
    std::vector<float> last_;
};

void usage() {
    std::fprintf(stderr,
        "usage: strata-gemma4 --model GGUF [--serve | --tokens ID,ID,... [--n N]] [--max-context N] [--cpu] [--gpu 0,1] [--eos ID,ID,...]\n"
        "                     [--dump-logits FILE]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, dump, gpus;
    int ctx = 8192, n_gen = 32;
    bool serve = false, cpu = false;
    std::set<int32_t> eos = {1, 50, 106};     // <eos>, <|tool_response> (the model hands over to the tool), <turn|>
    bool eos_given = false;
    std::vector<std::string> extra;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-gemma4: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--serve") serve = true;
        else if (a == "--cpu") cpu = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--tokens") tokens = next();
        else if (a == "--n" || a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--dump-logits") dump = next();
        else if (a == "--gpu") gpus = next();
        else if (a == "--eos") {
            if (!eos_given) { eos.clear(); eos_given = true; }
            for (int32_t t : g4::parse_ids(next().c_str())) eos.insert(t);
        } else if (a == "--help" || a == "-h") { usage(); return 0; }
        else extra.push_back(a);      // (the server's other engine flags: --prefill, --kv, --mtp ...: this engine has no such thing; ignored below)
    }
    if (model.empty()) { usage(); return 2; }
    g4::Model m;
    const std::string err = g4::load_model(model, m);
    if (!err.empty()) { std::fprintf(stderr, "strata-gemma4: %s\n", err.c_str()); return 1; }
    if (m.hp.eos > 0) eos.insert(m.hp.eos);
    if (ctx <= 0) ctx = 8192;
    ctx = std::min(ctx, m.hp.ctx_train > 0 ? m.hp.ctx_train : ctx);
    std::fprintf(stderr, "strata-gemma4: %s: %d layers (%d sliding-window, window %d), %d heads, n_embd %d, %d experts of which %d per token, vocab %d\n",
                 m.name.c_str(), m.hp.n_layer, m.hp.n_swa_layers(), m.hp.window, m.hp.n_head, m.hp.n_embd, m.hp.n_expert, m.hp.n_used, m.hp.n_vocab);
    g4::RefOpts ro;
    ro.threads = std::max(1u, std::thread::hardware_concurrency());
    if (!cpu) {
#ifdef STRATA_G4_CUDA
        return g4cuda::run(m, ctx, serve, tokens, n_gen, dump, gpus, eos, extra);
#else
        std::fprintf(stderr, "strata-gemma4: this build has no CUDA engine: running the CPU reference (slow)\n");
#endif
    }
    if (!extra.empty()) {
        std::string s;
        for (const auto& e : extra) s += " " + e;
        std::fprintf(stderr, "strata-gemma4: the CPU reference ignores:%s\n", s.c_str());
    }
    CpuEngine eng(m, ctx, ro);
    if (serve) {
        g4::Serve<CpuEngine> sv(eng, eos, STRATA_G4_VERSION);
        return sv.run();
    }
    if (tokens.empty()) { usage(); return 2; }
    const std::vector<int32_t> ids = g4::parse_ids(tokens.c_str());
    std::vector<float> lg;
    int64_t done = 0;
    if (!eng.process(ids.data(), (int64_t) ids.size(), [](int64_t) { return true; }, &done)) { std::fprintf(stderr, "strata-gemma4: the prompt does not fit the context\n"); return 1; }
    eng.logits(lg);
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        std::fwrite(lg.data(), 4, lg.size(), f);
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
        eng.logits(lg);
    }
    std::printf("\n");
    return 0;
}
