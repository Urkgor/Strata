// The Gemma 4 CUDA engine (src/gemma4/g4_cuda.cu, its kernels and its host code) run on the CPU emulation of CUDA (g4_emu.hpp) and compared with
// the CPU reference (g4_ref.hpp) on a GGUF of the tiny random model (tools/gemma4/make_tiny.py).  No GPU.  The scenarios are the engine's paths:
// one chunk; chunks smaller than the prompt with a sliding-window ring that wraps; one token at a time; the layers spread over two devices;
// a step back (truncate) and a re-read.  Build: -DSTRATA_G4_EMU (see tools/gemma4/README.md); run under AddressSanitizer for the memory checks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "g4_cuda.hpp"
#include "g4_ref.hpp"

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } else { std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static double rel_rms(const std::vector<float>& a, const float* b) {
    double d = 0, s = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double) (a[i] - b[i]) * (a[i] - b[i]); m += a[i]; }
    m /= (double) a.size();
    for (size_t i = 0; i < a.size(); ++i) s += (double) (a[i] - m) * (a[i] - m);
    return std::sqrt(d / std::max(s, 1e-30));
}
static int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: g4_emu_test tiny.gguf [quick]\n"); return 2; }
    const bool quick = argc > 2;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    g4::Model m;
    const std::string err = g4::load_model(argv[1], m);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const std::vector<int32_t> ids = {2, 259, 164, 302, 88, 105, 364, 107, 257, 61, 106, 107, 105, 368, 107, 59, 402, 288};   // 18 tokens, window 8 (a vocabulary of 512)
    const int n = (int) ids.size(), nv = m.hp.n_vocab;
    // the reference: float arithmetic, K / V rounded to half as the engine stores them
    g4::RefOpts ro;
    ro.cache_f16 = true;
    ro.threads = 4;
    g4::RefEngine ref(m, 64, ro);
    std::vector<float> all;
    ref.forward(ids.data(), n, nullptr, &all);
    auto row = [&](int t) { return all.data() + (size_t) t * (size_t) nv; };

    auto make = [&](int chunk, int slack, std::vector<int> devs, int host_experts = 0, int stage_slots = 32) {
        g4cuda::GpuOptions o;
        o.host_experts = host_experts;
        o.stage_slots = stage_slots;
        o.chunk = chunk;
        o.slack = slack;
        o.reserve_mib = 0;
        o.devices = devs;
        o.verbose = false;
        auto e = std::make_unique<g4cuda::GpuEngine>(m, 64, o);
        const std::string er = e->init();
        if (!er.empty()) { std::fprintf(stderr, "init: %s\n", er.c_str()); std::exit(1); }
        return e;
    };
    auto noop = [](int64_t) { return true; };
    std::vector<float> lg;

    {   // 1. the whole prompt in one chunk
        auto e = make(32, 1024, {0});
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4 && argmax(lg) == argmax(std::vector<float>(row(n - 1), row(n - 1) + nv)), "one chunk of %d tokens: logits within %.2e (rms / spread) of the reference, same best token", n, r);
        EXPECT(e->greedy() == argmax(lg), "greedy() is the arg-max of logits()");
    }
    {   // 2. chunks of 4 with a ring of window + chunk + slack = 8 + 4 + 2 = 14 cells: the sliding-window layers wrap during the prompt
        auto e = make(4, 2, {0});
        int64_t done = 0;
        int pieces = 0;
        e->process(ids.data(), n, [&](int64_t) { ++pieces; return true; }, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4 && done == n && pieces == 5, "chunks of 4 over a 14-cell ring (%d pieces): within %.2e", pieces, r);
    }
    if (!quick) {   // 3. one token at a time after a short prompt: every step against the reference's row
        auto e = make(8, 1024, {0});
        int64_t done = 0;
        e->process(ids.data(), 5, noop, &done);
        double worst = 0;
        bool same = true;
        for (int t = 4; t < n; ++t) {
            if (t > 4) e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
            same &= e->greedy() == argmax(lg);
        }
        EXPECT(worst < 2e-4 && same, "decode, one token at a time for %d steps: worst %.2e", n - 4, worst);
    }
    {   // 4. two devices (the layers split between them, the output head on the last one)
        setenv("STRATA_G4_EMU_FREE_MB", "6,8", 1);
        auto e = make(8, 1024, {0, 1});
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4, "two devices: within %.2e", r);
    }
    {   // 5. a step back: read 14 tokens, return to 9 (inside the ring's slack), read the rest again: the same logits as reading straight through
        auto e = make(4, 6, {0});
        int64_t done = 0;
        e->process(ids.data(), 14, noop, &done);
        EXPECT(e->can_keep(9) && e->can_keep(14) && !e->can_keep(15), "can_keep: 9 and 14 yes, 15 (past the end) no");
        e->truncate(9);
        e->process(ids.data() + 9, n - 9, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4 && e->pos() == n, "truncate to 9 and re-read: within %.2e", r);
    }
    {   // 6. a step back past the ring: refused.  18 tokens in a ring of window + chunk + slack = 8 + 4 + 2 = 14 cells hold the positions 4 .. 17;
        // the token at position q needs the keys from q - 7 on, so q >= 11
        auto e = make(4, 2, {0});
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        EXPECT(!e->can_keep(2) && !e->can_keep(10) && e->can_keep(11) && e->can_keep(18), "can_keep after a wrap: 2 and 10 refused, 11 and 18 kept (window 8, ring 14, 18 tokens)");
        e->truncate(11);
        e->process(ids.data() + 11, n - 11, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4, "truncate to 11 (the farthest back) after a wrap and re-read: within %.2e", r);
    }
    {   // 7. experts kept in host memory (the last 3 layers) and copied to the card per token: prompt in chunks (batches of 4 staging slots over 8 experts
        // or more per layer) then one token at a time
        auto e = make(8, 1024, {0}, 3, 4);
        EXPECT(e->streamed_layers() == 3, "3 layers streamed (%d)", e->streamed_layers());
        int64_t done = 0;
        e->process(ids.data(), n - 3, noop, &done);
        double worst = 0;
        for (int t = n - 3; t < n; ++t) {
            e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
        }
        EXPECT(worst < 2e-4, "3 layers' experts in host memory, 4 staging slots, prompt then 3 single tokens: worst %.2e", worst);
        e->profile(true);                                    // the timed pass changes nothing in the result, and reports the copies
        e->process(&ids[(size_t) n - 1], 1, noop, &done);
        const std::string rep = e->profile_report();
        e->logits(lg);
        EXPECT(rep.find("expert copies") != std::string::npos && rep.find(std::to_string(3 * m.hp.n_used) + " copies") != std::string::npos && rep.find("1 steps") != std::string::npos,
               "profile: one step, 3 streaming layers x %d experts = %d copies", m.hp.n_used, 3 * m.hp.n_used);
    }
    {   // 8. every layer streamed, one chunk, the whole prompt at once (the largest batches)
        auto e = make(32, 1024, {0}, 1000, 32);
        EXPECT(e->streamed_layers() == m.hp.n_layer, "all %d layers streamed (%d)", m.hp.n_layer, e->streamed_layers());
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4, "every layer's experts in host memory, one chunk: within %.2e", r);
    }
    {   // 9. two devices with streamed layers on the last one
        setenv("STRATA_G4_EMU_FREE_MB", "6,8", 1);
        auto e = make(8, 1024, {0, 1}, 2, 8);
        EXPECT(e->streamed_layers() == 2, "2 layers streamed (%d)", e->streamed_layers());
        int64_t done = 0;
        e->process(ids.data(), n, noop, &done);
        e->logits(lg);
        const double r = rel_rms(lg, row(n - 1));
        EXPECT(r < 2e-4, "two devices, the last 2 layers' experts in host memory: within %.2e", r);
    }
    {   // 10. the placement itself: a card with just too little room for the experts of every layer (5.9 MiB: the first layer keeps its experts, the others
        // stream, 2 staging slots): reading and then decoding give the reference's logits
        setenv("STRATA_G4_EMU_FREE_MB", "5.9", 1);
        auto e = make(8, 1024, {0}, 0, 2);
        EXPECT(e->streamed_layers() > 0 && e->streamed_layers() < m.hp.n_layer, "a short card streams some layers, not all of them (%d of %d)", e->streamed_layers(), m.hp.n_layer);
        int64_t done = 0;
        e->process(ids.data(), n - 2, noop, &done);
        double worst = 0;
        for (int t = n - 2; t < n; ++t) {
            e->process(&ids[(size_t) t], 1, noop, &done);
            e->logits(lg);
            worst = std::max(worst, rel_rms(lg, row(t)));
        }
        EXPECT(worst < 2e-4, "placement by free memory with streamed layers: worst %.2e", worst);
    }
    std::printf(fails ? "%d FAILED\n" : "g4_emu_test: all scenarios pass\n", fails);
    return fails ? 1 : 0;
}
